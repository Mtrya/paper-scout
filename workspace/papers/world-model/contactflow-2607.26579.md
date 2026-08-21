# 2607.26579 (from arXiv HTML; MinerU fallback)



# ContactFlow: A video action conditioning that transfers across embodiments

Sami Azirar

  
Enrico Pallotta

  
Jan Nogga

  
Jürgen Gall

  
Sven Behnke

  
Hermann Blum

Affiliation: [0.5em]
University of Bonn, Lamarr Institute

###### Abstract

World models offer a promising route toward robot planning by enabling agents to imagine and verify the consequences of actions before execution. However, current video-based world models often struggle to capture the physical constraints that govern manipulation, particularly contact. Further, their action conditioning is often constrained to specific embodiments such as parallel grippers. We propose Contact Flow, an embodiment-agnostic action representation that encodes manipulation through the trajectory of 3D contact points between an actor and a target object. By discarding actor-specific appearance and kinematics, Contact Flow provides a shared conditioning signal for both human demonstrations and robotic execution. Therefore, we can train a large-scale video generative model on both human and robotic object interaction videos conditioned on Contact Flow, yielding a world model that predicts physically plausible manipulation outcomes.
We integrate this model into a propose-imagine-verify-act pipeline, where generated rollouts are assessed by a vision-language model before execution. Experiments on the DROID dataset and real-world tabletop manipulation tasks demonstrate that Contact Flow enables transfer between human demonstrations and different robotic embodiments.

Keywords: World Modeling, Action Encoding, Interaction Learning

## 1 Introduction

Teaching robots to interact with the physical world remains one of the central challenges of embodied intelligence. A promising direction is to equip robots with a world model, a learned simulator of environment dynamics, that can be used to plan and verify actions before committing to execution. Such learned simulators have been used across a range of robotic applications:
for planning and control, by rolling out candidate action sequences in imagination and selecting those that reach the goal [10, 11], for policy learning, by pretraining or distilling manipulation policies from generated visual dynamics [47, 29], for data augmentation, by synthesizing novel demonstrations and experience to improve generalization [59, 18, 28], and for evaluation and verification, by acting as an action-conditioned simulator that scores or verifies behaviors before real-world execution [60, 43].
Recent advances in video generation have made this increasingly tractable:
large-scale generative models [1, 42] can now synthesize temporally coherent, photorealistic future frames conditioned on a variety of signals [19].
Yet physical faithfulness remains elusive [4, 34]. Visually plausible videos need not
respect the mechanics of contact and manipulation, but a world model that
hallucinates physically impossible outcomes is of limited use for robot control.

We argue that one of the missing ingredients is an appropriate interaction
representation, a conditioning signal that encodes not what the scene looks
like, but how the agent plans to engage with it physically. Existing approaches
fall into two families, each with a fundamental limitation. Joint-based
representations [1, 35] are expressive but inherently embodiment-specific: a
signal defined over a particular kinematic chain cannot transfer to a robot with
a different morphology. Actor mask or silhouette representations [26, 50] avoid this to some extent but introduce a different bias: because they encode the full silhouette of
the actor, they inadvertently place emphasis on the shape of the actor (hand or end-effector) rather than on its interaction with the target object. Neither representation isolates what physically matters: the contact between the agent and the world.

This observation motivates a simple but consequential insight: physical
manipulation is governed by contact. Regardless of the actor’s morphology, an
object’s movement is only determined by where and how force is applied to it.
This can be captured as a compact, local, geometric signal that is agnostic to the embodiment producing it.
We formalize this intuition as Contact Flow: the trajectory of 3D
contact points between a hand or robotic grasp and the target object over time,
projected into image space. Contact Flow is minimal by design, it discards
everything about the actor except the locus of interaction, and it is
naturally transferable across embodiments because it is defined entirely in terms
of the object-centric contact geometry.

We use Contact Flow to condition a large-scale video generative model
(Wan [42]), training it on a heterogeneous mixture of human hand-object interaction (HOI) videos [56] and teleoperated robot demonstrations. The resulting world model generates manipulation rollouts and serves as a trajectory verifier: candidate action sequences are rendered as synthetic videos and evaluated by a vision-language model (VLM), only trajectories judged successful in simulation are passed to the robot for execution.
Crucially, because Contact Flow abstracts away the actor, the very same signal is produced at inference directly from the robot’s planned grasp on the target object: we read off the 3D contact points between the gripper and the object and project them into the camera, exactly as we extract them from human hands during training.
This enables zero-shot deployment in environments and on objects unseen during training.

We validate this pipeline on a held-out split of DROID [23], on a suite of manipulation benchmarks unseen during training, and in real-world experiments on a fixed-arm manipulator observed from an exocentric camera, performing tabletop manipulation tasks in environments unseen during training.
Our results show that a model that learns interaction from human and robot demonstrations through a single embodiment-agnostic signal, can manipulate objects it has never encountered in environments unseen during training, a capability that we attribute directly to the embodiment-agnostic nature of Contact Flow. Our contributions are:

- •

Contact Flow, a novel action encoding shared by humans and robots: a compact, embodiment-agnostic representation based on the trajectory of 3D contact points between an actor and the target object, projected into image space.
- •

An approach for conditioning a video world model on Contact Flow, which we instantiate with both the ControlNet [55] and VACE [19] control-injection mechanisms and across multiple backbone scales.
- •

A scalable data-processing pipeline that extracts high-quality Contact Flow from heterogeneous human-demonstration and robot manipulation datasets as well as for zero-shot planning at inference.

## 2 Related Work

Video World Models for Robotics
Recent work has explored generative video models as world models for robot planning, policy learning, data augmentation, and evaluation. UniPi [10] uses predicted videos as visual plans from which actions can be inferred. GR-1 [47] shows that large-scale video generative pretraining can improve language-conditioned manipulation policies.
RoboDreamer [59] learns compositional video world models for robot imagination, Dreamitate [28] transfers generated human demonstration videos to real-world visuomotor policies, and DreamGen [18] generates neural trajectories to improve generalization in robot learning. Recent works [60, 29, 43] further treat video models as action-conditioned simulators or even policy representations.
A line of systems makes trajectories or explicit search the control interface for manipulation rollouts, such as depth-encoded trajectory-to-video generation [3] and search-guided generative world models[30].
We propose an embodiment-independent action representation to simulate manipulation trajectories for different agents rather than tying the world model to a specific robot or hand.

Controllable video generation provides the mechanism through which video world models can be steered toward task-relevant futures. Existing approaches differ mainly in the form of the conditioning signal. [11, 18, 57] rely on high-level semantic control, using language to specify goals or intermediate plans. A second line of work conditions generation on action or motion signals: [60, 1] use robot end-effector states, [35] uses full-body human pose sequences. Finally, spatial control methods guide generation with image-aligned cues, including hand masks [2], hand-object mask trajectories [26], hand keypoints [56], depth maps [44], optical-flow [20, 24], and motion-trajectory prompting [13].
These works show that the control signal is central to video generation for interaction. However, most existing signals are either high-level, embodiment-specific, or tied to the actor’s visible morphology. In contrast, our work conditions generation on the contact geometry between actor and object, providing a local and embodiment-agnostic representation of the interaction.

Hand-object interaction
A large body of work studies hand-object interaction and the contact that mediates it. Several datasets and methods capture where grasps touch objects: thermal contact maps in ContactDB [5], paired hand-pose and object-contact annotations in ContactPose [6], contact-driven grasp reconstruction in ContactOpt [14], and two-hand manipulation understanding in H2O [25]. In robotics, contact geometry underpins grasp synthesis [40] and contact-rich manipulation planning [36]. These works detect, predict, or plan over contact as an analysis or control target. We use and repurpose estimated contact geometry as a generative conditioning signal.

Cross-embodiments adaptation is a key challenge for video world models. Regarding robot-to-robot transfer, Kinema4D [49] represents robot motion as 4D pointmaps encoding spatio-temporal robot occupancy, while BridgeV2W [9] uses rendered embodiment masks to guide video generation across viewpoints, scenes, and robot platforms. These approaches reduce the dependence on raw robot actions, but still focus on the geometry of the acting embodiment. Other works use object-centric or flow-based interfaces, including flow as a cross-domain manipulation interface [48], any-point trajectory modeling [46], 3D object-flow [58], and object-centric 3D motion fields [52]. These share our object-centric motivation but predict whole-object or surface motion, Contact Flow instead isolates the active contact interface that produces that motion.

## 3 Method

In the following we describe contact flow and how we use it to condition a video world model. We then describe how to extract contact flow from existing robotic and human demonstration datasets in order to train models. Finally, we explain how to build contact flow inputs at inference time.

### 3.1 Contact Flow Encoding

Contact flow describes the surface interface where an agent makes contact with an object in the scene, as well as how that contact interface moves around in the scene over time. It captures where contact happens and how that contact region moves, but it does not include the object motion that happens afterward. For example, when someone turns a door handle, contact flow captures the motion of the handle surface where the fingers press and rotate it. It does not include the door swinging open afterward once the latch has released. Thus, it perfectly captures the active dynamics while excluding the passive dynamics. This is desirable because it focuses the learning signal on the interaction itself. It avoids training on signals that already contain the final outcome in the pixels, and it is not tied to a specific robot control interface. The same contact motion should produce the same representation, independent of whether a human hand or a robot gripper created it.
Formally, at each time step $t$ we represent contact flow as a set of contact points

|  | $$ \mathbf{C}_{t}=\big\{\,\mathbf{c}_{t}^{(i)}\,\big\}_{i=1}^{N_{t}},\qquad\mathbf{c}_{t}^{(i)}=\big(x,\,y,\,z,\;\Delta x,\,\Delta y,\,\Delta z,\;w\big)\in\mathbb{R}^{7}, $$ |  | (1) |
|---|---|---|---|

where $(x,y,z)$ is the 3D position of a contact point on the object surface in the camera frame, $(\Delta x,\Delta y,\Delta z)$ is its displacement to the following frame (the local contact motion, or “flow”), and $w\in[0,1]$ is a confidence weight. Each point is projected into the image plane through the camera intrinsics $\mathbf{K}$ as $\mathbf{u}_{t}^{(i)}=\pi\big(\mathbf{K},\,(x,y,z)\big)$, and its seven attributes are written to the pixel $\mathbf{u}_{t}^{(i)}$ to form a sparse $7$-channel control frame, stacking these frames over time yields the spatiotemporal contact-flow video $\mathbf{C}_{1:T}$ that conditions the generator, rendered as a control video for the ControlNet [55] branch or encoded as a video condition for VACE [19].

The confidence $w$ combines an estimate of how certain we are that a point is genuinely in contact with two geometric consistency cues. The base term is embodiment-specific: for robot data it is derived from the spatial closeness between the gripper and the object surface, whereas for human data it is the combined HaMeR hand-pose and HACO contact confidence (Sec. 3.3). We then modulate $w$ using the local spatio-temporal neighborhood: temporal alignment up-weights points whose displacement agrees in direction with that of their neighbors, and neighborhood density down-weights isolated points. Low-confidence points contribute less to both the conditioning signal and the training loss, which makes the representation robust to the noisy per-frame contact estimates produced by the extraction pipelines of Sec. 3.3.

### 3.2 CF World Model

We instantiate our Contact Flow world model with a latent video diffusion transformer following recent DiT-based video generators [42].
Given an input video $\mathbf{x}_{0:T}=\{x_{0},x_{1},\ldots,x_{T}\}$, we encode frames into a latent sequence $\mathbf{z}_{0:T}=\mathcal{E}(\mathbf{x}_{0:T})$, where $\mathcal{E}$ denotes the frozen video VAE encoder.
The model is conditioned on the first observed frame $x_{0}$ and our contact flow signal $\mathbf{C}_{1:T}$, which specifies the intended interaction over the prediction horizon, learning a conditional generative model

|  | $$ p_{\theta}(\mathbf{z}_{1:T}\mid z_{0},\mathbf{C}_{1:T}). $$ |  | (2) |
|---|---|---|---|

Training is performed with a flow-matching objective in latent space. We write the clean target latent sequence as $\mathbf{z}_{1:T}\sim q(\mathbf{z}_{1:T}\mid x_{1:T})$ and sample noise $\boldsymbol{\epsilon}\sim\mathcal{N}(0,I)$ of the same shape, the first-frame latent $z_{0}$ is kept clean as conditioning and is not noised.
For a sampled time $\tau\sim\mathcal{U}(0,1)$, we form the interpolant $\mathbf{u}_{\tau}=(1-\tau)\,\boldsymbol{\epsilon}+\tau\,\mathbf{z}_{1:T},$ and train the DiT denoising network $v_{\theta}$ to predict the target velocity

|  | $$ \mathcal{L}_{\mathrm{FM}}=\mathbb{E}_{\mathbf{z}_{1:T},\boldsymbol{\epsilon},\tau}\left[\left\|v_{\theta}(\mathbf{u}_{\tau},\tau,z_{0},\mathbf{C}_{1:T})-(\mathbf{z}_{1:T}-\boldsymbol{\epsilon})\right\|_{2}^{2}\right]. $$ |  | (3) |
|---|---|---|---|---|---|

At inference time, the model starts from noise in latent space and integrates the learned velocity field conditioned on $z_{0}$ and $\mathbf{C}_{1:T}$ to generate future latents $\hat{\mathbf{z}}_{1:T}$, which are decoded into predicted frames $\hat{\mathbf{x}}_{1:T}=\mathcal{D}(\hat{\mathbf{z}}_{1:T})$.
The goal is to condition future prediction on where and how the actor interacts with the object, rather than on the actor’s appearance, hand shape, robot morphology, or embodiment-specific action space.

Control injection in diffusion-based generative models was popularized by ControlNet [55], which introduces a trainable copy of the encoder blocks of a frozen U-Net, connected via zero-initialized convolution layers.
Conditioning signals such as depth maps, pose skeletons, or edge maps are fed through this parallel branch, whose outputs are added back to the main network, enabling fine-grained structural control without modifying the pretrained backbone.
More recently, VACE [19], introduces a Video Condition Unit (VCU) that encodes control signals, reference frames, and binary spatiotemporal masks into a unified token representation, which is injected into a frozen Diffusion Transformer via lightweight Context Adapter blocks distributed across the network layers, enabling simultaneous and compositional control over both spatial structure and temporal dynamics.

In this work, we test our Contact Flow representation with both conditioning mechanisms to verify that its effectiveness is not tied to a specific control architecture. In the ControlNet-based variant, Contact Flow is rendered as a spatiotemporal control video and processed by the trainable control branch, which injects contact-aware features into the frozen video generation backbone. In the VACE-based variant, the same Contact Flow signal is encoded as an additional video condition and injected through the VCU and Context Adapters. Both variants use the same underlying representation: a sparse, object-centric trace of contact points projected into the image plane over time.

### 3.3 Data Processing

We source training data from both human manipulation videos and teleoperated robot episodes.
Human data provides scale and diversity of contact-rich interactions, while robot data offer metrically-accurate, embodiment-specific signals. Both streams are processed to recover the same downstream targets: object masks, hand/gripper pose, dense pointmaps, and contact regions.
For a video sequence $\{x_{t}\}_{t=0}^{T}$ of RGB frames $x_{t}$, we first acquire corresponding pointmaps $P_{t}$. Depending on the available sensing modalities, we recover these pointmaps either from stereo pairs, estimating metric depth with FoundationStereo [45] and unprojecting it with the calibrated intrinsics, or directly from the RGB frames, optionally guided by noisy depth measurements, with MapAnything [22].

#### 3.3.1 Contact Flow from Human Demonstration

Hand Mesh. If not available in the dataset already, we first estimate hand poses in every frame using HaMeR [37]. We identify occluded hand keypoints by their assigned confidence. For visible keypoints, we infer their 3D coordinates from the 2D output of HaMeR and look up the corresponding 3D points in the pointmap, because we find HaMeR’s direct 3D estimation not consistent enough with the pointmap estimation. We then place the MANO model [39] into the scene based on the estimated hand pose and shape parameters and fit it with a least-squares regression to the 3D coordinates of the visible keypoints.
Object Mask. We segment the manipulated object in each frame by prompting SAM3 [7] with the target object’s name. How that name is obtained depends on the cues the dataset provides: when a language description of the task is available, we extract candidate object nouns with spaCy [16], when it is not, we instead query a vision-language model (Gemini [41]) on the first frame $I_{0}$ to name the object being manipulated. If the dataset already ships object masks or meshes, we use these directly and skip detection. The resulting per-frame masks index into the pointmaps $P_{t}$ to recover the object’s point cloud and provide the object support used during contact estimation.
Contact Estimation. We use HACO [21] to estimate which parts of the hand mesh make contact with the object. It is only trained on right hands, which is why we feed it a left-right flip of our observations for left hands. We find that it is reliable at predicting which fingers etc. make contact, but it is not reliable in predicting whether the hand makes contact at all, also predicting random contact if the hand is meters away from any object. Therefore, we filter its output with a binary estimation of whether the hand is in contact with the object or not. For this, we measure whether any hand point is closer than $\delta_{\textrm{contact dist}}$ from the object, as well as a non-empty 2D overlap between the (dilated) hand mask and the object mask in the image. Contact for a single frame is estimated if both criteria are met. We further apply a temporal smoothing filter over this per-frame estimation.
Dataset Specifics. With the above steps, we process TasteRob [56], TACO [33], and OakInk [54]. For TasteRob, we interpolate their provided object masks and depth to full frame rate and inpaint the holes in the depth map through nearest-neighbors. For TACO and OakInk, we process every camera view as a separate sample and take the provided hand and object meshes. We still run an additional quality check by passing the task annotation to GroundingDINO [32], verifying that the predicted bounding box covers the reprojected object mesh. We skip samples where this check fails.

#### 3.3.2 Contact Flow from Robotic Data

For robotic data we in general assume a calibrated, static, exocentric (i.e. looking at the arm, but not moving with the arm) camera, a URDF, and corresponding state vector available at every timestamp. Therefore, we can take a more geometric approach at contact estimation.
Segmentation. Given a text description of the task in the video, we enumerate candidate object nouns through spaCy [16], and estimate 3D bounding boxes for each using WildDet3D [17], with ground-truth depth when available. From these candidates, we identify the object that the robot actually interacts with by checking for which candidate the robot’s tool center point enters the 3D bounding box. We then take the pointmap $P_{0}$ and crop it with the 3D bounding box to retrieve the object’s pointcloud. In addition, we prompt SAM3 [7] with the selected object noun and $I_{0}$ to get a 2D object mask in the exocentric camera image. While in theory an accurate mask of the robot in the camera view can always be derived from the robot-to-camera calibration, robot state, and URDF rendering, we find that in practice small errors along this chain can easily lead to a few pixels offset. Therefore, we fine-tune a SAM3 model on the robot using the RoboEngine dataset [53] as well as 1000 hand-annotated images.
Contact Estimation. For contact, we distinguish robotic fingers that are between the camera and the object, and those that are, seen from the camera, behind the object. From the URDF and object pointcloud we know this distinction for each finger. For the rest of this paragraph, we assume a parallel gripper with one front and one back finger, but the method applies to any number of fingers.
For the front finger, we derive the overlap between the object mask and the robot mask and filter $P_{0}$ to this overlap region. The resulting points are points on the object that are in contact with the robot finger.
Since the back finger is not directly visible and also the object geometry is often incomplete on the backside, we have to take a different approach: We render the URDF into the image, if necessary adjusting its reprojection to the detected robot mask. We then take the rendering of the back finger and again check the overlap region with the object mask. The contact points are then taken from the rendering depth of the back finger within this overlap. Finally, we concatenate both sets to get the contact points for the given frame. For frames after gripper closing, we assume a rigid transform between the robot wrist and the last contact points, until the gripper is opened again. Point correspondences (“flow”) across frames are smoothed with a Hough-based
filter over a 3-frame window, which we empirically found to be the best operating point.We discard frames where the rendered robot and gripper disagree with the detected robot mask, and additionally filter with RobotInter [27], an interaction-prediction model that is not reliable enough to serve as a contact signal alone but lets us drop samples whose predicted interaction grossly disagrees with our estimate.

#### 3.3.3 Contact Flow at Inference

At inference we require only an end-effector trajectory together with a recovered model of the real-world scene, from the gripper geometry alone we synthesise the anticipated contact flow, so the conditioning signal is agnostic to whatever policy produced the trajectory.
To populate the scene we need a metrically-accurate 3D model and pose of the target object together with a camera-frame pointmap: we assume access to at least one external stereo camera, to which we apply FoundationStereo [45].
As in training, the task is passed to a VLM (Gemini [41]) that reduces it to an atomic instruction and proposes a bounding box for the target object, iteratively refining the box and optionally falling back to GroundingDINO [32], and the selected object noun then prompts SAM3 [7] for the object mask. We recover the object’s 3D geometry by running SAM 3D-Objects [8] on the initial frame conditioned on the pointmap, as this typically leaves a pose error of several centimetres, we apply a staged refinement consisting of a closed-form PCA alignment with sign disambiguation followed by trimmed ICP against the observed object point cloud (compensating for the half-shell bias between the fully generated splat and the single-view visible surface), a differentiable render-and-compare step that optimises the global rigid pose and uniform scale of the Gaussian splat through gsplat [51] differentiable rasterisation by minimising a composite loss of soft mask IoU, masked depth MSE, and per-pixel RGB cosine similarity, and a final per-Gaussian opacity pruning and colour-refinement step that sharpens silhouette and appearance, where each stage is accepted only if it improves all three strict render-space gatesmask IoU, depth inlier fraction, and RGB cosine similarity ($\geq 0.95$)ensuring the final splat faithfully reproduces the observed geometry and appearance under the calibrated camera. Finally, given the recovered scene, we apply the end-effector trajectory to the camera-frame pointmap through the gripper URDF, yielding the anticipated contact flow exactly as in our robotic-data extraction, and feed this contact flow to the video world model, which predicts how the real scene evolves under the proposed trajectory.

## 4 Experiments

We investigate two questions: (Q1) Is Contact Flow an effective conditioning
signal, producing accurate, coherent predictions of robot manipulation? (Q2) Does the
world model predict real-world outcomes well enough to act as a zero-shot verifier of
proposed trajectories in unseen scenes?

### 4.1 Experimental Setup

##### Training data.

We train on DROID, [23] Taste-ROB [56], TACO [33], OakInk [54], and LIBERO [31]. We process this mix of human and robotic datasets as described in Section 3.

##### Robot platform.

Real-robot experiments are conducted on a Franka Panda fixed-arm manipulator with a single exocentric RGBD camera.
At deployment, we recover the scene once and instantiate the symbolic twin
(Sec. 4) in a single offline pass.
From the exocentric stereo pair we estimate metric depth with
FoundationStereo [45], prompt SAM3 [7]
for the object mask, and fit a metric object mesh posed directly in the base
frame, refining its rigid pose and uniform scale until it is metrically
consistent with the observation through the differentiable procedure of
Sec. 3.3. With the twin in place, a $\pi_{0.5}$ [38]
vision-language-action policy rolls out a candidate end-effector trajectory
inside the twin. We convert it to the anticipated contact flow, render
the predicted rollout with our contact-flow video world model, and a VLM
(Gemini [41]) judges whether the task is solved.
Only then is the trajectory executed open-loop on the real robot.

##### Baselines and Metrics

TesserAct [57] is
language-conditioned, taking only the task description, with no geometric control toward a
specific trajectory. CTRL-World [15] conditions on low-dimensional action
embeddings, which are embodiment-specific and tied to the training robot’s action space,
hence not zero-shot. Kinema4D [49] conditions on a 4D pointmap of
the full robot, encoding the whole actor geometry rather than the contact locus alone.
Like ours it is zero-shot, and is our primary competitor. To test robustness independently
of backbone and control mechanism, we evaluate three configurations, Wan2.1-14B + VACE,
Wan2.2-5B + ControlNet, and Wan2.2-14B + ControlNet. Since Contact Flow conditions only on the object-side interface and never on the actor, we
compute the alignment metrics (PSNR, SSIM, LPIPS, DreamSim [12]) on the
agent-masked region, removing the agent (robot arm or human hand) from both prediction and
ground truth, with the same mask applied to the training loss. Masks come from RoboSeg
(robot) and SAM2 (human). FID and FVD are full-frame distribution metrics. All results use
$25$ held-out clips per dataset, each a $49$-frame window at $8$ FPS and $832\times 480$.
DreamSim is our primary metricFor Q2 our primary measure is prediction accuracy, whether the VLM forecast on the
imagined rollout matches real-robot execution for each trajectory $\pi_{0.5}$ proposes. A
trajectory succeeds only if the robot completes the task without unwanted
consequences (e.g. knocking over or dropping objects), i.e. the target is grasped and
placed on the designated region within the episode.

*Table 1: Evaluation on DROID.
All metrics are computed over $25$ held-out DROID clips after masking out the visible robot (via RoboSeg) in both the prediction and the ground truth. Arrows indicate the preferred direction of each metric.*

| Approach | Action Encoding | Backbone | DreamSim$\downarrow$ | FID$\downarrow$ | FVD$\downarrow$ | PSNR$\uparrow$ | SSIM$\uparrow$ | LPIPS$\downarrow$ |
|---|---|---|---|---|---|---|---|---|
| CF + VACE | 7ch | Wan 2.1 14B | 0.039 | 75.17 | 0.12 | 23.37 | 0.884 | 0.160 |
| CF + CTRL-Net | 7ch | Wan 2.2 5B | 0.036 | 40.41 | 0.03 | 23.28 | 0.904 | 0.165 |
| CF + CTRL-Net | 7ch | Wan 2.2 14B | 0.035 | 74.05 | 0.13 | 24.20 | 0.896 | 0.161 |
| Kinema4D | Robot Pointmap | Wan 2.1 14B 4DNeX | 0.043 | 67.06 | 0.08 | 20.17 | 0.830 | 0.220 |
| CTRL-World | Action Emb. | SVD 1.5B | 0.059 | 72.90 | 0.07 | 22.44 | 0.800 | 0.090 |
| TesserAct | Language Cond. | CogVideo-X 5B | 0.106 | 164.54 | 0.34 | 13.96 | 0.615 | 0.396 |

![Refer to caption](drafts/images/contactflow-2607.26579/sim_robomme_crop.jpg)

![Refer to caption](drafts/images/contactflow-2607.26579/rollout_droid_14b.jpg)

![Refer to caption](drafts/images/contactflow-2607.26579/rollout_droid_14b.jpg)

![Refer to caption](drafts/images/contactflow-2607.26579/rollout_droid_14b.jpg)

![Refer to caption](drafts/images/contactflow-2607.26579/rollout_droid_14b.jpg)

![Refer to caption](drafts/images/contactflow-2607.26579/rollout_droid_14b.jpg)

![Refer to caption](drafts/images/contactflow-2607.26579/rollout_droid_14b.jpg)

![Refer to caption](drafts/images/contactflow-2607.26579/rollout_ctrlnet14b_scen3.jpg)

![Refer to caption](drafts/images/contactflow-2607.26579/rollout_ctrlnet14b_scen3.jpg)

![Refer to caption](drafts/images/contactflow-2607.26579/rollout_ctrlnet14b_scen3.jpg)

![Refer to caption](drafts/images/contactflow-2607.26579/rollout_ctrlnet14b_scen3.jpg)

![Refer to caption](drafts/images/contactflow-2607.26579/rollout_ctrlnet14b_scen3.jpg)

![Refer to caption](drafts/images/contactflow-2607.26579/rollout_ctrlnet14b_scen3.jpg)

*Figure 1: Closed-loop deployment with Contact Flow.
On the left, a single RGBD view instantiates a symbolic twin simulator in which a
$\pi_{0.5}$ policy proposes an end-effector trajectory. On the right, our world model
imagines the rollout conditioned on the initial frame and Contact Flow from that trajectory,
showing the real rollout above and the simulated rollout below.*

##### Results

(Q1) We first evaluate Contact Flow as a conditioning signal for video prediction. On
held-out DROID scenes, each model receives the initial frame together with the
Contact Flow derived from the ground-truth end-effector trajectory and must
generate the full future video, which we score against the real recording
(Tab. 1). This measures prediction quality on held-out
scenes drawn from the training distribution, we probe generalisation to
entirely unseen embodiments, scenes, and rendering styles in the
cross-dataset evaluation below. We sample
these DROID frames at one third of the native rate, a stride of three, so a
single $49$-frame rollout forecasts roughly three times further into the future,
on the order of ten seconds of real interaction. We then extend
this evaluation beyond DROID to held-out splits of our human and robot training
data and to four datasets that are entirely unseen during training
(Tab. 2), spanning new embodiments,
scenes, and rendering styles, which probes whether the same contact-flow interface
transfers across domains without any per-dataset adaptation. (Q2) We deploy the full pipeline on a Franka Panda across $10$ unseen tabletop pick-and-place scenarios. With the Wan 2.2 14B backbone, the world model forecasts the real-robot outcome correctly in $8/10$ cases. This verification step is what makes deployment work. Deploying the $\pi_{0.5}$ model directly into the real world dooes not enable any successful run despite it being a state-of-the-art VLA. The minimal twin lets the policy propose a motion, but it is too crude to certify that motion.

(a) Robot manipulation

Benchmark
Model
DreamSim$\downarrow$
PSNR$\uparrow$
SSIM$\uparrow$
LPIPS$\downarrow$
FID$\downarrow$
FVD$\downarrow$

RLBench ood
Kinema4D
0.051
26.11
0.872
0.176
81.79
0.12

5B
0.044
25.30
0.821
0.160
124.96
0.15

14B (mix)
0.040
25.44
0.826
0.164
106.66
0.14

14B (DROID)
0.043
27.53
0.844
0.142
127.58
0.18

14B (human)
0.061
25.23
0.817
0.175
126.16
0.18

AgiBot ood
Kinema4D
0.075
20.79
0.839
0.226
101.69
0.37

5B
0.066
21.10
0.770
0.261
107.51
0.44

14B (mix)
0.054
20.29
0.772
0.264
91.83
0.35

14B (DROID)
0.082
19.97
0.801
0.233
107.92
0.44

14B (human)
0.084
18.70
0.689
0.346
108.03
0.39

GenieSim ood
Kinema4D
0.085
21.21
0.858
0.253
92.99
0.49

5B
0.050
24.73
0.876
0.202
100.03
0.48

14B (mix)
0.063
22.45
0.856
0.222
101.10
0.44

14B (DROID)
0.132
20.61
0.809
0.278
131.51
0.50

14B (human)
0.075
22.31
0.823
0.271
108.15
0.45

(b) Human-hand manipulation

Benchmark
Model
DreamSim$\downarrow$
PSNR$\uparrow$
SSIM$\uparrow$
LPIPS$\downarrow$
FID$\downarrow$
FVD$\downarrow$

TACO
Kinema4D
0.070
23.18
0.803
0.243
88.29
0.29

5B
0.027
27.44
0.852
0.174
44.47
0.20

14B (mix)
0.024
27.22
0.857
0.168
39.50
0.20

14B (DROID)
0.067
24.22
0.819
0.205
75.14
0.31

14B (human)
0.021
28.28
0.869
0.158
36.46
0.18

TASTE-Rob
Kinema4D
0.107
20.68
0.783
0.221
175.95
0.52

5B
0.016
30.30
0.882
0.134
36.90
0.17

14B (mix)
0.019
29.11
0.888
0.134
36.34
0.21

14B (DROID)
0.062
22.75
0.815
0.185
87.79
0.49

14B (human)
0.016
30.66
0.899
0.124
32.62
0.20

OakInk
Kinema4D
0.074
22.08
0.797
0.266
105.66
0.35

5B
0.040
25.96
0.839
0.217
75.35
0.35

14B (mix)
0.032
26.09
0.852
0.207
58.10
0.33

14B (DROID)
0.060
23.05
0.807
0.261
94.87
0.41

14B (human)
0.029
26.77
0.854
0.199
57.10
0.33

EgoDex† ood
Kinema4D
0.125
22.68
0.823
0.358
127.76
0.31

5B
0.127
22.11
0.818
0.372
128.33
0.36

14B (mix)
0.138
20.79
0.804
0.436
116.33
0.40

14B (DROID)
0.218
19.79
0.770
0.410
167.14
0.43

14B (human)
0.123
22.04
0.818
0.382
109.11
0.35

*Table 2: Cross-dataset evaluation.
All metrics are computed over $25$ held-out clips per dataset after masking out the visible hand/robot in both the prediction and the ground truth.*

We deploy the full pipeline on a Franka Panda across $10$ unseen tabletop pick-and-place scenarios. With the Wan 2.2 14B backbone, the world model forecasts the real-robot outcome correctly in $8/10$ cases. This verification step is what makes deployment work. Deploying the $\pi_{0.5}$ model directly into the real world dooes not enable any successful run despite it being a state-of-the-art VLA. The minimal twin lets the policy propose a motion, but it is too crude to certify that motion.

## 5 Limitations & Conclusion

We introduced Contact Flow, an embodiment-agnostic action representation encoding manipulation as the trajectory of 3D contact points between actor and object. Conditioned on this signal, a single world model trained on mixed human and robot data predicts plausible outcomes and serves as a zero-shot verifier, transferring across embodiments, scenes, and objects unseen at training.
The world model captures interaction outcomes more faithfully than the underlying physics, which suffices for verification but could be made more physically grounded for contact-rich tasks.
The inference cost is still far away from real-time capabilities.
In addition, contact quality depends on upstream calibration, metric depth, and the SAM3 masks, so the method assumes a calibrated exocentric camera with reliable depth.

#### Acknowledgments

If a paper is accepted, the final camera-ready version will (and probably should) include acknowledgments. All acknowledgments go at the end of the paper, including thanks to reviewers who gave useful comments, to colleagues who contributed to the ideas, and to funding agencies and corporate sponsors that provided financial support.

## References

- [1]
N. Agarwal, A. Ali, M. Bala, Y. Balaji, E. Barker, T. Cai, P. Chattopadhyay, Y. Chen, Y. Cui, Y. Ding, et al. (2025)

Cosmos world foundation model platform for physical ai.

arXiv preprint arXiv:2501.03575.

Cited by: §1,
§1,
§2.
- [2]
R. Akkerman, H. Feng, M. J. Black, D. Tzionas, and V. F. Abrevaya (2025)

Interdyn: controllable interactive dynamics with video diffusion models.

In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition,

pp. 12467–12479.

Cited by: §2.
- [3]
Y. Bai, L. Yang, G. Eskandar, F. Shen, M. Altillawi, Z. Liu, and G. Kutyniok (2025)

DRAW2ACT: turning depth-encoded trajectories into robotic demonstration videos.

arXiv preprint arXiv:2512.14217.

Cited by: §2.
- [4]
H. Bansal, Z. Lin, T. Xie, Z. Zong, M. Yarom, Y. Bitton, C. Jiang, Y. Sun, K. Chang, and A. Grover (2025)

Videophy: evaluating physical commonsense for video generation.

In International Conference on Learning Representations,

Vol. 2025, pp. 102075–102121.

Cited by: §1.
- [5]
S. Brahmbhatt, C. Ham, C. C. Kemp, and J. Hays (2019)

ContactDB: analyzing and predicting grasp contact via thermal imaging.

In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),

pp. 8709–8719.

Cited by: §2.
- [6]
S. Brahmbhatt, C. Tang, C. D. Twigg, C. C. Kemp, and J. Hays (2020)

ContactPose: a dataset of grasps with object contact and hand pose.

In Computer Vision – ECCV 2020,

pp. 361–378.

Cited by: §2.
- [7]
N. Carion, L. Gustafson, Y. Hu, S. Debnath, R. Hu, D. Suris, et al. (2025)

SAM 3: segment anything with concepts.

External Links: 2511.16719

Cited by: §3.3.1,
§3.3.2,
§3.3.3,
§4.1.
- [8]
X. Chen, F. Chu, P. Gleize, K. J. Liang, A. Sax, H. Tang, W. Wang, M. Guo, T. Hardin, X. Li, et al. (2025)

Sam 3d: 3dfy anything in images.

arXiv preprint arXiv:2511.16624.

Cited by: §3.3.3.
- [9]
Y. Chen, P. Li, J. Yang, K. He, X. Wu, Y. Xu, K. Wang, J. Liu, N. Liu, Y. Huang, et al. (2026)

BridgeV2W: bridging video generation models to embodied world models via embodiment masks.

arXiv preprint arXiv:2602.03793.

Cited by: §2.
- [10]
Y. Du, S. Yang, B. Dai, H. Dai, O. Nachum, J. Tenenbaum, D. Schuurmans, and P. Abbeel (2023)

Learning universal policies via text-guided video generation.

In Advances in Neural Information Processing Systems,

Vol. 36.

Cited by: §1,
§2.
- [11]
Y. Du, S. Yang, P. Florence, F. Xia, A. Wahid, P. Sermanet, T. Yu, P. Abbeel, J. B. Tenenbaum, L. Kaelbling, et al. (2024)

Video language planning.

In International Conference on Learning Representations,

Vol. 2024, pp. 31138–31155.

Cited by: §1,
§2.
- [12]
S. Fu, N. Tamir, S. Sundaram, L. Chai, R. Zhang, T. Dekel, and P. Isola (2023)

Dreamsim: learning new dimensions of human visual similarity using synthetic data.

arXiv preprint arXiv:2306.09344.

Cited by: §4.1.
- [13]
D. Geng, C. Herrmann, J. Hur, F. Cole, S. Zhang, T. Pfaff, T. Lopez-Guevara, Y. Aytar, M. Rubinstein, C. Sun, O. Wang, A. Owens, and D. Sun (2025)

Motion prompting: controlling video generation with motion trajectories.

In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),

Cited by: §2.
- [14]
P. Grady, C. Tang, C. D. Twigg, M. Vo, S. Brahmbhatt, and C. C. Kemp (2021)

ContactOpt: optimizing contact to improve grasps.

In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),

pp. 1471–1481.

Cited by: §2.
- [15]
Y. Guo, L. X. Shi, J. Chen, and C. Finn (2025)

Ctrl-world: a controllable generative world model for robot manipulation.

arXiv preprint arXiv:2510.10125.

Cited by: §4.1.
- [16]
M. Honnibal, I. Montani, S. Van Landeghem, and A. Boyd (2020)

spaCy: industrial-strength natural language processing in python.

 Zenodo.

External Links: [Document](https://dx.doi.org/10.5281/zenodo.1212303)

Cited by: §3.3.1,
§3.3.2.
- [17]
W. Huang, J. Zhang, S. Li, T. Jia, J. Duan, Y. Cheng, J. Cho, M. Wallingford, R. Soraki, C. D. Kim, S. Liu, D. Clay, T. Anderson, W. Han, A. Farhadi, B. Hariharan, Z. Ren, and R. Krishna (2026)

WildDet3D: scaling promptable 3D detection in the wild.

External Links: 2604.08626

Cited by: §3.3.2.
- [18]
J. Jang, S. Ye, Z. Lin, J. Xiang, J. Bjorck, Y. Fang, F. Hu, S. Huang, K. Kundalia, Y. Lin, L. Magne, A. Mandlekar, A. Narayan, Y. L. Tan, G. Wang, J. Wang, Q. Wang, Y. Xu, X. Zeng, K. Zheng, R. Zheng, M. Liu, L. Zettlemoyer, D. Fox, J. Kautz, S. Reed, Y. Zhu, and L. Fan (2025)

DreamGen: unlocking generalization in robot learning through video world models.

In Proceedings of The 9th Conference on Robot Learning,

Proceedings of Machine Learning Research, Vol. 305, pp. 5170–5194.

External Links: [Link](https://proceedings.mlr.press/v305/jang25a.html)

Cited by: §1,
§2,
§2.
- [19]
Z. Jiang, Z. Han, C. Mao, J. Zhang, Y. Pan, and Y. Liu (2025)

Vace: all-in-one video creation and editing.

In Proceedings of the IEEE/CVF International Conference on Computer Vision,

pp. 17191–17202.

Cited by: 2nd item,
§1,
§3.1,
§3.2.
- [20]
W. Jin, Q. Dai, C. Luo, S. Baek, and S. Cho (2025)

Flovd: optical flow meets video diffusion model for enhanced camera-controlled video synthesis.

In Proceedings of the Computer Vision and Pattern Recognition Conference,

pp. 2040–2049.

Cited by: §2.
- [21]
D. S. Jung and K. M. Lee (2025)

Learning dense hand contact estimation from imbalanced data.

Advances in Neural Information Processing Systems.

External Links: 2505.11152

Cited by: §3.3.1.
- [22]
N. Keetha, N. Müller, J. Schönberger, L. Porzi, Y. Zhang, T. Fischer, A. Knapitsch, D. Zauss, E. Weber, J. Luiten, M. Lopez-Antequera, S. Rota Bulò, C. Richardt, D. Ramanan, S. Scherer, and P. Kontschieder (2026)

MapAnything: universal feed-forward metric 3D reconstruction.

In International Conference on 3D Vision (3DV),

Cited by: §3.3.
- [23]
A. Khazatsky, K. Pertsch, S. Nair, A. Balakrishna, S. Dasari, S. Karamcheti, S. Nasiriany, M. K. Srirama, L. Y. Chen, K. Ellis, et al. (2024)

DROID: a large-scale in-the-wild robot manipulation dataset.

In Robotics: Science and Systems,

Cited by: §1,
§4.1.
- [24]
M. Koroglu, H. Caselles-Dupré, G. Jeanneret, and M. Cord (2025)

Onlyflow: optical flow based motion conditioning for video diffusion models.

In Proceedings of the Computer Vision and Pattern Recognition Conference,

pp. 6226–6236.

Cited by: §2.
- [25]
T. Kwon, B. Tekin, J. Stühmer, F. Bogo, and M. Pollefeys (2021)

H2O: two hands manipulating objects for first person interaction recognition.

In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV),

pp. 10138–10148.

Cited by: §2.
- [26]
G. Li, B. Zhao, J. Yang, and L. Sevilla-Lara (2026)

Mask2iv: interaction-centric video generation via mask trajectories.

In Proceedings of the AAAI Conference on Artificial Intelligence,

Vol. 40, pp. 6091–6099.

Cited by: §1,
§2.
- [27]
H. Li, Z. Wang, Z. Ding, S. Yang, Y. Chen, Y. Tian, X. Hu, T. Wang, D. Lin, F. Zhao, et al. (2026)

RoboInter: a holistic intermediate representation suite towards robotic manipulation.

In The Fourteenth International Conference on Learning Representations,

Cited by: §3.3.2.
- [28]
J. Liang, R. Liu, E. Ozguroglu, S. Sudhakar, A. Dave, P. Tokmakov, S. Song, and C. Vondrick (2025)

Dreamitate: real-world visuomotor policy learning via video generation.

In Proceedings of The 8th Conference on Robot Learning,

Proceedings of Machine Learning Research, Vol. 270, pp. 3943–3960.

External Links: [Link](https://proceedings.mlr.press/v270/liang25b.html)

Cited by: §1,
§2.
- [29]
J. Liang, P. Tokmakov, R. Liu, S. Sudhakar, P. Shah, R. Ambrus, and C. Vondrick (2025)

Video generators are robot policies.

arXiv preprint arXiv:2508.00795.

External Links: 2508.00795,
[Document](https://dx.doi.org/10.48550/arXiv.2508.00795)

Cited by: §1,
§2.
- [30]
W. Lin, J. Zhang, K. Cai, and K. Wang (2025)

STORM: search-guided generative world models for robotic manipulation.

arXiv preprint arXiv:2512.18477.

Cited by: §2.
- [31]
B. Liu, Y. Zhu, C. Gao, Y. Feng, Q. Liu, Y. Zhu, and P. Stone (2023)

LIBERO: benchmarking knowledge transfer for lifelong robot learning.

In Advances in Neural Information Processing Systems,

Vol. 36, pp. 44776–44791.

Cited by: §4.1.
- [32]
S. Liu, Z. Zeng, T. Ren, F. Li, H. Zhang, J. Yang, Q. Jiang, C. Li, J. Yang, H. Su, J. Zhu, and L. Zhang (2024)

Grounding DINO: marrying DINO with grounded pre-training for open-set object detection.

In Computer Vision – ECCV 2024,

pp. 38–55.

Cited by: §3.3.1,
§3.3.3.
- [33]
Y. Liu, H. Yang, X. Si, L. Liu, Z. Li, Y. Zhang, Y. Liu, and L. Yi (2024)

TACO: benchmarking generalizable bimanual tool-ACtion-object understanding.

In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),

pp. 21740–21751.

Cited by: §3.3.1,
§4.1.
- [34]
S. Motamed, L. Culp, K. Swersky, P. Jaini, and R. Geirhos (2026)

Do generative video models understand physical principles?.

In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision,

pp. 948–958.

Cited by: §1.
- [35]
E. Pallotta, S. M. Azar, L. Doorenbos, S. Ozsoy, U. Iqbal, and J. Gall (2026)

EgoControl: controllable egocentric video generation via 3d full-body poses.

In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),

pp. 4269–4279.

Cited by: §1,
§2.
- [36]
T. Pang, H. J. T. Suh, L. Yang, and R. Tedrake (2023)

Global planning for contact-rich manipulation via local smoothing of quasi-dynamic contact models.

IEEE Transactions on Robotics 39 (6), pp. 4691–4711.

Cited by: §2.
- [37]
G. Pavlakos, D. Shan, I. Radosavovic, A. Kanazawa, D. Fouhey, and J. Malik (2024)

Reconstructing hands in 3d with transformers.

In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition,

pp. 9826–9836.

Cited by: §3.3.1.
- [38]
Physical Intelligence, K. Black, N. Brown, J. Darpinian, K. Dhabalia, D. Driess, A. Esmail, M. Equi, C. Finn, K. Hausman, B. Ichter, S. Levine, K. Pertsch, Q. Vuong, H. Walke, et al. (2025)

$\pi_{0.5}$: a vision-language-action model with open-world generalization.

arXiv preprint arXiv:2504.16054.

Cited by: §4.1.
- [39]
J. Romero, D. Tzionas, and M. J. Black (2017)

Embodied hands: modeling and capturing hands and bodies together.

ACM Transactions on Graphics (Proc. SIGGRAPH Asia) 36 (6), pp. 245:1–245:17.

Cited by: §3.3.1.
- [40]
M. Sundermeyer, A. Mousavian, R. Triebel, and D. Fox (2021)

Contact-graspnet: efficient 6-DoF grasp generation in cluttered scenes.

In IEEE International Conference on Robotics and Automation (ICRA),

pp. 13438–13444.

Cited by: §2.
- [41]
G. Team, R. Anil, S. Borgeaud, J. Alayrac, J. Yu, R. Soricut, J. Schalkwyk, A. M. Dai, A. Hauth, K. Millican, et al. (2023)

Gemini: a family of highly capable multimodal models.

arXiv preprint arXiv:2312.11805.

Cited by: §3.3.1,
§3.3.3,
§4.1.
- [42]
A. Wang, B. Ai, B. Wen, C. Mao, C. Xie, D. Chen, F. Yu, H. Zhao, J. Yang, J. Zeng, et al. (2025)

Wan: open and advanced large-scale video generative models.

arXiv preprint arXiv:2503.20314.

Cited by: §1,
§1,
§3.2.
- [43]
Y. Wang, R. Syed, F. Wu, M. Zhang, A. Onol, J. Barreiros, H. Nayyeri, T. Dear, H. Zhang, and Y. Li (2026)

Interactive world simulator for robot policy training and evaluation.

In Robotics: Science and Systems,

External Links: [Link](https://www.yixuanwang.me/interactive_world_sim/)

Cited by: §1,
§2.
- [44]
Y. Wang, C. Wen, H. Guo, S. Peng, M. Qin, H. Bao, X. Zhou, and R. Hu (2025)

Precise action-to-video generation through visual action prompts.

In Proceedings of the IEEE/CVF International Conference on Computer Vision,

pp. 12713–12724.

Cited by: §2.
- [45]
B. Wen, M. Trepte, J. Aribido, J. Kautz, O. Gallo, and S. Birchfield (2025)

FoundationStereo: zero-shot stereo matching.

In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),

pp. 5249–5260.

Cited by: §3.3.3,
§3.3,
§4.1.
- [46]
C. Wen, X. Lin, J. So, K. Chen, Q. Dou, Y. Gao, and P. Abbeel (2024)

Any-point trajectory modeling for policy learning.

In Robotics: Science and Systems (RSS),

Cited by: §2.
- [47]
H. Wu, Y. Jing, C. Cheang, G. Chen, J. Xu, X. Li, M. Liu, H. Li, and T. Kong (2024)

Unleashing large-scale video generative pre-training for visual robot manipulation.

In International Conference on Learning Representations,

External Links: [Link](https://openreview.net/forum?id=NxoFmGgWC9)

Cited by: §1,
§2.
- [48]
M. Xu, Z. Xu, Y. Xu, C. Chi, G. Wetzstein, M. Veloso, and S. Song (2024)

Flow as the cross-domain manipulation interface.

In 8th Annual Conference on Robot Learning (CoRL),

Cited by: §2.
- [49]
M. Xu, T. Zhang, T. Liu, Z. Chen, X. Han, and Z. Liu (2026)

Kinema4D: kinematic 4d world modeling for spatiotemporal embodied simulation.

arXiv preprint arXiv:2603.16669.

Cited by: §2,
§4.1.
- [50]
H. Yan, H. Yu, Z. Zhong, W. Yuan, X. Gong, Z. Luo, C. Heyu, J. Li, W. Song, S. Zhou, et al. (2025)

Open-world hand-object interaction video generation based on structure and contact-aware representation.

arXiv preprint arXiv:2512.01677.

Cited by: §1.
- [51]
V. Ye, R. Li, J. Kerr, M. Turkulainen, B. Yi, Z. Pan, O. Seiskari, J. Ye, J. Hu, M. Tancik, and A. Kanazawa (2024)

gsplat: an open-source library for Gaussian splatting.

Journal of Machine Learning Research.

Cited by: §3.3.3.
- [52]
Z. Yin, S. Yang, and P. Abbeel (2026)

Object-centric 3d motion field for robot learning from human videos.

Advances in Neural Information Processing Systems 38, pp. 55923–55943.

Cited by: §2.
- [53]
C. Yuan, S. Joshi, S. Zhu, H. Su, H. Zhao, and Y. Gao (2025)

RoboEngine: plug-and-play robot data augmentation with semantic robot segmentation and background generation.

In 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS),

pp. 7622–7629.

Cited by: §3.3.2.
- [54]
X. Zhan, L. Yang, Y. Zhao, K. Mao, H. Xu, Z. Lin, K. Li, and C. Lu (2024)

OAKINK2: a dataset of bimanual hands-object manipulation in complex task completion.

In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),

pp. 445–456.

Cited by: §3.3.1,
§4.1.
- [55]
L. Zhang, A. Rao, and M. Agrawala (2023)

Adding conditional control to text-to-image diffusion models.

In Proceedings of the IEEE/CVF international conference on computer vision,

pp. 3836–3847.

Cited by: 2nd item,
§3.1,
§3.2.
- [56]
H. Zhao, X. Liu, M. Xu, Y. Hao, W. Chen, and X. Han (2025)

Taste-rob: advancing video generation of task-oriented hand-object interaction for generalizable robotic manipulation.

In Proceedings of the Computer Vision and Pattern Recognition Conference,

pp. 27683–27693.

Cited by: §1,
§2,
§3.3.1,
§4.1.
- [57]
H. Zhen, Q. Sun, H. Zhang, J. Li, S. Zhou, Y. Du, and C. Gan (2025)

Tesseract: learning 4d embodied world models.

arXiv preprint arXiv:2504.20995.

Cited by: §2,
§4.1.
- [58]
H. Zhi, P. Chen, S. Zhou, Y. Dong, Q. Wu, L. Han, and M. Tan (2025)

3DFlowAction: learning cross-embodiment manipulation from 3d flow world model.

arXiv preprint arXiv:2506.06199.

Cited by: §2.
- [59]
S. Zhou, Y. Du, J. Chen, Y. Li, D. Yeung, and C. Gan (2024)

RoboDreamer: learning compositional world models for robot imagination.

In Proceedings of the 41st International Conference on Machine Learning,

Proceedings of Machine Learning Research, Vol. 235, pp. 61885–61896.

External Links: [Link](https://proceedings.mlr.press/v235/zhou24f.html)

Cited by: §1,
§2.
- [60]
F. Zhu, H. Wu, S. Guo, Y. Liu, C. Cheang, and T. Kong (2025)

IRASim: a fine-grained world model for robot manipulation.

In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV),

pp. 9834–9844.

Cited by: §1,
§2,
§2.

