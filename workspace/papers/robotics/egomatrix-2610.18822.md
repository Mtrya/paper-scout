<!-- page 1 of 22 -->

arXiv:2610.18822v1 [cs.RO] 13 Oct 2026

# EGOMATRIX: PHYSICALLY FAITHFUL DEXTEROUS ROBOT DEMONSTRATIONS FROM EGOCENTRIC HU-MAN VIDEOS

**Ziheng Wang**<sup>1</sup>**, Yanming Shao**<sup>2</sup>**, Ting Mao**<sup>3</sup>

**Kaihan Chen**<sup>3</sup>**, Yiqun Wang**<sup>4</sup>**, Xuanye Wu**<sup>4</sup>**, Yao Mu**<sup>1,4†</sup>

<sup>1</sup>Shanghai Jiao Tong University <sup>2</sup>The University of Hong Kong

<sup>3</sup>Zhejiang University <sup>4</sup>Shanghai AI Laboratory

## ABSTRACT

Egocentric human videos offer a scalable alternative to robot teleoperation for collecting dexterous manipulation data. However, two challenges limit their use as reliable training data: accurate hand-object interaction (HOI) reconstruction from monocular RGB and high-quality robot demonstrations, especially visual observations. We present **EgoMatrix**, an end-to-end data engine that converts monocular egocentric human videos into bimanual dexterous trajectories and paired robot observation videos, with the trajectories directly replayable on real robots. It reconstructs HOI with emphasis on relative geometry and contact rather than exact metric scene recovery, then uses retargeting and reinforcement learning (RL) to reproduce the demonstrated tasks in simulation. We jointly render robot hands and objects from these simulated trajectories to produce physically faithful observations synchronized with state and control records. Rendering the same trajectories in diverse, task-compatible backgrounds expands paired training data while preserving robot-object motion and contact. Across TACO, OakInk2, and self-collected data, EgoMatrix achieves 54.84% end-to-end success in simulation versus 14.52% for the strongest baseline. Reconstruction reaches state-of-the-art performance on most metrics of the two public benchmarks. Qualitative results show more faithful hand-object interactions than prior methods, while feature-space analysis shows closer alignment with real-robot observations than GPT-6 Astra. Our project page is available at [https://spikew726.github.io/EgoMatrix/](https://spikew726.github.io/EgoMatrix/).

## 1 INTRODUCTION

Learning generalizable dexterous manipulation policies requires large, diverse datasets pairing visual observations with robot actions (Kim et al., 2025). Teleoperation and motion capture provide high-quality demonstrations, but specialized hardware and operator effort make them costly to scale, especially for multi-fingered hands (Wang et al., 2024; Xu et al., 2025). Egocentric videos offer a more accessible source of diverse human manipulation (Paliwal et al., 2026), yet converting them into robot demonstrations of comparable quality remains challenging (Liu et al., 2026).

Recent systems have shown that manipulation can be transferred from monocular human videos to dexterous robots (Paliwal et al., 2026; Mu et al., 2026; Ren et al., 2026). However, different combinations of 3D scale, depth, and hand-object placement can produce similar 2D projections, making relative geometry and contact difficult to recover reliably from monocular images. A reconstruction may therefore match the input images while still containing hand-object gaps or interpenetration. Such errors can conflict with the contacts required for successful manipulation.

Generating visual observations consistent with the robot’s actual manipulation poses a second challenge after successful task transfer in simulation. Data engines such as EgoEngine generate paired robot observations and trajectories (Liu et al., 2026), but retaining the original object imagery can

<small><span class="docvortex-page-footnote" data-block-type="page_footnote" style="color:#6b7280">†Corresponding author</span></small>

1

<!-- page 2 of 22 -->

![Image block](doc:5814607/tier:basic/page:2/block:1)

Figure 1: EgoMatrix: Any Task in Any Background. Rows denote source egocentric human manipulation tasks and columns denote target backgrounds; each cell shows the corresponding dexterous robot demonstration. EgoMatrix preserves physically faithful hand-object motion and contact across these task-background combinations, substantially increasing the visual diversity of robot training data.

leave depicted contacts and object motion inconsistent with the robot trajectory. For policy learning, visual observations should reflect how the robot actually manipulates the object.

We present **EgoMatrix**, an end-to-end data engine that converts monocular egocentric human videos into bimanual dexterous robot trajectories and synchronized observation videos. Our reconstruction prioritizes relative hand-object geometry and contact over exact metric scene recovery. We first use reconstructed hand geometry and relative-depth cues to establish the scene scale and initialize handobject alignment. Multi-start pose refinement then corrects wrist and object misalignments, followed by local contact refinement to close contact gaps and reduce interpenetration. The refined trajectories guide retargeting and RL, enabling robots to complete the demonstrated tasks in simulation.

We then jointly render the robot hands and objects from successful simulated trajectories to produce physically faithful observations synchronized with state and control records. These observations are composited into the source scene with simulated shadows, while support-plane alignment and target-view rendering extend the same manipulation to diverse, task-compatible backgrounds.

This enables demonstrations from large egocentric video datasets to be converted into paired robot data for a specified real-world workstation, potentially reducing additional robot data collection when adapting pretrained policies to that workstation.

Our contributions are threefold:

• We present **EgoMatrix**, an end-to-end data engine that converts monocular egocentric human videos into bimanual dexterous robot demonstrations with synchronized observations and state-control trajectories, supporting direct trajectory replay on real robots. EgoMatrix achieves 54.84% end-to-end success in simulation versus 14.52% for the strongest baseline.

• We develop **contact-aware hand-object reconstruction** with multi-start pose refinement and local contact refinement, achieving state-of-the-art performance on most reconstruction metrics. Ablations show that both stages improve end-to-end success in simulation.

• We introduce **physically faithful robot observation synthesis** by jointly rendering robot hands and objects from successful simulated trajectories. Support-plane alignment and target-view rendering expand the paired data across diverse, task-compatible backgrounds while preserving robot-object motion and contact.

2

<!-- page 3 of 22 -->

![Image block](doc:5814607/tier:basic/page:3/block:1)

Figure 2: Overview of EgoMatrix. Starting from monocular egocentric human videos, EgoMatrix performs contact-aware hand-object reconstruction, dexterous manipulation transfer via retargeting and RL, and physically faithful robot observation synthesis. It produces robot trajectories executable on real hardware and synchronized observation videos, with the same trajectories transferable to diverse, task-compatible backgrounds.

## 2 RELATED WORK

**Monocular Hand-Object Reconstruction.** Monocular hand-object reconstruction recovers 3D hand and object geometry from RGB videos, often by matching rendered reconstructions to image space observations (Fan et al., 2024; Aboukhadra et al., 2026). Occlusion limits this evidence, motivating learned shape and generative priors for hidden object regions (Wang et al., 2025; Chen et al., 2026c). Since geometry completion alone does not resolve ambiguous hand-object placement and contact, contact maps and temporally aggregated correspondences provide additional alignment cues (Grady et al., 2021; Ren et al., 2026). EgoMatrix similarly exploits contact cues, using contactguided multi-start pose optimization under image-space constraints and subsequent local contact refinement to better preserve the HOI demonstrated in the human videos.

**Human-to-Robot Manipulation Transfer.** After reconstruction, human hand-object motion must be adapted to the robot embodiment while preserving the demonstrated interaction. Existing methods combine geometric retargeting (Qin et al., 2022; Zhang et al., 2026) with grasp and motion planning (Mu et al., 2026), reference-tracking reinforcement learning (Li et al., 2025; Ren et al., 2026), or sampling-based trajectory optimization (Paliwal et al., 2026) to obtain executable robot motion. EgoMatrix follows the reference-tracking approach, using contact-aware reconstructed trajectories as motion references for policy learning.

**Robot Observation Synthesis.** Robot observations can be synthesized by replacing human hands while retaining source object content (Zhang et al., 2026; Liu et al., 2026), or by generative models conditioned on images or simulated geometry (Chen et al., 2025; 2026a). These image-level approaches may leave depicted contacts or object motion inconsistent with the robot trajectory. A different line of work starts from recorded robot demonstrations and uses physics simulation to generate additional training data (Jiang et al., 2025; Fang et al., 2025). EgoMatrix instead starts from monocular human videos, transfers the demonstrated manipulation to successful robot trajectories in simulation, and jointly renders the robot hands and objects from those trajectories to produce physically faithful observations.

## 3 METHOD

Given a monocular egocentric human video, EgoMatrix reconstructs hand-object interaction, transfers the demonstrated task to a dexterous robot via retargeting and RL, and synthesizes synchronized

3

<!-- page 4 of 22 -->

robot observations and motion data (Fig. 2). The resulting trajectories can also be rendered in different task-compatible backgrounds.

## 3.1 CONTACT-AWARE HAND-OBJECT RECONSTRUCTION

Without any external calibration or scene-specific metric references, monocular RGB observations cannot uniquely determine metric scene geometry. We therefore target two properties: **manipulation-relevant 3D consistency** for motion transfer and **source-view consistency** for video compositing. Our reconstruction proceeds from hand-anchored initialization to multi-start pose refinement and local contact refinement.

**Hand-Anchored Initialization** We use SAM 3.1 (Carion et al., 2026) with natural-language prompts to obtain hand and object masks, and SAM 3D (Chen et al., 2026b) to reconstruct an unscaled canonical mesh $\mathcal { V } _ { j }$ for each object. The temporal median of the per-frame intrinsics from MoGe-3 (Kong et al., 2026) defines a sequence-shared camera matrix $\bar { \mathbf { K } } ,$ while HaWoR (Zhang et al., 2025) provides the MANO (Romero et al., 2017) hand trajectories.

We use the human-scale prior encoded by the reconstructed MANO geometry as the scene-scale anchor. Let $\mathcal { P } _ { h , t }$ denote pixels shared by the visible hand mask and rendered MANO surface. We estimate a sequence-level scale from the MANO depth $z _ { h , t } ^ { H } ( u )$ and MoGe-3 depth $D _ { t } ^ { M } ( u )$

$$
\operatorname { l o g } s _ { \operatorname { s e q } } = \mathcal { R } \left( \left\{ \operatorname { l o g } z _ { h , t } ^ { H } ( u ) - \operatorname { l o g } D _ { t } ^ { M } ( u ) \right\} _ { h , t , u } \right) , \qquad \widetilde { D } _ { t } = s _ { \operatorname { s e q } } D _ { t } ^ { M } ,\tag{1}
$$

where R robustly aggregates observations over $u   \in   \mathcal { P } _ { h , t }$ Back-projecting the aligned depth $\widetilde { D } _ { t }$ within each object mask to 3D yields an object point cloud, from which we estimate a sequenceshared uniform scale $s _ { j }$ for $\mathcal { V } _ { j }$ . We also use $\widetilde { D } _ { t }$ as the metric-depth input to HaWoR’s SLAM module, so that the camera translation is recovered under the same scale.

Using the scaled meshes, aligned depth, and camera trajectory, we initialize object poses with FoundationPose (Wen et al., 2024), augmented by rigid-motion hypotheses from SpatialTracker (Xiao et al., 2024) to improve propagation under hand occlusion.

To first reduce large relative-placement errors with a lightweight alignment before multi-start refinement, we fix the right hand and optimize the frame-wise translations of the left hand and both objects, while keeping hand articulation, object rotation, and object scale fixed:

$$
\mathcal { L } _ { \operatorname { a l i g n } } = \lambda _ { \operatorname { 2 D } } \mathcal { L } _ { \operatorname { 2 D } } + \underbrace { \lambda _ { \operatorname { o r d } } \mathcal { L } _ { \operatorname { o r d } } + \lambda _ { \operatorname { g a p } } \mathcal { L } _ { \operatorname { g a p } } } _ { \mathcal { L } _ { \operatorname { d e p t h } } } + \lambda _ { \operatorname { t e m p } } \mathcal { L } _ { \operatorname { t e m p } } + \lambda _ { \operatorname { p r i o r } } \mathcal { L } _ { \operatorname { p r i o r } } .\tag{2}
$$

Here, $\mathcal { L } _ { \mathrm { 2 D } }$ encourages source-view consistency through visibility and silhouette constraints, while $\mathcal { L } _ { \mathrm { d e p t h } }$ preserves pairwise front-back ordering and relative depth differences induced by $\widetilde { D } _ { t } ;$ the remaining terms regularize the trajectory.

We estimate contact frames for each hand using HACO (Jung & Lee, 2025) and the proximity of the observed hand and object masks. Specifically, we project MANO vertices into the source image and use their HACO contact probabilities when they lie near both the hand mask and the corresponding object mask. We combine this evidence with the distance between the two masks, then apply temporal filtering to identify sustained contact intervals. Let $m _ { h , t }   \in   \{ 0 , 1 \}$ denote the resulting contact mask. We define

$$
\mathcal { T } _ { h } ^ { \mathrm { c } } = \{ t \in \mathcal { T } _ { h } \mid m _ { h , t } = 1 \} , \qquad h \in \{ L , R \} ,\tag{3}
$$

where $\mathcal { T } _ { h }$ denotes the task frames for hand h. The two hands are processed independently.

We additionally estimate gravity and a shared support surface from the background geometry. Frames outside $\mathcal { T } _ { h } ^ { \mathrm { c } }$ provide evidence for stable object support, while objects are not forced onto the support surface during contact. Further details are provided in Appendix A.1.

4

<!-- page 5 of 22 -->

**Multi-Start Pose Refinement** After the hand-anchored initialization stage, we refine the wrist and object rigid poses while keeping MANO articulation fixed. For contact frames $t \in \mathcal { T } _ { h } ^ { \mathrm { c } }$ , HACO (Jung & Lee, 2025) provides contact probabilities $p _ { h , t , i } ,$ from which we define likely contact vertices $\widetilde { \mathcal { I } _ { h , t } }   =   \{ i   \mid   \widetilde { p _ { h , t , i }   >   \tau _ { \mathrm { H } } } \}$ . For each MANO vertex $i \in \mathcal { I } _ { h , t }$ , we consider the camera ray passing through it. On concave or multi-layered objects, the ray may intersect multi-ple surfaces, and normal compatibility alone may still leave several plausible intersections on the object surface for the same hand vertex. Since pose refinement is sensitive to initialization (Grady et al., 2021), we construct multiple contact-guided starts that cover these alternative surface associations.

![Image block](doc:5814607/tier:basic/page:5/block:2)

As illustrated in Fig. 3, we generate starting poses by translating the hand along the ray from the camera center through the mean position of the initial MANO vertices. For each candidate k, we cast rays through the vertices in $\mathcal { I } _ { h , t }$ and select the intersection closest to the corresponding vertex. These intersections are stored in object coordinates and define the targets of $\mathcal { L } _ { \mathrm { r a y } } ^ { ( k ) }$ , which attracts the hand vertices toward them during subsequent pose optimization.

Figure 3: Contact-guided multi-start initialization. Different hand offsets associate candidate hand vertices with different points on the object surface, providing alternative starts for pose refinement.

To compare candidates, we group vertices in $\mathcal { I } _ { h , t }$ into fixed patches using MANO mesh connectivity. For each patch, we record how often vertices are paired with the first, second, or later intersection along the ray. For each candidate, we compute its difference from every retained start and use the smallest difference as its diversity measure. Starting from the unshifted pose, we greedily select additional candidates with large diversity while penalizing large distances between paired points. For task frames outside $\mathcal { T } _ { h } ^ { \mathrm { c } }$ , we use only the unshifted pose.

We then jointly refine the wrist and bound-object poses from each selected start. We collect the source-view, relative-depth, penetration, and regularization terms into $\mathcal { L } _ { \mathrm { b a s e } }$ . The objective is

$$
\mathcal { L } _ { \operatorname { c o a r s e } } ^ { ( k , s ) } = \mathcal { L } _ { \operatorname { b a s e } } + \lambda _ { \operatorname { c } } \left[ \eta _ { s } \mathcal { L } _ { \operatorname { r a y } } ^ { ( k ) } + ( 1 - \eta _ { s } ) \mathcal { L } _ { \operatorname { s u r f } } \right] ,\tag{4}
$$

where s indexes optimization steps and $\mathcal { L } _ { \mathrm { s u r f } }$ attracts vertices in $\mathcal { I } _ { h , i }$ toward the current nearest object surface. The weight $\eta _ { s }$ decreases from 1 to 0, so all starts are ultimately refined under the same contact objective. The contact attraction terms apply only to frames in $\mathcal { T } _ { h } ^ { \mathrm { c } }$ . For other task frames, optimization relies on the remaining terms in $\widehat { \mathcal { L } _ { \mathrm { b a s e } } }$ . Candidate selection and optimization details are provided in Appendix A.2.

**Local Contact Refinement** To improve the local hand-object geometry used for downstream manipulation transfer, we finally fix the object poses and jointly refine the wrist pose and MANO articulation. For frames $t   \in   \mathcal { T } _ { h } ^ { \mathrm { c } }$ , we cast rays through temporally filtered HACO contact vertices onto the object. When multiple intersections occur, we select those close to the hand vertex in depth with the strongest opposing normal match. These frame-wise contacts are stabilized in canonical object space following C2Dex (Ren et al., 2026). When stable patches provide insufficient opposing support, we supplement region-aware contacts according to the pad, palm, and side opposition types of the GRASP taxonomy (Feix et al., 2016).

For stable contact vertex $i ,$ let $\mathbf { v } _ { t , i }$ and $\mathbf { y } _ { t , i } ^ { * }$ denote the current hand vertex and its object-side target. For concise notation, we write the contact attraction as

$$
\mathcal { L } _ { \operatorname { c o n t a c t } } = \frac { 1 } { | \mathcal { T } _ { h } ^ { c } | } \sum _ { t \in \mathcal { T } _ { h } ^ { c } } \sum _ { i \in \mathcal { C } _ { t } } w _ { t , i } \left\| \mathbf { v } _ { t , i } - \mathbf { y } _ { t , i } ^ { * } \right\| _ { 2 } ^ { 2 } ,\tag{5}
$$

where $\mathcal { C } _ { t }$ contains vertices with valid contact targets, including supplementary opposition contacts when needed, and $w _ { t , i }$ is a contact weight derived from HACO confidence and temporal support. We combine $\mathcal { L } _ { \mathrm { c o n t a c t } }$ with bidirectional penetration penalties and regularization terms to form the final contact refinement objective. For frames outside $\mathcal { T } _ { h } ^ { \mathrm { c } }$ , wrist pose and MANO articulation are still optimized using penetration penalties and regularization, without contact attraction.Appendix A.3 provides the complete contact construction and optimization details.

5

<!-- page 6 of 22 -->

![Image block](doc:5814607/tier:basic/page:6/block:1)

Figure 4: Qualitative comparison of hand-object reconstruction. Each row shows the same input frame across ground truth, the three stages of EgoMatrix, and competing methods. The three examples illustrate how Local Contact Refinement addresses three contact defects that can hinder downstream dexterous-hand transfer: severe interpenetration (top), poorly localized contact (middle), and residual hand-object gaps (bottom). Across these examples, EgoMatrix produces more interaction-consistent hand-object reconstructions than the competing methods.

## 3.2 DEXTEROUS MANIPULATION TRANSFER

Given the reconstructed hand-object trajectories, we deliberately use a lightweight keypoint-based retargeting scheme rather than a more elaborate retargeting method. For each hand, we define fixed semantic correspondences between MANO joints and fingertips and robot-hand links. Let $\mathbf { x } _ { t , \ell } \in \mathbb { R } ^ { 3 }$ denote the 3D position of MANO keypoint ℓ at frame t, and let $\mathbf { p } _ { t } , \mathbf { R } _ { t } ,$ and $\mathbf { q } _ { t }$ denote the robot wrist position, wrist orientation, and finger configuration. With $\mathbf { f } _ { \ell } ( \mathbf { q } _ { t } )$ denoting the wrist-local forwardkinematics position of the corresponding robot keypoint, we optimize

$$
\mathcal { L } _ { \operatorname { r e t } } = \frac { 1 } { T N } \sum _ { t = 1 } ^ { T } \sum _ { \ell = 1 } ^ { N } w _ { \ell } \left\| \mathbf { p } _ { t } + \mathbf { R } _ { t } \mathbf { f } _ { \ell } ( \mathbf { q } _ { t } ) - \mathbf { x } _ { t , \ell } \right\| _ { 2 } + \mathcal { R } _ { \operatorname { r e t } } .\tag{6}
$$

subject to the robot joint limits. $\mathcal { R } _ { \mathrm { r e t } }$ regularizes wrist orientation, temporal smoothness, and table clearance. The two hands are retargeted independently, and the resulting wrist and joint trajectories are used to initialize physics-based policy learning.

We then adopt the RL framework of ManipTrans (Li et al., 2025), but omit its pretrained imitator and train a joint bimanual policy directly in Isaac Lab (Mittal et al., 2025). The retargeted robot trajectory is used for reference-state initialization, while the reconstructed MANO and object trajectories serve as tracking references. We train the policy with PPO (Schulman et al., 2017), using rewards for hand and object tracking, contact, and power. The learned policy is then rolled out in simulation to obtain a physically executable bimanual trajectory.

## 3.3 DEMONSTRATION SYNTHESIS AND BACKGROUND TRANSFER

Following a successful policy rollout, we replay the recorded simulator states and jointly render the dexterous hands and reconstructed objects using the source-video camera intrinsics and framealigned camera poses. Because both the hands and objects are rendered from the same physics rollout, their relative geometry directly reflects the executed interaction. We remove the human hands, arms, and manipulated objects from the source video using SVOR (Hu et al., 2026) with removal masks derived from SAM 3.1 segmentation. We then estimate tabletop shadow attenuation from paired simulator renders with and without the robot-object foreground, and apply it to the cleaned background.

6

<!-- page 7 of 22 -->

Let $I _ { t } ^ { \mathrm { s i m } }$ denote the simulator-rendered RGB image, $M _ { t } ^ { \mathrm { s i m } }$ the combined binary mask of the robot hands and manipulated objects, and $\tilde { I } _ { t } ^ { \mathrm { b g } }$ the shadow-adjusted background. The synthesized frame is

$$
I _ { t } ^ { \mathrm { s y n } } = M _ { t } ^ { \mathrm { s i m } } \odot I _ { t } ^ { \mathrm { s i m } } + \left( 1 - M _ { t } ^ { \mathrm { s i m } } \right) \odot \widehat { I } _ { t } ^ { \mathrm { b g } } .\tag{7}
$$

The same successful execution can be transferred to another task-compatible background without modifying its internal hand-object geometry. For source and target task frames ${ \mathcal { F } } _ { A }$ and ${ \mathcal { F } } _ { B } ,$ respectively defined from the support surface and initial object placement of each scene, we compute a single rigid transform $\mathbf { T } _ { B \leftarrow A } \in \operatorname { S E } ( 3 )$ and apply it to the entire interaction. Thus, both hands and all manipulated objects preserve their relative poses, with no rescaling or frame-wise realignment. We re-render the same robot and object trajectories using the target camera trajectory and supportsurface proxy. The rendered foreground is then composited with the cleaned target background using the same synthesis procedure. Different compatible backgrounds can therefore reuse the same robot and object trajectories and control records, changing only the common rigid transform applied to the trajectories, the camera viewpoint, and the visual background. Figure 1 illustrates this transfer across multiple source manipulation tasks and target backgrounds. Details of scene alignment and temporal synchronization are provided in Appendix B.2.

## 4 EXPERIMENTS

## 4.1 EXPERIMENTAL SETUP

**Datasets.** We evaluate on TACO (Liu et al., 2024), OakInk2 (Zhan et al., 2024), and a self-collected dataset. We use 4 TACO tasks with 37 demonstrations, 2 OakInk2 tasks with 14 demonstrations, and 11 self-collected tasks with one demonstration each. TACO and OakInk2 are used for reconstruction evaluation; all three datasets are used for end-to-end manipulation transfer.

**Baselines.** For end-to-end transfer, we compare with DexImit (Mu et al., 2026) and Do As I Do (Paliwal et al., 2026). For reconstruction, we compare with G-HOP (Ye et al., 2024), GHOST (Aboukhadra et al., 2026), MagicHOI (Wang et al., 2025), and ForeHOI (Chen et al., 2026c). For visual synthesis, we compare qualitatively with EgoEngine (Liu et al., 2026).

**Metrics.** For HOI reconstruction, we report root-aligned mean per-joint position error (MPJPE<sub>RA</sub>) for hand pose; Chamfer distance (CD) and F-score at a 5 mm threshold (F@5) for object geometry; absolute trajectory error (ATE), absolute rotation error (ARE), relative translation error (RTE), and relative rotation error (RRE) for object trajectories (Shin et al., 2026); and hand-relative Chamfer distance $( \mathrm { C D } _ { h } )$ , simulation displacement (SD), and maximum penetration depth $( D _ { \mathrm { p e n } } ^ { \mathrm { m a x } } )$ for interaction quality (Fan et al., 2024; Jiang et al., 2021; Hasson et al., 2019). We also report the Trajectory Transfer Success Rate, defined as the fraction of demonstrations converted into robot trajectories that complete the same task in simulation. For TACO and OakInk2, we align the simulated and ground-truth object trajectories as in ATE and count a transfer as successful if the translation and rotation errors remain within 0.10 m and 1.0 rad throughout the task. For self-collected videos, where ground-truth object poses are unavailable, we manually judge each rollout and require both a valid grasp and completion of the intended manipulation.

## 4.2 END-TO-END VIDEO-TO-ROBOT TRAJECTORY TRANSFER

First, we evaluate end-to-end trajectory transfer on selected TACO and OakInk2 tasks and 11 self-collected demonstrations spanning diverse object geometries, grasp classes, and motion patterns. As shown in Table 1, EgoMatrix achieves the highest success rate across all evaluated task groups, reaching 54.84% (34/62) overall, compared with 14.52% (9/62) for Do As I Do and 12.90% (8/62) for DexImit. The gains are consistent across all three data sources and include tasks where the baselines rarely or never succeed, such as screw, screwdriver, box and smear, glue gun, box. Overall, EgoMatrix converts monocular human demonstrations into robot trajectories that reproduce the demonstrated manipulation with substantially higher success rates.

7

<!-- page 8 of 22 -->

Table 1: End-to-end success rate of trajectory transfer across TACO, OakInk2, and self-collected demonstrations. Each entry reports the success rate and the number of successful trajectories over the total number of evaluated demonstrations.

<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Tasks</td><td colspan="3">Trajectory Transfer Success Rate (%) ↑</td></tr><tr><td>DexImit</td><td>Do As I Do</td><td>EgoMatrix</td></tr><tr><td rowspan="4">TACO</td><td>brush, brush, bowl</td><td>7.14% (0.5/7)</td><td>28.57% (2/7)</td><td>71.43% (5/7)</td></tr><tr><td>cut, knife, bowl</td><td>20.83% (2.5/12)</td><td>0.00% (0/12)</td><td>33.33% (4/12)</td></tr><tr><td>screw, screwdriver, box</td><td>0.00% (0/11)</td><td>9.09% (1/11)</td><td>45.45% (5/11)</td></tr><tr><td>smear, glue gun, box</td><td>0.00% (0/7)</td><td>0.00% (0/7)</td><td>57.14% (4/7)</td></tr><tr><td rowspan="2">OakInk2</td><td>stir spoon/rod</td><td>0.00% (0/8)</td><td>25.00% (2/8)</td><td>62.50% (5/8)</td></tr><tr><td>rearrange tableware</td><td>0.00% (0/6)</td><td>16.67% (1/6)</td><td>50.00% (3/6)</td></tr><tr><td>Self-Collected</td><td>All 11 tasks</td><td>45.45% (5/11)</td><td>27.27% (3/11)</td><td>72.73% (8/11)</td></tr><tr><td>Overall</td><td>All above tasks</td><td>12.90% (8/62)</td><td>14.52% (9/62)</td><td>54.84% (34/62)</td></tr></table>

<sup>\*</sup>For DexImit, a demonstration receives 0.5 credit when only one of the two hand-object trajectories satisfies the success criterion. Requiring both hands to succeed would give DexImit 0% on all public-dataset tasks.

Table 2: Quantitative comparison of hand-object reconstruction on TACO and OakInk2. Lower is better except for F@5. Best results are shown in bold and second-best results are underlined. $^ { \circ \circ } - ^ { \flat \flat }$ denotes metrics unavailable from the method outputs.

<table><tr><td rowspan="2">Dataset Method</td><td rowspan="2"></td><td colspan="2">Object Geometry</td><td>Hand Pose</td><td colspan="4">Object Trajectory</td><td colspan="3">Interaction</td></tr><tr><td>CD↓ (cm)</td><td>F@5↑ (%)</td><td> $\mathbf { M P J P E _ { R A } } \downarrow$  (mm)</td><td>ATE↓ (cm)</td><td>ARE↓ (°)</td><td>RTE↓ (cm)</td><td>RRE↓ (°)</td><td> $\mathbf { C D } _ { h } \downarrow \mathbf { S D } \downarrow \mathbf { D } _ { \operatorname { p e n } } ^ { \operatorname* { m a x } } \downarrow$  (cm)</td><td>(cm)</td><td>(mm)</td></tr><tr><td rowspan="7">TACO</td><td>G-HOP</td><td>2.54</td><td>30.87</td><td>37.28</td><td>38.35</td><td>65.32</td><td>7.69</td><td>8.62</td><td>37.66</td><td>9.23</td><td>1.84</td></tr><tr><td>MagicHOI</td><td>2.11</td><td>40.44</td><td>39.90</td><td></td><td></td><td></td><td></td><td>13.33</td><td>9.14</td><td>17.52</td></tr><tr><td>ForeHOI</td><td>1.52</td><td>51.28</td><td></td><td>16.11</td><td>69.14</td><td>2.33</td><td>8.26</td><td></td><td></td><td></td></tr><tr><td>GHOST</td><td>2.16</td><td>44.32</td><td>37.05</td><td>16.01</td><td>42.45</td><td>1.60</td><td>4.20</td><td>23.60</td><td>5.42</td><td>17.87</td></tr><tr><td>DexImit</td><td>0.64</td><td>83.08</td><td>29.12</td><td>8.43</td><td>37.53</td><td>1.35</td><td>9.26</td><td>17.35</td><td>10.39</td><td>0.35</td></tr><tr><td>Do As I Do</td><td>0.61</td><td>84.81</td><td>22.16</td><td>4.99</td><td>50.24</td><td>1.42</td><td>6.19</td><td>6.00</td><td>4.63</td><td>3.69</td></tr><tr><td>EgoMatrix</td><td>0.56</td><td>86.27</td><td>23.07</td><td>3.35</td><td>24.54</td><td>0.70</td><td>3.35</td><td>4.06</td><td>2.62</td><td>2.64</td></tr><tr><td rowspan="7"></td><td>G-HOP</td><td>1.55</td><td>42.36</td><td>27.39</td><td>46.08</td><td>63.19</td><td>3.44</td><td>6.79</td><td>11.63</td><td>9.51</td><td>0.45</td></tr><tr><td>MagicHOI</td><td>1.95</td><td>37.59</td><td>23.52</td><td></td><td></td><td></td><td></td><td>16.85</td><td>1.94</td><td>15.71</td></tr><tr><td>ForeHOI</td><td>1.00</td><td>58.36</td><td></td><td>9.31</td><td>83.09</td><td>1.43</td><td>7.13</td><td></td><td></td><td></td></tr><tr><td>OakInk2 GHOST</td><td>1.61</td><td>50.78</td><td>29.13</td><td>8.09</td><td>45.44</td><td>1.16</td><td>4.00</td><td>18.63</td><td>2.54</td><td>21.18</td></tr><tr><td>DexImit</td><td>1.20</td><td>71.47</td><td>25.27</td><td>12.55</td><td>78.75</td><td>1.57</td><td>10.66</td><td>30.71</td><td>9.66</td><td>0.18</td></tr><tr><td>Do As I Do</td><td>0.59</td><td>84.61</td><td>24.04</td><td>8.93</td><td>70.19</td><td>1.79</td><td>5.19</td><td>16.52</td><td>4.91</td><td>1.70</td></tr><tr><td>EgoMatrix</td><td>0.57</td><td>87.75</td><td>20.47</td><td>2.52</td><td>58.68</td><td>0.56</td><td>2.70</td><td>14.10</td><td>2.52</td><td>1.81</td></tr></table>

## 4.3 RECONSTRUCTION QUALITY

We next evaluate reconstruction quality, with particular attention to the hand-object interaction used for downstream transfer. As shown in Table 2, EgoMatrix achieves the best result on 8 of 10 metrics on TACO and 6 of 10 on OakInk2, including the best CD, F@5, ATE, and RTE on both datasets. It also achieves the best $\mathbf { C D } _ { h }$ and SD on TACO and remains competitive on OakInk2.

Interaction metrics should be interpreted jointly. DexImit, for example, obtains the lowest $D _ { \mathrm { p e n } } ^ { \mathrm { m a x } }$ on both datasets but substantially worse $\mathbf { C D } _ { h }$ and SD, showing that low penetration alone does not imply accurate hand-object interaction. Fig. 4 further shows more interaction-consistent reconstructions than Do As I Do and DexImit on representative examples.

## 4.4 VISUAL FIDELITY OF GENERATED ROBOT OBSERVATIONS

**Source-Background Fidelity.** Fig. 5 compares robot observations synthesized in the original source background. Kiwi-Edit (Lin et al., 2026) largely preserves the scene and object placement

8

<!-- page 9 of 22 -->

(b) Clean target background

Source RGB

Kiwi-Edit

GPT6-Astra

![Image block](doc:5814607/tier:basic/page:9/block:5)

EgoEngine

EgoMatrix

Figure 5: Qualitative comparison of robot observation synthesis in the source background. Each row compares Kiwi-Edit, GPT-6 Astra, EgoEngine, and EgoMatrix for the same source frame. Magnified regions highlight representative hand-object contacts. EgoMatrix produces more physically faithful robot observations with hand-object geometry consistent with the transferred manipulation.

but often fails to replace the human hands and arms with coherent robot geometry. Despite explicitly specifying a dual-UR5e system with XHand dexterous hands, GPT-6 Astra (OpenAI, 2026) still produces image distortion and mismatched robot hands or arms. EgoEngine better preserves robot appearance but exhibits local hand-object contact errors. EgoMatrix instead renders the robot hands and objects from the same successful simulation rollout, preserving the transferred interaction geometry while retaining the source background and viewpoint.

(d) GPT6-Astra

![Image block](doc:5814607/tier:basic/page:9/block:11)

Figure 6: Feature-space comparison for target-background transfer. Top: source human frame, clean target background, real-robot observation, GPT-6 Astra, and EgoMatrix. Bottom: feature distributions from ResNet18, VGG16, DINOv2, and DreamSim over 52 frames per domain. EgoMatrix more closely matches the real-robot distribution in three of the four feature spaces.

**Target-Background Fidelity.** We further compare GPT-6 Astra and EgoMatrix against ego-view real-robot observations in a target workspace. Using 52 frames from one manipulation sequence, we extract features with ResNet18 (He et al., 2016), VGG16 (Simonyan & Zisserman, 2015), DINOv2 (Oquab et al., 2024), and DreamSim (Fu et al., 2023). Fig. 6 visualizes the feature distributions, while Table 3

Table 3: Feature alignment (FD) with real robot observations. FD over 52 frames per domain; lower is better.

<table><tbody><tr><td rowspan="2">Source</td><td colspan="4">FD ↓</td></tr><tr><td>ResNet18</td><td>VGG16</td><td>DINOv2 D</td><td>reamSim</td></tr><tr><td>Human</td><td>273.10</td><td>1003.64</td><td>468.06</td><td>1.0178</td></tr><tr><td>GPT-6 Astra</td><td>119.81</td><td>711.24</td><td>240.54</td><td>0.5934</td></tr><tr><td>EgoMatrix</td><td>100.23</td><td>414.66</td><td>253.15</td><td>0.4975</td></tr></tbody></table>

reports Frechet distance (FD) to the real-robot distribution in each full feature space; lower is bet- ´ ter. EgoMatrix obtains the lowest FD for ResNet18, VGG16, and DreamSim, while GPT-6 Astra is slightly better on DINOv2. Qualitatively, Astra often preserves the source object’s image-space size and placement under the viewpoint change and fails to reproduce the correct bimanual robot configuration. EgoMatrix re-renders the transferred robot and object trajectories from the target camera,

9

<!-- page 10 of 22 -->

explicitly adapting them to the new viewpoint. The fixed Astra prompt and generation protocol are provided in Appendix D.

## 4.5 ABLATION STUDIES

Table 4: Ablation of reconstruction refinement. Each row cumulatively adds one stage. We report transfer success on self-collected tasks and TACO, and hand-pose and interaction metrics on TACO.

<table><tr><td rowspan="3">Method</td><td>Self</td><td colspan="5">TACO</td></tr><tr><td>Succ. ↑ (%)</td><td>Succ. ↑ (%)</td><td>MPJPERA ↓ (mm)</td><td> $\mathbf { C D } _ { h } \downarrow$  (cm)</td><td>SD↓ (cm)</td><td> $\mathbf { D } _ { \mathrm { p e n } } ^ { \mathrm { m a x } } \downarrow$  (mm)</td></tr><tr><td>Hand-Anchored Init.</td><td>18.18</td><td>5.41</td><td>21.97</td><td>7.14</td><td>4.00</td><td>3.82</td></tr><tr><td>+ Multi-Start Refinement</td><td>36.36</td><td>27.03</td><td>23.17</td><td>4.24</td><td>3.31</td><td>1.51</td></tr><tr><td>+ Local Contact Refinement</td><td>72.73</td><td>48.65</td><td>23.07</td><td>4.06</td><td>2.62</td><td>2.64</td></tr></table>

Table 4 shows that both refinement stages are important for downstream manipulation transfer. The hand-anchored initialization already achieves an $\mathrm { M P J P E _ { R A } }$ of 21.97 mm on TACO, outperforming all baselines in Table 2. The subsequent refinements slightly degrade this hand-pose metric but substantially improve interaction quality and end-to-end success: on TACO, success rises from 5.41% to 27.03% and then 48.65%, with the same trend on the self-collected tasks. This trade-off matches our goal of prioritizing manipulation-relevant hand-object geometry over hand pose accuracy alone.

Fig. 4 further illustrates the two stages. Multi-Start Pose Refinement corrects large initial pose errors in the first two examples while preserving the plausible initialization in the third. Local Contact Refinement then addresses interpenetration, poorly localized contact, and residual hand-object gaps. Together with the improved $\mathrm { C D } _ { h } ,$ SD, and transfer success, these results show the importance of interaction geometry for downstream transfer.

## 4.6 REAL-WORLD TRAJECTORY REPLAY

We further validate the transferred trajectories on two UR5e arms equipped with XHand dexterous hands. From the trajectories generated by EgoMatrix, we select representative tasks for hardware replay, with source demonstrations drawn from both self-collected videos and public egocentric datasets. As shown in Fig. 7, the transferred trajectories complete the corresponding manipulations on the physical system. Additional hardware details and full trajectory sequences are provided in Appendix E. More video demonstrations are available on our project page.

![Image block](doc:5814607/tier:basic/page:10/block:9)

Figure 7: Real-world trajectory replay. Each adjacent pair shows a source human demonstration frame and the corresponding real-robot replay for the same task.

## 5 CONCLUSION AND LIMITATIONS

We presented **EgoMatrix**, an end-to-end data engine that converts monocular egocentric human videos into bimanual dexterous robot demonstrations. Contact-aware reconstruction improves the hand-object references used for manipulation transfer, while joint rendering of simulated robot and object trajectories produces physically faithful visual observations and supports data generation across task-compatible backgrounds. Experiments demonstrate improved end-to-end manipulation transfer, strong reconstruction performance, and direct trajectory replay on real robots.

**Limitations.** EgoMatrix’s multi-stage pipeline introduces computational overhead for each demonstration. Although demonstrations can be processed in parallel with sufficient resources, further improvements in processing efficiency would facilitate larger-scale data generation. Moreover, monocular observations inherently involve depth and occlusion ambiguities. While our contactaware reconstruction helps address these challenges, incorporating additional information, such as

10

<!-- page 11 of 22 -->

ground-truth depth for selected frames or multi-view observations, could further improve reconstruction quality and manipulation transfer. Finally, although we evaluate the visual fidelity of synthesized robot observations, their effectiveness for downstream manipulation policy training remains to be explored.

## REFERENCES

Ahmed Tawfik Aboukhadra, Marcel Rogge, Nadia Robertini, Abdalla Arafa, Jameel Malik, Ahmed Elhayek, and Didier Stricker. Ghost: Fast category-agnostic hand-object interaction reconstruction from rgb videos using gaussian splatting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings, pp. 3394–3404, June 2026.

Nicolas Carion, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris Coll-Vinent, Chaitanya Ryali, Kalyan Vasudev Alwala, Haitham Khedr, Andrew Huang, Jie Lei, Tengyu Ma, Baishan Guo, Arpit Kalla, Markus Marks, Joseph Greer, Meng Wang, Peize Sun, Roman Radle, Triantafyllos Afouras, Effrosyni Mavroudi, Katherine Xu, Tsung-Han Wu, Yu Zhou, ¨ Liliane Momeni, Rishi Hazra, Shuangrui Ding, Sagar Vaze, Francois Porcher, Feng Li, Siyuan Li, Aishwarya Kamath, Ho Kei Cheng, Piotr Dollar, Nikhila Ravi, Kate Saenko, Pengchuan Zhang, ´ and Christoph Feichtenhofer. SAM 3: Segment anything with concepts. In International Conference on Learning Representations (ICLR), 2026.

Jason Chen, I-Chun Arthur Liu, Gaurav Sukhatme, and Daniel Seita. CRAFT: Video diffusion for bimanual robot data generation. In International Conference on Intelligent Robots and Systems (IROS), 2026a.

Lawrence Yunliang Chen, Chenfeng Xu, Karthik Dharmarajan, Richard Cheng, Kurt Keutzer, Masayoshi Tomizuka, Quan Vuong, and Ken Goldberg. RoVi-Aug: Robot and viewpoint augmentation for cross-embodiment robot learning. In Proceedings of The 8th Conference on Robot Learning, volume 270 of Proceedings of Machine Learning Research, pp. 209–233. PMLR, 2025. URL [https://proceedings.mlr.press/v270/chen25a.html](https://proceedings.mlr.press/v270/chen25a.html).

Xingyu Chen, Fu-Jen Chu, Pierre Gleize, Kevin J. Liang, Alexander Sax, Hao Tang, Weiyao Wang, Michelle Guo, Thibaut Hardin, Xiang Li, Aohan Lin, Jia-Wei Liu, Ziqi Ma, Anushka Sagar, Bowen Song, Xiaodong Wang, Jianing Yang, Bowen Zhang, Piotr Dollar, Georgia Gkioxari, Matt ´ Feiszli, and Jitendra Malik. Sam 3d: 3dfy anything in images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 7220–7232, 2026b.

Yuantao Chen, Jiahao Chang, Chongjie Ye, Chaoran Zhang, Zhaojie Fang, Chenghong Li, and Xiaoguang Han. Forehoi: Feed-forward 3d object reconstruction from daily hand-object interaction videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 8868–8879, June 2026c.

Zicong Fan, Maria Parelli, Maria Eleni Kadoglou, Xu Chen, Muhammed Kocabas, Michael J. Black, and Otmar Hilliges. HOLD: Category-agnostic 3d reconstruction of interacting hands and objects from video. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 494–504, June 2024.

Yu Fang, Yue Yang, Xinghao Zhu, Kaiyuan Zheng, Gedas Bertasius, Daniel Szafir, and Mingyu Ding. ReBot: Scaling robot learning with real-to-sim-to-real robotic video synthesis. In 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pp. 11351–11358, 2025. doi: 10.1109/IROS60139.2025.11246305.

Thomas Feix, Javier Romero, Heinz-Bodo Schmiedmayer, Aaron M. Dollar, and Danica Kragic. The grasp taxonomy of human grasp types. IEEE Transactions on Human-Machine Systems, 46 (1):66–77, 2016. doi: 10.1109/THMS.2015.2470657.

Stephanie Fu, Netanel Tamir, Shobhita Sundaram, Lucy Chai, Richard Zhang, Tali Dekel, and Phillip Isola. Dreamsim: Learning new dimensions of human visual similarity using synthetic data. In Advances in Neural Information Processing Systems, volume 36, pp. 50742–50768, 2023.

11

<!-- page 12 of 22 -->

Patrick Grady, Chengcheng Tang, Christopher D. Twigg, Minh Vo, Samarth Brahmbhatt, and Charles C. Kemp. ContactOpt: Optimizing contact to improve grasps. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 1471–1481, June 2021.

Yana Hasson, Gul Varol, Dimitrios Tzionas, Igor Kalevatykh, Michael J. Black, Ivan Laptev, and Cordelia Schmid. Learning joint reconstruction of hands and manipulated objects. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 11807–11816, 2019.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 770–778, June 2016. doi: 10.1109/CVPR.2016.90.

Jiagao Hu, Yuxuan Chen, Fuhao Li, Zepeng Wang, Fei Wang, Daiguo Zhou, and Jian Luan. From ideal to real: Stable video object removal under imperfect conditions. arXiv preprint arXiv:2603.09283, 2026.

Hanwen Jiang, Shaowei Liu, Jiashun Wang, and Xiaolong Wang. Hand-object contact consistency reasoning for human grasps generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 11107–11116, 2021.

Zhenyu Jiang, Yuqi Xie, Kevin Lin, Zhenjia Xu, Weikang Wan, Ajay Mandlekar, Linxi Fan, and Yuke Zhu. DexMimicGen: Automated data generation for bimanual dexterous manipulation via imitation learning. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pp. 11639–11646, 2025. doi: 10.1109/ICRA55743.2025.11127809.

Daniel Sungho Jung and Kyoung Mu Lee. Learning dense hand contact estimation from imbalanced data. Advances in Neural Information Processing Systems, 38:133424–133457, 2025. doi: 10. 52202/085713-4017.

Daniel Sungho Jung and Kyoung Mu Lee. Training-free dense hand contact estimation with multi-modal large language models. arXiv preprint arXiv:2605.05886, 2026.

Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan P Foster, Pannag R Sanketi, Quan Vuong, Thomas Kollar, Benjamin Burchfiel, Russ Tedrake, Dorsa Sadigh, Sergey Levine, Percy Liang, and Chelsea Finn. Open-VLA: An open-source vision-language-action model. In Pulkit Agrawal, Oliver Kroemer, and Wolfram Burgard (eds.), Proceedings of The 8th Conference on Robot Learning, volume 270 of Proceedings of Machine Learning Research, pp. 2679–2713. PMLR, 2025. URL [https://proceedings.mlr.press/v270/kim25c.html](https://proceedings.mlr.press/v270/kim25c.html).

Lingyu Kong, Ruicheng Li, Ruicheng Wang, Sicheng Xu, Chengtang Yao, Jianfeng Xiang, and Jiaolong Yang. Moge-3: Fine-detail monocular geometry estimation with self-guided sparse volumetric refinement. arXiv e-prints, pp. arXiv–2607, 2026.

Kailin Li, Puhao Li, Tengyu Liu, Yuyang Li, and Siyuan Huang. Maniptrans: Efficient dexterous bimanual manipulation transfer via residual learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 6991–7003, 2025.

Yiqi Lin, Guoqiang Liang, Ziyun Zeng, Zechen Bai, Yanzhe Chen, and Mike Zheng Shou. Kiwiedit: Versatile video editing via instruction and reference guidance, 2026.

Yangcen Liu, Shuo Cheng, Xinchen Yin, Woo Chul Shin, Alfred Cueva, Yiran Yang, Zhenyang Chen, Chuye Zhang, and Danfei Xu. Egoengine: From egocentric human videos to high-fidelity dexterous robot demonstrations. arXiv preprint arXiv:2606.12604, 2026.

Yun Liu, Haolin Yang, Xu Si, Ling Liu, Zipeng Li, Yuxiang Zhang, Yebin Liu, and Li Yi. Taco: Benchmarking generalizable bimanual tool-action-object understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 21740–21751, June 2024.

12

<!-- page 13 of 22 -->

Mayank Mittal, Pascal Roth, James Tigue, Antoine Richard, Octi Zhang, Peter Du, Antonio Serrano-Munoz, Xinjie Yao, Ren ˜ e Zurbr ´ ugg, Nikita Rudin, et al. Isaac lab: A gpu-accelerated simulation ¨ framework for multi-modal robot learning. arXiv preprint arXiv:2511.04831, 2025.

Juncheng Mu, Sizhe Yang, Yiming Bao, Hojin Bae, Tianming Wei, Linning Xu, Boyi Li, Huazhe Xu, and Jiangmiao Pang. Deximit: Learning bimanual dexterous manipulation from monocular human videos. In Proceedings of Robotics: Science and Systems, Sydney, Australia, July 2026. doi: 10.15607/RSS.2026.XXII.003.

OpenAI. GPT-6 Astra System Card, September 2026. OpenAI Deployment Safety Hub.

Maxime Oquab, Timothee Darcet, Th ´ eo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov, ´ Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mido Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Herve J ´ egou, Julien Mairal, Patrick Labatut, Ar- ´ mand Joulin, and Piotr Bojanowski. Dinov2: Learning robust visual features without supervision. Transactions on Machine Learning Research, 2024.

Bhawna Paliwal, Haritheja Etukuru, William Liang, Pieter Abbeel, Nur Muhammad Mahi Shafiullah, and Jitendra Malik. Do as i do: Dexterous manipulation data from everyday human videos. arXiv preprint arXiv:2606.19333, 2026.

Yuzhe Qin, Yueh-Hua Wu, Shaowei Liu, Hanwen Jiang, Ruihan Yang, Yang Fu, and Xiaolong Wang. DexMV: Imitation learning for dexterous manipulation from human videos. In Computer Vision – ECCV 2022, volume 13699 of Lecture Notes in Computer Science, pp. 570–587. Springer, 2022. doi: 10.1007/978-3-031-19842-7 33.

Jie Ren, Zhehao Jiang, Yinhong Yang, Haorui Jia, Han Jiang, Ben Li, Yao Yao, Cheng Lin, Qiu Shen, Zhenshan Bing, Xiao-Xiao Long, and Xun Cao. C2Dex: Contact-consistent reconstruction and retargeting for dexterous manipulation from monocular video. arXiv preprint arXiv:2608.07045, 2026.

Javier Romero, Dimitrios Tzionas, and Michael J. Black. Embodied hands: Modeling and capturing hands and bodies together. ACM Transactions on Graphics (Proc. SIGGRAPH Asia), 36(6): 245:1–245:17, 2017. doi: 10.1145/3130800.3130883.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Jisu Shin, Junoh Lee, JunGyu Lee, Inhwan Bae, Dohyeon Lee, Hokyun Im, Youngwoon Lee, and Hae-Gon Jeon. ComPose: When to trust hands for object pose tracking. arXiv preprint arXiv:2605.23523, 2026.

Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. In International Conference on Learning Representations (ICLR), 2015.

Chen Wang, Haochen Shi, Weizhuo Wang, Ruohan Zhang, Li Fei-Fei, and Karen Liu. DexCap: Scalable and portable mocap data collection system for dexterous manipulation. In Proceedings of Robotics: Science and Systems, Delft, Netherlands, July 2024. doi: 10.15607/RSS.2024.XX.043.

Shibo Wang, Haonan He, Maria Parelli, Christoph Gebhardt, Zicong Fan, and Jie Song. MagicHOI: Leveraging 3d priors for accurate hand-object reconstruction from short monocular video clips. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 5957–5968, October 2025.

Bowen Wen, Wei Yang, Jan Kautz, and Stan Birchfield. Foundationpose: Unified 6d pose estimation and tracking of novel objects. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 17868–17879, 2024.

Yuxi Xiao, Qianqian Wang, Shangzhan Zhang, Nan Xue, Sida Peng, Yujun Shen, and Xiaowei Zhou. Spatialtracker: Tracking any 2d pixels in 3d space. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 20406–20417, 2024.

13

<!-- page 14 of 22 -->

Mengda Xu, Han Zhang, Yifan Hou, Zhenjia Xu, Linxi Fan, Manuela Veloso, and Shuran Song. DexUMI: Using human hand as the universal manipulation interface for dexterous manipulation. In Joseph Lim, Shuran Song, and Hae-Won Park (eds.), Proceedings of The 9th Conference on Robot Learning, volume 305 of Proceedings of Machine Learning Research, pp. 437–459. PMLR, 2025. URL [https://proceedings.mlr.press/v305/xu25b.html](https://proceedings.mlr.press/v305/xu25b.html).

Yufei Ye, Abhinav Gupta, Kris Kitani, and Shubham Tulsiani. G-hop: Generative hand-object prior for interaction reconstruction and grasp synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 1911–1920, 2024. doi: 10.1109/CVPR52733.2024.00187.

Xinyu Zhan, Lixin Yang, Yifei Zhao, Kangrui Mao, Hanlin Xu, Zenan Lin, Kailin Li, and Cewu Lu. Oakink2: A dataset of bimanual hands-object manipulation in complex task completion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 445–456, June 2024. doi: 10.1109/CVPR52733.2024.00050.

Gu Zhang, Qicheng Xu, Haozhe Zhang, Jianhan Ma, Long He, Yiming Bao, Zeyu Ping, Zhecheng Yuan, Chenhao Lu, Chengbo Yuan, Tianhai Liang, Xiaoyu Tian, Maanping Shao, Feihong Zhang, Mingyu Ding, Yang Gao, Hao Zhao, Hang Zhao, and Huazhe Xu. Unidex: A robot foundation suite for universal dexterous hand control from egocentric human videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 1841–1852, June 2026.

Jinglei Zhang, Jiankang Deng, Chao Ma, and Rolandos Alexandros Potamias. Hawor: World-space hand motion reconstruction from egocentric videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 1805–1815, 2025.

## APPENDIX

## A ADDITIONAL DETAILS OF CONTACT-AWARE HAND-OBJECT RECONSTRUCTION

## A.1 HAND-ANCHORED INITIALIZATION

**Scale registration and camera consistency.** For hand h at frame t, let $\mathcal { P } _ { h , t }$ contain pixels supported by both the observed visible-hand mask and the rendered MANO surface. The sequence-level scale in Eq. 1 is estimated by robustly aggregating

$$
r _ { h , t } ( u ) = \operatorname { l o g } z _ { h , t } ^ { H } ( u ) - \operatorname { l o g } D _ { t } ^ { M } ( u ) , \qquad u \in \mathcal { P } _ { h , t } ,\tag{8}
$$

over valid observations from both hands and all frames. The resulting factor $s _ { \mathrm { s e q } }$ is applied to the MoGe-3 depth and point map.

For each object, we back-project the aligned depth within its visible masks to 3D and robustly estimate its observed spatial extent over multiple frames. A single uniform scale $s _ { j }$ is then applied to the canonical mesh $\mathcal { V } _ { j }$ for the entire sequence. The same aligned depth is also provided as the metric-depth input to the DROID-SLAM module used by HaWoR, so that camera translation is recovered under the same scale as the reconstructed hands and objects. Object tracking is subsequently performed using the scaled meshes and aligned scene geometry.

**Object pose initialization and tracking.** FoundationPose (Wen et al., 2024) initializes each object in its first valid frame and refines subsequent pose hypotheses. For later frames, we combine a camera-motion propagation hypothesis with rigid-motion hypotheses fitted from SpatialTracker (Xiao et al., 2024) point tracks. FoundationPose refines and scores these hypotheses to select the frame-wise object pose. Under severe hand occlusion, we avoid unconstrained full-pose recovery: translation-only refinement is used when sufficient image evidence remains, while otherwise the propagated pose is retained. This reduces abrupt pose changes when most of the object surface is unobserved.

14

<!-- page 15 of 22 -->

**Source-view alignment.** The lightweight alignment in Eq. 2 jointly considers the two hands and two manipulated objects. For $\mathcal { L } _ { \mathrm { 2 D } }$ , each entity is rendered independently and combined using depthaware soft visibility. The loss compares the resulting visible regions with the observed masks using semantic, silhouette, boundary, and per-entity coverage terms. Ambiguous mask overlaps and invalid hand observations are excluded. When an object projection loses most of the support from its observed mask, an additional projection-capture term discourages the optimizer from moving the object out of view.

**Pairwise relative-depth constraints.** We partition reliable visible regions into local patches and compare the relative depths of patch pairs. Let $Z _ { a , p }$ and $Z _ { b , q }$ denote their current rendered depths and let $\bar { D } _ { a , p }$ and $\bar { D } _ { b , q }$ denote robust estimates from the aligned MoGe-3 depth. We define

$$
\Delta Z _ { a p , b q } = Z _ { b , q } - Z _ { a , p } , \qquad \Delta D _ { a p , b q } = \bar { D } _ { b , q } - \bar { D } _ { a , p } .\tag{9}
$$

The ordering term penalizes disagreement with the observed front–back relation,

$$
\ell _ { \mathrm { o r d } } ^ { a p , b q } = w _ { \mathrm { o r d } } ^ { a p , b q } \rho \left( \frac { \left[ m _ { \mathrm { o r d } } - \mathrm { s i g n } ( \Delta D _ { a p , b q } ) \Delta Z _ { a p , b q } \right] _ { + } } { \sigma _ { \mathrm { o r d } } } \right) ,\tag{10}
$$

while the gap term penalizes disagreement in relative depth,

$$
\ell _ { \mathrm { g a p } } ^ { a p , b q } = w _ { \mathrm { g a p } } ^ { a p , b q } \rho \left( \frac { \Delta Z _ { a p , b q } - \Delta D _ { a p , b q } } { \sigma _ { \mathrm { g a p } } } \right) .\tag{11}
$$

The two losses are aggregated over valid hand-hand, hand-object, and object-object patch pairs. Missing rendered support receives a fixed penalty rather than being ignored, preventing the optimizer from reducing the loss by moving an entity outside the observed region.

**Contact-frame estimation.** For each hand, we combine HACO predictions with the proximity of the observed hand and its manipulated object. We first project MANO vertices into the source image and identify vertices near both the hand and object masks, excluding vertices more strongly associated with other objects. Let $H _ { h , t } \in [ 0 , 1 ]$ denote the aggregated HACO probability of these vertices, and let $P _ { h , t } \in [ \bar { 0 } , 1 ]$ denote the normalized proximity of the two masks. We compute

$$
s _ { h , t } = \left( 0 . 6 5 H _ { h , t } + 0 . 3 5 P _ { h , t } \right) P _ { h , t } .\tag{12}
$$

We apply a five-frame median filter to the resulting scores and identify contact intervals using a support threshold of 0.38 and a high-confidence threshold of 0.68. Each retained interval must contain at least three high-confidence frames. Short gaps of up to eight frames are merged when the hand and object remain sufficiently close. The resulting intervals define the contact-frame set $\mathcal { T } _ { h } ^ { \mathrm { c } }$ in Eq. 3.

**Temporal regularization and support grounding.** Frame-wise translation corrections are regularized temporally in the common world frame, while a weak initialization prior discourages unnecessary displacement. We estimate the world-up direction from dominant background surface normals. For each manipulated object, task frames outside the corresponding $\mathcal { T } _ { h } ^ { \mathrm { c } }$ provide evidence for estimating its stable support configuration and the shared support height. The resulting object poses are adjusted to the inferred support surface during non-contact intervals, while allowing the objects to leave the surface during contact. The hand trajectories remain unchanged during grounding.

## A.2 MULTI-START POSE REFINEMENT

**Starting-pose selection.** We generate candidates independently for each hand-object pair. All starts share an object baseline obtained by a short pose refinement under image, relative-depth, and support constraints. For frames $t \in \mathcal { T } _ { h } ^ { \mathrm { c } }$ , we select MANO vertices with HACO probabilities above 0.8 and scan hand translations over $[ - \overset { n } { 8 } 5 , 8 5 ]$ mm at 5 mm intervals. We retain five starts per contact frame, including the unshifted pose. For other task frames, only the unshifted pose is used.

The MANO patches remain fixed throughout each scan. For candidate k and patch $p ,$ we compute a histogram $\mathbf { h } _ { k , p }$ of the selected intersection ranks along the camera rays, normalized by the number

15

<!-- page 16 of 22 -->

of vertices with valid intersections. We also record the fraction $f _ { k , p }$ of rays that hit the object. The histogram uses nine bins: ranks $0 , \ldots , 7$ and a final bin for ranks $\geq   8 ;$ patches without hits have zero histograms. These quantities form the comparison vector $\mathbf { s } _ { k } = \operatorname { c o n c a t } _ { p } [ \mathbf { h } _ { k , p } , f _ { k , p } / 4 ]$

Let $\mathcal { S }$ contain the indices of retained starts, initially only the zero-offset candidate. While candidates remain sufficiently different, we select

$$
k ^ { \star } = \operatorname { a r g } \operatorname* { m a x } _ { k \notin \mathcal { S } } \left[ \operatorname* { m i n } _ { \ell \in \mathcal { S } } \frac { \| \mathbf { s } _ { k } - \mathbf { s } _ { \ell } \| _ { 1 } } { \operatorname { d i m } ( \mathbf { s } _ { k } ) } - \lambda _ { q } \widetilde { q } _ { k } \right] , \qquad \mathcal { S } \leftarrow \mathcal { S } \cup \{ k ^ { \star } \} ,\tag{13}
$$

where $\lambda _ { q }   =   0 . 0 5$ . The quality cost is $q _ { k } { = } f _ { k } \bar { g } _ { k } + ( 1 - f _ { k } ) c _ { \operatorname { m i s s } } ,$ , where $f _ { k }$ is the overall ray-hit fraction, $\bar { g } _ { k }$ is the mean distance between valid vertex-intersection pairs, and $c _ { \mathrm { m i s s } }$ is a fixed cost for missed rays. We obtain $\widetilde { q } _ { k }$ by min–max normalization over the entire scan, setting it to zero when all costs are equal. When the remaining candidates have negligible diversity, we instead select by increasing $q _ { k }$ , preferring offsets at least 10 mm from all retained offsets. If none satisfies this spacing, all remaining offsets are considered.

**Candidate optimization.** Each candidate jointly optimizes the wrist and its bound object, while the two hand-object pairs are solved independently. Translation corrections are defined in the camera frame, and rotation corrections left-multiply the initial orientation. MANO articulation and shape, object geometry and scale, and camera parameters remain fixed. Translation and rotation corrections are bounded by 100 mm and $1 0 ^ { \circ }$ for the wrist, and 50 mm and $1 2 ^ { \circ }$ for the object.

The source-view term in $\mathcal { L } _ { \mathrm { b a s e } }$ combines silhouette overlap and boundary alignment, using the other entity masks to exclude occluded regions. Relative-depth constraints follow $\mathrm { A _ { J } }$ ppendix A.1, but compare only the current hand-object pair using $\widetilde { D } _ { t }$ . Bidirectional penetration penalties limit interpenetration; table constraints preserve object support and discourage hand-table penetration.

We use the contact transition in Eq. 4. Ray targets remain fixed in object coordinates and follow the current object pose; vertices without valid ray targets use nearest-surface distance. Normal compatibility is introduced as the surface contact term replaces the ray term. Temporary pose anchors are reduced with the ray term. For task frames outside $\mathcal { T } _ { h } ^ { \mathrm { c } }$ , contact attraction is disabled, while the applicable image, depth, collision, and support constraints remain active. Temporal regularization is disabled during independent candidate optimization.

**Candidate validation and temporal selection.** After independent optimization, we first discard numerically invalid candidates and check contact and penetration quality. Candidates satisfying the contact-distance criterion are preferred, while penetration checks prevent strongly colliding solutions from dominating the selection. If no fully safe candidate is available, we retain the finite candidate with the smallest violation rather than discarding the frame. The zero-offset start receives no special preference, and candidates are not required to improve over the original initialization under a single global objective.

The remaining candidates are evaluated using a common geometric criterion that includes sourceview consistency, relative depth, contact geometry, penetration, and support constraints. Candidatespecific ray targets and temporary initialization anchors are excluded from this score. Let $U _ { t } ^ { k }$ denote the resulting unary cost for candidate k at frame t. We then select a temporally coherent path using second-order dynamic programming:

$$
\{ k _ { t } ^ { * } \} = \operatorname { a r g } \operatorname* { m i n } _ { \{ k _ { t } \} } \left[ \sum _ { t } U _ { t } ^ { k _ { t } } + \sum _ { t } E _ { \operatorname { p a i r } } ( k _ { t } , k _ { t + 1 } ) + \sum _ { t } E _ { \operatorname { a c c } } ( k _ { t - 1 } , k _ { t } , k _ { t + 1 } ) \right] .\tag{14}
$$

Here, $E _ { \mathrm { p a i r } }$ penalizes inconsistent relative hand-object states across adjacent frames, while $E _ { \mathrm { a c c } }$ penalizes abrupt wrist and object motion. These terms depend on the optimized geometry rather than on changes in candidate indices. The selected path is finally refined with temporal regularization in the common world frame. This refinement also covers task frames outside $\mathcal { T } _ { h } ^ { \mathrm { c } }$ , with additional motion-preserving corrections and swept collision checks where applicable.

## A.3 LOCAL CONTACT REFINEMENT

**Stable contact construction.** We follow the canonical-space stabilization strategy of C2Dex (Ren et al., 2026). Within $\mathcal { T } _ { h } ^ { \mathrm { c } }$ , temporally filtered HACO predictions are used to select candidate hand

16

<!-- page 17 of 22 -->

vertices, and small disconnected components are removed before ray casting. For each candidate vertex, we cast a ray from the camera center through its initial 3D position. Among the intersections with the bound object, we retain those within 15 mm in ray depth and with opposing-normal compatibility $- \mathbf { n } _ { h } ^ { \top } \mathbf { n } _ { o } \stackrel { \circ } { > } 0 . 5$ , then select the intersection with the strongest normal match.

The selected points are transformed to canonical object coordinates and grouped into locally stable interaction segments according to hand articulation and relative hand-object motion. For each MANO vertex within a segment, we cluster its canonical observations using DBSCAN with $\epsilon = 5$ mm and min samples = 3. The dominant non-noise cluster is retained, and its medoid defines the stable object-side target. Using the medoid keeps the target on an observed object surface rather than introducing an averaged point away from the mesh.

**Opposition supplementation.** Within $\mathcal { T } _ { h } ^ { \mathrm { c } }$ , when stable contacts provide insufficient opposing support, we supplement them according to the pad, palm, and side opposition types of the GRASP taxonomy (Feix et al., 2016). We use a simplified semantic partition of MANO adapted from the hand-part regions of ContactPrompt (Jung & Lee, 2026). Depending on the opposition type, supplementary vertices are drawn from the corresponding thumb, finger-pad, palm, or side regions and prioritized by HACO confidence. Their object-side targets are obtained from the nearest compatible surface point subject to distance and normal constraints.

**Optimization.** The object trajectory and MANO shape remain fixed. For each hand, we optimize the wrist translation, wrist rotation, and all 15 MANO joint rotations over all task frames $\mathcal { T } _ { h }$ , initialized from the Multi-Start Pose Refinement result. In implementation, the contact term in Eq. 5 is aggregated over contact patches:

$$
\mathcal { L } _ { \operatorname { c o n t a c t } } = \frac { 1 } { | \mathcal { T } _ { h } ^ { c } | } \sum _ { t \in \mathcal { T } _ { h } ^ { c } } \sum _ { p } W _ { t , p } \frac { \sum _ { i \in \mathcal { P } _ { t , p } } a _ { t , i } \left\| \mathbf { v } _ { t , i } - \mathbf { y } _ { t , i } ^ { * } \right\| _ { 2 } ^ { 2 } } { \sum _ { i \in \mathcal { P } _ { t , p } } a _ { t , i } } .\tag{15}
$$

where $\mathcal { P } _ { t , p }$ is a patch of vertices with valid contact targets, including supplementary opposition contacts when applicable. The vertex weight $a _ { t , i }$ combines HACO confidence and temporal support, while $W _ { t , p }$ controls the contribution of each patch. This prevents densely sampled contact regions from dominating the refinement.

We penalize penetration in both directions. For hand-to-object penetration, let $d _ { o , t } ( \mathbf { v } )$ denote signed distance to the object surface, positive outside:

$$
\mathcal { L } _ { h \to o } = \frac { 1 } { | \mathcal { T } _ { h } | } \sum _ { t \in \mathcal { T } _ { h } } \frac { 1 } { N _ { h } } \sum _ { i } \left[ - d _ { o , t } ( \mathbf { v } _ { t , i } ) \right] _ { + } .\tag{16}
$$

The reverse term penalizes sampled object-surface points that lie inside the closed hand mesh. Additional regularization limits deviation from the coarse wrist and articulation parameters, penalizes second-order temporal variation, enforces anatomical joint-angle plausibility, and prevents handtable penetration. For frames outside $\mathcal { T } _ { h } ^ { \mathrm { c } }$ , contact attraction is disabled, but wrist pose and articulation are still refined using the remaining penetration and regularization terms.

We optimize each hand independently with Adam for 1000 iterations. Wrist translation, wrist rotation, and articulation use learning rates 0.001, 0.005, and 0.0055, with residual bounds of 60 mm, 0.35 rad, and 0.45 rad per MANO joint, respectively.

## B DETAILS OF DEMONSTRATION SYNTHESIS AND BACKGROUND TRANSFER

## B.1 DEMONSTRATION SYNTHESIS

**Background removal and shadow transfer.** The removal masks combine the reconstructed hand and object masks with additional arm and sleeve segmentations. We apply video inpainting with SVOR (Hu et al., 2026) and use its native prediction as the background. The simulation foreground is the union of the two dexterous-hand and manipulated-object segmentation masks.

To preserve interaction shadows, we render the scene with and without the simulated hands and objects under identical camera and lighting conditions. Their linear-image intensity ratio defines

17

<!-- page 18 of 22 -->

a table-region shadow transmittance $\tau _ { t }$ . We apply the resulting attenuation to the inpainted background before the binary compositing in Eq. 7. The foreground itself is not geometrically warped or modified by a generative model.

**Temporal correspondence.** The synthesized video is rendered at 30 Hz, whereas policy control and physics simulation run at 60 Hz and 120 Hz, respectively. Each video frame stores its associated simulator-state and source-camera indices, while the complete control-rate trajectory is retained separately. The recorded trajectory contains robot and object states, policy actions, actuator targets, wrist wrenches, and transition outcomes.

## B.2 BACKGROUND TRANSFER

**Scene placement.** Let A and B denote the source and target task coordinate systems. For each scene, we define a table-aligned task frame F whose origin is the midpoint of the two initial object positions projected onto the support plane. Its x-axis follows the direction between the two objects, while the z-axis is the world-up direction. The initial source-to-target placement is

$$
\mathbf { T } _ { B \leftarrow A } ^ { 0 } = \mathbf { F } _ { B } \mathbf { F } _ { A } ^ { - 1 } .\tag{17}
$$

An optional fixed in-plane translation $\mathbf { D } _ { A }$ may be applied for framing,

$$
\mathbf { T } _ { B \leftarrow A } = \mathbf { T } _ { B \leftarrow A } ^ { 0 } \mathbf { D } _ { A } .\tag{18}
$$

The same transform is shared by both hands and all manipulated objects. No scale change or framewise alignment is applied.

Rather than rewriting the recorded source states, we equivalently express the target camera in the source task frame. If $\mathbf { C } _ { B } ( j )$ is the target camera-to-world pose at target frame $j ,$ the rendering camera for output frame i is

$$
\mathbf { C } _ { A } ( i ) = \mathbf { T } _ { B \leftarrow A } ^ { - 1 } \mathbf { C } _ { B } ( j ( i ) )   .\tag{19}
$$

Target intrinsics are used for rendering, and the target support proxy is transformed consistently with the camera and scene placement.

**Temporal mapping.** The source execution remains on its original 60 Hz state and control trajectory, while the synthesized video is rendered at 30 Hz. We therefore use every second recorded source state. When the available target-background segment is shorter than the source output, target RGB frames and camera poses are sampled uniformly over the available segment using the same frame mapping; the source execution itself is not time-warped.

The target support representation is taken from the configured background scene rather than reestimated during background transfer. Accordingly, this stage preserves the original executed handobject geometry and its motion relative to a compatible support plane, but does not re-simulate contact dynamics in the target scene.

## C RECONSTRUCTION EVALUATION PROTOCOL

We report ten reconstruction metrics: $\mathrm { M P J P E _ { R A } }$ , CD, F@5, ATE, ARE, RTE, RRE, $\mathrm { C D } _ { h }$ , SD, and $\hat { D } _ { \mathrm { p e n } } ^ { \mathrm { m a x } }$ . Unless otherwise specified below, all methods are evaluated using the same metric implementation and aggregation protocol.

**Hand pose and object geometry.** Following Fan et al. (2024), hand pose is evaluated using $\mathrm { M P J P E _ { R A } }$ over 21 MANO keypoints. Prediction and ground truth are independently translated to their wrist roots before comparison; no rotational, scale, or Procrustes alignment is applied.

For object geometry, we uniformly sample 20,000 surface points by triangle area from both the reconstructed and ground-truth canonical meshes. A proper Sim(3) transformation, consisting of rotation, translation, and uniform scale without reflection, is estimated once per object and kept fixed. CD is the sum of the two mean nearest-surface Euclidean distances and is therefore reported in centimeters rather than squared centimeters. F@5 uses a 5 mm nearest-surface threshold.

18

<!-- page 19 of 22 -->

**Object trajectory.** We evaluate object motion using ATE, ARE, RTE, and RRE following Shin et al. (2026). To avoid dependence on arbitrary canonical object origins, translation trajectories are represented by the area-weighted surface centroids of the posed meshes.

For each demonstration, all manipulated objects share a single fixed-scale rigid alignment between the predicted and ground-truth worlds. The alignment is estimated jointly from all valid objectcenter correspondences, with each object given equal total weight. No scale adjustment is allowed. This alignment is estimated once per demonstration and is never re-estimated independently for individual objects or frames.

The canonical orientation of a reconstructed object can differ from that of its ground-truth mesh by a constant rotation. We therefore remove one fixed orientation offset using the first common valid frame of each object and keep it unchanged throughout the sequence. No frame-wise rotational alignment is applied. ARE and RRE use the standard geodesic distance on $S O ( 3 )$ ; object symmetry is not explicitly modeled. RTE and RRE are computed only between consecutive valid prediction frames and never bridge missing frames.

DexImit produces trajectories at a temporal stride of three frames. Its trajectory metrics are therefore evaluated on its native sampled frames, and its relative errors correspond to three-frame intervals; these entries are marked accordingly in the table.

MagicHOI reconstructs its hand-object branches under independent SfM gauges and does not provide a common reconstructed camera trajectory. For MagicHOI only, we use the ground-truth camera trajectory to transform its valid camera-frame object predictions into the ground-truth world before computing the same trajectory metrics. These GT-camera-assisted results are marked with ∗ in the table.

**Hand-object interaction.** We evaluate relative hand-object geometry using $\mathbf { C D } _ { h }$ (Fan et al., 2024). At each valid frame, the predicted and ground-truth posed object surfaces are expressed relative to their corresponding wrist roots in the camera frame, after which the same unsquared bidirectional Chamfer distance as above is computed. No additional per-frame scale, rotation, or rigid alignment is applied.

Physical stability is evaluated using a GraspTTA-style simulation displacement (SD) protocol (Jiang et al., 2021). For each evaluated hand-object configuration, the hand is held static and the object is simulated dynamically in PyBullet for 35 steps at 1/240 s. The same gravity, object mass, contact, solver, and collision settings are used for all methods. SD measures the displacement of the simulated object between its initial and final states.

We additionally report maximum penetration depth $D _ { \mathrm { p e n } } ^ { \mathrm { m a x } }$ (Hasson et al., 2019). For each frame, we measure the largest penetration of MANO vertices into the posed object surface; frames without penetration contribute zero. The reported value is the mean of these frame-wise maxima over the evaluated sequence.

**Aggregation and validity.** Metrics are aggregated with equal weighting across the relevant hands or objects within each demonstration, followed by equal weighting across demonstrations. Geometry metrics are computed once for each reconstructed canonical object, whereas pose and interaction metrics are evaluated over their valid temporal support. Missing predictions are not interpolated. When a method does not produce the output required by a metric, the corresponding entry is reported as N/A. For methods with incomplete reconstruction coverage, we additionally report the number of valid demonstrations or branches used for evaluation.

## D VISUAL SYNTHESIS BASELINE PROTOCOLS

**Source-background synthesis.** For Kiwi-Edit (Lin et al., 2026), we provide the complete source human-manipulation video together with a reference image showing XHand dexterous hands mounted on Rainbow Robotics RB-Y1 arms. We use the following instruction for all videos:

<small><span class="docvortex-page-footnote" data-block-type="page_footnote" style="color:#6b7280">“Replace the human hands and arms with XHand dexterous hands mounted on Rainbow Robotics RB-Y1 robot arms, performing the same manipulation task. Keep the original background and all other scene elements unchanged.”</span></small>

19

<!-- page 20 of 22 -->

The generated video is used directly without additional geometric correction or post-processing.

For GPT-6 Astra (OpenAI, 2026), each source frame is edited using the human frame as the manipulation reference and the same RB-Y1–XHand reference image(s) used to specify the target embodiment. We use the following fixed prompt:

“Edit the first image. Replace the visible human hand(s) manipulating the object with XHand dexterous hand(s) mounted on a Rainbow Robotics RB-Y1 dual-arm robot, matching the reference image(s). If any wrist, forearm, or arm is visible, replace it with the corresponding RB-Y1 robot arm segment and keep the XHand-arm connection physically plausible. Preserve plausible hand-object contact. Do not add text, extra objects, or a collage layout.”

We retain the first generated result for each frame and perform no manual correction or result selection.

**Target-background synthesis.** For the target-background experiment, GPT-6 Astra receives three visual inputs for each sampled frame: the human source frame, one fixed image of the target realrobot workspace, and one fixed robot reference image based on the same UR5e and XHand assets used in EgoMatrix. The source frame specifies the manipulation action, manipulated objects, handobject interaction, and interaction stage, while the fixed target image specifies the camera viewpoint, workspace, background, and overall composition. The robot and object states already present in the target image are not treated as references for the desired manipulation. The robot reference specifies the robot geometry, proportions, mounting, and colors, rather than the desired pose or scene layout.

The target embodiment is fixed to two UR5e arms, each equipped with an XHand dexterous hand. We use the same target image, robot reference image, and following prompt for every source frame, with the three images supplied in the order described above:

“Use image 1, the current source frame, as the reference for the manipulation action, manipulated objects, hand-object interaction, and interaction stage. Edit image 2, the fixed target workstation image, so that a bimanual robot consisting of two XHand dexterous hands, each mounted on a UR5e robot arm, performs the same manipulation shown in image 1. Image 3 is a fixed visualization of the supplied assembled UR5e + XHand assets; use it only as a reference for the robot geometry, proportions, mounting, and colors, not for the robot pose, camera, lighting, background, or manipulation state.

Match the manipulated objects, their poses, the grasp or contact configurations, and the stage of the manipulation to image 1 as closely as possible, expressed from image 2’s camera viewpoint. Replace the robot configuration and manipulated-object state already visible in image 2 whenever necessary; they do not define the desired action. Reproduce both hands’ involvement when the source manipulation is bimanual. If only one hand is visibly involved, do not force the second hand to become prominent, but keep the embodiment consistent with two UR5e arms and two mounted XHands.

Preserve image 2’s camera viewpoint, workspace, background, table, overall scene composition, and visual appearance. Match the robot embodiment shown in image 3 while depicting it under the target workstation’s illumination. The UR5e arm segments must connect naturally to the XHand hands and remain physically plausible from the target viewpoint.

Do not copy image 1’s viewpoint or background. Do not preserve image 2’s robot pose when it conflicts with the source manipulation. Do not copy image 3’s demonstration pose or backdrop. Do not add text, unrelated objects, extra robot limbs, or a collage layout. Return a single edited target-workstation image at 1920x1080 pixels in landscape 16:9 format.”

Each frame is generated independently with the same prompt and fixed reference images. We use one generation per frame, retain the first result, and apply no manual selection or post-processing. No EgoMatrix output, previously generated Astra frame, or additional reference image is provided.

## E ROBOT DEPLOYMENT

We deploy the transferred trajectories on two real-robot platforms. The first platform consists of two UR5e arms, each equipped with an XHand dexterous hand. The second uses the dual-arm system

20

<!-- page 21 of 22 -->

of a Rainbow Robotics RB-Y1, also equipped with XHand dexterous hands. Fig. 8 shows the two hardware setups used in our experiments.

![Image block](doc:5814607/tier:basic/page:21/block:2)

(a) Dual UR5e arms with XHand dexterous hands.

![Image block](doc:5814607/tier:basic/page:21/block:4)

(b) Rainbow Robotics RB-Y1 with XHand.

Figure 8: **Real-robot platforms used for trajectory replay.** We evaluate transferred trajectories on two embodiments equipped with XHand dexterous hands: a dual-UR5e setup and the dual-arm Rainbow Robotics RB-Y1 platform.

Fig. 9 shows five representative real-world replay sequences. For each task, the upper row contains frames sampled from the source human demonstration, while the lower row shows the corresponding stages of the transferred trajectory replayed on the real robot. The examples include demonstrations collected by us as well as demonstrations from public egocentric datasets.

21

<!-- page 22 of 22 -->

Pour into Another Cup

![Image block](doc:5814607/tier:basic/page:22/block:2)

Figure 9: Real-world trajectory replay across representative tasks. For each task, the top row shows the source human demonstration and the bottom row shows the corresponding real-robot replay. Time progresses from left to right.

22