# 2607.05396 (from arXiv HTML; MinerU fallback)



# From Fixed to Free Cameras: Calibration-Free View-Robust Vision-Language-Action Model

Wenhao Li

Affiliation: Nanyang Technological University

  
Xueying Jiang

Affiliation: Nanyang Technological University

  
Quanhao Qian

Affiliation: DAMO Academy, Alibaba Group

Affiliation: HuPan Lab

  
Deli Zhao

Affiliation: DAMO Academy, Alibaba Group

Affiliation: HuPan Lab

  
Shijian Lu

Affiliation: Nanyang Technological University

  
Gongjie Zhang

Affiliation: Alibaba Group

  
Ran Xu

Affiliation: DAMO Academy, Alibaba Group

Affiliation: HuPan Lab

###### Abstract

Real-world robot deployment rarely maintains the training-stage camera setup, where cameras often experience repositioning or remounting depending on actual scenarios. Existing view-robust Vision-Language-Action (VLA) policies tolerate such camera variations only when the camera extrinsics are explicitly provided, making them fragile and hard to use especially when view robustness is critical. We argue that the policy should not be told where the camera is, but rather figure it out by itself. To this end, we introduce Camera-Centric VLA (CamVLA), a new VLA model that decouples manipulation controls from camera geometry by predicting (i) a camera-centric end-effector action expressed in the local camera frame, and (ii) a 6-DoF hand-eye matrix relating cameras to the robot base. A deterministic geometric transformation composes the two predictions into a robot base-frame action. This disentangles how I should move in pose-independent camera-centric action generation from where I am looking from in camera-perspective geometric grounding. The resulting policy is calibration-free, depth-free, and single-view, requiring only a single monocular RGB image as the visual observation and task instruction at deployment. Evaluations in both simulation and real-world robot data show that CamVLA consistently improves success rates across diverse unseen viewpoints. Project page: [https://alibaba-damo-academy.github.io/CamVLA/](https://alibaba-damo-academy.github.io/CamVLA/).

Keywords: Vision-Language-Action Models, Viewpoint Robustness, Calibration-Free Manipulation

## 1 Introduction

Vision-Language-Action (VLA) models [6, 25, 17, 4] have rapidly progressed toward generalist robot policies, leveraging internet-scale vision-language data and diverse robotic demonstrations to ground broad semantic knowledge into directly executable manipulation.
Yet despite their semantic competence, state-of-the-art VLAs exhibit a sharp and unexpected brittleness to camera viewpoint shifts.

As illustrated in Figure 1, $\pi_{0}$ [5] trained on a single canonical perspective achieves a $\sim$65.3% success rate under its training view on RLBench [20], yet collapses to a mere 6.3% under a 15∘ camera rotation.
This failure persists even when the scene remains fully observable and the semantic goal is unchanged.
Although large-scale multi-view training could mitigate this, acquiring such data is prohibitively expensive and hard to scale.
In practice, real-world robot deployment rarely matches the controlled camera setup of training time: sensors get bumped during operation, mounted on different platforms, hand held by operators, or affixed to mobile bases whose pose drifts continuously.
Consequently, without inherent view robustness, VLAs remain tethered to static laboratory setups, failing to generalize to the dynamic and unconstrained configurations of real-world deployment.

This brittleness has a structural origin.
Standard VLAs [26, 5, 4] rigidly predict actions in the robot base frame from camera-perspective visual observations.
However, this base-frame parameterization misaligns action outputs with camera-frame inputs, requiring the network to implicitly resolve the spatial transformation from the camera to the robot base (hand-eye transformation).
Without explicit geometric constraints, this hand-eye transformation remains a hidden variable, forcing the policy to memorize coordinate mappings by coupling manipulation control with camera geometry, which easily collapses under minor viewpoint shifts.
Recent works converge on a single recipe to fix this by telling the policy where the camera is.
For instance, OC-VLA [46] bypasses this frame-of-reference gap by re-expressing targets in calibrated camera coordinates;
Jiang et al. [23] rely on ray embeddings derived from known camera parameters; 4D-VLA [45] back-projects pixels into the robot frame using known intrinsics and extrinsics; and AnyCamVLA [16] synthesizes canonical views given both source and target camera poses.
Despite their architectural diversity, all of these methods require known and accurate camera extrinsics at deployment, which is precisely the assumption that breaks under hand-held, drifting, or remounted cameras.
Existing view-robust VLAs are therefore most fragile in the deployment regimes that motivate view robustness in the first place.

*Figure 1: The Viewpoint Trap in VLAs.
Conventional VLA (e.g., $\pi_{0}$) trained on a single viewpoint exhibits extreme spatial brittleness, where a mere 15∘ camera shift drops success rates to 6.3%.*

In this work, we argue that the policy should not be told where the camera is, but rather should figure it out by itself.
This self-localizing capability finds a natural analogue in human cognition: human vision-guided manipulation operates natively in an egocentric reference frame, while a complementary allocentric system maintains an implicit understanding of head pose relative to the torso [15, 12].
Inspired by this biological factorization, we decouple the VLA policy into two distinct subproblems, both inferable from a single monocular RGB image:
(i) Camera-Centric Action Generation answers “How should I move?” by predicting end-effector actions natively in the local camera coordinate frame, which aligns directly with visual observations, making the mapping inherently independent of the external camera pose.
(ii) Camera-Perspective Geometric Grounding answers “Where am I looking from?” by regressing the 6-DoF hand-eye matrix that relates the observing camera to the robot base, explicitly modeling the relative spatial geometry and isolating viewpoint-dependent variation within a single relative pose.

We instantiate this factorization as CamVLA (see Figure 2), a calibration-free, camera-centric VLA model.
From the extracted visual representations, two specialized heads predict the camera-centric action and the hand-eye matrix in parallel, and a deterministic geometric transformation composes them into a robot base-frame action.
Parameterizing actions natively in the camera frame aligns them directly with visual observations, avoiding the geometric entanglement of base-frame policies that map conflicting visual flows to base coordinates.
By absorbing all viewpoint variability into a single learned 6-DoF pose, CamVLA presents a simpler regression task than memorizing view-dependent coordinate mappings.
Crucially, it requires no external camera information at deployment (e.g., calibrated extrinsics, depth sensors, or view synthesis), relying exclusively on the same monocular RGB image and language command that standard VLA already receives.
By replacing the assumption of given geometry with the discipline of learned geometry, CamVLA closes the deployment gap left open by prior view-robust methods, enabling calibration-free manipulation under uncalibrated viewpoint shifts on real hardware.

Our contributions can be summarized as threefold.
First, we identify that existing view-robust VLA approaches share a common, deployment-fragile assumption of known camera extrinsics, and we argue for a self-localizing alternative that infers camera geometry directly from RGB.
Second, we propose CamVLA, a calibration-free, depth-free, single-view VLA framework that decouples camera-centric action generation from camera-perspective geometric grounding and recombines them via a deterministic geometric transformation.
Third, we comprehensively evaluate CamVLA in both simulation and real-world deployment, demonstrating substantial success rate improvements over strong VLA baselines (e.g., $\pi_{0}$ [5] and GR00T N1.7 [4]) across diverse unseen camera configurations.

## 2 Related Work

Vision-Language-Action Models.
The pursuit of generalist robot policies has catalyzed the development of VLAs [50, 14, 2, 11, 18], which adapt VLMs [30, 27, 49, 3] by predicting continuous robot actions via large-scale transformers.
Octo [40] utilizes a Transformer backbone to handle multi-modal observations and a diffusion head [9] for robust action generation.
Recent flow-matching architectures such as $\pi_{0}$ [5] and $\pi_{0.5}$ [19] have advanced high-frequency continuous control, open-world generalization, and learning from real-world experience.
GR00T N1 [4] employs a dual-system architecture to coordinate high-level reasoning with low-level motor commands for humanoid control.
Fast-in-Slow [8] further explores the unification of fast manipulation within a slow reasoning system to improve real-time responsiveness.
While these models possess strong semantic knowledge, their generalization capabilities remain limited under novel camera viewpoints.

Viewpoint Generalization in Robotics.
Achieving viewpoint robustness remains a critical bottleneck for robot policies, whose performance can degrade sharply when the deployment camera pose differs from the training setup [28].
One line of work introduces explicit 3D structure to improve geometric consistency.
For example, Perceiver-Actor [36] builds voxelized RGB-D scene representations, and recent VLA models incorporate 3D features, depth/point-cloud priors, or spatiotemporal 3D representations [48, 22, 47, 24, 13].
However, these approaches often introduce computational overhead and depend on depth sensing, multi-view observations, or calibrated camera geometry.
Alternative approaches improve robustness via novel-view synthesis or image-level augmentation [41, 10, 44], viewpoint selection [43], and view-invariant representations or latent actions [33, 21, 31, 29, 39, 1].
Several concurrent works mitigate viewpoint shifts via camera-aware policy conditioning, including grounding actions in the camera space [46], conditioning on camera parameters via ray embeddings [23], back-projecting pixels into the robot base frame using known camera parameters [45], or synthesizing canonical views based on relative camera poses [16].
Unlike these camera-aware methods that all heavily rely on known and accurate camera intrinsics or extrinsics at deployment, CamVLA self-predicts the hand-eye matrix directly from monocular RGB, achieving view-robust manipulation without external calibration, depth sensing, novel-view synthesis, or 3D reconstruction.

![Refer to caption](drafts/images/camvla-2607.05396/arch.png)

*Figure 2: Overview of the CamVLA Architecture.
Our CamVLA predicts the local camera-centric action and the 6-DoF hand-eye pose in parallel, which are then combined via a deterministic geometric transformation to execute the base-frame action.*

## 3 Methodology

As shown in Figure 2, CamVLA achieves viewpoint robustness by decoupling the policy into two parallel components: (i) an Action Head that predicts end-effector actions natively in the camera coordinate frame, and (ii) a Geometric Head that regresses the 6-DoF hand-eye matrix to estimate the camera pose relative to the robot base.
A deterministic transformation then composes these two outputs into a base-frame action for execution.
This decoupling design factorizes the VLA policy into pose-independent action generation and viewpoint-dependent hand-eye grounding, isolating viewpoint variations from the core visual-action mapping.
In the following subsections, we review the standard VLA formulation and detail our decoupling pipeline.

### 3.1 Standard VLA Action Formulation

At each timestep $t$, the VLA model receives a visual observation $I_{t}\in\mathbb{R}^{H\times W\times 3}$, a proprioceptive state $s_{t}=[p_{b,t},r_{b,t}]$, and a natural language goal $L$. Here, $p_{b,t}\in\mathbb{R}^{3}$ and $r_{b,t}\in\mathbb{R}^{3}$ denote the base-frame position and axis-angle orientation of the robot end-effector, respectively.
Standard VLAs [25, 5, 4] predict the base-frame delta action $\Delta A_{b,t}=[\Delta p_{b,t},\Delta r_{b,t},g_{t}]$, where $\Delta p_{b,t},\Delta r_{b,t}\in\mathbb{R}^{3}$ denote the translational displacement and axis-angle delta rotation, and $g_{t}\in[0,1]$ is the gripper state.
The network parameters $\theta$ are optimized with the action prediction objective $\mathcal{L}_{\text{act}}$; for regression-style policies, this can be written as $\sum_{t}\|f_{\theta}(I_{t},s_{t},L)-\Delta A_{b,t}\|_{2}^{2}$.

### 3.2 Camera-Centric Action Generation

When $f_{\theta}$ is trained to output $\Delta A_{b,t}$, it learns a difficult cross-frame mapping $F:(I_{t},s_{t},L)\mapsto\Delta A_{b,t}$ from camera-perspective visual observations to a base-frame action.
The bridge between these two frames is the hand-eye matrix $T_{t}$, a rigid transform relating the camera and robot coordinate systems [42, 35].
Absent from the input, $T_{t}$ is implicitly entangled in the weights of $f_{\theta}$.
While training on multi-view datasets commonly improves viewpoint generalization [34, 7], a base-frame policy forces the network to map conflicting visual flows to the same base-frame action, leading to geometric entanglement.
Consequently, the mapping $F$ collapses once the deployment $T_{t}$ diverges from the training distribution.

Rather than relying on this cross-frame mapping to absorb $T_{t}$ implicitly, our approach decouples the visual-to-action mapping from $T_{t}$ by letting the policy learn a simpler same-frame form in which both visual observations and actions are natively expressed in the local camera frame, where this mapping remains independent of how the camera is mounted.
We define the camera-centric delta action as:

|  | $$ \Delta A_{c,t}=[\Delta p_{c,t},\Delta r_{c,t},g_{t}], $$ |  | (1) |
|---|---|---|---|

where $\Delta p_{c,t}\in\mathbb{R}^{3}$ and $\Delta r_{c,t}\in\mathbb{R}^{3}$ are the relative translation and axis-angle rotation expressed in the camera frame.
Because both visual representations and actions share the same egocentric perspective, visual flows are naturally aligned with physical movement in the camera frame.
For instance, a leftward visual translation in the image consistently corresponds to a negative displacement along the local X-axis of the camera.
By resolving this conflict, this consistent spatial relationship prevents visual-action confusion, helping the policy generalize to unseen viewpoints.

### 3.3 Camera-Perspective Geometric Grounding

While predicting $\Delta A_{c,t}$ ensures robust visual alignment, robotic arms fundamentally operate based on kinematic models and inverse kinematics defined in their grounded base frame [37].
The hand-eye transformation defines the rigid-body mapping between the observing camera and the robot base coordinate system, which is essential for bridging the visual and physical action spaces.
Consequently, we cannot send $\Delta A_{c,t}$ directly to the low-level controller without first applying this transformation.

To bridge this gap, our architecture incorporates a specialized auxiliary Geometric Head that regresses the 6-DoF hand-eye matrix $T_{t}\in SE(3)$ from visual features.
This makes $T_{t}$ an explicit network output, whereas in the cross-frame baseline, it remains implicitly entangled in the weights.
We parameterize $T_{t}$ with a translation vector $\tau_{t}\in\mathbb{R}^{3}$ and an axis-angle rotation vector $\omega_{t}\in\mathbb{R}^{3}$, where $\omega_{t}$ is converted to a rotation matrix $R_{t}\in SO(3)$ during geometric fusion.
The network $f_{\theta}$ is therefore redefined to jointly output the camera-centric action $\Delta A_{c,t}$ from the Action Head and the hand-eye matrix $T_{t}$ from the Geometric Head:

|  | $$ (\Delta A_{c,t},T_{t})=f_{\theta}(I_{t},s_{t},L). $$ |  | (2) |
|---|---|---|---|

### 3.4 Deterministic Geometric Transformation

To execute actions on the physical robot, we combine the parallel predictions from the Action Head and the Geometric Head using a deterministic geometric transformation.
Specifically, since the relative translation $\Delta p_{c,t}$ and axis-angle rotation $\Delta r_{c,t}$ are free vectors, both transform linearly under the predicted rotation $R_{t}\in SO(3)$, independently of the translation vector $\tau_{t}$ [32]:

|  | $\displaystyle\Delta p_{b,t}$ | $\displaystyle=R_{t}\Delta p_{c,t},$ |  | (3) |
|---|---|---|---|---|
|  | $\displaystyle\Delta r_{b,t}$ | $\displaystyle=R_{t}\Delta r_{c,t}.$ |  | (4) |

The final base-frame delta action is then assembled as $\Delta A_{b,t}=[\Delta p_{b,t},\Delta r_{b,t},g_{t}]$.
Despite this mathematical independence, we still regress $\tau_{t}$ to enhance the geometric grounding of the visual backbone and to support potential absolute-action variants.
Consequently, test-time drift in $\tau_{t}$ has zero physical impact on execution, confining viewpoint errors exclusively to $R_{t}$.

## 4 Experiments

### 4.1 Simulation Experiments

![Refer to caption](drafts/images/camvla-2607.05396/sim_camera.png)

*Figure 3: Simulation camera configuration.
The training set (red cameras) covers discrete viewpoints, while evaluation (green cameras) is conducted on a dense set of unseen viewpoints.*

Experimental Setup.
We evaluate CamVLA on the RLBench benchmark [20].
To evaluate viewpoint robustness, we utilize the front camera and rotate it around the robot base from $-90^{\circ}$ to $90^{\circ}$ at $5^{\circ}$ intervals to generate a diverse set of viewpoints as shown in Figure 3.
The model is trained on a discrete subset of viewpoints with $15^{\circ}$ intervals ($0^{\circ},\pm 15^{\circ},\pm 30^{\circ},\dots,\pm 90^{\circ}$), while zero-shot generalization is evaluated on the remaining unseen viewpoints within the $5^{\circ}$ grid.
We evaluate CamVLA on six representative manipulation tasks: slide block to target, push buttons, take umbrella out of umbrella stand, close laptop lid, lamp off, and put knife on chopping board.
For each task and viewpoint, we collect 100 expert demonstrations for training and 50 episodes for evaluation.

*Table 1: Zero-shot viewpoint generalization on RLBench simulation experiments.
Success rate (%) comparison between VLA baselines and CamVLA across six tasks under unseen viewpoints.*

| Model | Slide Block | Push Buttons | Take Umbrella | Close Laptop | Lamp Off | Put Knife | Mean |
|---|---|---|---|---|---|---|---|
| $\pi_{0}$ [5] | 18.3 | 51.5 | 32.3 | 57.0 | 29.8 | 10.0 | 33.2 |
| $\pi_{0}$ + CamVLA (Ours) | 44.5 | 72.3 | 39.2 | 69.0 | 58.0 | 25.3 | 51.4 |
| GR00T N1.7 [4] | 27.5 | 13.5 | 41.8 | 47.7 | 28.2 | 11.5 | 28.4 |
| GR00T N1.7 + CamVLA (Ours) | 44.7 | 30.5 | 50.3 | 56.0 | 35.0 | 14.0 | 38.4 |

Zero-Shot Viewpoint Generalization.
As shown in Table 1, on RLBench simulation experiments, we integrate our CamVLA framework with two foundational VLA architectures, $\pi_{0}$ [5] and GR00T N1.7 [4], evaluating zero-shot generalization across six tasks under unseen camera configurations.
The reported success rates are averaged across all unseen viewpoints.
For $\pi_{0}$, integrating CamVLA improves the average success rate from 33.2% to 51.4% (+18.2% absolute gain) on unseen configurations.
Notably, on challenging tasks where the vanilla policy severely degrades under viewpoint variations, such as Slide Block and Lamp Off, CamVLA boosts the success rates from 18.3% to 44.5% and 29.8% to 58.0%, respectively.
Similarly, applying our method to the GR00T N1.7 architecture demonstrates consistent robustness enhancements, achieving +10.0% absolute gain (from 28.4% to 38.4%) on unseen configurations.
These results confirm that CamVLA generalizes zero-shot to unseen viewpoints, consistently outperforming conventional VLA policies under viewpoint shifts.

### 4.2 Real-World Experiments

![Refer to caption](drafts/images/camvla-2607.05396/real_camera.png)

*Figure 4: Real-world experimental setup.
Multi-view camera configuration used to verify viewpoint robustness in physical environments.*

Experimental Setup.
Our real-world setup uses a Franka Research 3 robot arm with a parallel gripper.
As shown in Figure 4, we use a multi-view camera setup where third-person perspectives are captured by calibrated Intel RealSense D435i cameras.
We collect training demonstration data from five different camera perspectives, which provides sufficient viewpoint diversity for VLA learning while avoiding the excessive setup complexity of denser arrays to facilitate easy real-world deployment.
For evaluation, we employ three testing cameras (Cam 2, Cam 3, and Cam 4).
To evaluate viewpoint robustness, we rotate each testing camera by $5^{\circ}$, $10^{\circ}$, and $15^{\circ}$ around the robot base from its default position ($0^{\circ}$).
We limit the viewpoint offsets up to $15^{\circ}$ to cover typical camera perturbations (e.g., sensor drifts) in daily operations, beyond which both the baseline and our model perform poorly, rendering further evaluation impractical.
We evaluate on five common household tasks: put the basket upright, pick up the banana and place to the circular basket, push the cabbage near the pineapple, wipe the table with the cloth, and pick up the cup and place it inside the bowl.
For each task, we collect 100 demonstrations per training viewpoint.
During evaluation, we run 20 episodes per task and viewpoint offset on each testing camera.

*Table 2:
Generalization to repositioned cameras on real-world robot experiments.
Success rates (%) under calibration-free viewpoint shifts, relative to the canonical view ($0^{\circ}$).*

| Model | Task | 0∘ (Canonical) | 5∘ | 10∘ | 15∘ |
|---|---|---|---|---|---|
| Baseline | Ours | Baseline | Ours | Baseline | Ours | Baseline | Ours |
| $\pi_{0}$ [5] | Basket Upright | 75.0 | 88.3 | 58.3 | 80.0 | 31.7 | 68.3 | 16.7 | 48.3 |
| Pick & Place Banana | 53.3 | 80.0 | 45.0 | 68.3 | 33.3 | 50.0 | 0.0 | 18.3 |
| Push Cabbage | 45.0 | 81.7 | 50.0 | 70.0 | 33.3 | 43.3 | 16.7 | 31.7 |
| Wipe Table | 75.0 | 68.3 | 51.7 | 71.7 | 41.7 | 51.7 | 23.3 | 18.3 |
| Pick & Place Cup | 68.3 | 76.7 | 61.7 | 50.0 | 56.7 | 63.3 | 23.3 | 30.0 |
|  | Mean | 63.3 | 79.0 | 53.3 | 68.0 | 39.3 | 55.3 | 16.0 | 29.3 |
| GR00T N1.7 [4] | Basket Upright | 95.0 | 96.7 | 88.3 | 100.0 | 36.7 | 73.3 | 0.0 | 45.0 |
| Pick & Place Banana | 48.3 | 70.0 | 28.3 | 53.3 | 13.3 | 30.0 | 0.0 | 8.3 |
| Push Cabbage | 65.0 | 73.3 | 53.3 | 81.7 | 65.0 | 66.7 | 30.0 | 50.0 |
| Wipe Table | 68.3 | 86.7 | 56.7 | 63.3 | 35.0 | 48.3 | 26.7 | 36.7 |
| Pick & Place Cup | 46.7 | 76.7 | 33.3 | 63.3 | 28.3 | 46.7 | 16.7 | 25.0 |
|  | Mean | 64.7 | 80.7 | 52.0 | 72.3 | 35.7 | 53.0 | 14.7 | 33.0 |

Generalization to Repositioned Cameras.
Table 2 reports the results of our real-world robot experiments, evaluating the generalization performance of CamVLA under repositioned cameras with calibration-free viewpoint shifts.
The reported success rates are averaged across all three cameras under the corresponding offset.
We observe that under the canonical viewpoint ($0^{\circ}$), CamVLA substantially outperforms the standard baselines, achieving average success rates of 79.0% (vs. 63.3% for $\pi_{0}$) and 80.7% (vs. 64.7% for GR00T N1.7).
Under the extreme $15^{\circ}$ offset, while the baseline success rates collapse to 16.0% ($\pi_{0}$) and 14.7% (GR00T), CamVLA maintains robust performance at 29.3% and 33.0%, respectively, demonstrating exceptional zero-shot generalization to uncalibrated viewpoint shifts.
This is enabled by our CamVLA framework, which predicts camera-centric actions and utilizes the auxiliary Geometric Head to dynamically calibrate action outputs from monocular inputs during inference, highlighting the practicality of CamVLA in unstructured settings where camera positions may shift.

To evaluate the accuracy of the Geometric Head in physical environments, we report the hand-eye errors across viewpoints in Table 4.
We measure these errors against the ground-truth extrinsics from hand-eye calibration, showing that CamVLA maintains stable geometric grounding even under rotated test viewpoints.
Despite the large errors at $15^{\circ}$, CamVLA outperforms the baseline because our relative-action formulation physically isolates translation errors, while the rotation error (under 10∘) remains within the tolerance of the closed-loop policy (as detailed in Sec. B of Supp.).

Computational Efficiency. We report the parameter count, FLOPs, and inference time of CamVLA in Table 4.
To evaluate its real-world deployment feasibility, we measure the inference speed on a single consumer-grade NVIDIA RTX 4090 GPU.
As shown, our auxiliary Geometric Head introduces only an extra 6.30 M parameters (0.19% overhead), 1.0 G FLOPs (0.15% overhead), and increases inference latency by only 1 ms (62 ms vs. 61 ms).
With a 20-step trajectory executed at 10 Hz, the 62 ms inference easily runs in parallel with robot motion, presenting no bottleneck for real-time deployment.

*Table 3: Real-world hand-eye matrix estimation errors.
Translation and rotation errors under different viewpoint offsets.*

| Metric | 0∘ | 5∘ | 10∘ | 15∘ |
|---|---|---|---|---|
| Trans. (cm) | 1.35 | 2.12 | 7.91 | 27.16 |
| Rot. (∘) | 2.49 | 4.73 | 5.98 | 9.39 |

*Table 4: Computational Efficiency.
Comparison of parameter count, FLOPs, and inference time on a single GeForce RTX 4090.*

| Model | Params (M) | FLOPs (G) | Inference (ms) |
|---|---|---|---|
| $\pi_{0}$ [5] | 3238.1 | 660.9 | 61 |
| CamVLA | 3244.4 | 661.9 | 62 |

### 4.3 Ablation Studies

We ablate key design choices of CamVLA on RLBench [20] with $\pi_{0}$ [5] as the baseline model, reporting the average success rate across unseen viewpoints for all evaluated tasks.

*Table 5: Generalization and hand-eye error across training distributions. Success rates (%) and pose estimation errors under different training intervals. CamVLA† uses ground-truth extrinsics.*

| Interval | Success Rate (%) | Estimation Error |
|---|---|---|
| $\pi_{0}$ | CamVLA | CamVLA† | Trans. (cm) | Rot. (∘) |
| 15∘ | 33.2 | 51.4 | 52.3 | 4.69 | 1.41 |
| 30∘ | 25.5 | 34.0 | 40.0 | 19.71 | 4.77 |
| 45∘ | 16.8 | 21.2 | 26.3 | 34.83 | 8.28 |

Density of Training Viewpoints.
We examine how the density of camera viewpoints in the training data affects generalization.
We compare three training configurations with viewpoint sampling intervals of 15∘, 30∘, and 45∘.
As shown in Table 5, denser viewpoint sampling leads to improved success rates under novel perspectives.
As training viewpoints become sparser (30∘ and 45∘ intervals), the performance of all configurations inevitably degrades.
Nevertheless, CamVLA consistently maintains a much higher unseen success rate (e.g., 34.0% under 30∘ compared to 25.5% of the baseline).
The small performance gap between our method and GT under the $15^{\circ}$ interval (51.4% vs. 52.3%) confirms the high accuracy of our self-predicted extrinsics ($1.41^{\circ}$ rotation error).
Even under the extremely sparse $45^{\circ}$ configuration, where self-localization errors increase ($8.28^{\circ}$), CamVLA still achieves a 21.2% success rate, which is close to the GT performance of 26.3% and outperforms the baseline by 4.4%.
These results demonstrate that our method is highly robust to self-localization errors.
Notably, even the configuration with ground-truth extrinsics (CamVLA†) suffers similar degradation, confirming that this decline is primarily driven by visual representation shifts under unseen perspectives rather than hand-eye regression errors.
Furthermore, while CamVLA operates without calibration, our framework is compatible with incorporating online hand-eye calibration systems to replace predicted extrinsics for better performance.

Viewpoint Intervals.
To provide a more granular understanding of the local generalization capabilities, we present the detailed relative drop for each individual unseen viewpoint interval in Table 6.
We define the Relative Drop (%) as $\text{Drop}=(\max(S_{\theta_{1}},S_{\theta_{2}})-S_{\text{unseen}})/\max(S_{\theta_{1}},S_{\theta_{2}})$, where $S_{\theta_{1}}$ and $S_{\theta_{2}}$ denote the success rates at the two adjacent training viewpoints, and $S_{\text{unseen}}$ represents the success rate at the unseen viewpoint.
CamVLA achieves an average relative drop of 4.1%, which outperforms the baseline (8.9%), showing exceptional viewpoint invariance.
In the viewpoint interval of $[30^{\circ},45^{\circ}]$, where the baseline suffers a severe performance drop of 20.8%, ours maintains an exceptionally stable drop of only 0.9%, showing robust invariance to viewpoint shifts.

State and Action Representations.
As shown in Table 7, we evaluate the impact of different action spaces, proprioceptive states, and calibration requirements.
Shifting the action output from the robot base frame to the camera frame yields a significant performance gain, boosting the success rate from 33.2% ($\pi_{0}$) to 51.4% (CamVLA).
Using self-predicted instead of ground-truth extrinsics leads to only a marginal performance drop (e.g., 0.9% gap between CamVLA† and CamVLA, and 0.2% gap between CamVLA‡ and CamVLA∗), validating the high accuracy of our auxiliary Geometric Head.
Although utilizing the base-frame proprioceptive state instead of the camera-frame state (comparing CamVLA with CamVLA∗) incurs a 0.3% performance drop, it represents a key design choice that completely eliminates the need for camera calibration in the state observation phase, enabling true calibration-free deployment.

*Table 6: Relative performance drop across individual viewpoint intervals. We compare the unseen success rates and their relative drops from adjacent seen viewpoints across unseen intervals.*

| Viewpoint Interval | $\pi_{0}$ | CamVLA |
|---|---|---|
| Seen (%) | Unseen (%) | Drop (%) | Seen (%) | Unseen (%) | Drop (%) |
| $[0^{\circ},15^{\circ}]$ | 42.0 | 32.8 | 21.8 | 70.0 | 58.7 | 16.2 |
| $[15^{\circ},30^{\circ}]$ | 42.0 | 44.0 | -4.8 | 70.0 | 61.5 | 12.1 |
| $[30^{\circ},45^{\circ}]$ | 37.7 | 29.8 | 20.8 | 57.7 | 57.2 | 0.9 |
| $[45^{\circ},60^{\circ}]$ | 35.7 | 37.0 | -3.7 | 50.0 | 51.5 | -3.0 |
| $[60^{\circ},75^{\circ}]$ | 35.0 | 30.0 | 14.3 | 43.0 | 43.8 | -1.9 |
| $[75^{\circ},90^{\circ}]$ | 26.7 | 25.3 | 5.0 | 35.7 | 35.7 | 0.0 |
| Average | 36.5 | 33.2 | 8.9 | 54.4 | 51.4 | 4.1 |

*Table 7: State and action space ablations.
Comparison of various proprioceptive state representations, action outputs, and calibration requirements.*

| Model Variant | State Input | Action Output | Hand-Eye Source | Calibration-Free? | Success Rate (%) |
|---|---|---|---|---|---|
| $\pi_{0}$ | Base | Base | None | ✓ | 33.2 |
| CamVLA‡ | Camera | Camera | Ground-Truth | ✗ | 51.9 |
| CamVLA∗ | Camera | Camera | Self-Predicted | ✗ | 51.7 |
| CamVLA† | Base | Camera | Ground-Truth | ✗ | 52.3 |
| CamVLA | Base | Camera | Self-Predicted | ✓ | 51.4 |

## 5 Conclusion and Limitations

In this work, we present CamVLA, a calibration-free VLA model that achieves viewpoint robustness without external camera information.
By decoupling the policy into an Action Head that predicts the camera-centric action and a Geometric Head that regresses the hand-eye matrix from a single monocular RGB image, CamVLA isolates camera-centric action generation from camera-perspective geometric grounding, composing their outputs into base-frame actions through a deterministic transformation.
Experiments in simulation and on real hardware demonstrate substantial gains over strong VLA baselines under unseen camera configurations.
By replacing given geometry with learned geometry, CamVLA brings VLA policies one step closer to robust and calibration-free manipulation under unseen camera viewpoints in unstructured environments.

While CamVLA achieves viewpoint robustness for the third-person perspective, it only utilizes a single third-person camera and does not consider viewpoint perturbations of the wrist-mounted camera in multi-camera systems. Furthermore, although robust to typical shifts, the model struggles under extreme viewpoint changes and high-precision tasks due to out-of-distribution visual features and hand-eye regression errors. In future work, we plan to improve geometric regression robustness under large viewpoint shifts and investigate wrist-mounted camera perturbations.

## References

- [1]
A. Abouzeid, M. Mansour, Q. Sun, Z. Sun, and D. Song (2025)

GeoAware-VLA: implicit geometry aware vision-language-action model.

arXiv preprint arXiv:2509.14117.

Cited by: §2.
- [2]
M. Ahn, A. Brohan, N. Brown, Y. Chebotar, O. Cortes, B. David, C. Finn, C. Fu, K. Gopalakrishnan, K. Hausman, et al. (2022)

Do as I can, not as I say: grounding language in robotic affordances.

In CoRL,

Cited by: §2.
- [3]
S. Bai, K. Chen, X. Liu, J. Wang, W. Ge, S. Song, K. Dang, P. Wang, S. Wang, J. Tang, H. Zhong, Y. Zhu, M. Yang, Z. Li, J. Wan, P. Wang, W. Ding, Z. Fu, Y. Xu, J. Ye, X. Zhang, T. Xie, Z. Cheng, H. Zhang, Z. Yang, H. Xu, and J. Lin (2025)

Qwen2.5-VL technical report.

arXiv preprint arXiv:2502.13923.

Cited by: §2.
- [4]
J. Bjorck, F. Castañeda, N. Cherniadev, X. Da, R. Ding, L. Fan, Y. Fang, D. Fox, F. Hu, S. Huang, et al. (2025)

GR00T N1: an open foundation model for generalist humanoid robots.

arXiv preprint arXiv:2503.14734.

Cited by: Appendix A,
§1,
§1,
§1,
§2,
§3.1,
§4.1,
Table 1,
Table 2.
- [5]
K. Black, N. Brown, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, L. Groom, K. Hausman, B. Ichter, et al. (2024)

$\pi_{0}$: A vision-language-action flow model for general robot control.

In RSS,

Cited by: Appendix A,
Appendix A,
§1,
§1,
§1,
§2,
§3.1,
§4.1,
§4.3,
Table 1,
Table 2,
Table 4.
- [6]
A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, J. Dabis, C. Finn, K. Gopalakrishnan, K. Hausman, A. Herzog, J. Hsu, et al. (2022)

RT-1: robotics transformer for real-world control at scale.

In RSS,

Cited by: §1.
- [7]
B. Cai, Q. Liang, J. Li, S. Weng, Z. Zhang, T. Lin, X. Chen, W. Zhang, J. Mao, W. Xu, et al. (2026)

Beyond viewpoint generalization: what multi-view demonstrations offer and how to synthesize them for robot manipulation?.

arXiv preprint arXiv:2603.26757.

Cited by: §3.2.
- [8]
H. Chen, J. Liu, C. Gu, Z. Liu, R. Zhang, X. Li, X. He, Y. Guo, C. Fu, S. Zhang, et al. (2025)

Fast-in-slow: a dual-system foundation model unifying fast manipulation within slow reasoning.

arXiv preprint arXiv:2506.01953.

Cited by: §2.
- [9]
C. Chi, Z. Xu, S. Feng, E. Cousineau, Y. Du, B. Burchfiel, R. Tedrake, and S. Song (2025)

Diffusion Policy: visuomotor policy learning via action diffusion.

The International Journal of Robotics Research 44 (10-11), pp. 1684–1704.

Cited by: §2.
- [10]
J. Coholich, J. Wit, R. Azarcon, and Z. Kira (2026)

Sim2real image translation enables viewpoint-robust policies from fixed-camera datasets.

arXiv preprint arXiv:2601.09605.

Cited by: §2.
- [11]
O. Collaboration, A. O’Neill, A. Rehman, A. Gupta, A. Maddukuri, A. Gupta, A. Padalkar, A. Lee, A. Pooley, A. Gupta, et al. (2023)

Open X-Embodiment: robotic learning datasets and RT-X models.

arXiv preprint arXiv:2310.08864 1 (2).

Cited by: §2.
- [12]
J. D. Crawford, W. P. Medendorp, and J. J. Marotta (2004)

Spatial transformations for eye–hand coordination.

Journal of neurophysiology.

Cited by: §1.
- [13]
S. Deng, M. Yan, Y. Zheng, J. Su, W. Zhang, X. Zhao, H. Cui, Z. Zhang, and H. Wang (2025)

StereoVLA: enhancing vision-language-action models with stereo vision.

arXiv preprint arXiv:2512.21970.

Cited by: §2.
- [14]
D. Driess, F. Xia, M. S. Sajjadi, C. Lynch, A. Chowdhery, B. Ichter, A. Wahid, J. Tompson, Q. Vuong, T. Yu, et al. (2023)

PaLM-E: an embodied multimodal language model.

In ICML,

Cited by: §2.
- [15]
M. A. Goodale and A. D. Milner (1992)

Separate visual pathways for perception and action.

Trends in neurosciences 15 (1), pp. 20–25.

Cited by: §1.
- [16]
H. Heo, S. Woo, S. M. Kim, J. Kim, J. Lee, Y. Lee, and Y. M. Kim (2026)

AnyCamVLA: zero-shot camera adaptation for viewpoint robust vision-language-action models.

arXiv preprint arXiv:2603.05868.

Cited by: §1,
§2.
- [17]
P. Intelligence, B. Ai, A. Amin, R. Aniceto, A. Balakrishna, G. Balke, K. Black, G. Bokinsky, S. Cao, T. Charbonnier, et al. (2026)

$\pi^{*}_{0.7}$: A steerable generalist robotic foundation model with emergent capabilities.

Cited by: §1.
- [18]
P. Intelligence, A. Amin, R. Aniceto, A. Balakrishna, K. Black, K. Conley, G. Connors, J. Darpinian, K. Dhabalia, J. DiCarlo, et al. (2025)

$\pi^{*}_{0.6}$: A VLA that learns from experience.

arXiv preprint arXiv:2511.14759.

Cited by: §2.
- [19]
P. Intelligence, K. Black, N. Brown, J. Darpinian, K. Dhabalia, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, et al. (2025)

$\pi_{0.5}$: A vision-language-action model with open-world generalization.

In CoRL,

Cited by: §2.
- [20]
S. James, Z. Ma, D. R. Arrojo, and A. J. Davison (2020)

RLBench: the robot learning benchmark & learning environment.

IEEE Robotics and Automation Letters 5 (2), pp. 3019–3026.

Cited by: §1,
§4.1,
§4.3.
- [21]
Y. Jeong, J. Chun, and T. Kim (2026)

Learning to act robustly with view-invariant latent actions.

arXiv preprint arXiv:2601.02994.

Cited by: §2.
- [22]
Y. Jia, J. Liu, S. Chen, C. Gu, Z. Wang, L. Luo, L. Lee, P. Wang, Z. Wang, R. Zhang, et al. (2024)

Lift3D foundation policy: lifting 2D large-scale pretrained models for robust 3D robotic manipulation.

arXiv preprint arXiv:2411.18623.

Cited by: §2.
- [23]
T. Jiang, J. Ji, X. Tan, J. Fang, A. Bhattad, V. Guizilini, and M. R. Walter (2025)

Do you know where your camera is? view-invariant policy learning with camera conditioning.

arXiv preprint arXiv:2510.02268.

Cited by: §1,
§2.
- [24]
T. Ke, N. Gkanatsios, and K. Fragkiadaki (2024)

3D Diffuser Actor: policy diffusion with 3D scene representations.

In CoRL,

Cited by: §2.
- [25]
M. J. Kim, C. Finn, and P. Liang (2025)

Fine-tuning vision-language-action models: optimizing speed and success.

In RSS,

Cited by: §1,
§3.1.
- [26]
M. J. Kim, K. Pertsch, S. Karamcheti, T. Xiao, A. Balakrishna, S. Nair, R. Rafailov, E. Foster, G. Lam, P. Sanketi, et al. (2024)

OpenVLA: an open-source vision-language-action model.

In CoRL,

Cited by: §1.
- [27]
J. Li, D. Li, S. Savarese, and S. Hoi (2023)

BLIP-2: bootstrapping language-image pre-training with frozen image encoders and large language models.

In ICML,

pp. 19730–19742.

Cited by: §2.
- [28]
W. Li, Q. Zhang, R. Zhai, L. Lin, and G. Wang (2025)

VLA models are more generalizable than you think: revisiting physical and spatial modeling.

arXiv preprint arXiv:2512.02902.

Cited by: §2.
- [29]
Z. Li, P. Qu, Y. Jia, S. Zhou, H. Ge, J. Cao, J. Zhou, G. Zhou, and J. Ma (2026)

Manivid-3D: generalizable view-invariant reinforcement learning for robotic manipulation via disentangled 3D representations.

IEEE Robotics and Automation Letters.

Cited by: §2.
- [30]
H. Liu, C. Li, Q. Wu, and Y. J. Lee (2023)

Visual instruction tuning.

NeurIPS 36, pp. 34892–34916.

Cited by: §2.
- [31]
M. Luo, Z. Xue, A. Dimakis, and K. Grauman (2025)

Viewpoint rosetta stone: unlocking unpaired ego-exo videos for view-invariant representation learning.

In CVPR,

pp. 15802–15812.

Cited by: §2.
- [32]
R. M. Murray, Z. Li, and S. S. Sastry (2017)

A mathematical introduction to robotic manipulation.

Cited by: §3.4.
- [33]
J. Pang, N. Tang, K. Li, Y. Tang, X. Cai, Z. Zhang, G. Niu, M. Sugiyama, and Y. Yu (2025)

Learning view-invariant world models for visual robotic manipulation.

In ICLR,

Cited by: §2.
- [34]
Q. Qian, G. Zhao, G. Zhang, J. Wang, R. Xu, J. Gao, and D. Zhao (2025)

GP3: a 3D geometry-aware policy with multi-view images for robotic manipulation.

In ICRA,

Cited by: §3.2.
- [35]
Y. C. Shiu and S. Ahmad (1987)

Calibration of wrist-mounted robotic sensors by solving homogeneous transform equations of the form ax= xb.

Cited by: §3.2.
- [36]
M. Shridhar, L. Manuelli, and D. Fox (2023)

Perceiver-Actor: a multi-task transformer for robotic manipulation.

In CoRL,

pp. 785–799.

Cited by: §2.
- [37]
B. Siciliano, L. Sciavicco, L. Villani, and G. Oriolo (2009)

Robotics: modelling, planning and control.

Cited by: §3.3.
- [38]
I. A. Sucan, M. Moll, and L. E. Kavraki (2012)

The open motion planning library.

IEEE Robotics & Automation Magazine 19 (4), pp. 72–82.

Cited by: Appendix A.
- [39]
J. Q. Sun, H. Weng, X. Xing, C. M. Yeum, and M. Crowley (2026)

View invariant learning for vision-language navigation in continuous environments.

IEEE Robotics and Automation Letters.

Cited by: §2.
- [40]
O. M. Team, D. Ghosh, H. Walke, K. Pertsch, K. Black, O. Mees, S. Dasari, J. Hejna, T. Kreiman, C. Xu, et al. (2024)

Octo: an open-source generalist robot policy.

In RSS,

Cited by: §2.
- [41]
S. Tian, B. Wulfe, K. Sargent, K. Liu, S. Zakharov, V. Guizilini, and J. Wu (2024)

View-invariant policy learning via zero-shot novel view synthesis.

In CoRL,

Cited by: §2.
- [42]
R. Y. Tsai R. K. Lenz et al. (1989)

A new technique for fully autonomous and efficient 3 d robotics hand/eye calibration.

IEEE Transactions on robotics and automation 5 (3), pp. 345–358.

Cited by: §3.2.
- [43]
S. Vasudevan, S. Sagar, and R. Senanayake (2025)

Viewpoint-agnostic manipulation policies with strategic vantage selection.

arXiv preprint arXiv:2506.12261.

Cited by: §2.
- [44]
Y. Xu, J. Yang, X. Wang, Y. Chen, Z. Zhu, B. Fang, G. Huang, X. Chen, Y. Ye, Q. Zhang, et al. (2025)

EgoDemoGen: novel egocentric demonstration generation enables viewpoint-robust manipulation.

arXiv preprint arXiv:2509.22578.

Cited by: §2.
- [45]
J. Zhang, Y. Chen, Y. Xu, Z. Huang, Y. Zhou, Y. Yuan, X. Cai, G. Huang, X. Quan, H. Xu, et al. (2025)

4D-VLA: spatiotemporal vision-language-action pretraining with cross-scene calibration.

arXiv preprint arXiv:2506.22242.

Cited by: §1,
§2.
- [46]
T. Zhang, H. Duan, H. Hao, Y. Qiao, J. Dai, and Z. Hou (2026)

Grounding actions in camera space: observation-centric vision-language-action policy.

In AAAI,

Vol. 40, pp. 18782–18790.

Cited by: §1,
§2.
- [47]
Z. Zhang, H. Li, Y. Dai, Z. Zhu, L. Zhou, C. Liu, D. Wang, F. E. Tay, S. Chen, Z. Liu, et al. (2025)

From spatial to actions: grounding vision-language-action model in spatial foundation priors.

arXiv preprint arXiv:2510.17439.

Cited by: §2.
- [48]
H. Zhen, X. Qiu, P. Chen, J. Yang, X. Yan, Y. Du, Y. Hong, and C. Gan (2024)

3D-VLA: a 3D vision-language-action generative world model.

In ICML,

Cited by: §2.
- [49]
D. Zhu, X. Shen, X. Li, M. Elhoseiny, et al. (2024)

MiniGPT-4: enhancing vision-language understanding with advanced large language models.

In ICLR,

Vol. 2024, pp. 18378–18394.

Cited by: §2.
- [50]
B. Zitkovich, T. Yu, S. Xu, P. Xu, T. Xiao, F. Xia, J. Wu, P. Wohlhart, S. Welker, A. Wahid, et al. (2023)

RT-2: vision-language-action models transfer web knowledge to robotic control.

In CoRL,

pp. 2165–2183.

Cited by: §2.

Supplementary Material

This supplementary material covers the following details:

- •

Additional Implementation details (Sec. A).
- •

Additional Mathematical derivation (Sec. B).
- •

Additional ablation studies (Sec. C).
- •

Additional visualization results (Sec. D).

## Appendix A Additional Implementation Details

Architecture and Training.
CamVLA is trained in a multi-task manner on 8 NVIDIA H100 80GB GPUs, and we subsequently evaluate its performance across different tasks.
We instantiate CamVLA upon foundational VLA architectures, specifically $\pi_{0}$ [5] and GR00T N1.7 [4], adhering to their respective training recipes and configurations.
Visual inputs are restricted to single third-person monocular RGB images resized to $224\times 224$ pixels.
To evaluate robustness to third-person viewpoint changes and prevent the model from relying on the unaffected wrist view to bypass them, we exclude wrist cameras, which also simplifies hardware routing across robot joints.
The predicted robot actions are parameterized as delta 6-DoF end-effector poses relative to the current state, where the rotation component is represented as a 3D axis-angle vector.
The auxiliary Geometric Head is implemented as a lightweight three-layer Multi-Layer Perceptron (MLP) with GELU activations and a hidden dimension of 1024.
It operates on the high-level semantic features extracted from the visual tokens of the backbone, which are aggregated via mean pooling before being fed into the MLP.

Simulation and Real-World Setup.
The simulation is powered by the CoppeliaSim engine, with expert demonstrations generated by the Open Motion Planning Library (OMPL) [38].
In simulation, the control frequency is 20 Hz and the model executes the first 5 steps of the predicted action trajectory per inference.
For real-world experiments, raw demonstrations are collected at 30 Hz, while training and deployment are conducted at 10 Hz to ensure stable hardware response, with the model executing the first 20 steps of the predicted trajectory per inference.
In both settings, ground-truth hand-eye matrices are used only as training supervision, and the model receives no extrinsic information at deployment.

Optimization.
We optimize the model end-to-end with a joint objective $\mathcal{L}=\mathcal{L}_{act}+\lambda\mathcal{L}_{ext}$, where $\mathcal{L}_{act}$ inherits the action prediction loss of the underlying baseline (e.g., flow-matching for $\pi_{0}$ [5]) and the geometric grounding loss is the mean squared error $\mathcal{L}_{ext}=\sum_{t}\big(\|\tau_{t}-\hat{\tau}_{t}\|_{2}^{2}+\|\omega_{t}-\hat{\omega}_{t}\|_{2}^{2}\big)$, with $\hat{\tau}_{t},\hat{\omega}_{t}$ denoting the ground-truth translation and axis-angle rotation vectors.
We set $\lambda=0.1$.

## Appendix B Additional Mathematical Derivation

We provide the derivation showing why the execution of camera-centric delta actions is independent of the hand-eye translation vector.
Let the hand-eye transform $T_{t}\in SE(3)$ from the camera frame to the robot base frame be parameterized by a translation vector $\tau_{t}\in\mathbb{R}^{3}$ and an axis-angle rotation vector $\omega_{t}\in\mathbb{R}^{3}$.
We convert $\omega_{t}$ to a rotation matrix $R_{t}\in SO(3)$ (e.g., via Rodrigues’ rotation formula) to construct the homogeneous transformation matrix:

|  | $$ T_{t}=\begin{bmatrix}R_{t}&\tau_{t}\\ 0&1\end{bmatrix},\qquad R_{t}\in SO(3),\ \tau_{t}\in\mathbb{R}^{3}. $$ |  | (5) |
|---|---|---|---|

For two end-effector positions expressed in the camera frame, $p_{c,t}^{(0)}$ and $p_{c,t}^{(1)}$ (denoting the initial and target positions of a delta action, respectively), their corresponding base-frame positions are

|  | $$ p_{b,t}^{(i)}=R_{t}p_{c,t}^{(i)}+\tau_{t},\qquad i\in\{0,1\}. $$ |  | (6) |
|---|---|---|---|

The base-frame relative translation is therefore

|  | $\displaystyle\Delta p_{b,t}$ | $\displaystyle=p_{b,t}^{(1)}-p_{b,t}^{(0)}$ |  |
|---|---|---|---|
|  |  | $\displaystyle=\left(R_{t}p_{c,t}^{(1)}+\tau_{t}\right)-\left(R_{t}p_{c,t}^{(0)}+\tau_{t}\right)$ |  |
|  |  | $\displaystyle=R_{t}\left(p_{c,t}^{(1)}-p_{c,t}^{(0)}\right)=R_{t}\Delta p_{c,t}.$ |  | (7) |

Thus, $\tau_{t}$ cancels exactly for relative translations.
Delta rotations are also independent of the hand-eye translation vector.
Let $Q_{c,t}^{(0)},Q_{c,t}^{(1)}\in SO(3)$ denote the two end-effector orientation matrices in the camera frame, with $Q_{b,t}^{(i)}=R_{t}Q_{c,t}^{(i)}$ in the base frame.
The relative rotation in the base frame satisfies

|  | $\displaystyle\Delta Q_{b,t}$ | $\displaystyle=Q_{b,t}^{(1)}\left(Q_{b,t}^{(0)}\right)^{\top}$ |  |
|---|---|---|---|
|  |  | $\displaystyle=R_{t}Q_{c,t}^{(1)}\left(Q_{c,t}^{(0)}\right)^{\top}R_{t}^{\top}=R_{t}\Delta Q_{c,t}R_{t}^{\top}.$ |  | (8) |

Using the equivariance of the matrix logarithm under rotation conjugation,

|  | $$ [\Delta r_{b,t}]_{\times}=\log(\Delta Q_{b,t})=R_{t}\log(\Delta Q_{c,t})R_{t}^{\top}=[R_{t}\Delta r_{c,t}]_{\times}, $$ |  | (9) |
|---|---|---|---|

which gives $\Delta r_{b,t}=R_{t}\Delta r_{c,t}$.
Consequently, both components of the executed delta action depend only on $R_{t}$:

|  | $$ \Delta A_{b,t}=[R_{t}\Delta p_{c,t},\ R_{t}\Delta r_{c,t},\ g_{t}]. $$ |  | (10) |
|---|---|---|---|

## Appendix C Additional Ablation Studies

Extrinsic Noise Robustness.
To evaluate the robustness of CamVLA to hand-eye matrix estimation errors, we conduct an ablation study with artificial rotation noise during evaluation.
At test time, we systematically inject random rotation noise into the ground-truth hand-eye rotation matrix across a wide range of noise levels, which consists of a $0^{\circ}$ reference, $1^{\circ}$–$20^{\circ}$ in $1^{\circ}$ increments, and $25^{\circ}$–$45^{\circ}$ in $5^{\circ}$ increments around a random 3D unit axis.
To closely simulate the dynamic prediction fluctuations of the geometric head, the rotation noise is independently re-sampled at each planning step rather than applying a static bias across the entire episode.
We explicitly omit translation noise because, as derived above, delta action execution is independent of the hand-eye translation vector under our relative-action formulation.
As shown in Figure 5, the policy is highly robust to small rotation perturbations: the success rate decreases only moderately from the 64.0% no-noise reference to 63.3% at $1^{\circ}$ and 58.7% at $5^{\circ}$.
Notably, even under injected rotation noise magnitudes of up to $12^{\circ}$, CamVLA still outperforms or remains highly comparable to the noise-free baseline $\pi_{0}$ (36.0%).
Crucially, our geometric head predicts hand-eye rotation with a mean error of less than $2^{\circ}$ (specifically $1.41^{\circ}$ as shown in the ablation table of the main text), a high-precision regime where CamVLA suffers almost no performance degradation.
This experiment validates the feasibility of our calibration-free framework, demonstrating that while closed-loop VLA policies naturally possess an inherent tolerance to small execution deviations, our decoupled design (which parallelly predicts camera-centric actions and hand-eye poses) successfully extends this feedback robustness to encompass tolerance against hand-eye rotation errors as well, explaining why CamVLA achieves high success rates despite using self-predicted, imperfect extrinsics.

*Figure 5: Robustness under artificial extrinsic rotation noise. Success rate (%) under varying levels of random rotation noise (ranging from $0^{\circ}$ to $45^{\circ}$) applied to the ground-truth hand-eye rotation matrix during execution.*

*Table 8: Ablation on hand-eye pose representation.
Comparison of success rates and geometric errors between predicting full 6-DoF extrinsics and regressing rotation only.*

| Pose Representation | Success (%) | Trans. (cm) | Rot. (∘) |
|---|---|---|---|
| Full 6-DoF | 51.4 | 4.7 | 1.4 |
| Rotation Only | 51.3 | - | 1.6 |

Hand-Eye Pose Representation.
We ablate the necessity of predicting the translation component of the hand-eye matrix.
Under our relative-action formulation, policy execution depends only on the rotation component.
To verify this, we compare the default 6-DoF configuration against a rotation-only variant.
As shown in Table 8, predicting only rotation yields a 51.3% success rate and a 1.6∘ rotation error, highly comparable to the default 6-DoF configuration (51.4% success rate and 1.4∘).
This demonstrates that additionally predicting the translation component does not degrade task success or rotation estimation accuracy.
Furthermore, regressing the full 6-DoF pose allows CamVLA to support potential absolute-action variants, enhancing framework versatility.

*Table 9: Ablation on the visual feature source for hand-eye matrix prediction.
Comparison of success rates and geometric errors using different visual features.*

| Feature Source | Success (%) | Trans. (cm) | Rot. (∘) |
|---|---|---|---|
| Image Encoder | 51.4 | 4.7 | 1.4 |
| Image Encoder (Detach) | 42.6 | 11.7 | 13.2 |
| VLM Backbone | 53.5 | 14.7 | 3.7 |
| VLM Backbone (Detach) | 20.9 | 45.0 | 36.0 |

Visual Feature Sources.
We investigate how the visual feature source and gradient propagation affect hand-eye matrix prediction.
As shown in Table 9, using features from the Image Encoder yields the most precise geometric localization, with only 4.7 cm translation error and $1.4^{\circ}$ rotation error, and achieves a 51.4% unseen success rate.
Extracting features from the deeper VLM Backbone slightly improves the task success rate to 53.5%, but substantially worsens geometric estimation (14.7 cm and $3.7^{\circ}$), indicating that high-level semantic features can support action generation while being less spatially precise for camera pose regression.
Although the VLM Backbone configuration achieves a slightly higher success rate, we select the Image Encoder as our default configuration.
The primary reason is that the VLM Backbone overfits to the discrete training viewpoints, producing geometric discontinuities and large localization spikes at unseen intermediate angles.
Such geometric instability poses significant risks during physical robot execution, where smooth and consistent hand-eye estimation is crucial for safety.
By contrast, the Image Encoder configuration offers stable and continuous spatial grounding across the entire viewpoint spectrum, making it a more reliable choice for real-world deployment.
Detaching gradients consistently hurts both localization and control.
Specifically, Image Encoder detachment reduces success from 51.4% to 42.6%, while VLM Backbone detachment collapses the success rate to 20.9% with severe pose errors (45.0 cm and $36.0^{\circ}$).
These results suggest that the Geometric Head benefits from end-to-end feature adaptation, and we therefore use Image Encoder features without detachment as the default configuration for its superior spatial grounding stability.

## Appendix D Additional Visualization Results

Visualization of Setup and Tasks.
Figure 6 provides a visual overview of the 10 RLBench simulation tasks used in our evaluation.
Figure 7 visualizes the visual range of both training and testing viewpoints in RLBench.
Figure 8 provides a detailed illustration of the camera placement around the robot base for the training viewpoint distribution (ablation study shown in Table 5).
Figure 9 illustrates the five representative household-style manipulation tasks used in our real-world evaluation.
Figure 10 illustrates the five training camera viewpoints (Cam 1–Cam 5) used for demonstration collection in our real-world hardware setup.
Figure 11 further visualizes the unseen testing viewpoints, where Cam 2, Cam 3, and Cam 4 are perturbed by horizontal rotations of $5^{\circ}$, $10^{\circ}$, and $15^{\circ}$ relative to their canonical $0^{\circ}$ training position.

Qualitative Comparison.
Figure 12 and Figure 13 provide qualitative comparisons of execution trajectories under unseen simulation viewpoints and repositioned real-world camera viewpoints, respectively.
We compare the execution trajectories of the baseline $\pi_{0}$ and our CamVLA under unseen viewpoints.
CamVLA maintains target-directed behavior and completes the tasks more robustly than the baseline $\pi_{0}$.
Additionally, Figure 14 visualizes the real-world deployment under continuously moving and hand-held cameras, echoing the drifting or hand-held camera scenarios and demonstrating the robust viewpoint generalization capability of CamVLA.

Failure Case Analysis.
Figure 15 visualizes representative failure cases of both $\pi_{0}$ and CamVLA.
These failures are primarily caused by target objects located at the boundary of the camera’s field of view, actions exceeding the physical workspace of the robot arm, or visual self-occlusion.
Notably, even in these failed cases, CamVLA’s execution trajectories still exhibit more target-directed and reasonable behaviors compared to the baseline $\pi_{0}$.

![Refer to caption](drafts/images/camvla-2607.05396/supp_sim_task.png)

*Figure 6: Simulation tasks on RLBench benchmark.
We evaluate CamVLA across a diverse set of manipulation tasks, requiring both high-level semantic understanding and precise low-level control.*

![Refer to caption](drafts/images/camvla-2607.05396/supp_sim_view.png)

*Figure 7: Visualization of training and testing viewpoints on RLBench.
The training set consists of views sampled at 15∘ intervals (top row), while the testing set covers a dense range of unseen viewpoints (middle and bottom rows) to evaluate the zero-shot generalization capability of CamVLA.*

![Refer to caption](drafts/images/camvla-2607.05396/supp_sim_camera.png)

*Figure 8: Detailed range of camera viewpoints in simulation.
(a) Training viewpoints sampled at 30∘ intervals and (b) training viewpoints sampled at 45∘ intervals.
Red and green cameras represent training and unseen testing viewpoints, respectively.*

![Refer to caption](drafts/images/camvla-2607.05396/supp_real_task.png)

*Figure 9: Real-world evaluation tasks.
Five representative manipulation tasks involving diverse objects and interaction requirements.*

![Refer to caption](drafts/images/camvla-2607.05396/supp_real_view.png)

*Figure 10: Real-world training camera viewpoints.
Five third-person training placements (Cam 1–Cam 5) used for demonstration collection.*

![Refer to caption](drafts/images/camvla-2607.05396/supp_real_camera.png)

*Figure 11: Real-world unseen testing viewpoints.
Cam 2, Cam 3, and Cam 4 are horizontally rotated by $5^{\circ}$, $10^{\circ}$, and $15^{\circ}$ from their canonical $0^{\circ}$ training position to form unseen testing views.*

![Refer to caption](drafts/images/camvla-2607.05396/supp_sim_compare.png)

*Figure 12: Qualitative comparison between $\pi_{0}$ and CamVLA on RLBench under unseen cameras.*

![Refer to caption](drafts/images/camvla-2607.05396/supp_real_compare.png)

*Figure 13: Qualitative comparison between $\pi_{0}$ and CamVLA on real-world robot experiments under repositioned cameras.*

![Refer to caption](drafts/images/camvla-2607.05396/supp_moving.png)

*Figure 14: Visualization of real-world experiments under dynamically hand-held moving cameras.
We visualize the robot’s execution and dynamic hand-eye pose tracking when the third-person camera is continuously moved by a human operator during deployment.*

![Refer to caption](drafts/images/camvla-2607.05396/supp_fail.png)

*Figure 15: Failure cases.
We illustrate common failure modes such as boundary objects, actions exceeding the robot’s physical workspace, and self-occlusion.*

