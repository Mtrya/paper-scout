# 2508.13103 (from arXiv HTML; MinerU fallback)



# Grounding Actions in Camera Space: Observation-Centric Vision-Language-Action Policy

Tianyi Zhang

Affiliation: College of Computer Science and Technology, Zhejiang University

Affiliation: College of Computer Science and Technology, Zhejiang University

Affiliation: Shanghai AI Lab

Thanks:  Equal Contribution

Thanks:  Equal Contribution

Thanks:  Corresponding Author

  
Haonan Duan

Affiliation: College of Computer Science and Technology, Zhejiang University

Affiliation: SenseTime Research

Thanks:  Equal Contribution

  
Haoran Hao

Affiliation: Shanghai AI Lab

Affiliation: Nanjing University

Thanks:  Corresponding Author

  
Yu Qiao

Affiliation: Shanghai AI Lab

Thanks:  Corresponding Author

  
Jifeng Dai

Affiliation: Tsinghua University

  
Zhi Hou

Affiliation: Shanghai AI Lab

Affiliation: Shanghai AI Lab

Thanks:  Corresponding Author

Thanks:  Corresponding Author

###### Abstract

Vision-Language-Action (VLA) models frequently encounter challenges in generalizing to real-world environments due to inherent discrepancies between observation and action spaces. Although training data are collected from diverse camera perspectives, the models typically predict end-effector poses within the robot base coordinate frame, resulting in spatial inconsistencies. To mitigate this limitation, we introduce the Observation-Centric VLA (OC-VLA) framework, which grounds action predictions directly in the camera observation space. Leveraging the camera’s extrinsic calibration matrix, OC-VLA transforms end-effector poses from the robot base coordinate system into the camera coordinate system, thereby unifying prediction targets across heterogeneous viewpoints. This lightweight, plug-and-play strategy ensures robust alignment between perception and action, substantially improving model resilience to camera viewpoint variations. The proposed approach is readily compatible with existing VLA architectures, requiring no substantial modifications. Comprehensive evaluations on both simulated and real-world robotic manipulation tasks demonstrate that OC-VLA accelerates convergence, enhances task success rates, and improves cross-view generalization. The code will be publicly available.

![Refer to caption](drafts/images/ocvla-2508.13103/Main_fig.png)

*Fig. 1: We introduce the Observation-Centric VLA (OC-VLA) framework. By transforming end-effector actions from the robot base coordinate to the third-person camera coordinate, OC-VLA aligns action predictions with visual observations across diverse viewpoints, enabling improved generalization and robustness in manipulation tasks.*

## I INTRODUCTION

Inspired by the remarkable progress of multimodal large models, recent advances in vision-language-action (VLA) models [1, 2, 3, 4, 5, 6] have focused on leveraging large-scale robot data from heterogeneous sources for pre-training, with the objective of enhancing generalization capabilities. Although this paradigm has achieved impressive performance across a variety of benchmarks, it remains fundamentally constrained by the intrinsic limitations of the robotics domain—namely, the relatively modest scale and high cost of data collection when compared to the web-scale corpora used in vision-language model (VLM) pre-training [7, 8]. Consequently, the ability of current VLA models to generalize effectively in real-world environments remains limited, leaving substantial room for further advancement.

A common practice in VLA modeling is to adapt pretrained vision-language or vision encoders for downstream robotic tasks [1, 7, 2, 6]. However, these vision models are primarily trained and supervised within the image or camera coordinate system, resulting in latent representations that are inherently aligned with camera viewpoints. In contrast, most robotic control signals are defined in the robot base coordinate system [1, 7, 2, 6]. This discrepancy introduces a misalignment between the perception and action spaces, which can hinder effective policy learning, especially during the transfer of pretrained vision models to robotic control tasks.

Moreover, robot datasets are typically collected under diverse camera viewpoints and heterogeneous hardware configurations [7, 9, 8], where the robot base is not always within the camera’s field of view. In such settings, the same action expressed in the robot base coordinate system must be inferred from different third-person camera views. This implicitly requires the model to reconstruct or reason about consistent 3D actions from limited 2D observations—a fundamentally ill-posed challenge when only single- or dual-view inputs are available. Predicting actions defined in the robot base coordinate system becomes even more challenging, as it necessitates an implicit understanding of the transformation between robot and camera spaces. Such inconsistencies are particularly detrimental during large-scale pre-training [10, 2], where diverse camera viewpoints are common: images capturing the same robot action from different angles are forced to share a single supervision signal in robot space, thereby introducing learning conflicts and hindering generalization.

To address these issues, we propose a novel paradigm that decouples the end-effector action from the robot base coordinate system and instead predicts actions directly in the third-person camera coordinate system, named Observation-Centric VLA (OC-VLA). Specifically, given the extrinsic transformation between the robot base and each camera, we transform the robot-space end-effector actions into their equivalent representations in the camera coordinate frame and adopt these as prediction targets. By anchoring the action target in the same space as the observation (i.e., the image plane), this formulation alleviates the misalignment between perception and action modalities and mitigates the ambiguity introduced by camera viewpoint variations. Furthermore, it explicitly encourages the model to learn the relative spatial relationships between the robot and the cameras, thereby enhancing its capacity to generalize effectively across diverse viewpoints and hardware configurations.

The proposed approach is evaluated across both simulated environments and real-world robotic platform. Experimental results consistently demonstrate that employing camera-space end-effector actions as prediction targets yields substantial performance gains over baselines that operate in robot coordinates. Notably, our method exhibits markedly improved adaptability to previously unseen camera viewpoints, underscoring its strong potential for robust generalization in diverse real-world deployment scenarios.

## II Related Work

### II-A Robotic Manipulation

Robotic manipulation has wide applications in the real world, but still faces significant challenges in complex environments and tasks. Compared with traditional methods, learning-based manipulation has gained significant attention in recent years [11]. A common strategy for learning to predict actions is reinforcement learning [12, 13, 14].Another approach is to provide offline expert demonstrations for supervised learning [1, 15, 16], which trains the model to imitate the action performed by experts. However, both approaches are data-driven and sensitive to environmental changes, limiting their effectiveness in open-world applications.
Recently, the development of large language models (LLMs) and vision-language models (VLMs) has made reasoning and planning possible for solving complex tasks that require human knowledge [17, 18, 19]. However, due to limitations in their pretraining data, these models are still unable to control robots and address real-world tasks effectively.
Vision-Language-Action (VLA) models [1, 20, 21, 2, 6, 22, 23, 5] are trained on large-scale observation-action pairs and have strong capabilities in unified perception, reasoning, planning, and control, making them a promising solution for achieving unified robotic manipulation. Nevertheless, the generalization of current VLA models is limited, and the observation space action prediction is poorly investigated.

### II-B Vision-Language-Action Model

Vision-Language-Action (VLA) models [1, 2, 3, 22, 7, 4, 24, 25, 6] have become the popular framework for generalist robot policies, enabling robots to interpret natural language instructions and visual observations for robust action generation. Recent advances leverage large-scale multi-modal backbones and foundation models [26, 23, 27, 28, 29, 30, 31, 32, 33, 34, 35], improving generalization across tasks and embodiments. Diffusion models [36, 37, 38, 39, 40] have shown strong performance in multi-modal action modeling [41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51], yet most existing approaches rely on U-Net or shallow cross-attention architectures, which limit scalability to more diverse tasks. To address complex scenarios, recent works integrate VLM embeddings with MLP diffusers [3, 52], or utilize Transformer-based [53] decoders for bimanual and multi-modal manipulation [50, 54], further pushing the frontier of unified VLA policy learning. Although VLA models have made great progress, most of them rely on a specific observation space to predict actions. However, differences in environment setups during data collection make it difficult to use large-scale web data directly for training, which limits their performance. Meanwhile, current manipulation datasets encompass a wide range of camera views, whereas existing VLA approaches typically focus on action prediction based on the robot’s base coordinates. The discrepancy between the coordinates of action prediction and observation poses a significant challenge for policy learning.

## III Method

In this section, we provide a detailed overview of OC-VLA, i.e., grounding actions in the observation (camera) space. We begin with the model structure and action modeling as preliminaries, followed by an introduction to the camera-centric action prediction approach. We then analyze the differences between camera-coordinate and robot-coordinate optimization.

### III-A Preliminary: Model Structure, Action Modeling

Vision-language-action (VLA) models have converged toward a common architectural pattern [3, 2, 5, 6], where action prediction is built upon a vision-language backbone. Following this paradigm, we adopt a lightweight 300M VLA model [6] for evaluation, which has demonstrated competitive performance using only a third-person camera image and language instructions as input. Specifically, we follow Dita [6], where the language instruction is encoded using a CLIP text encoder [55], and the third-person image is processed using DINOv2 [56]. The image features are further selected and modulated by the language instruction via a Q-Former [57] equipped with FiLM [58] conditioning layers.

Current VLA models typically employ one of two types of action spaces for end-effector control: discrete action spaces [2, 1] and continuous action spaces [3, 5]. To thoroughly evaluate the effectiveness of our proposed approach, we conduct experiments on models using both types of action spaces. Based on the baseline architecture, we implement a variant specifically designed for discrete action prediction or continuous action prediction.

![Refer to caption](drafts/images/ocvla-2508.13103/method_diff.png)

*Fig. 2: OC-VLA transforms the end effector pose whether defined in a discrete or continuous action space from the robot base coordinate to the third-person camera coordinate, unifying the observation and prediction targets across viewpoints, effectively replacing the usage of shared world action as prediction targets.*

### III-B Observation-Centric Action Prediction

In current robotic datasets, action/pose annotations are often defined at a low level, either as joint commands or end-effector poses within the robot base coordinate frame. While these representations are widely used as supervision signals for Vision-Language-Action (VLA) models, they are tightly coupled with specific robot embodiment configurations, rather than being derived from the observation space. Consequently, it is difficult for the model to achieve a reasonable projection from image observation to corresponding actions, and thus the model generalization is limited, especially for novel camera views with a large variance from the seen camera views in the training set.

To ground actions in the observation space, it is necessary to first transform the actions from the robot (world) coordinate system into the camera coordinate system. We utilize the extrinsics of the camera to conduct the transformation. Specifically, given two nearby end-effector poses in the world coordinate frame, denoted as $\mathbf{P}_{{\text{world}}_{1}}\in R^{4\times 4}$ and $\mathbf{P}_{{\text{world}}_{2}}\in R^{4\times 4}$, where the matrix can be converted from a 3D rotation and a translation, the corresponding action can be derived accordingly,

|  | $$ \mathbf{A}_{\text{world}}=\mathbf{P}_{{\text{world}}_{2}}\mathbf{P}_{{\text{world}}_{1}}^{-1} $$ |  | (1) |
|---|---|---|---|

Meanwhile, we can get the corresponding poses in the camera coordinate as follows,

|  | $$ \mathbf{P}_{{\text{cam}}_{2}}=\mathbf{T}\mathbf{P}_{{\text{world}}_{2}},\mathbf{P}_{{\text{cam}}_{1}}=\mathbf{T}\mathbf{P}_{{\text{world}}_{1}} $$ |  | (2) |
|---|---|---|---|

where $\mathbf{T}\in R^{4\times 4}$ represents the world-to-camera transformation matrix, consisting of a 3D rotation and a translation. $\mathbf{P}$ represents the corresponding matrix. Then, we can obtain the corresponding actions in the camera space.

|  | $$ \mathbf{A}_{\text{cam}}=\mathbf{P}_{{\text{cam}}_{2}}\mathbf{P}_{{\text{cam}}_{1}}^{-1} $$ |  | (3) |
|---|---|---|---|

Lastly, we convert $\mathbf{A}_{\text{cam}}\in R^{4\times 4}$ into the 7-dim actions $\langle\text{x},\text{y},\text{z},\text{roll},\text{pitch},\text{yaw},\text{gripper}\rangle$ for model optimization, where gripper is for the gripper position. Different from previous end-effector action prediction, the predicted action in our method is in the camera space.

During inference, we transform the actions in the camera space to robot coordinate space for robot control based on the camera calibration.

### III-C Analysis from Optimization Perspective

![Refer to caption](drafts/images/ocvla-2508.13103/coord_map.png)

*Fig. 3: Action translation between robot base coordinate and camera base coordinate. During training, actions are transformed from the robot base coordinate to the camera coordinate and used as groundtruth. During inference, the predicted actions are transformed back from the camera base coordinate to the robot base coordinate for execution on the real robot.*

In this section, we provide a detailed analysis of the advantages of camera-centric action prediction. In details, we can get $\mathbf{A}_{\text{cam}}$ from equations  1,  2 and  3 as follow,

|  | $$ \mathbf{A}_{\text{cam}}=\mathbf{T}\mathbf{A}_{\text{world}}\mathbf{T}^{-1} $$ |  | (4) |
|---|---|---|---|

where $\mathbf{A}_{\text{cam}}$ is the camera-based action, $\mathbf{A}_{\text{world}}$ is the robot-based action, and $\mathbf{T}$ is the camera world-to-camera transformation matrix.

Meanwhile, given an end-effector pose $\mathbf{P}_{\text{world}}$ of the robot, we can get,

|  | $$ \mathbf{P}_{\text{cam}}=\mathbf{T}\mathbf{P}_{\text{world}} $$ |  | (5) |
|---|---|---|---|

Equations 4 and  5 present that both the end effector pose and action in world space require the camera transformation matrix $\mathbf{T}$ to be driven from representations in observation space.

In particular, the transformation matrix $\mathbf{T}$ varies across different robot setups. For instance, Droid [9] features 1417 distinct camera viewpoints, requiring the model to internally infer the correct transformation $\mathbf{T}$ for each view to predict actions accurately in the robot’s coordinate frame.

Besides, the traditional perception task is based on UV coordinates (image coordinates). According to the intrinsics of the camera, we can obtain the UV coordinate from $(X_{\text{cam}},Y_{\text{cam}},Z_{\text{cam}})$. Given that the intrinsic matrix $\mathbf{K}$, the image coordinates $(u,v)$ can be calculated as:

|  | $$ u=\frac{f_{x}\cdot X_{\text{cam}}}{Z_{\text{cam}}}+c_{x} $$ |  | (6) |
|---|---|---|---|

|  | $$ v=\frac{f_{y}\cdot Y_{\text{cam}}}{Z_{\text{cam}}}+c_{y} $$ |  | (7) |
|---|---|---|---|

Where $f_{x},f_{y}$ are the focal lengths in the x and y directions, $c_{x},c_{y}$ are the principal point coordinates (usually the image center). We observe that the camera coordinate can be directly derived from the UV coordinate, and the intrinsic parameters are usually consistent across cameras of the same model. However, translating a point from the camera coordinate system to the robot base coordinate requires the corresponding rotation matrix, which varies with different camera placements. As a result, learning this translation for robot space action prediction becomes more challenging due to the diversity in camera poses.  In contrast, observation-centric action prediction inherently avoids these issues, offering a more consistent mapping between observation and action.

## IV Experiments

In this section, we first provide a detailed description of the pretraining data, followed by an overview of the model architecture for different action spaces. Next, we present the optimization process. Lastly, we present a comprehensive evaluation of the performance of our proposed method on both simulated benchmarks and real-world robotic platforms.

![Refer to caption](drafts/images/ocvla-2508.13103/setup.png)

*Fig. 4: The real-world robot platform with a Franka Emika Panda robot, a Robotiq 2F-85 gripper and multiple RealSense D435i RGB-D cameras.*

### IV-A Pretraining Data

To ensure a comprehensive and fair evaluation of our proposed approach, we incorporate a pretraining stage in selected experiments. Pretraining provides the model with a stronger initialization, which is particularly beneficial when handling complex multimodal inputs and diverse visual contexts. Since our method operates from a third-person perspective and explicitly requires camera extrinsics to transform robot-centric actions into the camera frame, it is crucial to select a dataset that includes such calibration information.

For this purpose, we choose the Droid dataset [9] for pretraining. This dataset consists of robotic manipulation trajectories captured from 1417 distinct third-person camera viewpoints, along with their corresponding extrinsic parameters, offering a wide range of visual perspectives and motion patterns. This diversity makes it an ideal choice for evaluating the generalizability and robustness of our observation-centric action prediction framework. Unless otherwise noted, all experiments involving pretrained models are initialized using weights obtained through pretraining on the Droid dataset [9].

### IV-B Model Details

In our experiments, we employ a typical lightweight VLM architecture, with distinct designs for the continuous and discrete action spaces. In the following, we detail the model implementations for each action space.

For the continuous action space model, we adopt a diffusion policy. In addition to language and image tokens, we concatenate the current timestep and the noise-perturbed action as inputs to the causal transformer. The entire transformer functions as a Diffusion Transformer (DiT) [39], which iteratively denoises the input over multiple steps to generate the final end-effector action.

For the discrete action space model, we pad zero vectors to align the action size after processing language and image inputs. The combined sequence is then fed into the Transformer. Despite using a causal mask during training, the model predicts the entire action sequence in a single pass, rather than autoregressively. This design enhances both semantic consistency across tokens and computational efficiency.

### IV-C Optimization Details

The training objectives vary depending on the type of action space used. For models with a continuous action space, the objective is to minimize the mean squared error (MSE) between the robot’s action (augmented with standard Gaussian noise) and the predicted noise, using DDPM [59] with 100 timesteps. In contrast, for models with a discrete action space, the robot actions are normalized to a predefined range and quantized into discrete bins. The objective here is to minimize the cross-entropy loss between the predicted discrete actions and the ground-truth labels.

For diffusion evaluation, we use DDIM [60] with 10 timesteps during inference. The model is optimized using AdamW [61] for 30,000 steps, with learning rates of $1e-4$ for both the causal Transformer and Q-Former, and $1e-5$ for DINOv2. Training is conducted with a batch size of 2048 across 8 NVIDIA A100 GPUs, with 256 samples per GPU. The model predicts actions in the third-person camera base coordinate, while the baseline model predicts actions in the robot base coordinate.

### IV-D Simulation Evaluation

*TABLE I: Comparison on ManiSkill2 under Success rate. SingleYCB indicates PickSingleYCB, ClutterYCB indicates PickClutterYCB, SingleEGAD indicates PickSingleEGAD. Coord indicates the selected coordinate while training. Continuous indicates whether the action prediction is continuous or discrete.*

| Coord | Continuous | All | PickC | StackC | SingleYCB | ClutterYCB | EGAD |
|---|---|---|---|---|---|---|---|
| Robot | ✓ | 45.2% | 71.0% | 62.0% | 30.0% | 15.0% | 48.0% |
| Camera | ✓ | 53.2% | 88.0% | 65.0% | 46.0% | 19.0% | 48.0% |
| Robot | × | 38.6% | 61.0% | 51.0% | 28.0% | 8.0% | 45.0% |
| Camera | × | 52.4% | 80.0% | 65.0% | 48.0% | 19.0% | 50.0% |

#### IV-D1 Simulation Dataset

For simulated evaluation, we select ManiSkill2 [62] to assess the effectiveness and generalization capabilities of our proposed approach. ManiSkill2, the successor to the original SAPIEN ManiSkill [63] benchmark, has become a widely recognized and authoritative platform for evaluating the generalization performance of embodied agents in robotic manipulation. Meanwhile, ManiSkill2 includes 20 diverse task families, covering a broad range of real-world manipulation scenarios. Additionally, ManiSkill2 supports rendering observations from randomly sampled camera viewpoints, making it a suitable choice for our evaluation.

#### IV-D2 Setup

To construct our benchmark, we select five representative tasks from the ManiSkill2 suite: PickCube-v0, StackCube-v0, PickSingleYCB-v0, PickClutterYCB-v0, and PickSingleEGAD-v0. We generate a pool of 300,000 randomly configured third-person camera viewpoints. For each trajectory, 20 cameras are randomly sampled to render the demonstration, resulting in a dataset comprising over 40,000 unique trajectories. We partition the generated data into training and validation sets using a 19:1 ratio. Care is taken to ensure that each task family is represented in both sets, and that trajectories rendered from different camera viewpoints are distributed across the splits, thereby preventing data leakage. To address data imbalance, we replicate trajectories from underrepresented task families to equalize the number of samples across tasks during training. For closed-loop evaluation, we randomly sample 100 trajectories from the validation set for each task family, resulting in an evaluation set of 500 trajectories. This evaluation benchmark is used to measure the success rate of the model across different manipulation tasks.

#### IV-D3 Comparison

Given the domain gap between Droid and Maniskill2, both the continuous and discrete action space models are trained from scratch in this evaluation. We conduct a comparative analysis of their performance under two supervision regimes: one using robot actions defined in the robot base coordinate frame, and the other using robot actions transformed into the third-person camera coordinate frame as the prediction targets. Table I shows the performance of the comparison of the different models with different prediction target. The results demonstrate that, regardless of the type of action space used, employing robot actions defined in the third-person camera coordinate frame as prediction targets consistently improves task success rates.  This improvement is particularly pronounced in models utilizing a discrete action space, where we observe an increase in success rate of about 14%.

### IV-E Real Robot Evaluation

*TABLE II: Quantitative results in Real robot experiments. Methods annotated with ”(var)” indicate results obtained under zero-shot camera evaluation, while those without the annotation correspond to evaluations conducted using the Training Cam 1. Robot Base and Camera Base indicates the model we built in robot base coordinates and third-person camera base coordinate following Dita [6], respectively. The mapping between task IDs and their corresponding simple task descriptions is as follows. Task 1: Pick up the carrot into the box. Task 2: Open box. Task 3: Close box. Task 4: Press button. Task 5: Pick up the cup into compartment. Task 6: Cook. Task 7: Bake the bread. Task 8: Push the book. Task 9: Put the marker and insert into pen box. Task 10: Stack dolls. Task 11: Stack bowls. Task 12: Push the toy car. Task 13: Pour the water. Task 14: Fold the towel. Task 15: Use the microwave. A more detailed description of each task is provided in the appendix.*

| Method | Avg | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 | Task 6 | Task 7 |
|---|---|---|---|---|---|---|---|---|
| OpenVLA-OFT | 63.3% | 100.0% | 80.0% | 90.0% | 80.0% | 80.0% | 80.0% | 60.0% |
| OpenVLA-OFT (var) | 42.0% | 90.0% | 70.0% | 60.0% | 40.0% | 50.0% | 50.0% | 10.0% |
| $\pi_{0}$ | 50.7% | 50.0% | 70.0% | 80.0% | 60.0% | 60.0% | 70.0% | 80.0% |
| $\pi_{0}$ (var) | 34.7% | 20.0% | 40.0% | 60.0% | 40.0% | 30.0% | 30.0% | 60.0% |
| Robot Base | 58.0% | 70.0% | 70.0% | 90.0% | 60.0% | 60.0% | 60.0% | 60.0% |
| Robot Base (var) | 41.3% | 40.0% | 50.0% | 70.0% | 60.0% | 40.0% | 60.0% | 60.0% |
| Camera Base (OC-VLA, ours) | 68.0% | 80.0% | 80.0% | 100.0% | 80.0% | 80.0% | 70.0% | 60.0% |
| Camera Base (var) | 54.0% | 70.0% | 70.0% | 100.0% | 70.0% | 60.0% | 60.0% | 70.0% |
| Method | Task 8 | Task 9 | Task 10 | Task 11 | Task 12 | Task 13 | Task 14 | Task 15 |
| OpenVLA-OFT | 70.0% | 50.0% | 20.0% | 20.0% | 80.0% | 50.0% | 60.0% | 30.0% |
| OpenVLA-OFT (var) | 40.0% | 50.0% | 10.0% | 10.0% | 30.0% | 50.0% | 50.0% | 20.0% |
| $\pi_{0}$ | 60.0% | 40.0% | 10.0% | 20.0% | 40.0% | 50.0% | 60.0% | 10.0% |
| $\pi_{0}$ (var) | 60.0% | 30.0% | 10.0% | 10.0% | 50.0% | 40.0% | 30.0% | 10.0% |
| Robot Base | 60.0% | 60.0% | 20.0% | 40.0% | 50.0% | 60.0% | 90.0% | 20.0% |
| Robot Base (var) | 30.0% | 30.0% | 10.0% | 20.0% | 30.0% | 50.0% | 50.0% | 20.0% |
| Camera Base (OC-VLA, ours) | 70.0% | 60.0% | 40.0% | 50.0% | 70.0% | 60.0% | 90.0% | 30.0% |
| Camera Base (var) | 40.0% | 50.0% | 20.0% | 30.0% | 20.0% | 60.0% | 70.0% | 20.0% |

*TABLE III: Real robot experiments of different camera views. Fixed Camera means no camera perturbations while data collection. The meanings of the methods and the Task ID mappings follow the same convention as in Table II.*

| Method | Avg | Task1 | Task2 | Task3 | Task4 | Task5 | Task6 | Task7 | Task8 |
|---|---|---|---|---|---|---|---|---|---|
| Robot Base(Fixed Camera, From Table II) | 66.3% | 70.0% | 70.0% | 90.0% | 60.0% | 60.0% | 60.0% | 60.0% | 60.0% |
| Cam Base(Fixed Camera, From Table II) | 77.5% | 80.0% | 80.0% | 100.0% | 80.0% | 80.0% | 70.0% | 60.0% | 70.0% |
| Robot Base(Camera Perturbations) | 61.3% | 80.0% | 70.0% | 50.0% | 70.0% | 40.0% | 60.0% | 50.0% | 70.0% |
| Cam Base(Camera Perturbations) | 73.8% | 80.0% | 80.0% | 70.0% | 90.0% | 80.0% | 60.0% | 60.0% | 70.0% |

#### IV-E1 Setup

We evaluate OC-VLA on a real-world Franka Robot setup, which comprises a 7-DoF tabletop Franka Emika Panda robot arm equipped with a Robotiq 2F-85 gripper as shown in Figure 4. Three RealSense D435i RGB-D cameras are positioned to capture the task environment from multiple third-person perspectives. Specifically, two cameras are used for both data collection and few-shot evaluation, while the remaining camera is reserved exclusively for zero-shot evaluation.

#### IV-E2 Data Collection and Model Finetuning

We adopt a demonstration-based approach to collect two datasets from different viewpoints using Training Camera 1 and Training Camera 2, respectively. For the dataset collected with Camera 1, we record trajectories for 15 distinct tasks while keeping the camera position fixed throughout the entire data collection process. In contrast, the dataset collected with Camera 2 consists of trajectories for 8 tasks, during which we introduce slight perturbations to the camera position to simulate minor viewpoint variations. The collected tasks span a diverse set of categories, including pick & place, pouring, stacking, pick & rotation, pull & push, as well as other long-horizon tasks, aiming to comprehensively evaluate the true performance of the model. A detailed list of tasks is provided in the appendix. Following Dita [6], for each task in both datasets, we collect 10 demonstration trajectories, aiming to evaluate the model fitting ability under a 10-shot setting.

For model finetuning, we fine-tune the model pretrained on the Droid dataset, using either end effector actions defined in the third-person camera coordinate or those in the robot base coordinate as prediction targets. Both models are optimized with AdamW [61] for 20,000 steps with a batch size of 512. For a fair performance comparison, we also fine-tune the pretrained versions of OpenVLA-OFT [2], $\pi_{0}$ [5] on our collected datasets, using their official training protocols. These models serve as baselines in our evaluation.

![Refer to caption](drafts/images/ocvla-2508.13103/qualitative_short.png)

*Fig. 5: A qualitative comparison in real-robot experiments. Failures are highlighted with red circles.*

#### IV-E3 Quantitative Evaluation and Comparison

The evaluations are organized into the following three main settings:

- •

Fixed Camera Viewpoint. We fine-tune all models using 15 task demonstrations collected from Camera 1 and perform a unified evaluation. For each task, we conduct 10 trials and measure performance by computing the task success rate. In this setting, the camera viewpoint remains fixed and identical throughout both the fine-tuning and evaluation phases.
- •

Slight Camera Perturbations To further validate the robustness of our method, we introduce slight variations in the camera viewpoints. Specifically, we fine-tune the models using 8 task demonstrations collected from Camera 2, each exhibiting minor differences in camera placement. For evaluation, we position the camera in a similar configuration to the fine-tuning setup and recalibrate the camera to obtain updated extrinsic parameters. The camera remains fixed throughout the evaluation process.
- •

Novel Camera Viewpoint. To assess the model’s robustness to changes in camera perspective, we conduct zero-shot evaluations using models fine-tuned with demonstrations from Camera 1. As illustrated in the Figure 4, we introduce a novel, previously unseen camera mounted near Camera 1, and perform all evaluations under this new fixed viewpoint without any additional fine-tuning.

Fix Camera View. As shown in the Table II, under the 10-shot setting with a fixed camera viewpoint, the model fine-tuned using robot base coordinate actions already demonstrates competitive performance. However, when the prediction target is switched from robot-base coordinate actions to camera-base coordinate actions, the model achieves a further 10% improvement in the metric of success rate, surpassing the best-performing baseline, OpenVLA-OFT, fine-tuned on the same data. This indicates that our method can partially compensate for the limited pretraining data and model size by improving data efficiency.

Novel Camera View. For novel camera view, all models exhibit varying degrees of performance degradation in Table II, as expected. Notably, OpenVLA-OFT, which performs as the best baseline under the 10-shot setting, suffers a performance drop of over 20%. In contrast, our method shows only a 14% decrease, outperforming all baselines in this setting.  These results highlight the added robustness to camera viewpoint changes when the model is trained to predict actions in the camera base coordinate frame.

Camera Perturbations. Furthermore, the results in Table III demonstrate the advantage of using camera-base coordinate actions as prediction targets when there is variance in camera viewpoints within the fine-tuning data. Although the overall performance is slightly lower than that under strictly fixed-view conditions, the relative benefit of camera-based supervision increases, underscoring the generalizability of our approach in more realistic and variable settings.

#### IV-E4 Qualitative Comparison

Figure 5 shows a comparison between OC-VLA and the baseline method under the robot base coordinate across different evaluation conditions and camera viewpoints. The results illustrate that OC-VLA offers improved robustness for fine-grained manipulation under a variety of settings. While baseline methods often fail to successfully complete tasks due to inaccurate grasp localization—especially under camera perturbations, OC-VLA consistently identifies more precise grasp positions. This advantage is particularly evident when there is variance in camera viewpoints: whereas baseline models begin to exhibit subtle errors, OC-VLA remains resilient and is able to complete the task successfully.

## V Conclusion

In this paper, we propose Observation-Centric VLA (OC-VLA), a simple yet effective framework that grounds action predictions in the camera base coordinate, addressing the spatial misalignment between perception and action in existing VLA models. OC-VLA introduces no architectural overhead and integrates seamlessly with existing pipelines. Extensive experiments show that OC-VLA significantly improves the cross-view generalization and enhance robustness under viewpoint shifts, showing the practical utility of OC-VLA and its strong potential for generalist robot policies.

## References

- [1]

A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, J. Dabis, C. Finn, K. Gopalakrishnan, K. Hausman, A. Herzog, J. Hsu, J. Ibarz, B. Ichter, A. Irpan, T. Jackson, S. Jesmonth, N. J. Joshi, R. Julian, D. Kalashnikov, Y. Kuang, I. Leal, K.-H. Lee, S. Levine, Y. Lu, U. Malla, D. Manjunath, I. Mordatch, O. Nachum, C. Parada, J. Peralta, E. Perez, K. Pertsch, J. Quiambao, K. Rao, M. Ryoo, G. Salazar, P. Sanketi, K. Sayed, J. Singh, S. Sontakke, A. Stone, C. Tan, H. Tran, V. Vanhoucke, S. Vega, Q. Vuong, F. Xia, T. Xiao, P. Xu, S. Xu, T. Yu, and B. Zitkovich, “Rt-1: Robotics transformer for real-world control at scale,” 2023. [Online]. Available: [https://arxiv.org/abs/2212.06817](https://arxiv.org/abs/2212.06817)
- [2]

M. J. Kim, K. Pertsch, S. Karamcheti, T. Xiao, A. Balakrishna, S. Nair, R. Rafailov, E. Foster, G. Lam, P. Sanketi, et al., “Openvla: An open-source vision-language-action model,” arXiv preprint arXiv:2406.09246, 2024.
- [3]

O. M. Team, D. Ghosh, H. Walke, K. Pertsch, K. Black, O. Mees, S. Dasari, J. Hejna, T. Kreiman, C. Xu, et al., “Octo: An open-source generalist robot policy,” arXiv preprint arXiv:2405.12213, 2024.
- [4]

S. Belkhale, T. Ding, T. Xiao, P. Sermanet, Q. Vuong, J. Tompson, Y. Chebotar, D. Dwibedi, and D. Sadigh, “Rt-h: Action hierarchies using language,” arXiv preprint arXiv:2403.01823, 2024.
- [5]

K. Black, N. Brown, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, L. Groom, K. Hausman, B. Ichter, et al., “$pi_{0}$: A vision-language-action flow model for general robot control,” arXiv preprint arXiv:2410.24164, 2024.
- [6]

Z. Hou, T. Zhang, Y. Xiong, H. Duan, H. Pu, R. Tong, C. Zhao, X. Zhu, Y. Qiao, J. Dai, and Y. Chen, “Dita: Scaling diffusion transformer for generalist vision-language-action policy,” arXiv preprint arXiv:2503.19757, 2025.
- [7]

A. O’Neill, A. Rehman, A. Maddukuri, A. Gupta, A. Padalkar, A. Lee, A. Pooley, A. Gupta, A. Mandlekar, A. Jain, et al., “Open x-embodiment: Robotic learning datasets and rt-x models: Open x-embodiment collaboration 0,” in 2024 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2024, pp. 6892–6903.
- [8]

H. R. Walke, K. Black, T. Z. Zhao, Q. Vuong, C. Zheng, P. Hansen-Estruch, A. W. He, V. Myers, M. J. Kim, M. Du, et al., “Bridgedata v2: A dataset for robot learning at scale,” in Conference on Robot Learning. PMLR, 2023, pp. 1723–1736.
- [9]

A. Khazatsky, K. Pertsch, S. Nair, A. Balakrishna, S. Dasari, S. Karamcheti, S. Nasiriany, M. K. Srirama, L. Y. Chen, K. Ellis, et al., “Droid: A large-scale in-the-wild robot manipulation dataset,” arXiv preprint arXiv:2403.12945, 2024.
- [10]

A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, X. Chen, K. Choromanski, T. Ding, D. Driess, A. Dubey, C. Finn, P. Florence, C. Fu, M. G. Arenas, K. Gopalakrishnan, K. Han, K. Hausman, A. Herzog, J. Hsu, B. Ichter, A. Irpan, N. Joshi, R. Julian, D. Kalashnikov, Y. Kuang, I. Leal, L. Lee, T.-W. E. Lee, S. Levine, Y. Lu, H. Michalewski, I. Mordatch, K. Pertsch, K. Rao, K. Reymann, M. Ryoo, G. Salazar, P. Sanketi, P. Sermanet, J. Singh, A. Singh, R. Soricut, H. Tran, V. Vanhoucke, Q. Vuong, A. Wahid, S. Welker, P. Wohlhart, J. Wu, F. Xia, T. Xiao, P. Xu, S. Xu, T. Yu, and B. Zitkovich, “Rt-2: Vision-language-action models transfer web knowledge to robotic control,” 2023. [Online]. Available: [https://arxiv.org/abs/2307.15818](https://arxiv.org/abs/2307.15818)
- [11]

O. Kroemer, S. Niekum, and G. Konidaris, “A review of robot learning for manipulation: Challenges, representations, and algorithms,” Journal of machine learning research, vol. 22, no. 30, pp. 1–82, 2021.
- [12]

M. Dalal, T. Chiruvolu, D. Chaplot, and R. Salakhutdinov, “Plan-seq-learn: Language model guided rl for solving long horizon robotics tasks,” arXiv preprint arXiv:2405.01534, 2024.
- [13]

J. Yamada, Y. Lee, G. Salhotra, K. Pertsch, M. Pflueger, G. Sukhatme, J. Lim, and P. Englert, “Motion planner augmented reinforcement learning for robot manipulation in obstructed environments,” in Conference on Robot Learning. PMLR, 2021, pp. 589–603.
- [14]

F. Xia, C. Li, R. Martín-Martín, O. Litany, A. Toshev, and S. Savarese, “Relmogen: Leveraging motion generation in reinforcement learning for mobile manipulation,” arXiv preprint arXiv:2008.07792, 2020.
- [15]

M. Shridhar, L. Manuelli, and D. Fox, “Cliport: What and where pathways for robotic manipulation,” in Conference on robot learning. PMLR, 2022, pp. 894–906.
- [16]

——, “Perceiver-actor: A multi-task transformer for robotic manipulation,” in Conference on Robot Learning. PMLR, 2023, pp. 785–799.
- [17]

X. Li, M. Zhang, Y. Geng, H. Geng, Y. Long, Y. Shen, R. Zhang, J. Liu, and H. Dong, “Manipllm: Embodied multimodal large language model for object-centric robotic manipulation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 18 061–18 070.
- [18]

Y. Jin, D. Li, J. Shi, P. Hao, F. Sun, J. Zhang, B. Fang, et al., “Robotgpt: Robot manipulation learning from chatgpt,” IEEE Robotics and Automation Letters, vol. 9, no. 3, pp. 2543–2550, 2024.
- [19]

I. Singh, V. Blukis, A. Mousavian, A. Goyal, D. Xu, J. Tremblay, D. Fox, J. Thomason, and A. Garg, “Progprompt: Generating situated robot task plans using large language models,” in 2023 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2023, pp. 11 523–11 530.
- [20]

X. Li, M. Liu, H. Zhang, C. Yu, J. Xu, H. Wu, C. Cheang, Y. Jing, W. Zhang, H. Liu, et al., “Vision-language foundation models as effective robot imitators,” arXiv preprint arXiv:2311.01378, 2023.
- [21]

H. Zhen, X. Qiu, P. Chen, J. Yang, X. Yan, Y. Du, Y. Hong, and C. Gan, “3d-vla: A 3d vision-language-action generative world model,” arXiv preprint arXiv:2403.09631, 2024.
- [22]

D. Qu, H. Song, Q. Chen, Y. Yao, X. Ye, Y. Ding, Z. Wang, J. Gu, B. Zhao, D. Wang, et al., “Spatialvla: Exploring spatial representations for visual-language-action model,” arXiv preprint arXiv:2501.15830, 2025.
- [23]

C.-L. Cheang, G. Chen, Y. Jing, T. Kong, H. Li, Y. Li, Y. Liu, H. Wu, J. Xu, Y. Yang, et al., “Gr-2: A generative video-language-action model with web-scale knowledge for robot manipulation,” arXiv preprint arXiv:2410.06158, 2024.
- [24]

H.-S. Fang, H. Fang, Z. Tang, J. Liu, J. Wang, H. Zhu, and C. Lu, “Rh20t: A robotic dataset for learning diverse skills in one-shot,” in RSS 2023 Workshop on Learning for Task and Motion Planning, 2023.
- [25]

D. Driess, F. Xia, M. S. Sajjadi, C. Lynch, A. Chowdhery, B. Ichter, A. Wahid, J. Tompson, Q. Vuong, T. Yu, et al., “Palm-e: An embodied multimodal language model,” arXiv preprint arXiv:2303.03378, 2023.
- [26]

H. Wu, Y. Jing, C. Cheang, G. Chen, J. Xu, X. Li, M. Liu, H. Li, and T. Kong, “Unleashing large-scale video generative pre-training for visual robot manipulation,” arXiv preprint arXiv:2312.13139, 2023.
- [27]

P. Li, H. Wu, Y. Huang, C. Cheang, L. Wang, and T. Kong, “Gr-mg: Leveraging partially-annotated data via multi-modal goal-conditioned policy,” IEEE Robotics and Automation Letters, 2025.
- [28]

S. Huang, L. Chen, P. Zhou, S. Chen, Z. Jiang, Y. Hu, P. Gao, H. Li, M. Yao, and G. Ren, “Enerverse: Envisioning embodied future space for robotics manipulation,” arXiv preprint arXiv:2501.01895, 2025.
- [29]

C. Lynch and P. Sermanet, “Language conditioned imitation learning over unstructured data,” arXiv preprint arXiv:2005.07648, 2020.
- [30]

M. Reuss, M. Li, X. Jia, and R. Lioutikov, “Goal-conditioned imitation learning using score-based diffusion policies,” arXiv preprint arXiv:2304.02532, 2023.
- [31]

H. Ha, P. Florence, and S. Song, “Scaling up and distilling down: Language-guided robot skill acquisition,” in Conference on Robot Learning. PMLR, 2023, pp. 3766–3777.
- [32]

V. Myers, A. W. He, K. Fang, H. R. Walke, P. Hansen-Estruch, C.-A. Cheng, M. Jalobeanu, A. Kolobov, A. Dragan, and S. Levine, “Goal representations for instruction following: A semi-supervised language interface to control,” in Conference on Robot Learning. PMLR, 2023, pp. 3894–3908.
- [33]

E. Zhang, Y. Lu, W. Wang, and A. Zhang, “Language control diffusion: Efficiently scaling through space, time, and tasks,” arXiv preprint arXiv:2210.15629, 2022.
- [34]

L. Chen, S. Bahl, and D. Pathak, “Playfusion: Skill acquisition via diffusion from language-annotated play,” in Conference on Robot Learning. PMLR, 2023, pp. 2012–2029.
- [35]

Y. Tian, S. Yang, J. Zeng, P. Wang, D. Lin, H. Dong, and J. Pang, “Predictive inverse dynamics models are scalable learners for robotic manipulation,” arXiv preprint arXiv:2412.15109, 2024.
- [36]

J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” Advances in neural information processing systems, vol. 33, pp. 6840–6851, 2020.
- [37]

R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “High-resolution image synthesis with latent diffusion models,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 10 684–10 695.
- [38]

P. Dhariwal and A. Nichol, “Diffusion models beat gans on image synthesis,” Advances in neural information processing systems, vol. 34, pp. 8780–8794, 2021.
- [39]

W. Peebles and S. Xie, “Scalable diffusion models with transformers,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 4195–4205.
- [40]

T. Brooks, B. Peebles, C. Holmes, W. DePue, Y. Guo, L. Jing, D. Schnurr, J. Taylor, T. Luhman, E. Luhman, C. Ng, R. Wang, and A. Ramesh, “Video generation models as world simulators,” 2024. [Online]. Available: [https://openai.com/research/video-generation-models-as-world-simulators](https://openai.com/research/video-generation-models-as-world-simulators)
- [41]

Z. Liang, Y. Mu, H. Ma, M. Tomizuka, M. Ding, and P. Luo, “Skilldiffuser: Interpretable hierarchical planning via skill abstractions in diffusion-based task execution,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 16 467–16 476.
- [42]

Z. Wang, Z. Li, A. Mandlekar, Z. Xu, J. Fan, Y. Narang, L. Fan, Y. Zhu, Y. Balaji, M. Zhou, et al., “One-step diffusion policy: Fast visuomotor policies via diffusion distillation,” arXiv preprint arXiv:2410.21257, 2024.
- [43]

J. Cao, Q. Zhang, J. Sun, J. Wang, H. Cheng, Y. Li, J. Ma, Y. Shao, W. Zhao, G. Han, et al., “Mamba policy: Towards efficient 3d diffusion policy with hybrid selective state models,” arXiv preprint arXiv:2409.07163, 2024.
- [44]

Y. Wang, Y. Zhang, M. Huo, R. Tian, X. Zhang, Y. Xie, C. Xu, P. Ji, W. Zhan, M. Ding, et al., “Sparse diffusion policy: A sparse, reusable, and flexible policy for robot learning,” arXiv preprint arXiv:2407.01531, 2024.
- [45]

B. Chen, D. M. Monso, Y. Du, M. Simchowitz, R. Tedrake, and V. Sitzmann, “Diffusion forcing: Next-token prediction meets full-sequence diffusion,” arXiv preprint arXiv:2407.01392, 2024.
- [46]

C. Chi, S. Feng, Y. Du, Z. Xu, E. Cousineau, B. Burchfiel, and S. Song, “Diffusion policy: Visuomotor policy learning via action diffusion,” arXiv preprint arXiv:2303.04137, 2023.
- [47]

Y. Ze, G. Zhang, K. Zhang, C. Hu, M. Wang, and H. Xu, “3d diffusion policy: Generalizable visuomotor policy learning via simple 3d representations,” in ICRA 2024 Workshop on 3D Visual Representations for Robot Manipulation, 2024.
- [48]

T.-W. Ke, N. Gkanatsios, and K. Fragkiadaki, “3d diffuser actor: Policy diffusion with 3d scene representations,” arXiv preprint arXiv:2402.10885, 2024.
- [49]

M. Reuss, Ö. E. Yağmurlu, F. Wenzel, and R. Lioutikov, “Multimodal diffusion transformer: Learning versatile behavior from multimodal goals,” in First Workshop on Vision-Language Models for Navigation and Manipulation at ICRA 2024, 2024.
- [50]

S. Liu, L. Wu, B. Li, H. Tan, H. Chen, Z. Wang, K. Xu, H. Su, and J. Zhu, “Rdt-1b: a diffusion foundation model for bimanual manipulation,” arXiv preprint arXiv:2410.07864, 2024.
- [51]

J. Wen, Y. Zhu, J. Li, Z. Tang, C. Shen, and F. Feng, “Dexvla: Vision-language model with plug-in diffusion expert for general robot control,” arXiv preprint arXiv:2502.05855, 2025.
- [52]

J. Wen, M. Zhu, Y. Zhu, Z. Tang, J. Li, Z. Zhou, C. Li, X. Liu, Y. Peng, C. Shen, et al., “Diffusion-vla: Scaling robot foundation models via unified diffusion and autoregression,” arXiv preprint arXiv:2412.03293, 2024.
- [53]

A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. u. Kaiser, and I. Polosukhin, “Attention is all you need,” in Advances in Neural Information Processing Systems, I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, Eds., vol. 30. Curran Associates, Inc., 2017. [Online]. Available: [https://proceedings.neurips.cc/paper˙files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
- [54]

S. Dasari, O. Mees, S. Zhao, M. K. Srirama, and S. Levine, “The ingredients for robotic diffusion transformers,” arXiv preprint arXiv:2410.10088, 2024.
- [55]

A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, et al., “Learning transferable visual models from natural language supervision,” in International conference on machine learning. PMLR, 2021, pp. 8748–8763.
- [56]

M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby, et al., “Dinov2: Learning robust visual features without supervision,” arXiv preprint arXiv:2304.07193, 2023.
- [57]

J. Li, D. Li, S. Savarese, and S. Hoi, “Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models,” in International conference on machine learning. PMLR, 2023, pp. 19 730–19 742.
- [58]

E. Perez, F. Strub, H. De Vries, V. Dumoulin, and A. Courville, “Film: Visual reasoning with a general conditioning layer,” in Proceedings of the AAAI conference on artificial intelligence, vol. 32, no. 1, 2018.
- [59]

J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” in Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual, H. Larochelle, M. Ranzato, R. Hadsell, M. Balcan, and H. Lin, Eds., 2020. [Online]. Available: [https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html](https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html)
- [60]

J. Song, C. Meng, and S. Ermon, “Denoising diffusion implicit models,” in 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net, 2021. [Online]. Available: [https://openreview.net/forum?id=St1giarCHLP](https://openreview.net/forum?id=St1giarCHLP)
- [61]

I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” in 7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019. OpenReview.net, 2019. [Online]. Available: [https://openreview.net/forum?id=Bkg6RiCqY7](https://openreview.net/forum?id=Bkg6RiCqY7)
- [62]

J. Gu, F. Xiang, X. Li, Z. Ling, X. Liu, T. Mu, Y. Tang, S. Tao, X. Wei, Y. Yao, et al., “Maniskill2: A unified benchmark for generalizable manipulation skills,” arXiv preprint arXiv:2302.04659, 2023.
- [63]

T. Mu, Z. Ling, F. Xiang, D. Yang, X. Li, S. Tao, Z. Huang, Z. Jia, and H. Su, “Maniskill: Generalizable manipulation skill benchmark with large-scale demonstrations,” arXiv preprint arXiv:2107.14483, 2021.
- [64]

H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux, T. Lacroix, B. Rozière, N. Goyal, E. Hambro, F. Azhar, et al., “Llama: Open and efficient foundation language models,” arXiv preprint arXiv:2302.13971, 2023.

## APPENDIX

### V-A Model Structure and Details

![Refer to caption](drafts/images/ocvla-2508.13103/model_detail.png)

*Fig. 6: Model Structure. We use the same model structure which is followed Dita to evaluate our method on both continuous action space and discrete action space, but treat the Transformer as different function module. The Transformer is served as a Diffusion Transformer (DiT) in the model predicting the actions in the continuous action space but as a non-autoregressive Transformer in which working in the discrete action space.*

In this section, we provide a detailed description of the model architecture and implementation used in our experiments. The overall structure follows the design of Dita without modification [6], as illustrated in Figure 6. Specifically, the model takes as input only a language description and an RGB image from a third-person camera.

The language description is encoded using a pretrained CLIP text encoder [55], while the RGB image is first resized to 224×224 and then processed through a pretrained DINOv2 [56] vision encoder. The resulting image features are passed into a 4-layer Q-Former [57], which serves to reduce the number of image tokens and control the overall model size. The number of image tokens is reduced to 32. The Q-Former is trained from scratch. Additionally, a FiLM layer [58] is injected into each Q-Former block, where the encoded language embedding is used as conditioning input to guide the selection and compression of visual features. The processed language and image features are then jointly fed into a LLaMA2-style Transformer [64], which produces the final predicted action for execution by the robot. This Transformer consists of 12 layers with a hidden size of 768, and operates under a causal masking scheme. The total model size is approximately 334M parameters, with only the CLIP text encoder frozen during training.

To ensure fair comparison, we adopt this unified architecture for both the continuous and discrete action space experiments, with only minor differences in how the Transformer component is used. For continuous action space models, we follow Dita and treat the Transformer as a Diffusion Transformer (DiT). During training, the ground-truth action is perturbed with noise using a 100-timestep DDPM scheduler [59], and then input into the DiT along with the timestep embedding and the preprocessed language and image features. The DiT is trained to predict the added noise. During inference, we use a 10-timestep DDIM scheduler [60] for efficient denoising, which previous work has shown to maintain strong performance with reduced computational cost. For discrete action space models, the Transformer is not autoregressive, predicting all action tokens in a single forward pass. The action is first normalized into a fixed range and tokenized. This formulation ensures efficient inference and explicitly decouples dependencies between individual action tokens.

### V-B Full Pipeline of Our method

![Refer to caption](drafts/images/ocvla-2508.13103/train-infer_detail.png)

*Fig. 7: Full Pipeline of our method. We introduce OC-VLA framework, aligning the observation space and the prediction target with the camera extrinsic calibration matrix. It is simple and efficient, improve the performance of the VLA models without any extra GPU consumption.*

In this work, we propose the Observation-Centric Vision-Language-Action (OC-VLA) framework, which leverages the extrinsic calibration matrix of a third-person camera to transform the end-effector pose from the robot base coordinate to the camera base coordinate . The transformed pose is then used as the prediction target for the model, thereby aligning the observation space with the action prediction target. The overall architecture of the proposed framework is illustrated in Figure 7.
Our framework introduces a minor distinction between the training and inference stages, which consists of the following key steps. During training, since the end-effector poses in most robotic datasets are defined in the robot base coordinate, we first apply the extrinsic calibration matrix of the third-person camera to transform the pose into the camera coordinate frame. This transformation process is described in detail in the Method Section. The transformed pose in the camera base coordinate is then used as the groundtruth for supervision, aligning the model’s prediction target with the visual observation space. During inference, the model outputs an end-effector pose in the camera base coordinate. However, real-world robotic systems typically require poses expressed in the robot base coordinate. To bridge this gap, we apply a post-processing step that transforms the predicted pose back from the camera base coordinate to the robot base coordinate using the same extrinsic matrix. The converted pose is then sent to the physical robot for execution.

Our approach is simple, efficient, and plug-and-play, requiring no additional GPU overhead and minimal integration effort. It offers strong potential for practical adoption in VLA systems, especially in settings involving diverse or dynamic camera viewpoints.

### V-C Simulation Benchmark Experiments

#### V-C1 Dataset Visualization

![Refer to caption](drafts/images/ocvla-2508.13103/manivisual.png)

*Fig. 8: Visualization of the ManiSkill2 Dataset. We generate a third-view camera pool in the Simulated Environment and sample 20 cameras for each of the trajectory to render the data as our dataset.*

For the simulation experiments, we utilize the ManiSkill2 dataset [62], a benchmark built on the SAPIEN simulator that supports flexible third-person camera placement and trajectory rendering—making it particularly well-suited for evaluating our proposed method.

Leveraging these features, we construct a new dataset by selecting five task families from ManiSkill2: PickCube, StackCube, PickSingleYCB, PickSingleEGAD, and PickClutterYCB. To introduce sufficient visual diversity, we generate a camera pool containing 300,000 randomly sampled third-person viewpoints, and for each trajectory, we randomly select 20 camera poses from this pool to render demonstrations.

The resulting dataset consists of approximately 40,000 unique trajectories, with 5% held out as a validation set. Throughout the dataset construction process, we ensure balanced distribution across task families and strict separation between training and validation sets to prevent data leakage. Figure 8 provides representative visual examples from our generated dataset.

#### V-C2 Qualitative Comparison

![Refer to caption](drafts/images/ocvla-2508.13103/maniqualitative_v2.png)

*Fig. 9: Qualitative Comparison on ManiSkill2 of OC-VLA and Baseline. OC-VLA show better performance on the grasp pose and searching for the goal point.*

In Figure 9, we demonstrate some qualitative results of our method. As illustrated in the figure, the alignment between the observation space and the prediction target enables OC-VLA to produce more accurate grasp poses and better end-effector alignment. The model consistently identifies the correct goal point with higher precision, which contributes to the superior performance observed on this benchmark.

#### V-C3 Ablation Study

To comprehensively evaluate the performance of OC-VLA under diverse conditions, we conduct a series of ablation studies on the model configured with a discrete action space. The results are summarized in Table IV.

Specifically, we investigate the impact of three key factors during training: the observation sequence length, the trajectory length, and whether the ViT encoder is frozen during optimization. Across all settings, OC-VLA, which uses action targets represented in the camera base coordinate, consistently outperforms the baseline model that operates in the robot base coordinate. These results highlight the generalizability and robustness of OC-VLA across different training conditions.

*TABLE IV: Ablation Study on ManiSkill2. SingleYCB indicates PickSingleYCB, ClutterYCB indicates PickClutterYCB, SingleEGAD indicates PickSingleEGAD. Coord indicates the selected coordinate while training. #Obs, #Traj, #Freeze ViT indicates the observation length, the trajectory length and whether freezeing the ViT backbone while training.*

| Coord | #Obs | #Traj | # Freeze ViT | All | PickC | StackC | SingleYCB | ClutterYCB | EGAD |
|---|---|---|---|---|---|---|---|---|---|
| Robot | 2 | 2 | × | 38.6% | 61.0% | 51.0% | 28.0% | 8.0% | 45.0% |
| Camera | 2 | 2 | × | 52.4% | 80.0% | 65.0% | 48.0% | 19.0% | 50.0% |
| Robot | 2 | 2 | ✓ | 16.6% | 34.0% | 24.0% | 10.0% | 6.0% | 9.0% |
| Camera | 2 | 2 | ✓ | 27.8% | 51.0% | 49.0% | 14.0% | 9.0% | 16.0% |
| Robot | 2 | 16 | × | 16.4% | 23.0% | 29.0% | 15.0% | 1.0% | 14.0% |
| Camera | 2 | 16 | × | 39.4% | 63.0% | 58.0% | 27.0% | 15.0% | 34.0% |
| Robot | 3 | 3 | × | 33.0% | 54.0% | 32.0% | 28.0% | 6.0% | 45.0% |
| Camera | 3 | 3 | × | 51.8% | 77.0% | 75.0% | 43.0% | 9.0% | 55.0% |

### V-D Real Robot Experiments

#### V-D1 Tasks Details for Real Franka Arm Evaluation

| ![Refer to caption](drafts/images/ocvla-2508.13103/Pick_up_the_carrot_into_the_box.png) |
|---|
| Pick up the carrot into the box. |
| ![Refer to caption](drafts/images/ocvla-2508.13103/Open_the_storage_box.png) |
| Open the storage box. |
| ![Refer to caption](drafts/images/ocvla-2508.13103/Close_the_storage_box.png) |
| Close the storage box. |
| ![Refer to caption](drafts/images/ocvla-2508.13103/Press_the_blue_red_yellow_green_button.png) |
| Press the blue button. |
| ![Refer to caption](drafts/images/ocvla-2508.13103/Pick_up_the_cup_into_the_wood_grid.png) |
| Pick up the cup into the wood grid. |

*Fig. 10: Data Samples of the Dataset on the Real Franka Emika Panda Robot Arm.*

| ![Refer to caption](drafts/images/ocvla-2508.13103/Prepare_the_dishes.png) |
|---|
| Prepare the dishes. |
| ![Refer to caption](drafts/images/ocvla-2508.13103/Bake_the_bread.png) |
| Bake the bread. |
| ![Refer to caption](drafts/images/ocvla-2508.13103/Push_the_book_into_the_black_square.png) |
| Push the book into the black square. |
| ![Refer to caption](drafts/images/ocvla-2508.13103/Put_the_marker_into_the_pen_container.png) |
| Put the marker into the pen container. |
| ![Refer to caption](drafts/images/ocvla-2508.13103/Fold_the_nesting_dolls_together.png) |
| Fold the nesting dolls together. |

*Fig. 11: Data Samples of the Dataset on the Real Franka Emika Panda Robot Arm.*

| ![Refer to caption](drafts/images/ocvla-2508.13103/Stack_the_bowls.png) |
|---|
| Stack the bowls. |
| ![Refer to caption](drafts/images/ocvla-2508.13103/Push_the_toy_car_into_the_black_square.png) |
| Push the toy car into the black square. |
| ![Refer to caption](drafts/images/ocvla-2508.13103/Pour_the_water_from_the_teapot_to_the_cup.png) |
| Pour the water from the teapot to the cup. |
| ![Refer to caption](drafts/images/ocvla-2508.13103/Fold_the_towel.png) |
| Fold the towel. |
| ![Refer to caption](drafts/images/ocvla-2508.13103/Put_the_bone_into_the_plate_then_put_the_plate_intothe_stove_then_close_the_stove.png) |
| Put the bone into the plate then put the plate intothe stove then close the stove. |

*Fig. 12: Data Samples of the Dataset on the Real Franka Emika Panda Robot Arm.*

To assess the real-world performance of our model, we conduct experiments on a Franka Emika Panda robotic arm using a privately collected dataset. We utilize two distinct camera setups for data collection: For Camera 1, we collect 15 tasks, labeled from Task 1 to Task 15. For Camera 2, we collect 8 tasks, labeled from Task 1 to Task 8. These task indices are consistently referenced throughout the Experiment Section in the paper. Representative examples from each task are shown in the Figure 10, Figure 11 and Figure 12.

In the following, we provide the full list of tasks along with their corresponding names and descriptions for reproducibility and clarity.

- •

Task1: Pick up the carrot into the box. This is a pick & place task, pick up the carrot first then move it to the box successfully.
- •

Task2: Open the storage box. This is a pick & place task, the robot should precisely grasp the small handle of the box then open the box.
- •

Task3: Close the storage box. This is a pick & place task, grasp the small handle and move it to the top of the box.
- •

Task4: Press the (blue/red/yellow/green) button. This is a press task. Press the center of the button and make the button flash.
- •

Task5: Pick up the cup into the wood grid. This is a challenging pick & place task due to the narrow rim of the cup and its high susceptibility to tipping over.
- •

Task6: Prepare the dishes. This is a long horizon task. It can be divided in several parts: pick up the potatoes, move the potatoes into the pot, pick up the shovel and make a cook action above the pot, drop the shovel on the towel, push the pot off the chopping board.
- •

Task7: Bake the bread. This is a challenging pick & place task, pick up the bread and insert the bread into the toaster.
- •

Task8: Push the book into the black square. This is a push task, push a large book into the selected area surrounding by the black tape.
- •

Task9: Put the marker into the pen container. This is a pick & place and large rotation task. Pick up the marker firstly then insert the marker into the pen container.
- •

Task10: Fold the nesting dolls together. This is a long horizon task with hard grasping.
- •

Task11: Stack the bowls. This is a long horizon task to stack three bowls together.
- •

Task12: Push the toy car into the black square. This is a push task, pushing the small toy car precisely with the closed gripper.
- •

Task13: Pour the water from the teapot to the cup. This is a pour task with hard grasping and large rotation.
- •

Task14: Fold the towel. This is a soft-body manipulation task, fold the towel by grasp the corner of the towel and drag it to the suitable position.
- •

Task15: Put the bone into the plate then put the plate into the stove then close the stove. This is a long horizon task. Pick up the bone, put it into the plate, grasp the plate and move it into the microwave,
close the microwave.

