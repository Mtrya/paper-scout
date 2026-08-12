# GeniWorld: A Generalizable Interactive World Model for Robotic Manipulation via Visual Actions

Chenghao Gu<sup>1,2,∗</sup>, Hanyang Yu<sup>2,3,∗</sup>, Jingbo Zhang<sup>2,†</sup>, Haitao Lin<sup>2</sup>, Wenyao Zhang<sup>2</sup>, Jinghe Wang<sup>1</sup>, Hanglei Jin<sup>1</sup>, Shuzhao Xie<sup>1</sup>, Jingyan Jiang<sup>4</sup>, and Zhi Wang<sup>1,†</sup>

<sup>1</sup>Shenzhen International Graduate School, Tsinghua University <sup>2</sup>Tencent Robotics X

<sup>3</sup>The Hong Kong University of Science and Technology <sup>4</sup>Shenzhen Technology University <sup>∗</sup>Equal contribution <sup>†</sup>Corresponding authors

Project Page: https://chenghaogu.github.io/GeniWorld/

![](images/a653ff51cdc5c2f2348885113c30645a1eb5dcb6a01a6f3811d34bfc64a8130f.jpg)  
Fig. 1. Overview of GeniWorld. GeniWorld is an autoregressive robotic world model that transforms action inputs into visual action representations, enabling closed-loop interaction with human operators and robot policies. Trained on limited scene-specific demonstrations, GeniWorld generalizes to out-of-distribution (OOD) scenarios to produce high-fidelity observation predictions. Furthermore, it synthesizes rich manipulation trajectories to boost downstream policy performance and robustness under diverse conditions.

Abstract— Generalist robot policies exhibit strong capabilities, but their robustness in complex and unseen environments remains limited. Scaling robot learning and evaluation in diverse real-world environments remains costly and challenging. Action-conditioned world models offer a promising alternative, but they often suffer from limited action controllability and poor generalization to out-of-distribution (OOD) scenarios. To this end, we present GeniWorld, an interactive world model for robots that generalizes robustly across unseen scenarios. Building on pretrained video generative models, we use URDFbased rendering to transform numerical actions into visual action representations, enabling spatially grounded action control. By explicitly decoupling embodiment kinematics from environmental dynamics, our model mitigates scene overfitting and facilitates modeling of robot-environment interactions. To achieve closed-loop control, we construct an autoregressive video prediction model integrated with high-frequency robot kinematic control, enabling interaction with both robot policies and human teleoperators. In our experiments, even when trained solely on limited fixed-scene data, our model achieves superior in-domain performance and robust zero-shot generalization to highly randomized, unseen environments. For downstream applications, GeniWorld serves as a scalable policy evaluator that remains reliable under environmental perturbations. Furthermore, even with limited real-world demonstrations, GeniWorld generates diverse manipulation trajectories within the world model, improving downstream policy performance and robustness in complex environments.

## I. INTRODUCTION

Recent advances in generalist robot policies [1]–[12] have demonstrated impressive multi-task capabilities. However, these policies often remain brittle when deployed in complex and unstructured environments [13]–[15]. Changes in backgrounds, object instances, spatial layouts, and task configurations can lead to substantial performance degradation [16], [17]. Scaling robot learning and evaluation to cover such variations typically requires constructing new physical environments, which is costly and severely limits scene diversity.

Action-conditioned world models offer a promising alternative by enabling robots to interact within an imagination space, thereby facilitating scalable policy learning and evaluation. Existing approaches [18]–[22] typically condition generative models directly on numerical action vectors. However, this paradigm suffers from two critical limitations. First, numerical action vectors lack explicit spatial grounding, limiting the accurate modeling of complex robot motions. Second, low-dimensional action conditioning entangles embodiment motion with environmental changes, forcing the model to focus excessively on interaction-irrelevant details and generalize poorly to unseen scenes. We argue that directly conditioning the world model on robot motion provides precise spatial guidance for manipulation and captures finegrained interaction details. Meanwhile, conditioning scene dynamics on robot motion more directly reflects the physical changes driven by robot–environment interactions.

To this end, we present GeniWorld, a generalizable interactive world model conditioned on embodied visual actions that enables closed-loop interaction with policies and human operators. Even with limited demonstrations, our model generalizes robustly across diverse scenarios and enables the generation of novel behaviors.

For the target robotic system, an embodiment-specific kinematic model first converts numerical action sequences into dense robot motion sequences. Building on a pretrained video generative model, GeniWorld encodes visual actions and scene observations into spatially aligned latent representations. We then build an autoregressive model with causal attention, ensuring that future predictions depend strictly on current robot actions and historical states. During inference, we leverage KV caching to maintain high-quality video generation while enabling closed-loop interaction.

We conduct comprehensive experiments to evaluate Geni-World from three perspectives. First, GeniWorld achieves superior generative performance over the baselines and demonstrates strong zero-shot interaction modeling across OOD scenarios. Second, across multiple real-world manip ulation tasks, policy success rates measured in GeniWorld correlate positively with real-world performance, supporting its utility as a scalable policy evaluator. Third, given a limited set of real demonstrations, GeniWorld generates rich and varied manipulation data that improve downstream policy performance under novel layouts and in complex visual environments.

Our main contributions are summarized as follows:

• We introduce GeniWorld, an interactive world model that learns from fixed-scene demonstrations and generalizes robustly to diverse unseen scenarios, supporting robot manipulation in rich synthesized imagination spaces.

• We propose an autoregressive generative model condi tioned on visual actions to decouple embodiment motion from scene dynamics, enabling explicit interaction modeling and closed-loop interaction with both robot policies and human teleoperators.

• We demonstrate that GeniWorld serves as a robust policy evaluator and enables the synthesis of rich, varied manipulation data from limited real-world demonstrations, thereby enhancing downstream policy performance across diverse robotic systems.

## II. RELATED WORK

a) Action-Conditioned World Models for Robot Manipulation: Action-conditioned world models, which predict future observations given control actions, have emerged as a pivotal paradigm in robot learning [23]–[25]. Recent advances further scale this paradigm by leveraging highcapacity video generation backbones to simulate high-fidelity visual interactions and environmental changes [26]–[33].

Existing efforts for robotics [18]–[21] adapt these video generative models into physical simulators by incorporating robot control actions as conditioning signals for manipulation. However, such methods commonly represent robot motion using implicit, low-dimensional numerical vectors. Passing these vectors directly to the video backbone conflates robot kinematics with scene dynamics, ultimately hindering the model from learning precise, interaction-dependent environmental transitions [34].

b) Pixel-Grounded Visual Action Representations: Many works inject action signals into video generative mod els by introducing numerical joint or end-effector parameters through AdaLN or cross-attention layers [18]–[21], [33], [35], [36]; however, these implicit representations struggle to capture 3D spatial dynamics accurately. To address this, subsequent works incorporate explicit pixel-aligned conditions, such as projected end-effector poses [37], [38] or skeletal trajectories [39], [40]. Another line of work leverages URDFrendered abstractions, such as segmentation masks [41] and 3D point clouds [42], to provide spatial grounding for video synthesis. However, such explicit representations often fail to capture fine-grained geometric details and structuregrounded interactions. Crucially, physical interactions stem from environmental dynamics induced by the robot’s spatial motion, which align naturally with the strong spatiotemporal dynamics learned by pretrained video generative models [43]. Therefore, directly embedding embodiment motion into future-frame prediction allows the model to faithfully reflect interaction-driven environmental changes.

c) Data Synthesis for Policy Improvement: Data synthesis is an effective paradigm for enhancing robot policies. One line of work constructs real-to-sim environments to enable data synthesis across diverse scenes without modifying physical setups [44]–[47]. However, these approaches require complex simulation setups and inevitably suffer from the reality gap. Alternatively, other methods employ generative models for visual augmentation of existing action trajectories [48]–[50]. Nevertheless, these augmented videos do not support controllable interactions with the environment, thereby limiting the generation of novel robot behaviors. World models enable scalable data synthesis within imagination spaces to improve manipulation-policy performance [18], [21], [22]. However, existing methods are largely restricted to in-distribution data generation and remain confined to seen environments and familiar objects. Several methods leverage pretraining on large-scale robot datasets to synthesize richer robot-specific data [28], [51]– [53]. However, most robotic systems lack access to such large-scale datasets. Even with limited laboratory-scale data, GeniWorld learns a highly generalizable world model capable of synthesizing diverse manipulation data across novel objects, environments, and spatial layouts. This capability enables broad application across robotic systems for generalist robot learning.

![](images/a1347e4623c9c3f3af7a8d4cf53f96dca023f7383a08d2c4a0e97bccf0f0986e.jpg)  
Fig. 2. Overview of the GeniWorld method. We convert robot actions into visual motions via URDF rendering. These motions are subsequently encoded into latent representations and concatenated with noisy video latents. The combined representation serves as input to a causal DiT that predicts future videos via flow matching. During inference, the initial scene image is provided as the first frame. Actions are passed through the URDF-based renderer to produce motion conditions, and the model predicts future observations while leveraging KV caching to maintain historical context.

## III. METHOD

## A. Problem Formulation

Given a target robotic system, we aim to learn an interactive world model from an offline dataset $\mathcal { D } = \{ \tau _ { i } \} _ { i = 1 } ^ { N }$ consisting of paired observation–action trajectories $\begin{array} { r l } { \tau _ { i } } & { { } = } \end{array}$ $\left\{ \big ( o _ { 0 } , a _ { 0 } \big ) , \big ( o _ { 1 } , a _ { 1 } \big ) , \dots , \big ( o _ { T } , a _ { T } \big ) \right\}$ . Given the current observation $o _ { t }$ and an incoming action sequence $a _ { t + 1 : t + H } =$ $( a _ { t + 1 } , \ldots , a _ { t + H } )$ , our goal is to construct a world model W that predicts the future visual observation sequence $\begin{array} { r l } { \mathrm { ~ } } & { { } O _ { t + 1 : t + H } \colon } \end{array}$

$$
o _ { t + 1 : t + H } = \mathcal { W } \big ( o _ { t } , a _ { t + 1 : t + H } \big ) .
$$

By autoregressively modeling action-sequence-conditioned future prediction, W enables closed-loop interaction with both human operators and downstream policies.

## B. World Modeling with Visual Actions

a) From Numerical to Visual Actions: We assume that the physical dynamics in manipulation videos are predominantly driven by embodiment kinematics. To this end, we construct the dense visual motion sequence m<sub>t+1:t+H</sub> by mapping the numerical actions $a _ { t + 1 : t + H }$ through the robot’s URDF model and forward-kinematics system and rendering the corresponding embodiment motion from the target camera viewpoint. Crucially, this rendering isolates the articulated robot structure and excludes objects and background appearance.

We then define the world model as

$$
o _ { t + 1 : t + H } = \mathcal { W } ( o _ { t } , m _ { t + 1 : t + H } ) ,
$$

where conditioning on visual action sequences enables the system to learn scene interactions driven directly by embodiment motion.

b) Spatially Aligned Action Conditioning: We construct our world model based on a video diffusion backbone [54]. As shown in Fig. 2, visual actions are processed via a causal 3D VAE encoder into action latents $\mathbf { \Psi } _ { \mathcal { Z } _ { a } } ^ { \bullet } \in \mathbb { R } ^ { C \times L \times H ^ { \prime } \times W ^ { \prime } }$ which are channel-wise concatenated with the original video latents $\boldsymbol { z } _ { v } \in \mathbb { R } ^ { C \times L \times H ^ { \prime } \times W ^ { \prime } }$ to form a spatially aligned joint representation $z = [ z _ { v } ; z _ { a } ] \in \mathbb R ^ { 2 C \times L \times \dot { H ^ { \prime } } \times W ^ { \prime } }$

This design aligns the embodiment dynamics with the pretrained visual latent space, ensuring strict spatial correspondence between each position in the action and scene latents. By leveraging this action-embedding mechanism, we introduce minimal architectural modifications to the pretrained video backbone, thereby preserving its rich generative priors for interaction modeling [9].

c) Causal Modeling: We formulate future observation prediction as an autoregressive process. During training, we initially condition the model on ground-truth frames and subsequently feed model-generated predictions into the observation context to mitigate exposure bias in future prediction [55]. We enforce a causal attention mask that restricts each token to attending only to preceding tokens, ensuring that each predicted observation depends strictly on historical states and past action tokens.

d) Training Objective: Using flow matching [56], we predict the next observation conditioned on the history and the corresponding action token. For the next observation latent $z _ { t + 1 }$ , the training objective is formulated as follows:

$$
\begin{array} { r } { \mathcal { L } = \mathbb { E } _ { t , s , \mathbf { z } _ { t + 1 } , \epsilon } \left\| \mathbf { v } _ { \theta } \left( \mathbf { z } _ { t + 1 } ^ { ( s ) } , s , \mathbf { z } _ { \leq t } \mid \mathbf { z } _ { a , t + 1 } , \mathbf { c } \right) - \dot { \mathbf { z } } _ { t + 1 } ^ { ( s ) } \right\| _ { 2 } ^ { 2 } , } \end{array}\tag{1}
$$

where $s \in [ 0 , 1 ]$ denotes the flow timestep, $\mathbf { z } _ { t + 1 } ^ { ( s ) } = ( 1 -$ $\mathbf { \Phi } _ { s } ) \mathbf { \epsilon } + s \mathbf { z } _ { t + 1 }$ defines the interpolated state with Gaussian noise $\epsilon \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ , and $\dot { \mathbf { z } } _ { t + 1 } ^ { ( s ) } = \mathbf { z } _ { t + 1 } - \epsilon$ specifies the groundtruth velocity field. Here, $\mathbf { z } _ { \leq t }$ represents the historical token sequence, while c denotes the language instruction. Only the observation latent is noised and predicted; the visual action serves as a clean conditioning signal.

## C. Efficient Closed-Loop Interaction

a) Autoregressive Prediction: To enable closed-loop interaction with human operators and downstream policies, the world model predicts the next observation in response to input actions and autoregressively feeds generated observations back as historical context for subsequent rollouts. Following [9], we cache the key-value pairs of preceding tokens as historical context while applying full self-attention within the newly generated token block at each prediction.

TABLE I  
QUANTITATIVE EVALUATION OF WORLD-MODEL PERFORMANCE ON THE ROBOTWIN BENCHMARK ACROSS CLEAN-TO-CLEAN AND CLEAN-TO-RANDOM SETTINGS. ↑ (↓) INDICATES THAT HIGHER (LOWER) IS BETTER. BEST RESULTS ARE SHOWN IN BOLD.
<table><tr><td rowspan="2">Method / action representation</td><td colspan="6">Clean-to-Clean</td><td colspan="6">Clean-to-Random</td></tr><tr><td>LPIPS↓ PSNR↑</td><td></td><td>SSIM↑</td><td>FID↓</td><td>FVD↓</td><td></td><td>EWMScore↑ LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>FID↓</td><td>FVD↓</td><td>EWMScore↑</td></tr><tr><td>Ctrl-World</td><td>0.165</td><td>18.65</td><td>0.879</td><td>9.62</td><td>15.52</td><td>58.74</td><td>0.285</td><td>20.41</td><td>0.791</td><td>21.66</td><td>35.85</td><td>51.47</td></tr><tr><td>IRASim</td><td>0.323</td><td>13.26</td><td>0.801</td><td>42.98</td><td>48.98</td><td>48.83</td><td>0.766</td><td>8.15</td><td>0.476</td><td>174.52</td><td>191.26</td><td>46.41</td></tr><tr><td>EnerVerse-AC</td><td>0.378</td><td>10.89</td><td>0.780</td><td>40.35</td><td>54.28</td><td>52.92</td><td>0.751</td><td>12.25</td><td>0.433</td><td>74.55</td><td>99.14</td><td>31.28</td></tr><tr><td>Ours w/ numerical actions</td><td>0.3773</td><td>11.59</td><td>0.7836</td><td>42.69</td><td>48.03</td><td>57.69</td><td>0.3659</td><td>13.74</td><td>0.6961</td><td>40.91</td><td>53.69</td><td>51.49</td></tr><tr><td>Ours w/ EE trajectory</td><td>0.1848</td><td>17.17</td><td>0.8632</td><td>12.19</td><td>18.35</td><td>60.00</td><td>0.3000</td><td>14.17</td><td>0.7158</td><td>37.69</td><td>45.89</td><td>61.36</td></tr><tr><td>Ours w/ skeleton</td><td>0.1801</td><td>17.38</td><td>0.8609</td><td>12.49</td><td>18.82</td><td>60.04</td><td>0.2568</td><td>18.48</td><td>0.7956</td><td>31.78</td><td>39.88</td><td>61.61</td></tr><tr><td>Ours w/ ControlNet-style conditioning</td><td>0.094</td><td>23.99</td><td>0.927</td><td>8.82</td><td>12.36</td><td>59.26</td><td>0.416</td><td>14.51</td><td>0.625</td><td>40.02</td><td>59.95</td><td>56.13</td></tr><tr><td>Ours (visual actions)</td><td>0.055</td><td>27.57</td><td>0.942</td><td>5.59</td><td>7.59</td><td>61.80</td><td>0.144</td><td>22.71</td><td>0.873</td><td>13.08</td><td>20.15</td><td>63.54</td></tr></table>

![](images/dcb76ec5405ceb23956049faf1a0236f96c8b4344b64b455ef81378d3309e0d0.jpg)  
Fig. 3. Qualitative Clean-to-Random world-modeling results. All models are trained on clean tabletop data and evaluated in unseen scenes. GeniWorld preserves the commanded robot motion and produces interaction outcomes that closely match the ground truth, whereas competing methods fail to generate the expected results in out-of-distribution (OOD) scenarios. Furthermore, the ablation results show that our dense visual actions enable more accurate interaction modeling.

b) Interaction via Motion Rendering: For both human teleoperators and robot policies, the executed actions are first rendered as dense visual motions using URDF-based forward kinematics aligned with the camera viewpoint. These motions serve as conditioning signals for the world model. Conditioned on the visual action and historical context, the world model synthesizes the corresponding future observation, which is fed back to the agent to guide subsequent decision-making in a continuous control loop.

## IV. EXPERIMENTS

In this section, we design comprehensive experiments to evaluate GeniWorld by addressing the following questions:

1) Generative Fidelity & Generalization: Can Geni-World generate high-fidelity video predictions and generalize effectively across diverse, unseen scenarios?

2) Policy Evaluation Reliability: Can GeniWorld reliably evaluate downstream policies while maintaining a strong correlation with real-world performance under scene perturbations and out-of-domain settings?

3) Data Synthesis for Policy Improvement: Can Geni-World synthesize effective training data from limited recorded trajectories to improve policy learning and performance without the cost of constructing new realworld setups?

## A. World Modeling Quality

a) Experimental Setup: Following [57], we establish two settings for evaluating action-conditioned world modeling based on RoboTwin2.0 [15]. The Clean-to-Clean setting follows the standard WorldArena protocol. The clean data are split, with 45 trajectories per task used for training and 5 used for evaluation. Across 50 tasks, this produces 2,250 training episodes and 250 held-out test episodes. To evaluate out-of-distribution generalization, we introduce the Clean-to-Random setting for zero-shot out-of-distribution evaluation. Models are trained on the same 2,250 Clean episodes and evaluated, without adaptation, on 250 episodes from the Random subset. These episodes introduce substantial variations in scene appearance, object instances, object placements, and task-relevant spatial layouts. Each episode contains 121 frames rendered at 24 fps. In both settings, the model receives an initial observation $I _ { 0 } ,$ , a language instruction ℓ, and a future action trajectory $a _ { 0 : T - 1 }$ , and predicts the corresponding future observations.

We evaluate visual prediction quality using PSNR, SSIM [58], LPIPS [59], FID [60], FVD [61] and EWM-Score [57]. We compare GeniWorld with Ctrl-World [18], IRASim [19], and EnerVerse-AC [37] under the same training split. For the ablation analysis, we evaluate four actionconditioning designs built on the same video backbone: numerical actions, projected end-effector poses, projected skeleton actions, and our dense visual actions. We further compare our concatenation-based action embedding with ControlNet-style conditioning [62].

b) World-Model Quality and OOD Generalization: As shown in Table I, GeniWorld achieves the highest generation quality in both the Clean-to-Clean and Clean-to-Random settings. In the in-domain evaluation, our method consistently achieves superior performance across all six metrics. Compared to ablation baselines using numerical action control or alternative explicit representations (e.g., end-effector pose and skeleton), our model demonstrates significantly higher alignment with ground-truth observations. Furthermore, under domain shifts from clean to randomized environments, prior works such as IRASim experience severe degradation, with FID and FVD dropping to 174.52 and 191.26, respectively. In contrast, GeniWorld maintains remarkable fidelity, retaining low FID and FVD scores of 13.08 and 20.15. These results demonstrate that GeniWorld achieves superior generation quality even when scene appearance, object instances, object placements, and task-relevant spatial layouts differ significantly from those seen during training.

As illustrated in Fig. 3, despite being trained solely on a fixed, plain scene, GeniWorld synthesizes accurate interactions and dynamics across novel randomized scenes, successfully executing target tasks guided by input actions. In contrast, baseline methods such as Ctrl-World suffer from severe visual artifacts in complex textures and cause unintended changes in task-irrelevant objects during manipulation. Furthermore, implicit numerical actions fail to generate accurate robot motions in unseen environments, while skeleton-based control often generates physically implausible interactions, such as contactless grasping.

c) Training Efficiency: We evaluate the training efficiency of our visual action representation under identical backbone architectures and experimental setups. Fig. 4 presents the evaluation metrics across training iterations for different action input modalities. Compared to baseline representations, visual actions achieve faster convergence and consistently superior performance, producing high-fidelity outputs even in early training stages and consistently outperforming other action representations. This demonstrates that visual actions provide a more effective input modality by leveraging the generative priors of pretrained video models.

d) Inference Efficiency: For robotic manipulation tasks, we observe that visual action conditioning remains highly effective with substantially fewer flow-matching sampling steps. As shown in Fig. 5, reducing the number of sampling steps from 50 to 10 or even 5 incurs negligible quality degradation across both in-domain and out-of-domain scenarios. Specifically, our model achieves a ∼ 10× inference speedup with an FVD degradation of only ∼ 2%, whereas the numerical-action-conditioned baseline deteriorates considerably with an FVD drop of ∼ 22%. Furthermore, as illustrated in Fig. 5(b), visual action conditioning provides strong motion priors for interactive dynamics generation; in contrast, numerical action conditioning suffers severe visual degradation under few-step sampling. This demonstrates that our design effectively increases the interaction rate.

![](images/2b5b2a065423c2384e22b7667cf98ea75e87326ceb094a41cfd1a843940f41a8.jpg)  
Fig. 4. Training convergence across different action representations. Under identical training configurations, we compare generation quality (1− LPIPS; y-axis) against the number of training steps (x-axis) for various action representations.

![](images/867cacdc19229b62b679d831effaedd7f53f072dca40b937fc64a113d97b9dd6.jpg)

(b) Performance Degradation with Reduced Sampling Steps Numerical Actions  
![](images/244d306cae22a6c836c261903b43286d368b48bda119833dac1e2ee9646c9abd.jpg)  
Fig. 5. Performance with different numbers of flow-matching sampling steps. We evaluate world-modeling quality using 50, 10, and 5 inference steps, where fewer steps correspond to faster inference. Visual action conditioning consistently maintains superior generation quality with fewer denoising steps than the other action representations.

## B. Real-World Platform Setup

We construct our experimental platform using a dual-arm Xtrainer robotic system as shown in Figs. 6. Following hand– eye calibration, we integrate the robot’s URDF into Isaac Sim for motion control and replicate the physical camera setup to render visually consistent observations.

We design four representative manipulation tasks involving diverse objects and complex physical dynamics: Move Bowl, Fold Towel, Place Mug, and Open Drawer. Demonstrations are collected using the Xtrainer teleoperation system. During training-data collection, we maintain a clean tabletop environment and randomize only the poses of the manipulated objects across episodes.

![](images/93bbfdac0b9334c100b619897160d2237aba8b99c4dad7501df261e8de082e6b.jpg)  
Fig. 6. Real-world experimental setup. Our platform uses a dual-arm Xtrainer robotic system and includes four manipulation tasks.

## C. World Model for Policy Evaluation

We investigate whether GeniWorld can serve as an effective offline evaluator for downstream policies. By executing the same policy checkpoints in both GeniWorld and realworld setups, we quantitatively measure the correlation between simulated rollouts and physical task success rates.

![](images/dad55c9ff3c809c10a95d84f583c28d6e66d036db05c76e27cdd48908ee1627b.jpg)  
Fig. 7. Correlation between policy performance in GeniWorld and the real world. (a) Joint evaluation of policy performance in in-domain scenarios and OOD scenes with visual distractions. Here, \* denotes tasks with environmental distractions. (b) Comparison showing that our world simulator is more robust in OOD scenarios.

Policies are fine-tuned from the $\pi _ { 0 }$ vision-language-action (VLA) model [1] using the collected real-world demonstrations. For each trial, the policy is initialized with the same first-frame visual observation in GeniWorld and the realworld environment. Human evaluators assess binary task success, while a vision-language model (VLM) [63], [64] serves as an automated judge of manipulation plausibility.

Fig. 7(a) plots real-world success rates (x-axis) against success rates predicted by the world models (y-axis) across the evaluated tasks. GeniWorld exhibits a strong positive correlation between real-world and simulated performance in standard in-domain settings. This alignment remains consistent under environmental perturbations, demonstrating the model’s stability in noisy conditions. As illustrated in

Fig. 7(b), environmental perturbations severely corrupt the generated interactions involving the manipulated object in Ctrl-World. In contrast, our model is substantially more robust to visual distractors and consistently generates accurate manipulation observations. These results demonstrate that our world model provides reliable policy evaluation under diverse conditions.

## D. Data Synthesis for Policy Improvement

We next ask whether GeniWorld can leverage limited demonstrations from a narrow distribution to synthesize diverse manipulation data, thereby improving policy performance and generalization under complex conditions.

a) Experimental Setup: We train the π<sub>0</sub> VLA model on the four tasks using only 25 demonstrations per task. In addition to the standard in-domain setting, we evaluate task execution under challenging out-of-distribution (OOD) conditions: (i) Spatial rearrangement, where target-object positions are randomized over a broader area; (ii) Novel object instances, where target objects are replaced with unseen instances; (iii) Visual distractors, where random objects are placed on the tabletop; and (iv) Lighting shifts, where ambient illumination is varied.

![](images/f48a6eba5eea80f0c28feb60becc8c67d62e529980cb272d3757f7d501a65c5a.jpg)  
Fig. 8. Teleoperation with the interactive world model. A human operator teleoperates the robot in real time based on the generated scene, while the model autoregressively generates the corresponding visual responses.

b) Diverse Data Synthesis via World Model: Using advanced image-generation models (e.g., GPT-Image [65] and Qwen-Image [66]), we synthesize high-variance manipulation scenarios through instruction-driven editing, as shown in Fig. 10. Specifically, we modify the initial frame using targeted instructions, such as placing target objects in edge-case spatial configurations, introducing complex environmental changes, and replacing target objects. As shown in Figs. 8, we construct a system for teleoperating the world model by linking teleoperation hardware to URDFbased robot kinematics in simulation. The system streams rendered visual-action frames directly to the world model during interactive control. With 5 sampling steps, the model achieves an interactive inference rate of approximately 8 Hz on an NVIDIA H20 GPU. The edited images serve as initial frames for conditioning the world model. For existing trajectories, we replay the recorded action sequences through the simulator. To generate novel behaviors, operators interactively control the world model and synthesize new manipulation trajectories. For each task, we synthesize 65 indomain trajectories with spatial randomization and 65 diverse trajectories under varied conditions.

![](images/30ed085fc081bd97907237c71b4eaec3dab2082f6669a9ee26fb8f5529618e1d.jpg)  
Fig. 9. Real-world policy success rates across different settings. Real denotes policies trained strictly on real-world data; Real + Spatial-Gen adds GeniWorld-synthesized data focused on spatial generalization; and Real + Spatial-Gen + Diverse-Gen adds both spatially randomized data and data with diverse scene variations. The combined regime achieves the strongest overall performance.

![](images/4f14c6dc1b560ecbeeeb4ad75e5f7a4d7474270eaef719a57429bfa2f7b43fa6.jpg)  
Fig. 10. Data generation for real-world robot manipulation. We use image-editing models to modify the original scene images and create diverse, complex conditions. New manipulation trajectories are synthesized through action replay or teleoperation.

c) Policy Performance Enhancement: We train the policy under four data regimes for each task: (i) Real-Only, using 25 real-world demonstrations; (ii) + Spatial-Gen, adding 65 spatially randomized trajectories generated by GeniWorld; and (iii) + Spatial-Gen + Diverse-Gen, further adding 65 synthetic trajectories generated under diverse environmental conditions. All policy variants are evaluated under identical physical and OOD test settings to ensure a fair comparison.

Fig. 9 compares the three training regimes. The results shows that Spatial-Gen and Diverse-Gen provide complementary benefits. Spatial-Gen improves success under spatial rearrangement from 37.5% to 62.5%, compared with 42.5% for Diverse-Gen, whereas Diverse-Gen produces substantially larger gains under distractors (63.8% vs. 38.8%), novel instances (62.5% vs. 42.5%), and lighting shifts (42.5% vs. 35.0%). Combining both sources performs best across all settings, reaching 70.0% under spatial rearrangement, 72.5% with distractors, 70.0% on novel instances, and 52.5% under lighting shifts, while increasing the overall success rate from 40.8% to 69.0%. Representative policy behaviors under different OOD conditions are shown in Fig. 11. These results demonstrate that GeniWorld effectively mitigates data scarcity in robot policy learning by synthesizing highly diverse data without the high cost of manual scene construction and data collection. Consequently, our approach substantially improves overall policy performance in challenging out-ofdistribution scenarios.

![](images/2d28c177a0402dfbfbf1a78bfed56058c664ca9c37c272be3ea541e3b2e909dd.jpg)  
Fig. 11. Performance of policies augmented with GeniWorldsynthesized data. Even with minimal collected real-world data, policies augmented with GeniWorld-synthesized data demonstrate superior performance across diverse OOD conditions.

## V. CONCLUSION

In this work, we introduce GeniWorld, a generalizable interactive world model conditioned on visual actions that enables spatially grounded and closed-loop robot interaction. Despite training on limited fixed-scene demonstrations, GeniWorld achieves strong in-domain performance and reliable zero-shot generalization to unseen scenarios. It enables robust policy evaluation and generates scalable, highly diverse synthetic trajectories that improve downstream policy performance. These results highlight the strong potential of generalizable world models to provide scalable imagination spaces for embodied robot learning.

## REFERENCES

[1] K. Black, N. Brown, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, L. Groom, K. Hausman, B. Ichter et al., “π<sub>0</sub>: A visionlanguage-action flow model for general robot control,” arXiv preprint arXiv:2410.24164, 2024.

[2] P. Intelligence, A. Amin, R. Aniceto, A. Balakrishna, K. Black, K. Conley, G. Connors, J. Darpinian, K. Dhabalia, J. DiCarlo et $a l . , \ ^ {  } \pi _ { 0 . 6 } ^ { * } .$ A vla that learns from experience,” arXiv preprint arXiv:2511.14759, 2025.

[3] P. Intelligence, B. Ai, A. Amin, R. Aniceto, A. Balakrishna, G. Balke, K. Black, G. Bokinsky, S. Cao, T. Charbonnier et al., “π<sub>0.7</sub>: A steerable generalist robotic foundation model with emergent capabilities,” arXiv preprint arXiv:2604.15483, 2026.

[4] P. Intelligence, K. Black, N. Brown, J. Darpinian, K. Dhabalia, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai et al., “π<sub>0.5</sub>: A vision-language-action model with open-world generalization,” arXiv preprint arXiv:2504.16054, 2025.

[5] J. Bjorck, F. Castaneda, N. Cherniadev, X. Da, R. Ding, L. Fan,˜ Y. Fang, D. Fox, F. Hu, S. Huang et al., “Gr00t n1: An open foundation model for generalist humanoid robots,” arXiv preprint arXiv:2503.14734, 2025.

[6] G. R. Team, S. Abeyruwan, J. Ainslie, J.-B. Alayrac, M. G. Arenas, T. Armstrong, A. Balakrishna, R. Baruch, M. Bauza, M. Blokzijl et al., “Gemini robotics: Bringing ai into the physical world,” arXiv preprint arXiv:2503.20020, 2025.

[7] H. Lin, H. Yu, J. Huang, H. Zhang, Y. Ling, P. Tan, X. Xue, and Y. Fu, “Posevla: Universal pose pretraining for generalizable visionlanguage-action policies,” arXiv preprint arXiv:2602.19710, 2026.

[8] W. Zhang, H. Liu, Z. Qi, Y. Wang, X. Yu, J. Zhang, R. Dong, J. He, H. Wang, Z. Zhang et al., “Dreamvla: a vision-language-action model dreamed with comprehensive world knowledge,” Advances in Neural Information Processing Systems, vol. 38, pp. 24 195–24 228, 2026.

[9] L. Li, Q. Zhang, Y. Luo, S. Yang, R. Wang, F. Han, M. Yu, Z. Gao, N. Xue, X. Zhu et al., “Causal world modeling for robot control,” arXiv preprint arXiv:2601.21998, 2026.

[10] S. Ye, Y. Ge, K. Zheng, S. Gao, S. Yu, G. Kurian, S. Indupuru, Y. L. Tan, C. Zhu, J. Xiang et al., “World action models are zeroshot policies,” arXiv preprint arXiv:2602.15922, 2026.

[11] W. Zhang, B. Zhang, Z. Qi, W. Zeng, X. Jin, and L. Zhang, “Disentangled robot learning via separate forward and inverse dynamics pretraining,” arXiv preprint arXiv:2604.16391, 2026.

[12] T. Yuan, Z. Dong, Y. Liu, and H. Zhao, “Fast-wam: Do world action models need test-time future imagination?” arXiv preprint arXiv:2603.16666, 2026.

[13] X. Zhou, Y. Xu, G. Tie, Y. Chen, G. Zhang, D. Chu, P. Zhou, and L. Sun, “Libero-pro: Towards robust and fair evaluation of vision-language-action models beyond memorization,” arXiv preprint arXiv:2510.03827, 2025.

[14] S. Fei, S. Wang, J. Shi, Z. Dai, J. Cai, P. Qian, L. Ji, X. He, S. Zhang, Z. Fei et al., “Libero-plus: In-depth robustness analysis of visionlanguage-action models,” arXiv preprint arXiv:2510.13626, 2025.

[15] T. Chen, Z. Chen, B. Chen, Z. Cai, Y. Liu, Z. Li, Q. Liang, X. Lin, Y. Ge, Z. Gu et al., “Robotwin 2.0: A scalable data generator and benchmark with strong domain randomization for robust bimanual robotic manipulation,” arXiv preprint arXiv:2506.18088, 2025.

[16] H. Yuan, Z. Liang, A. Chen, Y. Wang, H. Li, P. Lin, Y. Huang, Z. Lei, T. Zhang, J. Zhang et al., “Qwen-robotmanip technical report: Alignment unlocks scale for robotic manipulation foundation models,” arXiv preprint arXiv:2606.17846, 2026.

[17] H. Yu, H. Lin, J. Zhang, W. Zhang, C. Gu, H. Li, and P. Tan, “Maskwam: Unifying mask prompting and prediction for world-action models,” arXiv preprint arXiv:2606.13515, 2026.

[18] Y. Guo, L. X. Shi, J. Chen, and C. Finn, “Ctrl-World: A controllable generative world model for robot manipulation,” arXiv preprint arXiv:2510.10125, 2025.

[19] F. Zhu, H. Wu, S. Guo, Y. Liu, C. Cheang, and T. Kong, “IRASim: A fine-grained world model for robot manipulation,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025.

[20] Y. Li, Y. Zhu, J. Wen, C. Shen, and Y. Xu, “WorldEval: World model as real-world robot policies evaluator,” arXiv preprint arXiv:2505.19017, 2025.

[21] J. Quevedo, A. K. Sharma, Y. Sun, V. Suryavanshi, P. Liang, and S. Yang, “WorldGym: World model as an environment for policy evaluation,” arXiv preprint arXiv:2506.00613, 2025.

[22] Y. Wang, R. Syed, F. Wu, M. Zhang, A. Onol, J. Barreiros, H. Nayyeri, T. Dear, H. Zhang, and Y. Li, “Interactive world simulator for robot policy training and evaluation,” arXiv preprint arXiv:2603.08546, 2026.

[23] E. Goodman, “World Models - David Ha, Jurgen Schmidhuber.”¨

[24] D. Hafner, T. Lillicrap, J. Ba, and M. Norouzi. Dream to Control: Learning Behaviors by Latent Imagination.

[25] B. Hou, G. Li, J. Jia, T. An, X. Guo, S. Leng, H. Geng, Y. Ze, T. Harada, P. Torr, O. Mees, M. Pollefeys, Z. Liu, J. Wu, P. Abbeel, J. Malik, Y. Du, and J. Yang. World Model for Robot Learning: A Comprehensive Survey.

[26] J. Bruce, M. D. Dennis, A. Edwards, J. Parker-Holder, Y. Shi, E. Hughes, M. Lai, A. Mavalankar, R. Steigerwald, C. Apps et al., “Genie: Generative interactive environments,” in Forty-first International Conference on Machine Learning, 2024.

[27] M. Assran, A. Bardes, D. Fan, Q. Garrido, R. Howes, M. Muckley, A. Rizvi, C. Roberts, K. Sinha, A. Zholus et al., “V-jepa 2: Selfsupervised video models enable understanding, prediction and planning,” arXiv preprint arXiv:2506.09985, 2025.

[28] NVIDIA, N. Agarwal, A. Ali, M. Bala, Y. Balaji, E. Barker, T. Cai, P. Chattopadhyay, Y. Chen, Y. Cui, Y. Ding, D. Dworakowski, J. Fan, M. Fenzi, F. Ferroni, S. Fidler, D. Fox, S. Ge, Y. Ge, J. Gu, S. Gururani, E. He, J. Huang, J. Huffman, P. Jannaty, J. Jin, S. W. Kim, G. Klar, G. Lam, S. Lan, L. Leal-Taixe, A. Li, Z. Li, C.-H.´ Lin, T.-Y. Lin, H. Ling, M.-Y. Liu, X. Liu, A. Luo, Q. Ma, H. Mao, K. Mo, A. Mousavian, S. Nah, S. Niverty, D. Page, D. Paschalidou, Z. Patel, L. Pavao, M. Ramezanali, F. Reda, X. Ren, V. R. N. Sabavat, E. Schmerling, S. Shi, B. Stefaniak, S. Tang, L. Tchapmi, P. Tredak, W.-C. Tseng, J. Varghese, H. Wang, H. Wang, H. Wang, T.-C. Wang, F. Wei, X. Wei, J. Z. Wu, J. Xu, W. Yang, L. Yen-Chen, X. Zeng, Y. Zeng, J. Zhang, Q. Zhang, Y. Zhang, Q. Zhao, and A. Zolkowski. Cosmos World Foundation Model Platform for Physical AI.

[29] [2512.15840] Large Video Planner Enables Generalizable Robot Control. [Online]. Available: https://arxiv.org/abs/2512.15840

[30] C. Zhu, R. Yu, S. Feng, B. Burchfiel, P. Shah, and A. Gupta. Unified World Models: Coupling Video and Action Diffusion for Pretraining on Large Robotic Datasets.

[31] J. Wu, S. Yin, N. Feng, X. He, D. Li, J. Hao, and M. Long, “ivideogpt: Interactive videogpts are scalable world models,” Advances in Neural Information Processing Systems, vol. 37, pp. 68 082–68 119, 2024.

[32] G. R. Team, K. Choromanski, C. Devin, Y. Du, D. Dwibedi, R. Gao, A. Jindal, T. Kipf, S. Kirmani, I. Leal et al., “Evaluating gemini robotics policies in a veo world simulator,” arXiv preprint arXiv:2512.10675, 2025.

[33] S. Yang, Y. Du, K. Ghasemipour, J. Tompson, L. Kaelbling, D. Schuurmans, and P. Abbeel, “Learning interactive real-world simulators,” arXiv preprint arXiv:2310.06114, 2023.

[34] J. Gao, Y. Guo, Z. Guan, W. Huang, W. Ma, X. Xiao, J. Xiong, and S. Wen, “Sword: Style-robust world models as simulators via dynamic latent bootstrapping for vla policy post-training,” arXiv preprint arXiv:2605.07288, 2026.

[35] Z. Jiang, S. Zhou, Y. Jiang, Z. Huang, M. Wei, Y. Chen, T. Zhou, Z. Guo, H. Lin, Q. Zhang et al., “Wovr: World models as reliable simulators for post-training vla policies with rl,” arXiv preprint arXiv:2602.13977, 2026.

[36] F. Zhu, Z. Yan, Z. Hong, Q. Shou, X. Ma, and S. Guo, “Wmpo: World model-based policy optimization for vision-language-action models,” arXiv preprint arXiv:2511.09515, 2025.

[37] Y. Jiang, S. Chen, S. Huang, L. Chen, P. Zhou, Y. Liao et al., “EnerVerse-AC: Envisioning embodied environments with action condition,” arXiv preprint arXiv:2505.09723, 2025.

[38] B. Qiu, L. Chen, Y. Liao, N. Wang, L. Wang, J. Luo, W. Zhao, S. Chen, D. Chen, Y. Li et al., “Ge-sim 2.0: A roadmap towards comprehensive closed-loop video world simulators for robotic manipulation,” arXiv preprint arXiv:2605.27491, 2026.

[39] Y. Wang, C. Wen, H. Guo, S. Peng, M. Qin, H. Bao et al., “Precise action-to-video generation through visual action prompts,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 12 713–12 724.

[40] Z. Wu and J. Gao, “OSCAR: Omni-embodiment action-conditioned world model for robotics,” arXiv preprint arXiv:2606.04463, 2026.

[41] Y. Chen, P. Li, J. Yang, K. He, X. Wu, Y. Xu et al., “BridgeV2W: Bridging video generation models to embodied world models via embodiment masks,” arXiv preprint arXiv:2602.03793, 2026.

[42] M. Xu, T. Zhang, T. Liu, Z. Chen, X. Han, and Z. Liu, “Kinema4d: Kinematic 4d world modeling for spatiotemporal embodied simulation,” arXiv preprint arXiv:2603.16669, 2026.

[43] R. Chu, Y. He, Z. Chen, S. Zhang, X. Xu, D. WANG, H. Yi, X. Liu, H. Zhao, Y. Liu et al., “Wan-move: Motion-controllable video generation via latent trajectory guidance,” Advances in Neural Information Processing Systems, vol. 38, pp. 404–432, 2026.

[44] X. Li, J. Li, Z. Zhang, R. Zhang, F. Jia, T. Wang, H. Fan, K.-K. Tseng, and R. Wang, “Robogsim: A real2sim2real robotic gaussian splatting simulator,” arXiv preprint arXiv:2411.11839, 2024.

[45] H. Zhao, C. Zeng, L. Zhuang, Y. Zhao, S. Xue, H. Wang, X. Zhao, Z. Li, K. Li, S. Huang et al., “High-fidelity simulated data generation for real-world zero-shot robotic manipulation learning with gaussian splatting,” IEEE Robotics and Automation Letters, 2026.

[46] S. Yang, W. Yu, J. Zeng, J. Lv, K. Ren, C. Lu, D. Lin, and J. Pang, “Novel demonstration generation with gaussian splatting enables robust one-shot manipulation,” arXiv preprint arXiv:2504.13175, 2025.

[47] C. Gu, H. Kang, J. Lin, J. Wang, D. Wu, S. Xie, F. Huang, J. Ge, Z. Gong, L. Li et al., “Igen: Scalable data generation for robot learning from open-world images,” arXiv preprint arXiv:2512.01773, 2025.

[48] C. Yuan, S. Joshi, S. Zhu, H. Su, H. Zhao, and Y. Gao, “Roboengine: Plug-and-play robot data augmentation with semantic robot segmentation and background generation,” in 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2025, pp. 7622–7629.

[49] J. Ye, R. Xue, B. Van Hoorick, P. Tokmakov, M. Z. Irshad, Y. Wang, and V. Guizilini, “Anchordream: Repurposing video diffusion for embodiment-aware robot data synthesis,” arXiv preprint arXiv:2512.11797, 2025.

[50] H. A. Alhaija, J. Alvarez, M. Bala, T. Cai, T. Cao, L. Cha, J. Chen, M. Chen, F. Ferroni, S. Fidler et al., “Cosmos-transfer1: Conditional world generation with adaptive multimodal control,” arXiv preprint arXiv:2503.14492, 2025.

[51] S. Gao, W. Liang, K. Zheng, A. Malik, S. Ye, S. Yu et al., “Dream-Dojo: A generalist robot world model from large-scale human videos,” arXiv preprint arXiv:2602.06949, 2026.

[52] J. Jang, S. Ye, Z. Lin, J. Xiang, J. Bjorck, Y. Fang, F. Hu, S. Huang, K. Kundalia, Y.-C. Lin et al., “Dreamgen: Unlocking generalization in robot learning through video world models,” arXiv preprint arXiv:2505.12705, 2025.

[53] G. R. Team, K. Choromanski, C. Devin, Y. Du, D. Dwibedi, R. Gao, A. Jindal, T. Kipf, S. Kirmani, I. Leal, F. Liu, A. Majumdar, A. Marmon, C. Parada, Y. Rubanova, D. Shah, V. Sindhwani, J. Tan, F. Xia, T. Xiao, S. Yang, W. Yu, and A. Zhou. Evaluating Gemini Robotics Policies in a Veo World Simulator.

[54] Wan Team, “Wan: Open and advanced large-scale video generative models,” arXiv preprint arXiv:2503.20314, 2025.

[55] X. Huang, Z. Li, G. He, M. Zhou, and E. Shechtman, “Self forcing: Bridging the train-test gap in autoregressive video diffusion,” Advances in Neural Information Processing Systems, vol. 38, pp. 167 283– 167 308, 2026.

[56] Y. Lipman, R. T. Chen, H. Ben-Hamu, M. Nickel, and M. Le, “Flow matching for generative modeling,” arXiv preprint arXiv:2210.02747, 2022.

[57] Y. Shang, Z. Li, Y. Ma, W. Su, X. Jin, Z. Wang et al., “WorldArena: A unified benchmark for evaluating perception and functional utility of embodied world models,” arXiv preprint arXiv:2602.08971, 2026.

[58] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, “Image quality assessment: from error visibility to structural similarity,” IEEE transactions on image processing, vol. 13, no. 4, pp. 600–612, 2004.

[59] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, “The unreasonable effectiveness of deep features as a perceptual metric,” in 2018 IEEE/CVF conference on computer vision and pattern recognition. IEEE, 2018, pp. 586–595.

[60] M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter, “Gans trained by a two time-scale update rule converge to a local nash equilibrium,” Advances in neural information processing systems, vol. 30, 2017.

[61] T. Unterthiner, S. Van Steenkiste, K. Kurach, R. Marinier, M. Michalski, and S. Gelly, “Towards accurate generative models of video: A new metric & challenges,” arXiv preprint arXiv:1812.01717, 2018.

[62] L. Zhang, A. Rao, and M. Agrawala, “Adding conditional control to text-to-image diffusion models,” in 2023 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE, 2023, pp. 3813–3824.

[63] S. Bai, Y. Cai, R. Chen, K. Chen, X. Chen, Z. Cheng, L. Deng, W. Ding, C. Gao, C. Ge et al., “Qwen3-vl technical report,” arXiv preprint arXiv:2511.21631, 2025.

[64] J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman, D. Almeida, J. Altenschmidt, S. Altman, S. Anadkat et al., “Gpt-4 technical report,” arXiv preprint arXiv:2303.08774, 2023.

[65] OpenAI, “Introducing our latest image generation model in the API,” https://openai.com/index/image-generation-api/, Apr. 2025.

[66] C. Wu, J. Li, J. Zhou, J. Lin, K. Gao, K. Yan, S.-m. Yin, S. Bai, X. Xu, Y. Chen et al., “Qwen-image technical report,” arXiv preprint arXiv:2508.02324, 2025.

## APPENDIX I TRAINING DETAILS

## A. Model Architecture & Training Setup

Our world model backbone is built upon the Wan2.2- TI2V-5B model and optimized using Flow Matching with an autoregressive causal attention mechanism. To integrate action conditioning, visual actions are encoded into latent representations using the same VAE encoder, matching the visual latent dimensions $( C ~ = ~ 4 8 )$ . We perform channelwise concatenation of the action and video latents to form a 96-channel input (48 + 48). Accordingly, the original 3D convolutional patch embedding layer (channels = 48) is replaced with a expanded layer (channels = 96). The first 48 input channels copy the pre-trained weights to preserve the original visual pathways, while the 48 newly added action channels are initialized using Kaiming initialization and scaled by $0 . 1 \times \mathrm { t o }$ stabilize early gradient flow. The bias parameters are inherited directly from the original layer.

The diffusion process utilizes 1000 timesteps with a noise schedule shift parameter of 5.0. We train the model using a learning rate of $1 \times 1 0 ^ { - 5 }$ and a global batch size of 4 on 4 NVIDIA H20 GPUs.. To enhance visual robustness, color overlay augmentation is applied to both the groundtruth video sequences and the initial frame with a probability of $p = 0 . 5$

## B. Inference & Autoregressive Generation

During inference, we employ KV-cache to enable autoregressive prediction. Visual sequences are generated at a spatial resolution of $4 8 0 \times 6 4 0$ , where 121 raw video frames correspond to 31 latent frames after VAE encoding. Generation proceeds chunk-by-chunk with a size of 3 latent frames per autoregressive step, while the initial frame (I<sub>0</sub>) is processed independently as a single block. We apply Classifier-Free Guidance (CFG) with a scale of 3.0. A detailed summary of the hyperparameters is provided in Table II.

TABLE II  
IMPLEMENTATION AND HYPERPARAMETER DETAILS.
<table><tr><td>Parameter / Configuration</td><td>Value</td></tr><tr><td>Backbone Architecture</td><td>Wan2.2-TI2V-5B</td></tr><tr><td>Optimization Objective</td><td>Flow Matching</td></tr><tr><td>Learning Rate</td><td> $1 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Global Batch Size</td><td>4</td></tr><tr><td>Diffusion Timesteps (T)</td><td>1000</td></tr><tr><td>Noise Schedule Shift</td><td>5.0</td></tr><tr><td>Inference Spatial Resolution</td><td>480 × 640</td></tr><tr><td>Frame Count</td><td>121 pixel frames (31 latent frames)</td></tr><tr><td>Autoregressive Block Size</td><td>3 latent frames / block</td></tr><tr><td>Classifier-Free Guidance (CFG)</td><td>3.0</td></tr></table>

## APPENDIX II EVALUATION METRICS.

To evaluate visual fidelity and temporal consistency, we report pixel-level quality via frame-wise PSNR and SSIM (↑), perceptual distance via a pre-trained AlexNet-backed LPIPS (↓), frame-level distribution distance via FID (↓) using an ImageNet pre-trained Inception-v3, and video-level temporal distance via FVD (↓), which computes Frechet distance over´ Inception-v3 features aggregated by concatenated temporal mean and standard deviation.

## APPENDIX III

## DETAILS OF POLICY IMPROVEMENT

This section provides policy-training details, the evaluation protocol, and per-task results for the policy-improvement experiments. Table III reports real-world success rates for the three training-data regimes across all evaluation settings. Figs. 14 and 15 present qualitative comparisons of real-world rollouts from $\pi _ { 0 }$ policies trained on real data alone and on real data augmented with GeniWorld-generated data.

a) Policy training details: For the policy-improvement experiments, we independently fine-tune the publicly released $\pi _ { 0 }$ base model using the official OpenPI implementation. Given an observation $\mathbf { o } _ { t }$ and an action chunk ${ \bf A } _ { t } = { \bf \nabla }$ $\left[ \mathbf { a } _ { t } , \ldots , \mathbf { a } _ { t + H - 1 } \right]$ with $H \ : = \ : 1 6$ , the noisy action input is constructed as

$$
{ \bf A } _ { t } ^ { \tau } = \tau { \bf A } _ { t } + ( 1 - \tau ) \epsilon , \qquad \epsilon \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } ) ,\tag{2}
$$

where $\tau \in [ 0 , 1 ]$ is the flow timestep. The policy is trained using the conditional flow-matching objective

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { F M } } = \mathrm { E } \left[ \| \mathbf { v } _ { \theta } ( \mathbf { A } _ { t } ^ { \tau } , \mathbf { o } _ { t } ) - ( \mathbf { A } _ { t } - \epsilon ) \| _ { 2 } ^ { 2 } \right] . } \end{array}\tag{3}
$$

All policies are initialized from the same $\pi _ { 0 }$ checkpoint and use identical training configurations; only the composition of the training data differs across regimes.

b) Evaluation settings: We evaluate each policy under one in-distribution setting and four out-of-distribution settings. In the Standard setup, the scene, object instances, and lighting remain consistent with the data-collection environment, while object poses are sampled within the training range. Spatial rearrangement retains the original objects and scene appearance but expands the range of task-relevant object placements. Distractors introduces additional taskirrelevant objects on the tabletop. Novel instances replaces the target object with an unseen instance from the same functional category. Lighting shift changes the ambient illumination while preserving the task configuration.

c) Trial protocol and success criteria: All policy variants are evaluated under identical test conditions. We conduct 20 physical trials for each task–setting pair, resulting in $4 \times 5 \times 2 0 = 4 0 0$ trials per training regime. The environment is reset after each trial, and task success is recorded as a binary outcome. Table IV summarizes the execution stages and completion criteria, where Progress indicates the average normalized temporal position of each stage over successful demonstrations. A trial is considered successful only when the final Done criterion is satisfied.

TABLE III  
PER-TASK REAL-WORLD POLICY SUCCESS RATES (%) WITH GENERATED TRAINING DATA. EACH TASK IS EVALUATED OVER 20 TRIALS IN EACH SETTING. MEAN AVERAGES THE FOUR TASKS, WHILE OVERALL AVERAGES THE FIVE EVALUATION SETTINGS.
<table><tr><td rowspan="2">Training data</td><td rowspan="2">Task (n = 20)</td><td colspan="5">Evaluation setting</td><td rowspan="2">Overall</td></tr><tr><td>Standard</td><td>Spatial</td><td>Distractors</td><td>Novel inst.</td><td>Lighting</td></tr><tr><td rowspan="5">Real</td><td>Move Bowl</td><td>90</td><td>50</td><td>45</td><td>40</td><td>40</td><td>53</td></tr><tr><td>Fold Towel</td><td>65</td><td>30</td><td>25</td><td>20</td><td>20</td><td>32</td></tr><tr><td>Place Mug</td><td>75</td><td>35</td><td>35</td><td>35</td><td>35</td><td>43</td></tr><tr><td>Open Drawer</td><td>60</td><td>35</td><td>30</td><td>25</td><td>25</td><td>35</td></tr><tr><td>Mean</td><td>72.5</td><td>37.5</td><td>33.8</td><td>30.0</td><td>30.0</td><td>40.8</td></tr><tr><td rowspan="5">Real + Spatial-Gen</td><td>Move Bowl</td><td>90</td><td>75</td><td>50</td><td>55</td><td>45</td><td>63</td></tr><tr><td>Fold Towel</td><td>70</td><td>55</td><td>30</td><td>30</td><td>25</td><td>42</td></tr><tr><td>Place Mug</td><td>80</td><td>65</td><td>40</td><td>45</td><td>40</td><td>54</td></tr><tr><td>Open Drawer</td><td>65</td><td>55</td><td>35</td><td>40</td><td>30</td><td>45</td></tr><tr><td>Mean</td><td>76.3 </td><td>62.5</td><td>38.8</td><td>42.5</td><td>35.0</td><td>51.0</td></tr><tr><td rowspan="5">Real + Spatial-Gen + Diverse-Gen</td><td>Move Bowl</td><td>100</td><td>80</td><td>85</td><td>85</td><td>65</td><td>83</td></tr><tr><td>Fold Towel</td><td>65</td><td>65</td><td>65</td><td>60</td><td>40</td><td>59</td></tr><tr><td>Place Mug</td><td>90</td><td>70</td><td>75</td><td>75</td><td>55</td><td>73</td></tr><tr><td>Open Drawer</td><td>65</td><td>65</td><td>65</td><td>60</td><td>50</td><td>61</td></tr><tr><td>Mean</td><td>80.0</td><td>70.0</td><td>72.5</td><td>70.0</td><td>52.5</td><td>69.0</td></tr></table>

TABLE IV  
STAGE-WISE EXECUTION PROTOCOL FOR THE FOUR REAL-WORLD MANIPULATION TASKS. PROGRESS INDICATES THE AVERAGE NORMALIZED TEMPORAL POSITION AT WHICH EACH STAGE IS COMPLETED OVER SUCCESSFUL DEMONSTRATIONS, RATHER THAN ITS SUCCESS RATE.
<table><tr><td>Task</td><td>Stage</td><td>Progress</td><td>Completion criterion</td></tr><tr><td>Move Bowl</td><td>S1: Approach and grasp</td><td>45%</td><td>The robot approaches the target bowl and establishes a stable grasp.</td></tr><tr><td></td><td>S2: Lift and Transport</td><td>87%</td><td>The bowl is picked up and moved toward the target.</td></tr><tr><td></td><td>Done: Place on the target</td><td>100%</td><td>The bowl is placed at the designated target position and remains stable.</td></tr><tr><td>Fold Towel</td><td>S1: Approach and grasp</td><td>33%</td><td>The left and right grippers approach the towel and grasp the upper side required for the first fold.</td></tr><tr><td></td><td>S2: Fold top to bottom</td><td>68%</td><td>The upper part of the towel is folded downward over the lower part.</td></tr><tr><td></td><td>S3: Regrasp the right side</td><td>77%</td><td>The robot repositions its grasp on the right side to prepare for the second fold.</td></tr><tr><td></td><td>Done: Fold right to left</td><td>100%</td><td>The right part is folded toward the left, producing the specified two-stage folded configuration.</td></tr><tr><td>Place Mug</td><td>S1: Approach and grasp</td><td>49%</td><td>The robot approaches the mug, aligns the gripper with its handle, and establishes a stable grasp on the handle.</td></tr><tr><td></td><td>S2: Lift and Transport</td><td>89%</td><td>The mug is picked up and moved toward the target.</td></tr><tr><td></td><td>Done: Move to the target</td><td>100%</td><td>The mug is moved to the designated target location and remains stably placed within the target region.</td></tr><tr><td>Open Drawer</td><td>S1: Grasp the handle</td><td>20%</td><td>The robot&#x27;s right arm grasps the drawer handle</td></tr><tr><td></td><td>S2: Pull the drawer</td><td>38%</td><td>The robot opens the drawer sufficiently for object placement.</td></tr><tr><td></td><td>S3: Grasp the marker</td><td>55%</td><td>The left robot arm approaches the marker and establishes a stable grasp.</td></tr><tr><td></td><td>S4: Place the marker</td><td>76%</td><td>The marker is transported into the open drawer and released inside it.</td></tr><tr><td></td><td>Done: Close the drawer</td><td>100%</td><td>The drawer is fully closed with the marker remaining inside, completing the long-horizon task.</td></tr></table>

Replace the table surface with a chessboard pattern, and replace the bowls.  
![](images/944a17f86055de3aab9031217b3a5b32187fbcd184e3ea331a082ec8cc70301d.jpg)

Change the scene to a cozy study room, and replace the cabinet with a new one.  
![](images/36235f517c4845607d7fc9563c832c9e4abff2d7dc6d36f291caeb5ee3c14f8c.jpg)

Add cluttered items on top of the table.  
![](images/44e30d2cac8a8f9e0b17b747d99edd49a757432b4d0cd35d2bf98f6c9a6a6115.jpg)

Add a tablecloth to the table.  
![](images/a9fc41627f1ae2e6d9098ee562fc46dfbc2a768a48f6e67528816c933e686e0a.jpg)

Change the scene to a cozy cafe, and replace cups in different styles.  
![](images/f9c6e774e391a9d7c2222aaf1d7cd8e9354032dd2365ed0fe35c464cac3a4859.jpg)  
Fig. 12. Real-world video rollouts under out-of-domain (OOD) conditions. GeniWorld maintains precise motion trajectories and realistic physical interactions even when exposed to unseen object instances, novel backgrounds, and complex environmental distractors.

![](images/6319baef9d0bafb3bf6765db314723b2826e0a38f5873691768f8fe33baa9218.jpg)  
Fig. 13. Qualitative results of GeniWorld on in-domain real-world tasks. The predicted video sequences demonstrate accurate spatial trajectories fine-grained object interaction, and strict visual temporal consistency aligned with ground-truth rollouts.

![](images/29b6260a8ab48aec59915120099e250d70570f5f5607ed6b674725396a249adb.jpg)  
Fig. 14. Comparison of real-world policy rollouts. We visualize rollouts of π<sub>0</sub> policies trained on real data alone and on real data augmented with GeniWorld-generated trajectories. We found that trained on both real and generated data, π<sub>0</sub> shows a better performance on OOD scenarios such a distractors, novel instances, spatial rearrangement and lighting shift.

![](images/c58faa8ecaaad61eff816f6699bb927d60b60556d2ab74da1a0f51122142a08c.jpg)  
Fig. 15. More visualization of real-world policy rollouts.

![](images/82ff80dc2d2e4c5633127962753c8269c972a072cec0abffb1922ef3422aa58d.jpg)  
Fig. 16. Qualitative visualizations of generated rollouts on RoboTwin. We show the predictions across Clean-to-Clean (in-domain) and Clean-to-Random (out-of-distribution) setups. GeniWorld maintains high visual fidelity and precise temporal consistency even under randomized environmental perturbations.

![](images/ec36d3f531a22310569cd700dda11a42cdf18ecec9cdb93d5c94e54903fdd6cb.jpg)  
Fig. 17. Qualitative comparison of generation rollouts across different action representations.