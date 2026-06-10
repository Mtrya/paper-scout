# OASIS: From Simulation Data Collection to Real-World Humanoid Loco-Manipulation

Zehao Yu1,2 Jiakun Zheng1,3 Weiji Xie1,4 Jiyuan Shi1 Chenyun Zhang1 Chenjia Bai1† Xuelong Li1

1Institute of Artificial Intelligence (TeleAI), China Telecom 2Fudan University 3East China University of Science and Technology 4Shanghai Jiao Tong University †Corresponding author

![](images/eca38a04515b237a1296bb669cfafcd79c2ab0d1f3302c585d5a91cbe1210222.jpg)  
Figure 1: OASIS collects whole-body demonstrations entirely in simulation and deploys the visuomotor policy zero-shot on the real Unitree G1 humanoid across diverse loco-manipulation tasks.

Abstract: Recent progress in robot manipulation has been largely driven by learning from large-scale demonstrations. For humanoid robot loco-manipulation tasks, however, existing data sources force an unsatisfying tradeoff between trajectory quality and scalability. Real-world teleoperation provides the highest-quality trajectories but requires dedicated physical space and time-consuming scene resets. Simulation offers an alternative way out of this dilemma: it can produce clean, embodiment-aligned data at scale without any physical hardware. In this paper, we propose OASIS, a simulation-data-driven framework for humanoid loco-manipulation. OASIS automatically reconstructs realistic object assets from real-world images using a 3D generative model. Based on these assets, trajectories are first collected through teleoperation in simulation, and then augmented under diverse domain randomizations in a post-processing stage. With the resulting simulation data, we further design a hierarchical visuomotor policy for humanoid loco-manipulation. Extensive experiments on the real humanoid robot show that, under zero-shot deployment, the policy trained on our simulation data achieves higher success rates on most tasks than that trained on real-robot teleoperation data, owing largely to the broad lighting and environmental variations covered by our simulation rendering, which real-robot data fails to capture.

Keywords: Humanoid Loco-Manipulation, Simulation Data Collection

## 1 Introduction

Humanoid robots are expected to take on a wide range of tasks in everyday human environments [1], where locomotion and manipulation must be tightly coordinated to act effectively [2, 3]. However, robust and generalizable loco-manipulation ultimately depends on large-scale, high-quality demonstration data, which current humanoid platforms still largely lack [4, 5, 6].

To obtain the demonstration data required for humanoid manipulation, prior work has explored a range of sources, including human videos [7, 8], egocentric recordings [9, 10, 11, 12], and real-robot teleoperation [13, 14, 15, 16, 17]. Among these, real-robot teleoperation has been the most widely used [18, 19, 20, 21], as the operator directly drives the robot to complete the task, yielding trajectories that are precisely aligned with the robot’s embodiment and inherently accompanied by action supervision. However, collecting teleoperation data on real robots is time-consuming, resource-intensive, and hard to scale. First, large-scale collection requires a substantial number of expensive robots and supporting equipment, along with correspondingly large physical spaces, resulting in high financial and spatial overhead. Second, physical interaction itself makes the collection process fragile and inefficient. In long-

![](images/bd782159e912179a11321db0e2b92420f38d64ae7bc61af3fd885c6fe5f907cb.jpg)  
Figure 2: Failure recovery: real robot vs. simulation. Recovering from a failure on a real humanoid requires a tedious manual sequence — (a) falling down, (b) lifted by the operator, (c) rearranging props, and (d) resuming. In contrast, simulation supports one-click restart, restoring the scene instantaneously.

horizon tasks, any failure or need for repositioning requires manual reconfiguration of both the robot and the environment, significantly slowing collection, whereas simulation [22, 23, 24] allows instantaneous reset, as shown in Fig. 2. Moreover, operational errors during physical interaction often damage robot hardware or objects in the scene.

To address these limitations, we introduce OASIS, a framework that learns humanoid locomanipulation policies from data collected entirely in simulation. Given reference images of real objects, it synthesizes 3D meshes with a 3D generative model and estimates their physical dimensions and material properties with a vision-language model (VLM). The resulting assets closely match their real counterparts, enabling diverse and physically plausible simulation scenes to be built at scale. Built on these assets, OASIS adopts a two-stage decoupled design. In the first stage, the operator teleoperates a humanoid robot in simulation in real time from a first-person view through VR devices such as PICO 4U [25], a portable virtual reality system that captures the operator’s full-body pose through a headset, a pair of handheld controllers, and two ankle-mounted trackers, obviating the need for dedicated motion-capture studios. To preserve real-time responsiveness, the VR headset receives a lightweight rendering for operator feedback, while only the state sequences of the robot and the objects in the scene are recorded. In the second stage, the recorded states are replayed offline and rendered at high fidelity for training. Textures, lighting, and camera extrinsics are randomized in the process, turning each teleoperated trajectory into a diverse set of visually distinct training samples. This decoupling separates the cost of teleoperation from the size of the resulting dataset, so a small amount of operator time produces a large and visually diverse training set.

We build a hierarchical visuomotor policy based on our system. The high-level planner is a Flow Matching policy that predicts reference motion commands from visual observations, and the lowlevel controller converts these commands into target joint angles. We validate our system on a real humanoid robot. The high-level planner is trained purely on simulation data and successfully accomplishes several tasks zero-shot. It also demonstrates adaptability to real-world perturbations such as camera motion blur and background clutter.

The main contributions of this paper are as follows. First, we present OASIS, a humanoid locomanipulation framework that implements a novel pipeline in which control policies are learned entirely from teleoperation data collected in simulation. Second, we design a scalable data collection system that enables efficient demonstration collection in simulation. Third, through real-robot experiments, we demonstrate that data collected with OASIS enables effective zero-shot transfer to real robots on multiple tasks.

## 2 Related Work

## 2.1 Humanoid Loco-Manipulation

Humanoid loco-manipulation requires coordinated locomotion, whole-body manipulation, and tasklevel reasoning, and remains challenging from both the execution and the data-supervision sides. To shoulder the engineering burden, recent works standardize humanoid policy learning into reproducible workflows [26]. In parallel, a growing body of work builds generalist humanoid visionlanguage-action (VLA) policies trained on heterogeneous mixtures of human videos, synthetic data and teleoperated trajectories, typically pairing a vision-language backbone with a fast action expert and a dedicated whole-body tracking controller [15, 16, 17, 27]. More recent efforts further refine this paradigm by introducing humanoid-aligned state representations for cross-embodiment learning [14] or unified latent VLAs for manipulation-aware locomotion [28]. Alongside this, to bypass the bottleneck of robot teleoperation, another line of work collects robot-free demonstrations through portable rigs, wearable exoskeletons, or egocentric capture devices, and bridges the human-humanoid embodiment gap via view and action alignment [29, 30, 13]; even teleoperationbased systems have shifted toward portable, mocap-free setups to make whole-body data collection scalable [19]. Despite this progress, both robot-centric teleoperation and robot-free human capture remain time-consuming and physically expensive. In contrast, we treat high-fidelity simulation as a scalable source of whole-body loco-manipulation data: by automatically constructing physically plausible scenes from generative 3D assets and decoupling trajectory collection from photorealistic rendering, each teleoperated trajectory is expanded into a large number of visually diverse training samples, on which we train a Flow Matching policy.

## 2.2 Simulation Data Collection For Robot Learning

The high cost of real-robot data collection has motivated growing efforts to use simulation as a scalable training source [31, 32]. One line of work automates the construction of simulated tasks and assets via foundation models [33]. Another scales data through trajectory augmentation, replaying a few human demonstrations into many new initial conditions. MimicGen [34] establishes this paradigm for tabletop manipulation, and DexMimicGen [35] extends it to bimanual dexterous setups. However, both remain restricted to fixed-base or upper-body settings. Recent humanoidspecific efforts further explore simulation data for whole-body policies, yet each carries its own bottleneck. GR00T N1 [15] augments its corpus with synthetic data, but the simulated portion is dominated by simple bimanual tabletop tasks. VIRAL [36] collects data via reinforcement learning in simulation, but the RL acquisition process itself is expensive, requiring carefully shaped rewards and long training per behavior. In contrast, we automatically construct physically plausible scenes from generative 3D assets, collect whole-body trajectories via simulation teleoperation, and successfully train policies that transfer to real-robot.

## 3 Method

## 3.1 Overview

OASIS is a simulation-data-driven framework for humanoid loco-manipulation, consisting of automated simulation scene construction, a two-stage teleoperation-and-rendering data collection pipeline, and a hierarchical whole-body policy trained on the resulting data for zero-shot real-robot deployment. In this section, we detail (i) how OASIS collects scalable simulation data, and (ii) how we learn a hierarchical whole-body policy that transfers to the real robot.

![](images/954750b226fc7532c4c6feb989b1b87d0d4bce52dc55dc07bf498e75516dd9a9.jpg)  
Figure 3: Overview of OASIS. Our framework consists of four stages. First, we reconstruct physicsready simulation assets from single-view photos of real objects. Second, demonstration trajectories are collected in simulation via VR teleoperation. Third, these trajectories are replayed with texture, lighting, and camera-extrinsics randomization for visual augmentation. Finally, a hierarchical policy is trained on the augmented data, where a high-level Flow Matching predicts reference motion command from multimodal observations, and a low-level controller tracks them as joint angles in a closed loop.

## 3.2 Data Collection

## 3.2.1 Simulation Scene Construction

Moving data collection into simulation removes the dependence on physical hardware, but it introduces a new bottleneck: every task requires a corresponding simulation scene with realistic, physically plausible objects, and constructing such scenes by hand is itself labor-intensive. To eliminate this bottleneck, we build an automated asset generation pipeline.

Real-to-Sim Asset Generation. Given reference images of real-world objects, we first leverage Hunyuan3D [37], an advanced large-scale 3D synthesis system for generating high-resolution textured 3D assets. The outputs of the generative model consist solely of meshes and texture maps, lacking both physical scale and material properties. To recover these attributes, we further leverage the strong prior knowledge of Qwen3-VL [38], a vision-language model with strong visual reasoning capabilities over object geometry, materials, and physical properties. Given the reference image and a category description of the object, it is prompted with a structured template to produce reasonably accurate estimates of the object’s physical dimensions and material category.

Physics Parameter Assignment. The predicted dimensions rescale the normalized mesh to its physical size. Meanwhile, the material category serves as an index into a predefined table to retrieve the effective density, friction, and restitution coefficients, as detailed in Appendix A. From these, mass and inertia are computed under a uniform-density assumption, while friction and restitution are attached to the collision body. To account for estimation errors, all physical properties are randomized around their predicted values during data collection.

## 3.2.2 Teleop Trajectory Collection

With the simulation scene built from the generated assets, we collect humanoid manipulation trajectories through VR-based teleoperation. Human operators control the simulated humanoid via VR devices such as PICO 4U, while the robot’s head-camera stream is transmitted to the headset in real time as a first-person view.

The operator’s motions are retargeted to the humanoid by GMR [39] to produce reference wholebody motions, which are then input to Teleopit [40], an open-source reinforcement learning-based whole-body controller, to drive the simulated humanoid to execute the corresponding actions. To maintain low-latency teleoperation, this stage employs the Real-Time rendering mode of Isaac-Sim [23], which substantially reduces the rendering overhead while preserving sufficient visual fidelity, allowing the simulator to run at a high frame rate.

During data collection, two categories of data are recorded. The first is the whole-body kinematic state of the robot, together with the kinematic states of all interactive rigid bodies in the scene. These states are used to replay the trajectory in the second stage. The second is the reference motions retargeted by GMR, which are used to train the high-level policy.

## 3.2.3 Scalable Trajectory Rendering

We then collect diverse image observations paired with these recorded trajectories to construct the training dataset. Each trajectory is replayed offline and rendered under randomized visual conditions, expanding a single demonstration into a large number of visually diverse samples. Free from the real-time constraint of teleoperation, the offline setting enables Path-Tracing rendering mode in IsaacSim, which produces higher-fidelity images. Specifically, we randomize background textures, the intensity and color temperature of environmental lighting, and the extrinsic parameters of the cameras.

## 3.3 Whole-Body Policy Learning

## 3.3.1 Model Architecture

Following TextOp [41], we represent the per-frame reference motion command $m _ { t } \in \mathbb { R } ^ { 6 7 }$ as:

$$
m _ { t } = \Big [ \phi ( r _ { t } ) , ~ \Delta \psi _ { t } , ~ \Delta p _ { t } ^ { \mathrm { l o c a l } } , ~ h _ { t } , ~ q _ { t } , ~ \Delta q _ { t } \Big ] ,\tag{1}
$$

where $\phi ( r _ { t } ) = [ \mathrm { s i n } ( { \mathrm { r o l l } _ { t } } ) , \mathrm { c o s } ( { \mathrm { r o l l } _ { t } } ) - 1 , \mathrm { s i n } ( \mathrm { p i t c h } _ { t } ) , \mathrm { c o s } ( \mathrm { p i t c h } _ { t } ) - 1 ]$ represents the trigonometric encoding of roll and pitch, $\Delta \psi _ { t } = \mathrm { y a w } _ { t + 1 } - \mathrm { y a w } _ { i }$ denotes the per-frame yaw difference, $\Delta p _ { t } ^ { \mathrm { l o c a l } } =$ $R _ { z } ( \mathbf { y a w } _ { t } ) ^ { \top } ( p _ { t + 1 } - p _ { t } )$ is the root translation in the local frame, $h _ { t }$ represents root height, $q _ { t } \in \mathbb { R } ^ { 2 9 }$ represents joint positions, and $\Delta q _ { t } = q _ { t + 1 } - q _ { t }$ represents joint increments.

As illustrated in Fig. 3, our high-level planner is a Transformer-based, action-chunking policy that generates future motion sequences with Flow Matching [42], and is coupled with a low-level controller in a hierarchical design. The denoiser takes three inputs: the text instruction, encoded by a frozen CLIP [43] text encoder; three-view images, encoded by a frozen DINOv2 [44] visual encoder; and robot proprioception over the most recent $H = 2$ frames, encoded by an MLP. These features are concatenated into a condition token sequence $c ,$ on which the denoiser predicts the whole-body reference motion $\mathbf { m } _ { t : t + F } \in \mathbb { R } ^ { F \times 6 7 }$ over the next $F = 3 2$ frames.

We train the denoiser $v _ { \theta }$ with the Flow Matching objective, which regresses the constant-velocity field along the linear path between a Gaussian prior $\mathbf { a } _ { 1 } \sim \mathcal { N } ( 0 , I )$ and the target action chunk ${ \bf a } _ { 0 }$

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { F M } } ( \theta ) = \mathbb { E } _ { \tau , \mathbf { a } _ { 0 } , \mathbf { a } _ { 1 } } \Big [ \big | \mathbf { \nabla } v _ { \theta } ( \mathbf { a } _ { \tau } , \tau , c ) - ( \mathbf { a } _ { 1 } - \mathbf { a } _ { 0 } ) \big | \big | _ { 2 } ^ { 2 } \Big ] , } \end{array}\tag{2}
$$

where ${ \bf a } _ { \tau } = \left( 1 - \tau \right) { \bf a } _ { 0 } + \tau { \bf a } _ { 1 }$ and $\tau \sim \mathcal { U } ( 0 , 1 )$ . At inference, we generate actions by integrating the learned velocity field with an Euler solver using 10 denoising steps. Consistent with teleoperation, the low-level controller Teleopit converts the reference motion into 29-DoF body joint angles; together with the 14-DoF hand joints, the system outputs 43-DoF whole-body joint angles.

## 3.3.2 Training Recipe

Reference Motion Commands as Proprioception. For the proprioception input, we use the reference motion commands rather than the robot state. The robot state reflects the trajectory already executed by the low-level controller, which inevitably carries tracking errors and noise; conditioning the planner on such signals lets these errors accumulate and feed back into planning. The reference commands, in contrast, provide a consistent and noise-free history, keeping the planner’s input distribution identical between simulation and deployment.

![](images/2b4c0b86aeed1effa767c7d900b2d9e983c462fd61cb84e1c15e781f12a29f04.jpg)  
Figure 4: Real-robot experiments on loco-manipulation tasks across different difficulty levels.

Curriculum-based Rollout Training. Since the planner predicts F frames in a single pass, training only on ground-truth history leaves it unable to cope with the accumulated errors of its own predictions at inference, causing instability over long horizons. We therefore adopt a curriculum-based rollout mechanism: at each training step we sample $P = 4$ consecutive segments from the same sequence, where the first segment uses ground-truth history and each subsequent segment reuses its predecessor’s last H predicted frames with probability $p _ { \mathrm { r o l l o u t } }$ . This probability stays at 0 for the first 20% of training, letting the model first fit the conditional distribution on clean history, then increases linearly to 0.8. By exposing the model to its own prediction errors during training, this mechanism maintains stability under long-horizon autoregressive rollout at deployment.

## 3.4 Deployment

We deploy our system on a 29-DoF Unitree G1 humanoid, equipped with 7-DoF three-fingered dexterous hands. In addition to a Realsense D435i camera on the head, each wrist is fitted with an additional Realsense D405 camera. The high-level planner operates at 25 Hz on an NVIDIA RTX 4090 GPU, while the low-level controller executes the predicted 32-step action chunk at 50 Hz.

## 4 Experiments

In this section, we conduct experiments on the Unitree G1 humanoid to answer the following questions: Q1: Can OASIS achieve higher data collection efficiency than real-robot teleoperation? Q2: How does each component of the OASIS data augmentation stage affect sim-to-real transfer? Q3: How effective is simulation data from OASIS compared with real-robot data for humanoid locomanipulation?

## 4.1 Data Collection Efficiency

To address Q1, we measure data collection efficiency in both simulation and the real world. Specifically, we use the same low-level controller and the same operator, and collect the same number of successful trajectories on the same tasks. As shown in Table 1, collecting data with OASIS is significantly faster than real-robot collection across all tasks, and the speedup grows with task difficulty.

Since both settings drive the same humanoid through the same interface, the time spent executing each task is comparable; the efficiency gap arises almost entirely from the overhead beyond each trajectory, which is unavoidable in the real world but nearly zero in simulation. In real-robot collection, after each attempt the operator must enter the workspace and reset every object to its initial configuration before the next trajectory can begin, and this overhead grows with the number of objects and the task length. In simulation, resets are instantaneous and fully automatic.

Moreover, physical interaction is fragile, and this fragility extends even to the manipulated objects. In our screen-wiping task, the robot makes frequent contact with a fragile monitor, and during realrobot collection any deviation in force or timing risks damaging it—in fact, we damaged a monitor due to excessive contact force, forcing the operator to proceed slowly and cautiously. In simulation, none of this is a concern: a damaged object can simply be reset, so the operator does not need to hold back.

## 4.2 Ablations on Data Augmentation

OASIS augments each trajectory by applying vision randomization and rendering it multiple times, expanding a single demonstration into a large set of visually diverse training samples. To address Q2, we examine this component from two angles: the contribution of each randomization factor, and the number of renderings needed per trajectory.

For the randomization factors, we compare disabling all randomization (w/o All), removing one factor at a time (texture, lighting, or camera extrinsics), and the full configuration (Ours). As shown in Table 2, disabling all randomization causes the policy to almost completely fail to transfer, confirming that randomization is indispensable. Among the individual components, lighting contributes the most, since illumination differences are among the largest sim-to-real visual gaps. Importantly, the full combination outperforms every ablated variant, indicating that these randomizations target complementary aspects of the sim-to-real gap and are most effective when applied jointly.

The success rate rises steadily with more renderings and approaches saturation around 15–20, beyond which the gains taper off. We therefore render each trajectory into 20 environments to balance performance and overhead.

## 4.3 Effectiveness Of Simulation Data

To address Q3, we evaluate policies on the real Unitree G1 across the loco-manipulation tasks shown in Fig. 4, which span tabletop manipulation, whole-body lifting, and kneeling under-table wiping. For each task, we compare three sources of training data under the same total number of trajectories: simulation data from OASIS, real-robot only, and an equal mixture of both.

As shown in Fig. 5, the policy trained on simulation data alone achieves a real-robot success rate comparable to, and on some tasks higher than, the one trained on real-robot data. Since both use the same number of trajectories, this shows that the simulation data collected by OASIS rivals real-robot data in supervision quality and can serve as an effective substitute, without the high time and hardware costs of real-robot collection. We attribute the cases where simulation even surpasses real data to visual diversity: real-robot data is collected in a relatively fixed environment, so the policy struggles once deployment conditions deviate from collection time, whereas the large-scale randomized re-rendering in simulation covers far richer visual conditions and yields stronger robustness.

<table><tr><td>Task</td><td>OASIS (min)</td><td>Real (min)</td><td>Speedup</td></tr><tr><td>Place Cup in Box</td><td>15.2</td><td>17.5</td><td>1.15×</td></tr><tr><td>Wipe Monitor</td><td>19.1</td><td>26.8</td><td>1.40×</td></tr><tr><td>Lift Basket and Place Cup</td><td>25.2</td><td>40.2</td><td>1.60×</td></tr><tr><td>Kneel and Wipe Under Table</td><td>28.4</td><td>44.8</td><td>1.84×</td></tr></table>

Table 1: Time taken to collect 50 successful trajectories per task with OASIS versus real-robot teleoperation. OASIS is faster on every task, and the gap is larger on harder ones.

<table><tr><td rowspan="2">Task</td><td colspan="4">Domain Randomization</td><td colspan="4">Rendered Envs. per Traj.</td></tr><tr><td>w/o All</td><td>w/o Tex.</td><td>w/o Light</td><td>w/o Cam.</td><td>5</td><td>10</td><td>15</td><td>Ours</td></tr><tr><td>Place Cup in Box</td><td>0/10</td><td>5/10</td><td>3/10</td><td>7/10</td><td>4/10</td><td>5/10</td><td>8/10</td><td>8/10</td></tr><tr><td>Lift Basket and Place Cup</td><td>0/10</td><td>3/10</td><td>1/10</td><td>5/10</td><td>2/10</td><td>4/10</td><td>5/10</td><td>7/10</td></tr><tr><td>Wipe Monitor</td><td>1/10</td><td>5/10</td><td>4/10</td><td>7/10</td><td>5/10</td><td>7/10</td><td>7/10</td><td>8/10</td></tr><tr><td>Kneel and Wipe Under Table</td><td>1/10</td><td>4/10</td><td>4/10</td><td>6/10</td><td>4/10</td><td>7/10</td><td>10/10</td><td>10/10</td></tr><tr><td>Average Success Rate</td><td>0.05</td><td>0.43</td><td>0.30</td><td>0.63</td><td>0.38</td><td>0.58</td><td>0.75</td><td>0.83</td></tr></table>

Table 2: Ablations on the data-augmentation stage. All numbers are real-robot zero-shot success rates over 10 trials. The Ours column denotes our final configuration, which applies all randomization and renders each trajectory under 20 randomized environments.

Moreover, mixing the two sources under the same trajectory budget outperforms either alone. As the total data is unchanged, this gain stems not from more data but from their complementarity: simulation contributes largescale, visually diverse samples for generalization, while real-robot data supplies the real interaction and perception characteristics that simulation cannot fully capture. Overall, simulation data alone supports high-performance real-robot deployment and further improves performance when combined with real data, highlighting the value of OASIS as a scalable data source.

![](images/5609449f64c796a82e31511995366b583bcb1c6ed03a7f47db6d3cd6fd89b87f.jpg)  
Figure 5: Real-world zero-shot success rates of policies trained on simulation data from OASIS, real-robot data, and their equal mixture, using the same total of 50 trajectories per setting.

## 5 Conclusion

OASIS grounds simulated scenes in 3D-generated assets recovered from real-world images, and separates VR-based teleoperation from offline photorealistic rendering, so that each demonstration is expanded into a large set of visually diverse training samples without additional operator effort. On the Unitree G1 humanoid, data collection with OASIS runs up to 1.84× faster than real-robot teleoperation. Policies trained entirely on OASIS-generated data transfer zero-shot to the real robot, matching or surpassing those trained on real-robot data under the same trajectory budget. These results suggest that high-fidelity simulation, when paired with realistic asset generation and large-scale visual randomization, can serve as a practical and scalable alternative to real-robot teleoperation for humanoid loco-manipulation.

## 6 Limitations

While OASIS transfers from simulation to the real world zero-shot on loco-manipulation tasks, several limitations remain.

First, our augmentation only randomizes visual appearance and leaves trajectories unchanged, since perturbing whole-body states easily breaks balance. Motion diversity is thus bounded by what the operator demonstrates, and physics-aware trajectory augmentation is a natural next step.

Second, our simulation fidelity depends on automatically generated assets, whose geometry and physical parameters may be inaccurate for visually complex objects, widening the sim-to-real gap

on contact-rich tasks. Better asset reconstruction and physical-parameter calibration could help close this gap.

## Acknowledgments

This work is supported by the National Key Research and Development Program of China (Grant No.2024YFE0210900), the National Natural Science Foundation of China (Grant No.62306242), the Young Elite Scientists Sponsorship Program by CAST (Grant No. 2024QNRC001), and the Yangfan Project of the Shanghai (Grant No.23YF11462200).

## References

[1] Z. Gu, J. Li, W. Shen, W. Yu, Z. Xie, S. McCrory, X. Cheng, A. Shamsah, R. Griffin, C. K. Liu, et al. Humanoid locomotion and manipulation: Current progress and challenges in control, planning, and learning. IEEE/ASME Transactions on Mechatronics, 31(2):2300–2330, 2026.

[2] Z. Fu, Q. Zhao, Q. Wu, G. Wetzstein, and C. Finn. Humanplus: Humanoid shadowing and imitation from humans. arXiv preprint arXiv:2406.10454, 2024.

[3] T. He, Z. Luo, X. He, W. Xiao, C. Zhang, W. Zhang, K. Kitani, C. Liu, and G. Shi. Omnih2o: Universal and dexterous human-to-humanoid whole-body teleoperation and learning. arXiv preprint arXiv:2406.08858, 2024.

[4] A. O’Neill, A. Rehman, A. Maddukuri, A. Gupta, A. Padalkar, A. Lee, A. Pooley, A. Gupta, A. Mandlekar, A. Jain, et al. Open x-embodiment: Robotic learning datasets and rt-x models: Open x-embodiment collaboration 0. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 6892–6903. IEEE, 2024.

[5] A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, J. Dabis, C. Finn, K. Gopalakrishnan, K. Hausman, A. Herzog, J. Hsu, et al. Rt-1: Robotics transformer for real-world control at scale. arXiv preprint arXiv:2212.06817, 2022.

[6] B. Zitkovich, T. Yu, S. Xu, P. Xu, T. Xiao, F. Xia, J. Wu, P. Wohlhart, S. Welker, A. Wahid, et al. Rt-2: Vision-language-action models transfer web knowledge to robotic control. In Conference on Robot Learning, pages 2165–2183. PMLR, 2023.

[7] S. Nair, A. Rajeswaran, V. Kumar, C. Finn, and A. Gupta. R3m: A universal visual representation for robot manipulation. arXiv preprint arXiv:2203.12601, 2022.

[8] C. Wang, L. Fan, J. Sun, R. Zhang, L. Fei-Fei, D. Xu, Y. Zhu, and A. Anandkumar. Mimicplay: Long-horizon imitation learning by watching human play. arXiv preprint arXiv:2302.12422, 2023.

[9] K. Grauman, A. Westbury, E. Byrne, Z. Chavis, A. Furnari, R. Girdhar, J. Hamburger, H. Jiang, M. Liu, X. Liu, et al. Ego4d: Around the world in 3,000 hours of egocentric video. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18995–19012, 2022.

[10] S. Kareer, D. Patel, R. Punamiya, P. Mathur, S. Cheng, C. Wang, J. Hoffman, and D. Xu. Egomimic: Scaling imitation learning via egocentric video. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pages 13226–13233. IEEE, 2025.

[11] R. Zheng, D. Niu, Y. Xie, J. Wang, M. Xu, Y. Jiang, F. Castaneda, F. Hu, Y. L. Tan, L. Fu, et al. ˜ Egoscale: Scaling dexterous manipulation with diverse egocentric human data. arXiv preprint arXiv:2602.16710, 2026.

[12] R. Hoque, P. Huang, D. J. Yoon, M. Sivapurapu, and J. Zhang. Egodex: Learning dexterous manipulation from large-scale egocentric video. arXiv preprint arXiv:2505.11709, 2025.

[13] M. Shi, S. Peng, J. Chen, H. Jiang, Y. Li, D. Huang, P. Luo, H. Li, and L. Chen. Egohumanoid: Unlocking in-the-wild loco-manipulation with robot-free egocentric demonstration. arXiv preprint arXiv:2602.10106, 2026.

[14] S. Bai, M. Li, X. Lv, J. Wang, X. Wang, F. Liao, C. Hou, L. Gu, W. Zhou, K. Wu, et al. Hex: Humanoid-aligned experts for cross-embodiment whole-body manipulation. arXiv preprint arXiv:2604.07993, 2026.

[15] J. Bjorck, F. Castaneda, N. Cherniadev, X. Da, R. Ding, L. Fan, Y. Fang, D. Fox, F. Hu, ˜ S. Huang, et al. Gr00t n1: An open foundation model for generalist humanoid robots. arXiv preprint arXiv:2503.14734, 2025.

[16] P. Ding, J. Ma, X. Tong, B. Zou, X. Luo, Y. Fan, T. Wang, H. Lu, P. Mo, J. Liu, et al. Humanoid-vla: Towards universal humanoid control with visual integration. arXiv preprint arXiv:2502.14795, 2025.

[17] S. Wei, H. Jing, B. Li, Z. Zhao, J. Mao, Z. Ni, S. He, J. Liu, X. Liu, K. Kang, et al. Ψ0: An open foundation model towards universal humanoid loco-manipulation. arXiv preprint arXiv:2603.12263, 2026.

[18] Y. Ze, Z. Chen, J. P. Araujo, Z. ang Cao, X. B. Peng, J. Wu, and C. K. Liu. Twist: Teleoperated ´ whole-body imitation system. arXiv preprint arXiv:2505.02833, 2025.

[19] Y. Ze, S. Zhao, W. Wang, A. Kanazawa, R. Duan, P. Abbeel, G. Shi, J. Wu, and C. K. Liu. Twist2: Scalable, portable, and holistic humanoid data collection system. arXiv preprint arXiv:2511.02832, 2025.

[20] Z. Luo, Y. Yuan, T. Wang, C. Li, S. Chen, F. Castaneda, Z.-A. Cao, J. Li, D. Minor, Q. Ben, ˜ X. Da, R. Ding, C. Hogg, L. Song, E. Lim, E. Jeong, T. He, H. Xue, W. Xiao, Z. Wang, S. Yuen, J. Kautz, Y. Chang, U. Iqbal, L. Fan, and Y. Zhu. Sonic: Supersizing motion tracking for natural humanoid whole-body control. arXiv preprint arXiv:2511.07820, 2025.

[21] Y. Li, L. Ma, Y. Lin, Y. Du, M. Liu, K. Hu, J. Cui, Y. Zhu, W. Liang, B. Jia, et al. Omniclone: Engineering a robust, all-rounder whole-body humanoid teleoperation system. arXiv preprint arXiv:2603.14327, 2026.

[22] V. Makoviychuk, L. Wawrzyniak, Y. Guo, M. Lu, K. Storey, M. Macklin, D. Hoeller, N. Rudin, A. Allshire, A. Handa, et al. Isaac gym: High performance gpu-based physics simulation for robot learning. arXiv preprint arXiv:2108.10470, 2021.

[23] NVIDIA. Isaac Sim. URL https://github.com/isaac-sim/IsaacSim.

[24] E. Todorov, T. Erez, and Y. Tassa. Mujoco: A physics engine for model-based control. In 2012 IEEE/RSJ international conference on intelligent robots and systems, pages 5026–5033. IEEE, 2012.

[25] PICO Immersive Pte. Ltd. PICO 4 Ultra: An All-New Mixed Reality Experience. https: //www.picoxr.com/global/products/pico4-ultra, 2023.

[26] H. Zhao, R. Cathomen, L. Gulich, W. Liu, E. A. Ongan, M. Lin, S. Jain, S. Pouya, and Y. Chang. Agile: A comprehensive workflow for humanoid loco-manipulation learning. arXiv preprint arXiv:2603.20147, 2026.

[27] Y. Fu, F. Xie, C. Xu, J. Xiong, H. Yuan, and Z. Lu. Demohlm: From one demonstration to generalizable humanoid loco-manipulation. IEEE Robotics and Automation Letters, 2026.

[28] H. Jiang, J. Chen, Q. Bu, L. Chen, M. Shi, Y. Zhang, D. Li, C. Suo, C. Wang, Z. Peng, et al. Wholebodyvla: Towards unified latent vla for whole-body loco-manipulation control. arXiv preprint arXiv:2512.11047, 2025.

[29] R. Nai, B. Zheng, J. Zhao, H. Zhu, S. Dai, Z. Chen, Y. Hu, Y. Hu, T. Zhang, C. Wen, et al. Humanoid manipulation interface: Humanoid whole-body manipulation from robot-free demonstrations. arXiv preprint arXiv:2602.06643, 2026.

[30] R. Zhong, Y. Sun, J. Wen, J. Li, C. Cheng, W. Dai, Z. Zeng, H. Lu, Y. Zhu, and Y. Xu. Humanoidexo: Scalable whole-body humanoid manipulation via wearable exoskeleton. arXiv preprint arXiv:2510.03022, 2025.

[31] S. Nasiriany, A. Maddukuri, L. Zhang, A. Parikh, A. Lo, A. Joshi, A. Mandlekar, and Y. Zhu. Robocasa: Large-scale simulation of everyday tasks for generalist robots. arXiv preprint arXiv:2406.02523, 2024.

[32] L. Wang, Y. Ling, Z. Yuan, M. Shridhar, C. Bao, Y. Qin, B. Wang, H. Xu, and X. Wang. Gensim: Generating robotic simulation tasks via large language models. In International Conference on Learning Representations, volume 2024, pages 4890–4924, 2024.

[33] Y. Wang, Z. Xian, F. Chen, T.-H. Wang, Y. Wang, K. Fragkiadaki, Z. Erickson, D. Held, and C. Gan. Robogen: towards unleashing infinite data for automated robot learning via generative simulation. In Proceedings of the 41st International Conference on Machine Learning, pages 51936–51983, 2024.

[34] A. Mandlekar, S. Nasiriany, B. Wen, I. Akinola, Y. Narang, L. Fan, Y. Zhu, and D. Fox. Mimicgen: A data generation system for scalable robot learning using human demonstrations. arXiv preprint arXiv:2310.17596, 2023.

[35] Z. Jiang, Y. Xie, K. Lin, Z. Xu, W. Wan, A. Mandlekar, L. J. Fan, and Y. Zhu. Dexmimicgen: Automated data generation for bimanual dexterous manipulation via imitation learning. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pages 16923– 16930. IEEE, 2025.

[36] T. He, Z. Wang, H. Xue, Q. Ben, Z. Luo, W. Xiao, Y. Yuan, X. Da, F. Castaneda, S. Sas- ˜ try, et al. Viral: Visual sim-to-real at scale for humanoid loco-manipulation. arXiv preprint arXiv:2511.15200, 2025.

[37] Z. Zhao, Z. Lai, Q. Lin, Y. Zhao, H. Liu, S. Yang, Y. Feng, M. Yang, S. Zhang, X. Yang, et al. Hunyuan3d 2.0: Scaling diffusion models for high resolution textured 3d assets generation. arXiv preprint arXiv:2501.12202, 2025.

[38] S. Bai, Y. Cai, R. Chen, K. Chen, X. Chen, Z. Cheng, L. Deng, W. Ding, C. Gao, C. Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025.

[39] J. P. Araujo, Y. Ze, P. Xu, J. Wu, and C. K. Liu. Retargeting matters: General motion retargeting for humanoid motion tracking. arXiv preprint arXiv:2510.02252, 2025.

[40] BotRunner64. Teleopit: A lightweight and scalable whole-body teleoperation framework for humanoid robots, 2025. URL https://github.com/BotRunner64/Teleopit. Accessed: 2026-04-15.

[41] W. Xie, J. Zheng, J. Han, J. Shi, W. Zhang, C. Bai, and X. Li. Textop: Real-time interactive text-driven humanoid robot motion generation and control. arXiv preprint arXiv:2602.07439, 2026.

[42] Y. Lipman, R. T. Chen, H. Ben-Hamu, M. Nickel, and M. Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.

[43] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021.

[44] M. Oquab, T. Darcet, T. Moutakanni, H. V. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby, R. Howes, P.-Y. Huang, H. Xu, V. Sharma, S.-W. Li, W. Galuba, M. Rabbat, M. Assran, N. Ballas, G. Synnaeve, I. Misra, H. Jegou, J. Mairal, P. Labatut, A. Joulin, and P. Bojanowski. Dinov2: Learning robust visual features without supervision, 2023.

## A Material Density

To enable contact-rich manipulation in simulation, each generated asset is assigned a mass based on its mesh volume and a category-level material density. Table 3 lists the density values used in our experiments.

<table><tr><td>Material</td><td>Density (kg/m3)</td><td>Example Object</td></tr><tr><td>Polypropylene (PP)</td><td>910</td><td>Box</td></tr><tr><td>Polyurethane (foam)</td><td>50</td><td>Sponge</td></tr><tr><td>ABS</td><td>1050</td><td>Monitor</td></tr><tr><td>Wicker</td><td>200</td><td>Basket</td></tr><tr><td>Wood</td><td>700</td><td>Cup</td></tr></table>

Table 3: Density values used for assigning physical mass to generated assets in simulation.

## B Domain Randomization

We apply domain randomization during offline rendering. Table 4 lists the randomized parameters.

<table><tr><td>Parameter</td><td>Distribution</td></tr><tr><td>Background Materials Wall diffuse texture Floor diffuse texture</td><td>U(Concrete, Wood, Terrazzo, Metal) U(Concrete, Wood, Terazzo)</td></tr><tr><td>Table diffuse texture Roughness</td><td>U(Wood) u(0.1, 0.65)</td></tr><tr><td>Metallic constant Texture rotation deg]</td><td>u(0.25, 1.0) U(0, 45)</td></tr><tr><td>Texture translation</td><td></td></tr><tr><td>UVW projection</td><td>u(0.1, 1.0)</td></tr><tr><td></td><td>B(0.9)</td></tr><tr><td>Lighting</td><td></td></tr><tr><td>Dome light intensity</td><td></td></tr><tr><td>Dome light color temperature</td><td>U(1000, 3000)</td></tr><tr><td></td><td>U(4500, 6500)</td></tr><tr><td>Dome light color (RGB)</td><td> $\mathcal { U } ( 0 . 8 5 , 1 . 0 ) \times \mathcal { U } ( 0 . 8 5 , 1 . 0 ) \times \mathcal { U } ( 0 . 8 5 , 1 . 0 )$ </td></tr><tr><td>Indoor light intensity</td><td>U(20000,200000)</td></tr><tr><td>Indoor light color temperature</td><td>U(4500,6500)</td></tr><tr><td>Indoor light color (RGB)</td><td></td></tr><tr><td>Camera Extrinsics</td><td> $\mathcal { U } ( 0 . 8 5 , 1 . 0 ) \times \mathcal { U } ( 0 . 8 5 , 1 . 0 ) \times \mathcal { U } ( 0 . 8 5 , 1 . 0 )$ </td></tr><tr><td></td><td></td></tr><tr><td>Position offset  $( x , y , z )$  [m]</td><td> $\mathcal { U } ( - 0 . 0 1 , 0 . 0 1 ) \times \mathcal { U } ( - 0 . 0 1 , 0 . 0 1 ) \times \mathcal { U } ( - 0 . 0 1 , 0 . 0 1 )$ </td></tr><tr><td>Rotation offset (roll, pitch, yaw) [deg]</td><td> $\mathcal { U } ( - 1 . 5 , 1 . 5 ) \times \mathcal { U } ( - 1 . 5 , 1 . 5 ) \times \mathcal { U } ( - 1 . 5 , 1 . 5 )$ </td></tr></table>

Table 4: Domain randomization parameters used during offline rendering. $\textstyle { \mathcal { U } } ( a , b )$ denotes a uniform distribution over $[ a , b ]$ , and $B ( p )$ denotes a Bernoulli distribution with success probability p.

## C Real-to-Sim Asset Generation Details

## C.1 Prompt Template for Physical Attribute Estimation

We query Qwen3-VL with the reference image and a category description of the object, using the following prompt:

This is a 3D model of {category}. Estimate its real-world dimensions in centimeters (length × width × height). Consider typical sizes of this object category. Output JSON: {“length cm”: X, “width cm”: Y, “height cm”: Z, “material”: “”}

The model’s output is parsed as JSON to populate the physical dimensions and material category of the corresponding 3D asset.

## C.2 Dimension Accuracy Evaluation

To assess the reliability of Qwen3-VL’s physical dimension estimation, we compare the predicted dimensions against ground-truth measurements obtained with a caliper on 5 real-world objects. Table 5 reports the per-object predicted and measured dimensions, along with the relative error.
<table><tr><td>Object</td><td>Predicted (cm)  $( L \times W \times H )$ </td><td>Measured (cm)  $( L \times W \times H )$ </td><td>Avg. Error (cm)</td></tr><tr><td>Box</td><td> $2 6 \times 1 9 \times 1 1$ </td><td> $2 2 \times 2 1 \times 1 0$ </td><td>2.3</td></tr><tr><td>Sponge</td><td> $2 4 \times 1 1 \times 6$ </td><td> $2 0 \times 9 . 5 \times 4 . 5$ </td><td>2.3</td></tr><tr><td>Monitor</td><td> $6 1 \times 2 1 \times 4 6$ </td><td> $6 1 \times 2 3 \times 4 5$ </td><td>1.0</td></tr><tr><td>Basket</td><td> $2 6 \times 2 4 \times 1 8$ </td><td> $3 0 \times 2 5 \times 2 2$ </td><td>3.0</td></tr><tr><td>Cup</td><td> $7 \times 7 \times 1 2$ </td><td> $7 \times 7 \times 1 1$ </td><td>0.3</td></tr></table>

Table 5: Comparison between Qwen3-VL predicted dimensions and real-world measurements.

## D Ablation on Curriculum-based Rollout

To validate the necessity of the curriculum-based rollout mechanism, we compare two training variants:

• w/o Rollout: the planner is trained exclusively on ground-truth history.

• w/ Rollout: the curriculum-based rollout described in Sec. 3.3.2.

We evaluate both variants on four manipulation tasks and report success rate per task as well as the overall average. Results are summarized in Table 6.
<table><tr><td>Variant</td><td>Place Cup in Box</td><td>Wipe Monitor</td><td>Lift Basket and Place Cup</td><td>Kneel and Wipe Under Table</td></tr><tr><td>w/o Rollout</td><td>2/10</td><td>1/10</td><td>0/10</td><td>0/10</td></tr><tr><td>w/ Rollout</td><td>8/10</td><td>7/10</td><td>8/10</td><td>10/10</td></tr></table>

Table 6: Ablation on the curriculum-based rollout mechanism. Training without rollout leads to compounding errors over long horizons, resulting in consistently lower success rates across all tasks.