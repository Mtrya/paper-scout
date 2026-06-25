# Foresight: Failure Detection for Long-Horizon Robotic Manipulation with Action-Conditioned World Model Latents

Haoran Zhang<sup>1,∗</sup>, Yifu Lu<sup>2,∗</sup>, Boyang Wang<sup>3</sup>, Xuhui Kang<sup>3</sup>, Yen-Ling Kuo<sup>3</sup>, Zezhou Cheng<sup>3</sup>, Mengdi Wang<sup>2</sup>, Odest Chadwicke Jenkins<sup>1,†</sup>

<sup>1</sup>University of Michigan <sup>2</sup>Princeton University <sup>3</sup>University of Virginia

<sup>∗</sup>Equal contribution. <sup>†</sup>Corresponding author.

Project Page: Foresight.github.io

Abstract: Long-horizon tasks are common in real-world robotic deployments, yet failure detection for such tasks remains underexplored. Detecting failures in long-horizon robotic tasks is particularly challenging because failure onset is often ambiguous and dense temporal annotations are typically unavailable. We present Foresight, a failure detection framework that monitors manipulation trajectories using latent representations from an action-conditioned world model. Foresight is trained using only final task-level success or failure labels. By leveraging predictive world-model embeddings, our method provides a unified framework for failure detection across different policies. We further use functional conformal prediction (FCP) to calibrate detection thresholds adaptively. We evaluate Foresight with state-of-the-art vision-language-action policies in simulation on LIBERO-Long, ManiSkill-Long, and BEHAVIOR-1K, compare it against state-of-the-art failure detection methods, and validate it on real robots with three long-horizon tasks on a ReactorX-200 arm and one task on a Franka arm. Our results suggest that action-conditioned world-model embeddings provide a scalable representation for reliable failure monitoring in long-horizon manipulation.

Keywords: Failure Detection, Long-Horizon Tasks, World Models

## 1 Introduction

Robots operating over long horizons must recognize not only when a task has failed, but also when an ongoing execution has drifted toward failure. We study failure detection: given the observations and actions available up to time t, a detector assigns a failure score to the current rollout. Prior work has estimated this score using policy uncertainty [1], policy-internal representations [2, 3], vision-language judgments of visible mistakes [4], embedding distribution [5], or world-model latents [6, 7]. These methods show that failures can often be detected before a terminal outcome is observed. However, most focus on short-horizon settings, isolated visual anomalies, or policyspecific confidence signals, leaving failure detection in long-horizon robotic tasks underexplored.

Long-horizon failure detection is challenging because the meaning of a visual state depends on the action history and task stage. The same object resting on a table may be expected before a grasp, evidence of a missed grasp after a lift command, or correct after a placement action. In multi-stage tasks lasting hundreds or thousands of steps, small deviations can accumulate and only later become irreversible. Effective detectors must therefore look beyond whether the current image appears unusual; they must judge whether the observed trajectory remains consistent with the progress implied by the robot’s actions.

This connection motivates us to use an action-conditioned (AC) world model [8] as the backbone for long-horizon failure detection. Latent representations from action-conditioned world models compactly encode task-relevant state cues, including spatial relationships, motion patterns, interaction dynamics, and action-conditioned scene changes. By condensing task-relevant state cues into a small set of informative tokens, these representations are well-suited for monitoring long-horizon tasks.

We introduce Foresight, a policy-interface-agnostic failure detector for long-horizon robotic tasks that leverages latent representations from an action-conditioned world model. We freeze the visual encoder of the pretrained world model [8] and then attach and train an action-conditioned predic tor from scratch. The resulting action-conditioned world model produces predicted latent features for rollouts, which are then passed to the downstream failure detector, a simple yet effective causal Transformer [9]. Finally, we calibrate the detector with conformal prediction on held-out successful rollouts, yielding time-varying thresholds [10, 11]. Foresight does not require policy logits, hidden states, token probabilities, or access to a policy-specific uncertainty head; it only uses the rollout interface of visual observations and the corresponding action chunks. As a result, the same framework can be applied to different vision-language-action (VLA) and visuomotor policies once the dataset-specific AC predictor and detector are trained.

To fully demonstrate the effectiveness of Foresight, we comprehensively evaluate on challenging long-horizon simulation benchmarks and real-robot rollouts. In simulation, we adopt LIBERO-Long [12], ManiSkill-Long [13], and BEHAVIOR-1K [14], covering tabletop manipulation and mobile household tasks. These settings include horizons from hundreds of steps to BEHAVIOR 1K rollouts averaging more than 8,000 steps. We collect rollouts from multiple policy families, including OpenVLA [15], SmolVLA [16], π<sub>0</sub>-FAST [17, 18], and $\pi _ { 0 . 5 }$ policy [19]. We also test real ReactorX and Franka robot rollouts with ACT [20], $\pi _ { 0 . 5 }$ , SmolVLA, and GR00T N1.5 policies [21]. Across these benchmarks, we compare against multiple state-of-the-art baselines.

Our main contributions are:

• We propose Foresight, a failure detection framework for long-horizon robotic manipulation that feeds latent representations from an action-conditioned world model consisting of a frozen visual encoder and trained AC predictor as inputs to a causal transformer failure detector.

• We show that action-conditioned world model embeddings enable failure detection with supervision from only final task success/failure labels across different vision-language-action policies, and we incorporate functional conformal prediction to adaptively calibrate detection thresholds for reliable long-horizon failure detection.

• We provide a comprehensive evaluation of long-horizon failure detection across diverse manipulation tasks, policies, robotic embodiments, simulation benchmarks, and real-world experiments, demonstrating the effectiveness of Foresight against state-of-the-art failure detection methods.

## 2 Related Work

## 2.1 Failure Detection

Failure detection aims to identify unsuccessful robot executions from partial or complete rollout observations. Prior work has explored different monitoring signals. Some works [22, 4, 23, 24, 25, 26] frame failure detection as a vision-language reasoning problem, using a vision-language model to detect manipulation failures and provide natural-language explanations. ReDiffuser [5] learns a confidence function based on Random Network Distillation (RND) to measure the reliability of sampled decisions. FAIL-Detect [1] formulates failure detection for imitation-learning policies as sequential out-of-distribution detection, extracting scalar failure scores from policy observations and predicted actions, and calibrating time-varying thresholds with conformal prediction. SAFE [2] instead studies multitask failure detection for vision-language-action policies, training lightweight detectors on policy-internal representations to predict per-step failure scores from trajectory-level outcome labels. More recently, some works explore using world models as a signal [27, 7, 28].

For instance, Gauge [7] uses compressed video world-model latents [29] with conformal prediction thresholds to classify executions as success, known failure, or out-of-distribution anomaly.

## 2.2 Foundational World Models

Foundation world models aim to learn general-purpose representations or simulators of physical dynamics from large-scale video data. This has motivated a growing line of video-based world models [30, 31] that repurpose generative video prediction for simulating future observations. Cosmo World Foundation Models [29] introduce a generative world model platform for physical AI, with Cosmos-Predict models supporting future video or world-state prediction from text, image, or video conditions. Cosmos-Predict2.5 [32] extends this line with video foundation models for world simulation, supporting conditional generation from text, images, and videos. In parallel, joint-embedding predictive architectures learn world representations without reconstructing pixels. V-JEPA [6] predicts masked video regions in latent space, and V-JEPA 2 [8] scales this idea with internet scale video pretraining and post-trains an action-conditioned variant, V-JEPA 2-AC, on robot trajectories for physical prediction and planning. Because V-JEPA 2-AC predicts future latent states conditioned on robot actions, its representations can capture whether the observed execution is consistent with the policy’s intended behavior. We therefore use these action-conditioned world-model features as signals for detecting failures in long-horizon manipulation tasks.

## 3 Problem Formulation

We study failure detection for robot policies in long-horizon manipulation tasks. In this work, we define long-horizon tasks as those requiring multiple subgoals, typically involving multiple symbolic manipulation actions such as pick, place, open, and close. At timestep t, the robot receives an image observation $I _ { t }$ . Let $c _ { t }$ denote the observation context available at timestep t, which may consist of the current image $I _ { t }$ or a short history of recent images.

A robot policy π maps the observation context $c _ { t }$ to a predicted action chunk

$$
A _ { t } = \left( a _ { t | t } , a _ { t + 1 | t } , \dotsc , a _ { t + H - 1 | t } \right) ,\tag{1}
$$

where H denotes the prediction horizon and $\displaystyle a _ { t + k | t }$ denotes the action predicted for timestep $t + k$ at replanning timestep t. The robot executes the first $H ^ { \prime } \leq H$ actions from this chunk before replanning.

Each completed rollout is annotated with a trajectory-level binary outcome label

$$
y = { \left\{ \begin{array} { l l } { 1 , } & { { \mathrm { i f ~ t h e ~ r o b o t ~ f a i l s ~ t o ~ c o m p l e t e ~ t h e ~ t a s k , } } } \\ { 0 , } & { { \mathrm { i f ~ t h e ~ r o b o t ~ s u c c e s s f u l l y ~ c o m p l e t e s ~ t h e ~ t a s k . } } } \end{array} \right. }\tag{2}
$$

We assume access only to trajectory-level success or failure labels, without annotations of the precise timestep at which a failure occurs.

At execution time, our goal is to predict whether the ongoing rollout will eventually fail using only information available before executing the next action chunk. Given the current observation context $c _ { t }$ and the policy-predicted action chunk $A _ { t } .$ , we formulate failure detection as learning a scoring function

$$
D _ { \theta } \colon \{ ( c _ { i } , A _ { i } ) \} _ { i = 1 } ^ { t } = s _ { t }\tag{3}
$$

where $s _ { t } \in [ 0 , 1 ]$ is the predicted failure score at timestep t.

A failure alarm is triggered when the score exceeds a time-varying decision threshold $\delta _ { t }$

$$
\hat { y } _ { t } = \left\{ \begin{array} { l l } { 1 , } & { \mathrm { i f } \ s _ { t } \ge \delta _ { t } , } \\ { 0 , } & { \mathrm { o t h e r w i s e . } } \end{array} \right.\tag{4}
$$

The objective is to detect impending or ongoing failures during execution, while using only the current observation context and the policy’s next predicted action chunk.

![](images/8a1554c162e520335c62f2fff3ec75009d26c47590c1758fc206e0e27b9894cb.jpg)  
Figure 1: Overview of Foresight. Foresight consists of three stages. Stage 1: we fine-tune an action-conditioned world model (WM-AC) on robot rollouts consisting of image observations $I _ { 1 : T }$ and actions $a _ { 1 : T - 1 }$ . Stage 2: for each timestep $t ,$ the world model encodes the current observation context into hidden latents $z _ { t } ^ { h }$ and predicts action-conditioned future latents $z _ { t } ^ { p }$ using the policypredicted action chunk $A _ { t }$ . These latent tokens, together with positional encodings, are passed into a causal sequence model to produce per-timestep failure scores $s _ { t }$ . Stage 3: a conformal calibration set is used to construct a time-varying threshold $\delta _ { t }$ (orange line), and a rollout is flagged as failure once the failure score (blue line) is higher than the threshold $s _ { t } \geq \delta _ { t }$

## 4 Methodology

## 4.1 System Overview

We propose Foresight, a failure detection framework for policies executing long-horizon robotic tasks. Given the current observation context $c _ { t }$ and the policy-predicted action chunk $A _ { t }$ , Foresight uses an action-conditioned world model to extract execution-aware latent features and predicts a per-timestep failure score $s _ { t }$ . At deployment, a calibrated time-varying threshold $\delta _ { t }$ converts this score into a binary failure alarm. Figure 1 shows the overall pipeline.

Compared with prior failure detection methods that rely on policy-internal features [2, 3], Foresight uses features from an action-conditioned video world model. This design encourages the detector to capture execution-level failure cues rather than policy-specific artifacts, enabling cross-policy generalization.

## 4.2 World Model Feature Extraction

We use V-JEPA 2-AC [8] as our action-conditioned world model backbone. At each timestep $t ,$ the world model receives the observation context $c _ { t }$ and the policy-predicted action chunk $A _ { t }$ . The temporal feature encoder produces an observed latent representation

$$
z _ { t } ^ { h } = \operatorname { P o o l } \left( f _ { \phi } ( c _ { t } ) \right) ,\tag{5}
$$

where $f _ { \phi }$ is the visual encoder and Pool(·) averages over spatial patch embeddings. The action predictor then produces an action-conditioned predicted latent

$$
z _ { t } ^ { p } = \mathrm { P o o l } \left( g _ { \psi } ( z _ { t } ^ { h } , A _ { t } ) \right) ,\tag{6}
$$

where $g _ { \psi }$ predicts future latent states conditioned on the proposed action chunk.

The hidden latent $z _ { t } ^ { h }$ captures what is currently observed, while the predicted latent $z _ { t } ^ { p }$ captures what the world model expects to happen under the policy’s next action chunk. We use the predicted latent to form the timestep token

$$
u _ { t } = W z _ { t } ^ { p } + p _ { t } ,\tag{7}
$$

where $p _ { t } ~ \in ~ \mathbb { R } ^ { d }$ is a fixed sinusoidal positional encoding. We compare $z _ { t } ^ { p }$ and $z _ { t } ^ { h }$ as inputs in Appendix 12.2 to validate the importance of action conditioning in feature selection.

## 4.3 Failure Scoring with Causal Sequence Models

Given latent tokens up to timestep t,

$$
U _ { \leq t } = \{ u _ { 1 } , u _ { 2 } , \ldots , u _ { t } \} ,\tag{8}
$$

we use a causal sequence model to predict a per-timestep failure score:

$$
s _ { t } = D _ { \theta } ( U _ { \leq t } ) \in [ 0 , 1 ] .\tag{9}
$$

The causal mask ensures that the detector only uses information available up to the current timestep. We implement $D _ { \theta }$ using a causal Transformer with positional encodings and masked self-attention, and compare it against MLP and LSTM variants in the experiments.

The detector is trained using trajectory-level binary labels. Since failure timestamps are not annotated, each timestep inherits the rollout-level label, with early-detection weighting applied to encourage high scores before or during failure events.

## 4.4 Conformal Prediction Thresholding

Following previous works [1, 7, 2], we adopt functional conformal prediction (FCP) [11] to convert the continuous failure score $s _ { t }$ into a binary alarm with statistical guarantees. FCP constructs a one-sided time-varying upper band $\delta _ { t } = \mu _ { t } + h _ { t }$ calibrated on successful rollouts from a calibration split, where $\mu _ { t }$ is the time-varying mean score and $h _ { t }$ is a calibrated bandwidth term. We provide the detailed construction of $h _ { t }$ in Appendix 9.

A failure is declared at the first step where the score exceeds the band:

$$
\hat { y } _ { t } = \mathbf { 1 } [ s _ { t } \geq \delta _ { t } ] .\tag{10}
$$

Under mild exchangeability assumptions [10], this guarantees that the false positive rate, i.e., the probability of flagging a truly successful rollout as a failure at any point during execution, is con trolled at level α. We evaluate across a range of significance levels α in Section 5.

## 5 Experiment

## 5.1 Evaluation Benchmarks

We evaluate our method on three long-horizon manipulation benchmark suites: LIBERO-Long [12], ManiSkill-Long [13], and BEHAVIOR-1K [14]. These benchmarks cover different task horizons, environment complexity, robot embodiments, and policy sources. Table 1 summarizes the main properties of each benchmark.

LIBERO-Long LIBERO-Long is a widely used benchmark for evaluating vision-language-action policies on long-horizon tabletop manipulation. Its tasks typically require multiple object-interaction steps, such as placing two objects into a target container. We evaluate OpenVLA [15] and π<sub>0</sub>- FAST [17] on LIBERO-Long using their officially released checkpoints.

ManiSkill-Long To evaluate longer and more compositional manipulation behaviors, we construct four tasks in ManiSkill [13], referred to as ManiSkill-Long. These tasks require at least eight symbolic subgoals to accomplish. For example, stack 6 cube requires the robot to sequentially stack six cubes, which involves 12 pick-and-place actions. We evaluate π<sub>0</sub>-FAST [17] using self-collected rollouts, where the policy is fine-tuned from the corresponding checkpoints using our collected data.

<table><tr><td>Benchmark</td><td>#Tasks</td><td>Embodiment</td><td>Evaluated Policies</td><td>Avg. Steps</td></tr><tr><td>LIBERO-Long [12]</td><td>10</td><td>Franka</td><td> $\mathrm { O p e n V L A } / \pi _ { 0 } – \mathrm { F A S T }$ </td><td>253</td></tr><tr><td>ManiSkill-Long [13]</td><td>4</td><td>Franka</td><td> $\pi _ { 0 } { \mathrm { - F A S T } }$ </td><td>1,484</td></tr><tr><td>BEHAVIOR-1K [14]</td><td>4</td><td>R1Pro</td><td> $\operatorname { r e v i s e d } \ \pi _ { 0 . 5 }$ </td><td>8,557</td></tr><tr><td>Real-World Exp</td><td>4</td><td>ReactorX/Franka</td><td> $\mathrm { A C T } / \pi _ { 0 . 5 } / \mathrm { G R 0 0 T N 1 . 5 / S m o l V L A }$ </td><td>1,175</td></tr></table>

Table 1: Summary of the evaluation benchmarks. Each row corresponds to one benchmark suite. Average simulation steps are computed over successful rollouts; when multiple evaluated policies have different rollout horizons, we report the observed range.

BEHAVIOR-1K We further evaluate on four mobile manipulation tasks selected from the BEHAVIOR-1K challenge long-horizon benchmark. Unlike LIBERO-Long and ManiSkill-Long, BEHAVIOR-1K requires both navigation and manipulation in larger household environments. We evaluate a revised version of $\pi _ { 0 . 5 }$ [19] based on the best solution [33] from the BEHAVIOR-1K challenge.

Real-World Experiment Beyond simulation, we validate our approach in real-world manipulation experiments. We collect rollouts via teleoperation and evaluate three policies, ACT [20], $\pi _ { 0 . 5 }$ [19], and SmolVLA [16], on a ReactorX-200 arm across three tabletop arrangement tasks. To assess cross-embodiment generalization, we additionally evaluate GR00T N1.5 [21] on a toy pick-up task using a Franka arm.

Additional task-level details, rollout statistics, and policy information are provided in Appendix 11.

## 5.2 Baselines

We compare Foresight against four representative runtime failure detection baselines. Rollout-level ROC-AUC and balanced accuracy are computed using the protocol in Section 5.3. When a baseline requires a detection threshold, we calibrate it using the same held-out successful rollouts and sweep over the same significance levels α. We report the best-performing variant within each baseline:

FAIL-Detect [1] is an uncertainty-based OOD detection method using only successful rollouts. SAFE [2] trains on both success and failure rollouts. Predictions are calibrated with the same functional conformal prediction procedure used for Foresight. SAFE serves as the policy-internalrepresentation baseline. RND [5] models the embedding distribution of successful rollouts for OOD detection. Gauge [7] uses compressed video world-model latents together with conformal decision functions to classify executions as success, known failure, or out-of-distribution anomaly. We adapt it to our binary failure detection setting by collapsing all non-success outputs into failures and training on success data only. This baseline compares Foresight against a recent world-model approach that uses video latents and conformal thresholding.

## 5.3 Evaluation Metrics

Following the evaluation protocol of previous work [2], we assess rollout-level failure prediction using ROC-AUC and balanced accuracy, which respectively measure threshold-independent score separability and threshold-dependent classification performance.

ROC-AUC Given a per-timestep failure score $s _ { t } ,$ we aggregate it into a rollout-level score by taking the maximum value over the trajectory:

$$
\bar { s } = \operatorname* { m a x } _ { t = 1 , \dots , T } s _ { t } .\tag{11}
$$

We then compute ROC-AUC using s¯ to evaluate how well the score separates failed rollouts from successful ones across all possible thresholds. A higher ROC-AUC indicates stronger thresholdindependent discriminative ability.

Balanced accuracy For threshold-based evaluation, each rollout is classified as successful or failed according to the selected detection threshold. We report balanced accuracy,

$$
{ \mathrm { B a l A c c } } = { \frac { 1 } { 2 } } ( { \mathrm { T P R + T N R } } ) ,\tag{12}
$$

![](images/3f6896fa747d33f0ebc9c4032c333f0865e5c701ed354828bca0db741a6a66f7.jpg)

![](images/22256d6a6799852c822ead6535fa5e04532f10c4740ac84cec164b27f3ca18f3.jpg)  
Figure 2: Real-Robot Setup. Left: real-world robot setting for three table-top manipulation tasks using ReactorX-200 arm. Right: real-world robot setting for a three-toy picking task using Franka arm.

where TPR denotes the true positive rate and TNR denotes the true negative rate. Balanced accuracy assigns equal weight to successful and failed rollouts, making it robust to class imbalance.

We evaluate all baselines and our method using 3-fold cross-validation. In experiments, we sweep α to evaluate multiple operating points and report the value that gives the best cross-validation balanced accuracy. To ensure conclusions are not driven only by this threshold choice, we also report ROC-AUC, which evaluates threshold-independent score separability.

## 5.4 Experiment Results

Simulation failure detection. Table 2 compares Foresight with baselines on the three simulated benchmarks. Across all three datasets, the strongest Foresight-Transformer achieves the best calibrated balanced accuracy: $0 . 9 4 \pm 0 . 0 6$ on LIBERO-Long, $0 . 8 0 \pm 0 . 1 0$ on ManiSkill-Long, and $0 . 7 8 \pm 0 . 0 2$ on BEHAVIOR-1K. On the two longer-horizon benchmarks, Foresight-Transformer also obtains the best threshold-independent ROC-AUC, reaching $0 . 8 4 \pm 0 . 0 3$ on ManiSkill-Long and $0 . 7 6 \pm 0 . 0 2$ on BEHAVIOR-1K.

The gains are most pronounced on BEHAVIOR-1K, the longest and most challenging benchmark in our evaluation. As summarized in Table 1, BEHAVIOR-1K rollouts average 8,557 simulation steps, roughly 34× longer than LIBERO-Long and 5.8× longer than ManiSkill-Long. This setting requires detecting failures over extended executions rather than short rollouts, and it goes beyond the horizons for which the baseline methods were originally designed and evaluated. In this regime, the best non-Foresight baseline reaches $0 . 7 2 \pm 0 . 0 2 \mathrm { R O C - A U C }$ and $0 . 6 4 \pm 0 . 0 5$ balanced accuracy, whereas Foresight-Transformer reaches 0.76 ± 0.02 ROC-AUC and $0 . 7 8 \pm 0 . 0 2$ balanced accuracy. This 0.14 balanced-accuracy and 0.04 ROC-AUC improvement suggests that action-conditioned world-model features are especially useful over a long trajectory.

Real-world rollout monitoring. Table 3 shows that Foresight transfers to real-robot rollout monitoring across policies and embodiments. Foresight-Transformer achieves the best ROC-AUC in three of the four settings: ReactorX / ACT (0.93 ± 0.01), ReactorX $/ \pi _ { 0 . 5 } ( 0 . 8 7 \pm 0 . 0 3 )$ , and Franka / GR00T N1.5 (0.89 ± 0.10). Across settings, Foresight-LSTM is also consistently strong, whereas Foresight-MLP remains near chance (0.50–0.59). These results suggest that action-conditioned world model features are useful for real-world failure detection, but robust rollout monitoring requires sequence-level detectors rather than independent frame-level classification.

Cross-policy generalization. Table 4 evaluates whether a detector trained on one policy distribution can transfer to another. The results show that cross-policy generalization is feasible with Foresight, as detectors trained on $\pi _ { 0 } { \mathrm { - F A S T } }$ or $\pi _ { 0 . 5 }$ can successfully transfer to different test policies. However, transfer is policy-dependent and can be asymmetric. In the real-world setting, training on $\pi _ { 0 . 5 }$ transfers well to ACT, while training on ACT transfers poorly to $\pi _ { 0 . 5 }$ . One possible reason is that $\pi _ { 0 . 5 }$ rollouts contain broader behaviors, including recovery trajectories. For example, in a sequential task where the policy should pick the lion first and the banana second, ACT or SmolVLA may fail after missing the lion, whereas $\pi _ { 0 . 5 }$ may recover by picking the banana and then returning to pick the lion. A detector trained only on ACT-like rollouts may not see such recovery behavior and may misclassify it as failure. Overall, these results suggest that Foresight can generalize across policies, but the strength of transfer depends on whether the training policy covers the behaviors and failure modes of the target policy.

<table><tr><td rowspan="2">Method</td><td colspan="2">LIBERO-Long</td><td colspan="2">ManiSkill-Long</td><td colspan="2">BEHAVIOR-1K</td></tr><tr><td>ROC-AUC</td><td> $\mathbf { B a l A c c }$ </td><td> ${ \mathrm { R O C } } – \mathbf { A U C }$ </td><td> $\mathbf { B a l A c c }$ </td><td>ROC-AUC</td><td> $\mathbf { B a l A c c }$ </td></tr><tr><td>FAIL-Detect [1]</td><td> $0 . 9 0 \pm 0 . 0 2$ </td><td> $0 . 8 2 \pm 0 . 0 6$ </td><td> $0 . 7 1 \pm 0 . 0 2$ </td><td> $0 . 5 0 \pm 0 . 0 1$ </td><td> $0 . 5 4 \pm 0 . 0 6$ </td><td> $0 . 5 2 \pm 0 . 0 1$ </td></tr><tr><td>SAFE-MLP [2]</td><td> $0 . 5 2 \pm 0 . 0 1$ </td><td> $0 . 5 0 \pm 0 . 0 1$ </td><td> $0 . 6 1 \pm 0 . 0 2$ </td><td> $0 . 5 3 \pm 0 . 0 2$ </td><td> $0 . 5 0 \pm 0 . 0 0$ </td><td> $0 . 5 0 \pm 0 . 0 0$ </td></tr><tr><td>SAFE-LSTM [2]</td><td> ${ \bf 0 . 9 1 \pm 0 . 0 2 }$ </td><td> $0 . 8 8 \pm 0 . 0 2$ </td><td> $0 . 8 2 \pm 0 . 0 1$ </td><td> $0 . 7 4 \pm 0 . 0 1$ </td><td> $0 . 7 2 \pm 0 . 0 2$ </td><td> $0 . 6 4 \pm 0 . 0 5$ </td></tr><tr><td>RND [5]</td><td> $0 . 9 0 \pm 0 . 0 2$ </td><td> $0 . 8 3 \pm 0 . 0 4$ </td><td> $0 . 8 3 \pm 0 . 0 2$ </td><td> $0 . 6 8 \pm 0 . 1 8$ </td><td> $0 . 6 5 \pm 0 . 0 1$ </td><td> $0 . 5 4 \pm 0 . 0 4$ </td></tr><tr><td>Gauge [7]</td><td> $0 . 8 8 \pm 0 . 0 1$ </td><td> $0 . 8 1 \pm 0 . 0 6$ </td><td> $0 . 8 0 \pm 0 . 0 2$ </td><td> $0 . 7 7 \pm 0 . 0 3$ </td><td> $0 . 6 1 \pm 0 . 0 3$ </td><td> $0 . 6 0 \pm 0 . 0 3$ </td></tr><tr><td>Foresight-MLP</td><td> $0 . 8 8 \pm 0 . 0 1$ </td><td> $0 . 8 0 \pm 0 . 0 2$ </td><td> $0 . 7 0 \pm 0 . 0 3$ </td><td> $0 . 7 1 \pm 0 . 1 8$ </td><td> $0 . 7 3 \pm 0 . 0 2$ </td><td> $0 . 5 6 \pm 0 . 0 3$ </td></tr><tr><td>Foresight-LSTM</td><td> $0 . 8 6 \pm 0 . 0 2$ </td><td> $0 . 8 9 \pm 0 . 0 3$ </td><td> $0 . 7 6 \pm 0 . 0 0$ </td><td> $0 . 7 9 \pm 0 . 1 6$ </td><td> $0 . 7 5 \pm 0 . 0 4$ </td><td> $0 . 7 5 \pm 0 . 0 9$ </td></tr><tr><td>Foresight-Transformer</td><td> $0 . 8 9 \pm 0 . 0 2$ </td><td> ${ \bf 0 . 9 4 \pm 0 . 0 6 }$ </td><td> ${ \bf 0 . 8 4 \pm 0 . 0 3 }$ </td><td> ${ \bf 0 . 8 0 \pm 0 . 1 0 }$ </td><td> ${ \bf 0 . 7 6 \pm 0 . 0 2 }$ </td><td> ${ \bf 0 . 7 8 \pm 0 . 0 2 }$ </td></tr></table>

Table 2: Main rollout-level failure detection results. Values are reported as mean ± standard deviation across folds and rounded to two decimals. ROC-AUC is computed from the maximum failure score over each rollout on the test split. Balanced accuracy is computed after calibrating the detection threshold on the calibration split and selecting the tuned α. Best results are shown in bold blue, and second-best results are shown in orange. All Foresight methods use action-conditioned latent predictions. LIBERO-Long and ManiSkill-Long use $\pi _ { 0 } { \mathrm { - F A S T } }$ rollouts [17, 18], while BEHAVIOR-1K uses rollouts from a $\pi _ { 0 . 5 }$ model revised for the best BEHAVIOR-1K solution [33]. Gauge reports the best performance among its seven methods. Full experiment results with the selected alpha values are provided in Appendix 9.

<table><tr><td>Method</td><td>ReactorX / ACT</td><td> $\operatorname { R e a c t o r X } / \pi _ { 0 . 5 }$ </td><td>ReactorX / SmolVLA</td><td>Franka / GR00T N1.5</td></tr><tr><td>FAIL-Detect [1]</td><td> $0 . 8 5 \pm 0 . 0 7$ </td><td> $0 . 6 4 \pm 0 . 0 6$ </td><td> $0 . 7 1 \pm 0 . 0 5$ </td><td> $0 . 8 8 \pm 0 . 0 5$ </td></tr><tr><td>SAFE-MLP [2]</td><td> $0 . 8 9 \pm 0 . 0 5$ </td><td> $0 . 6 6 \pm 0 . 3 6$ </td><td> $0 . 6 4 \pm 0 . 1 9$ </td><td> $0 . 5 0 \pm 0 . 1 0$ </td></tr><tr><td>SAFE-LSTM [2]</td><td> $0 . 7 0 \pm 0 . 0 7$ </td><td> $0 . 7 5 \pm 0 . 1 4$ </td><td> $0 . 4 3 \pm 0 . 1 0$ </td><td> $0 . 7 9 \pm 0 . 1 0$ </td></tr><tr><td>RND [5]</td><td> $0 . 8 6 \pm 0 . 0 4$ </td><td> $0 . 7 8 \pm 0 . 0 6$ </td><td> $0 . 8 2 \pm 0 . 0 3$ </td><td> $0 . 6 4 \pm 0 . 1 5$ </td></tr><tr><td>Foresight-MLP</td><td> $0 . 5 0 \pm 0 . 0 0$ </td><td> $0 . 5 5 \pm 0 . 0 5$ </td><td> $0 . 5 3 \pm 0 . 2 2$ </td><td> $0 . 5 9 \pm 0 . 2 0$ </td></tr><tr><td>Foresight-LSTM</td><td> $0 . 8 5 \pm 0 . 0 5$ </td><td> $0 . 8 5 \pm 0 . 0 3$ </td><td> $0 . 6 4 \pm 0 . 0 8$ </td><td> $0 . 6 6 \pm 0 . 0 8$ </td></tr><tr><td>Foresight-Transformer</td><td> $0 . 9 3 \pm 0 . 0 1$ </td><td> $0 . 8 7 \pm 0 . 0 3$ </td><td> $0 . 7 9 \pm 0 . 0 9$ </td><td> $0 . 8 9 \pm 0 . 1 0$ </td></tr></table>

Table 3: Real-world manipulation results. We collect teleoperated rollouts and evaluate rollout-level failure detection across ACT [20], $\pi _ { 0 . 5 }$ [19], and SmolVLA [16] on a ReactorX arm over tabletop arrangement tasks. To assess cross-embodiment generalization, we further evaluate GR00T N1.5 [21] on a toy pick-up task using a Franka arm. We compare Foresight against the same baselines as in the simulation experiments and report ROC-AUC as mean ± standard deviation. Blue indicates the best result and orange indicates the second-best result in each column.

<table><tr><td>Benchmark</td><td>Train distribution</td><td>Test distribution</td><td>ROC-AUC</td><td>BalAcc</td></tr><tr><td>LIBERO-Long</td><td>π0-FAST rollouts</td><td>OpenVLA rollouts</td><td> $0 . 6 4 \pm 0 . 0 2$ </td><td> $0 . 9 0 \pm 0 . 0 1$ </td></tr><tr><td>Real-World Exp.</td><td> $\pi _ { 0 . 5 }$  rollouts</td><td>ACT rollouts</td><td> $0 . 9 4 \pm 0 . 0 2$ </td><td> $0 . 8 2 \pm 0 . 0 8$ </td></tr><tr><td>Real-World Exp.</td><td>ACT rollouts</td><td>π0.5 rollouts</td><td> $0 . 5 6 \pm 0 . 0 7$ </td><td> $0 . 5 2 \pm 0 . 0 3$ </td></tr><tr><td>Real-World Exp.</td><td>SmolVLA rollouts</td><td>ACT rollouts</td><td> $0 . 9 2 \pm 0 . 0 4$ </td><td> $0 . 7 3 \pm 0 . 0 7$ </td></tr><tr><td>Real-World Exp.</td><td> $\pi _ { 0 . 5 }$  rollouts</td><td>SmolVLA rollouts</td><td> $0 . 6 7 \pm 0 . 0 2$ </td><td> $0 . 6 2 \pm 0 . 0 1$ </td></tr></table>

Table 4: Generalization experiments. Cross-policy transfer evaluates whether the detector learns execution-level failure cues rather than policy-specific artifacts.

## 6 Conclusion

We presented Foresight, a failure detection framework for long-horizon robotic manipulation that monitors rollouts using action-conditioned world-model representations. By combining V-JEPAstyle latent prediction with causal failure detectors and functional conformal calibration, Foresight detects failures using only trajectory-level success/failure labels and does not require access to policy-internal states or uncertainty estimates. Across LIBERO-Long, ManiSkill-Long, BEHAVIOR-1K, and real-robot experiments, our results show that action-conditioned predicted latents provide effective signals for identifying execution failures, particularly on longer-horizon tasks. These findings suggest that action-grounded world-model embeddings are a promising interface for scalable and policy-adaptable runtime monitoring in robotic manipulation.

Limitations A key limitation is the computational cost and latency of pretrained world models, which makes on-device deployment challenging and may limit applicability to highly reactive or agile tasks requiring fast closed-loop control. In addition, while conformal calibration helps control false alarms under held-out successful rollouts, its guarantees depend on the calibration distribution matching deployment conditions.

## References

[1] C. Xu, T. K. Nguyen, E. Dixon, C. Rodriguez, P. Miller, R. Lee, P. Shah, R. Ambrus, H. Nishimura, and M. Itkina. Can we detect failures without failure data? uncertainty-aware runtime failure detection for imitation learning policies. arXiv preprint arXiv:2503.08558, 2025.

[2] Q. Gu, Y. Ju, S. Sun, I. Gilitschenski, H. Nishimura, M. Itkina, and F. Shkurti. Safe: Multitask failure detection for vision-language-action models. arXiv preprint arXiv:2506.09937, 2025.

[3] J.-F. Yeh, K.-H. Hung, P.-C. Lo, C.-M. Chung, T.-H. Wu, H.-T. Su, Y.-T. Chen, and W. H. Hsu. Aed: Adaptable error detection for few-shot imitation policy. In The Thirty-eighth Annual Conference on Neural Information Processing Systems (NeurIPS), 2024.

[4] J. Duan, W. Pumacay, N. Kumar, Y. R. Wang, S. Tian, W. Yuan, R. Krishna, D. Fox, A. Mandlekar, and Y. Guo. Aha: A vision-language-model for detecting and reasoning over failures in robotic manipulation. arXiv preprint arXiv:2410.00371, 2024.

[5] N. He, S. Li, Z. Li, Y. Liu, and Y. He. ReDiffuser: Reliable decision-making using a diffuser with confidence estimation. In R. Salakhutdinov, Z. Kolter, K. Heller, A. Weller, N. Oliver, J. Scarlett, and F. Berkenkamp, editors, Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 17921– 17933. PMLR, 2024. URL https://proceedings.mlr.press/v235/he24e.html.

[6] A. Bardes, Q. Garrido, J. Ponce, X. Chen, M. Rabbat, Y. LeCun, M. Assran, and N. Ballas. Revisiting feature prediction for learning visual representations from video. arXiv preprint arXiv:2404.08471, 2024.

[7] M. Ho, M. F. Ginting, I. R. Ward, A. Reinke, M. J. Kochenderfer, A.-a. Agha-Mohammadi, and S. Omidshafiei. World model failure classification and anomaly detection for autonomous inspection, 2026. URL https://arxiv.org/abs/2602.16182.

[8] M. Assran, A. Bardes, D. Fan, Q. Garrido, R. Howes, M. Komeili, M. Muckley, A. Rizvi, C. Roberts, K. Sinha, A. Zholus, S. Arnaud, A. Gejji, A. Martin, F. Robert Hogan, D. Dugas, P. Bojanowski, V. Khalidov, P. Labatut, F. Massa, M. Szafraniec, K. Krishnakumar, Y. Li, X. Ma, S. Chandar, F. Meier, Y. LeCun, M. Rabbat, and N. Ballas. V-jepa 2: Selfsupervised video models enable understanding, prediction and planning. arXiv preprint arXiv:2506.09985, 2025.

[9] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin. Attention is all you need. In Proceedings of the 31st International Conference on Neural Information Processing Systems, NIPS’17, page 6000–6010, Red Hook, NY, USA, 2017. Curran Associates Inc. ISBN 9781510860964.

[10] V. Vovk, A. Gammerman, and G. Shafer. Algorithmic Learning in a Random World. Springer, New York, NY, 2005. ISBN 978-0-387-00152-4. doi:10.1007/b106715.

[11] J. Diquigiovanni, M. Fontana, and S. Vantini. The importance of being a band: Finite-sample exact distribution-free prediction sets for functional data. Statistica Sinica, 35(2):853–871, 2025. doi:10.5705/ss.202022.0087.

[12] B. Liu, Y. Zhu, C. Gao, Y. Feng, Q. Liu, Y. Zhu, and P. Stone. Libero: Benchmarking knowledge transfer for lifelong robot learning. arXiv preprint arXiv:2306.03310, 2023.

[13] S. Tao, F. Xiang, A. Shukla, Y. Qin, X. Hinrichsen, X. Yuan, C. Bao, X. Lin, Y. Liu, T. kai Chan, Y. Gao, X. Li, T. Mu, N. Xiao, A. Gurha, V. N. Rajesh, Y. W. Choi, Y.-R. Chen, Z. Huang, R. Calandra, R. Chen, S. Luo, and H. Su. Maniskill3: Gpu parallelized robotics simulation and rendering for generalizable embodied ai. Robotics: Science and Systems, 2025.

[14] C. Li, R. Zhang, J. Wong, C. Gokmen, S. Srivastava, R. Mart´ın-Mart´ın, C. Wang, G. Levine, W. Ai, B. Martinez, H. Yin, M. Lingelbach, M. Hwang, A. Hiranaka, S. Garlanka, A. Aydin, S. Lee, J. Sun, M. Anvari, M. Sharma, D. Bansal, S. Hunter, K.-Y. Kim, A. Lou, C. R. Matthews, I. Villa-Renteria, J. H. Tang, C. Tang, F. Xia, Y. Li, S. Savarese, H. Gweon, C. K. Liu, J. Wu, and L. Fei-Fei. Behavior-1k: A human-centered, embodied ai benchmark with 1,000 everyday activities and realistic simulation. arXiv preprint arXiv:2403.09227, 2024.

[15] M. Kim, K. Pertsch, S. Karamcheti, T. Xiao, A. Balakrishna, S. Nair, R. Rafailov, E. Foster, G. Lam, P. Sanketi, Q. Vuong, T. Kollar, B. Burchfiel, R. Tedrake, D. Sadigh, S. Levine, P. Liang, and C. Finn. Openvla: An open-source vision-language-action model. arXiv preprint arXiv:2406.09246, 2024.

[16] M. Shukor, D. Aubakirova, F. Capuano, P. Kooijmans, S. Palma, A. Zouitine, M. Aractingi, C. Pascal, M. Russi, A. Marafioti, S. Alibert, M. Cord, T. Wolf, and R. Cadene. Smolvla: A vision-language-action model for affordable and efficient robotics. arXiv preprint arXiv:2506.01844, 2025.

[17] K. Black, N. Brown, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, L. Groom, K. Hausman, B. Ichter, S. Jakubczak, T. Jones, L. Ke, S. Levine, A. Li-Bell, M. Mothukuri, S. Nair, K. Pertsch, L. X. Shi, J. Tanner, Q. Vuong, A. Walling, H. Wang, and U. Zhilinsky. π<sub>0</sub>: A vision-language-action flow model for general robot control. arXiv preprint arXiv:2410.24164, 2024. doi:10.48550/arXiv.2410.24164.

[18] K. Pertsch, K. Stachowicz, B. Ichter, D. Driess, S. Nair, Q. Vuong, O. Mees, C. Finn, and S. Levine. Fast: Efficient action tokenization for vision-language-action models. arXiv preprint arXiv:2501.09747, 2025. doi:10.48550/arXiv.2501.09747.

[19] K. Black, N. Brown, J. Darpinian, K. Dhabalia, D. Driess, A. Esmail, M. R. Equi, C. Finn, N. Fusai, M. Y. Galliker, D. Ghosh, L. Groom, K. Hausman, B. Ichter, S. Jakubczak, T. Jones, L. Ke, D. LeBlanc, S. Levine, A. Li-Bell, M. Mothukuri, S. Nair, K. Pertsch, A. Z. Ren, L. X. Shi, L. Smith, J. T. Springenberg, K. Stachowicz, J. Tanner, Q. Vuong, H. Walke, A. Walling, H. Wang, L. Yu, and U. Zhilinsky. π : A vision-language-action model with open-world generalization. In Proceedings of The 9th Conference on Robot Learning, volume 305 of Proceedings of Machine Learning Research, pages 17–40. PMLR, 2025.

[20] T. Z. Zhao, V. Kumar, S. Levine, and C. Finn. Learning fine-grained bimanual manipulation with low-cost hardware. arXiv preprint arXiv:2304.13705, 2023.

[21] NVIDIA, J. Bjorck, N. C. Fernando Castaneda, X. Da, R. Ding, L. J. Fan, Y. Fang, D. Fox,˜ F. Hu, S. Huang, J. Jang, Z. Jiang, J. Kautz, K. Kundalia, L. Lao, Z. Li, Z. Lin, K. Lin, G. Liu, E. Llontop, L. Magne, A. Mandlekar, A. Narayan, S. Nasiriany, S. Reed, Y. L. Tan, G. Wang, Z. Wang, J. Wang, Q. Wang, J. Xiang, Y. Xie, Y. Xu, Z. Xu, S. Ye, Z. Yu, A. Zhang, H. Zhang, Y. Zhao, R. Zheng, and Y. Zhu. GR00T N1: An open foundation model for generalist humanoid robots. In ArXiv Preprint, March 2025.

[22] C. Agia, R. Sinha, J. Yang, Z. Cao, R. Antonova, M. Pavone, and J. Bohg. Unpacking failure modes of generative policies: Runtime monitoring of consistency and progress. In Proceedings of The 8th Conference on Robot Learning, volume 270 of Proceedings of Machine Learning Research, pages 689–723. PMLR, 2025.

[23] P. Pacaud, R. Garcia, S. Chen, and C. Schmid. Scaling cross-environment failure reasoning data for vision-language robotic manipulation, 2026. URL https://arxiv.org/abs/2512. 01946.

[24] P. Yi, Y. Ma, W. Xu, Y. Hao, S. Gan, W. Li, and S. Zhong. Critic in the loop: A tri-system vla framework for robust long-horizon manipulation, 2026. URL https://arxiv.org/abs/ 2603.05185.

[25] E. Zhou, Q. Su, C. Chi, Z. Zhang, Z. Wang, T. Huang, L. Sheng, and H. Wang. Code-asmonitor: Constraint-aware visual programming for reactive and proactive robotic failure detection, 2025. URL https://arxiv.org/abs/2412.04455.

[26] C. Grislain, H. Rahimi, O. Sigaud, and M. Chetouani. I-failsense: Towards general robotic failure detection with vision-language models. In Proceedings of the International Conference on Robotics and Automation (ICRA), 2026. URL https://arxiv.org/abs/2509.16072.

[27] I. R. Ward, M. Ho, H. Liu, A. Feldman, J. Vincent, L. Kruse, S. Cheong, D. Eddy, M. J. Kochenderfer, and M. Schwager. Foundational world models accurately detect bimanual manipulator failures, 2026. URL https://arxiv.org/abs/2603.06987.

[28] H. Liu, Y. Zhang, V. Betala, E. Zhang, J. Liu, C. Ding, and Y. Zhu. Multi-task interactive robot fleet learning with visual world models, 2024. URL https://arxiv.org/abs/2410.22689.

[29] N. Agarwal, A. Ali, M. Bala, Y. Balaji, E. Barker, T. Cai, P. Chattopadhyay, Y. Chen, Y. Cui, Y. Ding, et al. Cosmos world foundation model platform for physical ai. arXiv preprint arXiv:2501.03575, 2025.

[30] Y. Du, S. Yang, B. Dai, H. Dai, O. Nachum, J. Tenenbaum, D. Schuurmans, and P. Abbeel. Learning universal policies via text-guided video generation. Advances in neural information processing systems, 36:9156–9172, 2023.

[31] B. Wang, N. Sridhar, C. Feng, M. Van der Merwe, A. Fishman, N. Fazeli, and J. J. Park. This&that: Language-gesture controlled video generation for robot planning. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pages 12842–12849. IEEE, 2025.

[32] NVIDIA, A. Ali, J. Bai, M. Bala, Y. Balaji, A. Blakeman, T. Cai, J. Cao, T. Cao, E. Cha, Y.-W. Chao, P. Chattopadhyay, M. Chen, Y. Chen, Y. Chen, S. Cheng, Y. Cui, J. Diamond, Y. Ding, J. Fan, L. Fan, L. Feng, F. Ferroni, S. Fidler, X. Fu, R. Gao, Y. Ge, J. Gu, A. Gupta, S. Gururani, I. El Hanafi, A. Hassani, Z. Hao, J. Huffman, J. Jang, P. Jannaty, J. Kautz, G. Lam, X. Li, Z. Li, M. Liao, C.-H. Lin, T.-Y. Lin, Y.-C. Lin, H. Ling, M.-Y. Liu, X. Liu, Y. Lu, A. Luo, Q. Ma, H. Mao, K. Mo, S. Nah, Y. Narang, A. Panaskar, L. Pavao, T. Pham, M. Ramezanali, F. Reda, S. Reed, X. Ren, H. Shao, Y. Shen, S. Shi, S. Song, B. Stefaniak, S. Sun, S. Tang, S. Tasmeen, L. Tchapmi, W.-C. Tseng, J. Varghese, A. Z. Wang, H. Wang, H. Wang, H. Wang, T.-C. Wang, F. Wei, J. Xu, D. Yang, X. Yang, H. Ye, S. Ye, X. Zeng, J. Zhang, Q. Zhang, K. Zheng, A. Zhu, and Y. Zhu. World simulation with video foundation models for physical ai. arXiv preprint arXiv:2511.00062, 2025.

[33] I. Larchenko, G. Zarin, and A. Karnatak. Task adaptation of vision-language-action model: 1st place solution for the 2025 behavior challenge, 2025. URL https://arxiv.org/abs/ 2512.06951.

[34] J. J. Kuffner and S. M. LaValle. Rrt-connect: An efficient approach to single-query path planning. In Proceedings 2000 IEEE International Conference on Robotics and Automation (ICRA), volume 2, pages 995–1001, 2000. doi:10.1109/ROBOT.2000.844730.

[35] I. A. S¸ucan, M. Moll, and L. E. Kavraki. The Open Motion Planning Library. IEEE Robotics & Automation Magazine, 19(4):72–82, December 2012. doi:10.1109/MRA.2012.2205651. https://ompl.kavrakilab.org.

## Appendix

## 7 More Implementation Details

World-model feature extraction. We use V-JEPA 2-AC [8] as the action-conditioned worldmodel backbone, initialized from the pretrained $\mathtt { v j e p a 2 - a c - v i t g . p t }$ checkpoint (ViT-Giant encoder). The visual encoder is frozen throughout; only the action-conditioned predictor is trained on robot rollouts from the corresponding benchmark. Images are resized to $2 5 6 \times 2 5 6$ and normalized with ImageNet statistics. The encoder uses a patch size of $1 6 \times 1 6$ with tubelet size 2, yielding 256 spatial patch tokens per frame. At each replanning step, the model receives a sliding window of 8 frames (non-overlapping) as the observation context, together with the policy predicted action chunk, whose action dimensionality depends on the benchmark and robot embodiment: 7D for LIBERO and the real-world $\mathrm { { A C T / \pi _ { 0 . 5 } } }$ policies, 8D for ManiSkill-Long, 10D for the real-world Franka setup, and 23D for the BEHAVIOR-1K R1Pro robot. The predictor has 24 transformer layers with embedding dimension 1024 and 16 attention heads, and is frame-causal (pred is frame causal=True). Its output patch tokens are mean-pooled over all 256 spatial patches to produce a 1408-dimensional latent vector per frame (matching the ViT-Giant encoder embedding dimension), which is passed to the failure detector.

Action-conditioned predictor training. The predictor is trained with a combined teacher-forcing and autoregressive-rollout objective following the V-JEPA 2-AC training procedure [8]. Briefly, at each iteration the predictor is run in teacher-forcing mode (ground-truth target-encoder features as context) and in autoregressive rollout mode (n=2 steps), and the L1 losses on LayerNormnormalized representations are summed: $\mathcal { L } = \mathcal { L } _ { \mathrm { T F } } + \mathcal { L } _ { \mathrm { A R } }$ . We use AdamW $( \beta _ { 1 } { = } 0 . 9 , \beta _ { 2 } { = } 0 . 9 9 9 )$ with weight decay $0 . 0 4 , \mathrm { ~ a ~ }$ linear LR warmup over 10 epochs, and cosine annealing to 0 for 200 epochs total. The visual encoder remains frozen, and the predictor is trained from scratch for all ex periments. For LIBERO we train on a single H200 GPU with batch size 256 and peak LR $2 \times 1 0 ^ { - 4 }$ (warmup from $2 . 5 { \times } 1 0 ^ { - 5 } )$ . For BEHAVIOR-1K and Maniskill-Long we use 2×H200 GPUs with effective batch size 512 and the same LR schedule. For real-world benchmarks (Franka, ACT, $\pi _ { 0 . 5 } ,$ SmolVLA) we use 2×H200 GPUs with effective batch size 32 (16 per GPU) and peak LR $5 \times 1 0 ^ { - 5 }$ (warmup from $5 \times 1 0 ^ { - 6 } )$ .

Failure detector architectures. We evaluate three detector architectures on V-JEPA 2-AC features: MLP, LSTM, and causal Transformer. All three share the following hyperparameters: input dimension 1408 (the world-model latent), 2 layers, hidden dimension 256, learning rate $1 0 ^ { - 4 }$ (Adam), $\ell _ { 2 }$ regularization $\lambda { = } 1 0 ^ { - 2 }$ , dropout 0.1, and 300 training epochs with batch size 512 on a single H200 GPU.

The MLP projects the input through two linear layers (Linear→ReLU→Linear→Sigmoid), treating each timestep independently.

The LSTM is a 2-layer LSTM with hidden dimension 256, followed by a Linear→Sigmoid output head. It processes the full episode sequence with dropout applied between layers and on the final hidden state.

The causal Transformer applies a learned linear projection to dimension 256, adds sinusoidal positional encodings, then passes through 2 pre-norm TransformerEncoder layers with 4 attention heads, feedforward dimension $1 0 2 4 \left( = 4 \times 2 5 6 \right)$ , and dropout 0.1. A causal attention mask ensures that the score at timestep t depends only on features up to and including t. A final Linear→Sigmoid head produces per-step failure probabilities.

## 8 Data Splits and Calibration Protocol

We first randomly shuffle all rollouts and partition them into three equal-sized folds, yielding three experimental rounds in accordance with the standard 3-fold cross-validation protocol. In each round, one fold is held out as the test set, while the remaining two folds are used for model development. Specifically, these two folds are further split into a training set, validation set, and calibration set in a 6:1:1 ratio. The training set is used to fit the downstream detector, the validation set is used for model selection and hyperparameter tuning, and the calibration set is used to construct time-varying conformal thresholds. For the AC predictor, however, we use all non-test data available in each round, including the training, validation, and calibration sets. We assume the AC predictor has ful access to all data except the held-out test set.

<table><tr><td rowspan="2">Method</td><td colspan="2">LIBERO-Long</td><td colspan="2">ManiSkill-Long</td><td colspan="2">BEHAVIOR-1K</td></tr><tr><td>BalAcc</td><td>Best α</td><td> $\mathbf { B a l A c c }$ </td><td>Best α</td><td>BalAcc</td><td>Best α</td></tr><tr><td>FAIL-Detect [1]</td><td> $0 . 8 2 \pm 0 . 0 6$ </td><td>0.10</td><td> $0 . 5 0 \pm 0 . 0 1$ </td><td>0.15</td><td> $0 . 5 2 \pm 0 . 0 1$ </td><td>0.02</td></tr><tr><td>SAFE-MLP [2]</td><td> $0 . 5 0 \pm 0 . 0 1$ </td><td>0.02</td><td> $0 . 5 3 \pm 0 . 0 2$ </td><td>0.20</td><td> $0 . 5 0 \pm 0 . 0 0$ </td><td>0.02</td></tr><tr><td>SAFE-LSTM [2]</td><td> $0 . 8 8 \pm 0 . 0 2$ </td><td>0.02</td><td> $0 . 7 4 \pm 0 . 0 1$ </td><td>0.25</td><td> $0 . 6 4 \pm 0 . 0 5$ </td><td>0.10</td></tr><tr><td>RND [5]</td><td> $0 . 8 3 \pm 0 . 0 4$ </td><td>0.02</td><td> $0 . 6 8 \pm 0 . 1 8$ </td><td>0.05</td><td> $0 . 5 4 \pm 0 . 0 4$ </td><td>0.02</td></tr><tr><td>Gauge [7]</td><td> $0 . 8 1 \pm 0 . 0 6$ </td><td>0.20</td><td> $0 . 7 7 \pm 0 . 0 3$ </td><td>0.25</td><td> $0 . 6 0 \pm 0 . 0 3$ </td><td>0.20</td></tr><tr><td>Foresight-MLP</td><td> $0 . 8 0 \pm 0 . 0 2$ </td><td>0.10</td><td> $0 . 7 1 \pm 0 . 1 8$ </td><td>0.15</td><td> $0 . 5 6 \pm 0 . 0 3$ </td><td>0.02</td></tr><tr><td>Foresight-LSTM</td><td> $0 . 8 9 \pm 0 . 0 3$ </td><td>0.05</td><td> $0 . 7 9 \pm 0 . 1 6$ </td><td>0.02</td><td> $0 . 7 5 \pm 0 . 0 9$ </td><td>0.02</td></tr><tr><td>Foresight-Transformer</td><td> $0 . 9 4 \pm 0 . 0 6$ </td><td>0.02</td><td> $0 . 8 0 \pm 0 . 1 0$ </td><td>0.02</td><td> $0 . 7 8 \pm 0 . 0 2$ </td><td>0.20</td></tr></table>

Table 5: Balanced accuracy and selected α on simulation benchmarks. For each method and benchmark, α is chosen by maximizing balanced accuracy over the candidate set using 3-fold crossvalidation.

## 9 Conformal Prediction Thresholding

As described in the main text, FCP constructs a one-sided time-varying upper threshold

$$
\delta _ { t } = \mu _ { t } + h _ { t } ,\tag{13}
$$

where $\mu _ { t }$ is the mean score trajectory estimated from successful calibration rollouts and $h _ { t }$ is a calibrated bandwidth term. We now describe how $h _ { t }$ is instantiated.

Let $\{ s _ { t } ^ { ( i ) } \} _ { i = 1 } ^ { n }$ denote the score trajectories of the n successful rollouts in a held-out calibration set. We estimate the mean trajectory as

$$
\mu _ { t } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } s _ { t } ^ { ( i ) } .\tag{14}
$$

We further estimate a time-varying modulation term $\sigma _ { t } .$ which captures how scores deviate from the mean across calibration trajectories. For each calibration rollout, we compute the normalized nonconformity score

$$
R _ { i } = \operatorname* { s u p } _ { t } \frac { s _ { t } ^ { ( i ) } - \mu _ { t } } { \sigma _ { t } } .\tag{15}
$$

Let $\hat { q }$ be the (1 − α)-quantile of the calibration nonconformity scores $\{ R _ { i } \} _ { i = 1 } ^ { n }$ . The bandwidth term in the main text is then given by

$$
h _ { t } = \hat { q } \sigma _ { t } ,\tag{16}
$$

which yields the time-varying threshold

$$
\delta _ { t } = \mu _ { t } + h _ { t } = \mu _ { t } + \hat { q } \sigma _ { t } .\tag{17}
$$

A failure alarm is declared at the first step where the score exceeds the band:

$$
\hat { y } _ { t } = \mathbf { 1 } [ s _ { t } \geq \delta _ { t } ] .\tag{18}
$$

Selection of α. We sweep a fixed candidate set

$$
\alpha \in \{ 0 . 0 2 , 0 . 0 5 , 0 . 1 0 , 0 . 1 5 , 0 . 2 0 , 0 . 2 5 , 0 . 3 0 , 0 . 3 5 , 0 . 4 0 , 0 . 4 5 , 0 . 5 0 , 0 . 6 0 , 0 . 7 0 , 0 . 8 0 , 0 . 9 0 \} .
$$

For each value of α, the time-varying threshold $\delta _ { t }$ is computed solely from the dedicated calibration split. The operating α is selected per method and benchmark by maximizing balanced accuracy aggregated across three cross-validation folds; the selected value is then fixed before reporting test results. Table 5 reports the selected α alongside balanced accuracy for simulation.

## 10 Baseline Implementation Details

Baseline details are in Table 6. To ensure fair comparison, all baselines are evaluated using the same train/calibration/test splits whenever applicable. When a method requires thresholding, we calibrate it using the same held-out rollouts used for Foresight.

For Gauge, we use the authors’ released code and default hyperparameters, but we use only the success model because our setting contains success and failure labels but no separate OOD split, and the original paper reports that the success model outperforms the failure model. Thus, for this baseline, we train and calibrate only on successful rollouts. Gauge also reports multiple CP scoring methods, such as reconstruction error and latent distance (L2), so we report the best-performing variant for each dataset.
<table><tr><td>Method</td><td>Input signal</td><td>Uses failures for training?</td><td>Uses policy internals?</td></tr><tr><td>FAIL-Detect†</td><td>VLA internal latent</td><td>No</td><td>Yes</td></tr><tr><td>RND†</td><td>VLA internal latent + action predictions</td><td>No</td><td>Yes</td></tr><tr><td>SAFE-MLP</td><td>VLA internal latent</td><td>Yes</td><td>Yes</td></tr><tr><td>SAFE-LSTM</td><td>VLA internal latent</td><td>Yes</td><td>Yes</td></tr><tr><td>Gauge</td><td>World-model video latents</td><td>No</td><td>No</td></tr><tr><td>Foresight</td><td>Action-conditioned world-model latents</td><td>Yes</td><td>No</td></tr></table>

Table 6: Baseline comparison summary. This table clarifies which information each method is allowed to use and how thresholds are calibrated. <sup>†</sup>We adapt the original method by replacing its image observation input with the VLA’s internal latent, making it directly comparable with the other policy-internal baselines.

## 11 Additional Benchmark Details

This section provides additional details for the benchmark suites used in our experiments, including task names, number of rollouts, success rates of the evaluated policies, and rollout horizon statistics. These details complement the benchmark summary in Section 5.1. A visualization can be found in Fig. 3.

![](images/a904417d3ed711097aef2c8c9b8bbbb838264e820f614271d794f81db8bfb8db.jpg)  
Figure 3: Benchmark tasks overview

## 11.1 LIBERO-Long

LIBERO-Long contains 10 long-horizon tabletop manipulation tasks (as shown in Fig. 4). We collect 50 rollouts for each task, resulting in 500 rollouts per policy. The benchmark consists of multistage manipulation tasks where the robot must complete multiple object interactions within a single rollout.

We collect rollouts from two policies on LIBERO-Long. OpenVLA uses the publicly released openvla-7b-finetuned-libero-10 checkpoint, a 7B-parameter vision-language-action model fine-tuned on the LIBERO-10 dataset [15]. π<sub>0</sub>-FAST uses the pi0 fast libero configuration, initialized from the pi0 fast base pretrained model and fully fine-tuned for 30,000 steps on the physical-intelligence/libero dataset, with replan steps=5 (5 simulator steps executed per inference call, action dimension 7, action horizon 10).

Table 7 reports the per-task rollout statistics for OpenVLA, and Table 8 reports the corresponding statistics for $\pi _ { 0 } { \mathrm { - F A S T } }$

![](images/b1626652eed1f429743f5d412ce11b998171d52a4410b8b9057a4ecba01023bd.jpg)  
Figure 4: LIBERO-Long tasks overview

<table><tr><td>Task ID</td><td>Task Name</td><td>Success Rate</td><td>Avg. Policy Calls</td><td>Avg. Sim. Steps</td></tr><tr><td>0</td><td>Put both the alphabet soup and the tomato sauce in the basket</td><td>50.0% (25/50)</td><td>300.5</td><td>300.5</td></tr><tr><td>1</td><td>Put both the cream cheese box and the butter in the basket</td><td>64.0% (32/50)</td><td>269.5</td><td>269.5</td></tr><tr><td>2</td><td>Turn on the stove and put the moka pot on it</td><td>58.0% (29/50)</td><td>289.0</td><td>289.0</td></tr><tr><td>3</td><td>Put the black bowl in the bottom drawer of the cabinet and close it</td><td>42.0% (21/50)</td><td>285.8</td><td>285.8</td></tr><tr><td>4</td><td>Put the white mug on the left plate and the yellow and white mug on the right plate</td><td>38.0% (19/50)</td><td>251.7</td><td>251.7</td></tr><tr><td>5</td><td>Pick up the book and place it in the back compartment of the caddy</td><td>72.0% (36/50)</td><td>192.8</td><td>192.8</td></tr><tr><td>6</td><td>Put the white mug on the plate and put the chocolate pudding to the right of the plate</td><td>56.0% (28/50)</td><td>258.0</td><td>258.0</td></tr><tr><td>7</td><td>Put both the alphabet soup and the cream cheese box in the basket</td><td>64.0% (32/50)</td><td>291.2</td><td>291.2</td></tr><tr><td>8</td><td>Put both moka pots on the stove</td><td>32.0% (16/50)</td><td>410.3</td><td>410.3</td></tr><tr><td>9</td><td>Put the yellow and white mug in the mi- crowave and close it</td><td>54.0% (27/50)</td><td>323.3</td><td>323.3</td></tr><tr><td>Average</td><td></td><td>53.0%</td><td>287.2</td><td>287.2</td></tr></table>

Table 7: Per-task performance and rollout statistics for OpenVLA (openvla-7b-finetuned-libero-10) on LIBERO-Long with replan steps=1. Each task is evaluated with 50 rollouts. Since replan steps=1, one policy call corresponds to one policycontrolled simulation step.

<table><tr><td>Task ID</td><td>Task Name</td><td>Success Rate</td><td>Avg. Policy Calls</td><td>Avg. Sim. Steps</td></tr><tr><td>0</td><td>Put both the alphabet soup and the tomato sauce in the basket</td><td>80.0% (40/50)</td><td>54.6</td><td>280</td></tr><tr><td>1</td><td>Put both the cream cheese box and the butter in the basket</td><td>100.0% (50/50)</td><td>51.5</td><td>265</td></tr><tr><td>2</td><td>Turn on the stove and put the moka pot on it</td><td>30.0% (15/50)</td><td>50.2</td><td>258</td></tr><tr><td>3</td><td>Put the black bowl in the bottom drawer of the cabinet and close it</td><td>42.0% (21/50)</td><td>47.3</td><td>243</td></tr><tr><td>4</td><td>Put the white mug on the left plate and the yellow and white mug on the right plate</td><td>88.0% (44/50)</td><td>48.8</td><td>251</td></tr><tr><td>5</td><td>Pick up the book and place it in the back compartment of the caddy</td><td>62.0% (31/50)</td><td>35.2</td><td>183</td></tr><tr><td>6</td><td>Put the white mug on the plate and put the chocolate pudding to the right of the plate</td><td>80.0% (40/50)</td><td>48.0</td><td>247</td></tr><tr><td>7</td><td>Put both the alphabet soup and the cream</td><td>94.0% (47/50)</td><td>52.2</td><td>269</td></tr><tr><td>8</td><td>cheese box in the basket Put both moka pots on the stove</td><td>2.0% (1/50)</td><td>74.0</td><td>378</td></tr><tr><td>9</td><td>Put the yellow and white mug in the mi-</td><td>34.0% (17/50)</td><td>51.7</td><td>266</td></tr><tr><td>Average</td><td>crowave and close it</td><td>61.2%</td><td>49.2</td><td>253</td></tr></table>

Table 8: Per-task performance and rollout statistics for π -FAST (pi0 fast libero, fine-tuned from pi0 fast base) on LIBERO-Long with replan steps=5. Each task is evaluated with 50 rollouts.

## 11.2 ManiSkill-Long

ManiSkill-Long consists of four long-horizon manipulation tasks (as shown in Fig. 5) constructed in ManiSkill [13]. These tasks require longer chains of symbolic actions than LIBERO-Long, including exploration, packing, stacking, opening, closing, picking, and placing. The tasks are evaluated using the Franka arm embodiment. We adopt π<sub>0</sub>-FAST with the $\mathtt { p i 0 }$ maniskill rlds finetune

<table><tr><td>Task ID Short Name</td><td></td><td>Language Instruction</td><td>#Episodes</td><td>SR</td><td>Avg Policy Calls</td><td>Avg Sim Steps</td></tr><tr><td>0</td><td>Screwdriver &amp; Cup</td><td>pick up the screwdriver and cup out of the</td><td>59</td><td>15%</td><td>187</td><td>1807</td></tr><tr><td>1</td><td>Cubes into Bowl</td><td>drawer put three cubes into the bowl</td><td>100</td><td>50%</td><td>152</td><td>1646</td></tr><tr><td>2</td><td>Stack 3 Cubes</td><td>stack 3 cubes together, start with red cube</td><td>100</td><td>50%</td><td>142</td><td>1336</td></tr><tr><td>3</td><td>Stack 6 Cubes</td><td>stack 6 cubes together, start with red cube</td><td>60</td><td>17%</td><td>178</td><td>1122</td></tr></table>

Table 9: Task descriptions and rollout statistics for ManiSkill-Long. Exec horizon = 16 sim steps per policy call. Avg sim steps computed over successful rollouts only.

checkpoint, initialized from pi0 fast base and LoRA fine-tuned for approximately 100,000 step on the ManiSkill RLDS dataset. Training used a cosine learning rate schedule (peak $1 0 ^ { - 4 }$ , decay $1 0 ^ { - 5 }$ , 2,000 warmup steps), with action dimension 8 and action horizon 16. The training demonstrations were generated automatically without human teleoperation: a PDDL-based task planner decomposes each task into a sequence of symbolic actions (pick, place, stack, etc.), which are then executed by MPlib [13], ManiSkill’s built-in motion planning library that uses the RRTConnect [34] algorithm from OMPL [35] to compute collision-free joint trajectories. Table 9 summarizes the

![](images/14d93cfc9d4c8208d5295288dfc90d0e32a61068538f1398354899d50a2878c3.jpg)  
Figure 5: ManiSkill-Long tasks overview

task-level rollout statistics for $\pi _ { 0 } { \mathrm { - F A S T } } .$ In total, we collect 319 valid rollouts across four tasks. Compared with LIBERO-Long, ManiSkill-Long requires longer execution horizons. Successful $\pi _ { 0 } \cdot$ FAST rollouts require 93 policy calls and 1,484 simulation control steps on average.

![](images/74e9c731f72726eb35292329d57f80ac055d3879c4783422e441b52d98e0b482.jpg)  
Figure 6: Behavior-1k tasks overview

BEHAVIOR-1K evaluates long-horizon mobile manipulation in large-scale household environments. We select four tasks (as shown in Fig. 6) from the BEHAVIOR-1K challenge long-horizon benchmark. Unlike LIBERO-Long and ManiSkill-Long, which use a fixed-base Franka arm, BEHAVIOR-1K uses the R1Pro mobile manipulator and requires both navigation and manipulation.(See Table 10 for details)

We use a revised version of $\pi _ { 0 . 5 }$ , initialized from the pi0.5 base pretrained checkpoint and finetuned on the BEHAVIOR-1K challenge demonstration dataset [14]. This model was the 1st-place solution in the 2025 BEHAVIOR-1K Challenge (26% overall q-score) [33]. Fine-tuning used 200,000 steps with delta actions, action horizon 30, action dimension 23 (zero-padded to 32 internally), and 50 trainable task embeddings replacing text conditioning. The four evaluated tasks used the same checkpoint.

We collect 100 rollouts per task, resulting in 400 rollouts in total. Table 10 reports the task-level rollout statistics. Rollouts were collected targeting 50 successes and 50 failures per task for Foresight training; the reported success rate reflects the policy’s natural success rate observed during collection. The average successful rollout requires 427.4 policy calls and 8,557 simulation steps (1 policy call = 20 simulator steps). The longest task, setting mousetraps, requires 13,657 simulation steps on average.

## 11.4 Real-World Benchmarks

We evaluate on four real-robot platforms spanning three policies on a ReactorX arm and one policy on a Franka arm (as shown in Fig. 7). Table 11 summarizes task descriptions, episode counts, success rates, and rollout lengths.

ReactorX / ACT. We collect 40 episodes per task (banana, lego, arrange) using an ACT policy [20] trained for 50k gradient steps. Each episode consists of 12 policy calls at an execution horizon of 100 steps per call (∼1150 total executed steps). Success rates range from 8% (lego, a precision-intensive task) to 50% (arrange).

ReactorX / π0.5 and SmolVLA. We collect 40 episodes per task for both $\pi _ { 0 . 5 }$ [19] and SmolVLA [16], each with 14 policy calls per episode (∼1190 total executed steps). SmolVLA achieves notably higher success on arrange (65%) compared to $\pi _ { 0 . 5 }$ (22%), reflecting differences in policy capability on the more structured placement task.

<table><tr><td>Task ID</td><td>Short Name</td><td>Language Instruction</td><td>#Episodes</td><td>SR</td><td>Avg Policy Calls</td><td>Avg Sim Steps</td></tr><tr><td>3</td><td>Setting Mousetraps</td><td>Take the four mousetraps from the cabinet in the bathroom and place them on the bathroom floor, at</td><td>100</td><td>50%</td><td>849</td><td>13657</td></tr><tr><td>4</td><td>Hiding Easter Eggs</td><td>least two next to the same sink. Take the three Easter eggs out of the wicker basket on the lawn and place them next to a single tree, none</td><td>100</td><td>50%</td><td>595</td><td>8540</td></tr><tr><td>10</td><td>Turning On Radio</td><td>left in the basket. Turn on the radio receiver that&#x27;s on the table in the living room.</td><td>100</td><td>50%</td><td>167</td><td>2375</td></tr><tr><td>47</td><td>Cook Hot Dogs</td><td>Take the two hot dogs out of the refrigerator in the kitchen and cook them in the microwave until both are cooked.</td><td>100</td><td>50%</td><td>695</td><td>9654</td></tr></table>

Table 10: Task descriptions and rollout statistics for BEHAVIOR-1K. Exec horizon = 20 sim steps per policy call. All 4 tasks are seen (3-fold cross-validation). Language instructions from the official BEHAVIOR-1K challenge [14].

![](images/1c1f0abd462a0bc103f909391b5165074bcb255bcc87bfd0da11cae08fa1fea7.jpg)  
Figure 7: Real-world experiment task overview.

Franka / GR00T N1.5. We collect 44 episodes of the “pick 3 toys” task using GR00T N1.5 [21] on a Franka arm, with an average of 38 policy calls and an exec horizon of 45 steps per call (∼1700 total executed steps), achieving 48% success.

## 12 Ablation Studies

This section studies which components of Foresight are responsible for performance.

## 12.1 World-Model Backbone

Cosmos-Predict2.5-2B finetuning and feature extraction. We finetune the pretrained nvidia/Cosmos-Predict2.5-2B robot action-conditioned checkpoint [29] on LIBERO-Long using LoRA (rank 32, α = 32) applied to the attention and MLP projection layers of the video DiT (q/k/v/output proj, mlp.layer1/2), yielding approximately 20M trainable parameters out of 2B total. Each fold is trained for 5,000 iterations (batch size 1) at 256×320 resolution. At inference time, for each policy timestep t we feed the current observation frame and a look-ahead action chunk of A=12 future 7-DoF scaled delta-EEF actions to the model, which predicts A+1=13 total latent frames at spatial resolution 32×40 with 16 channels. The conditioning frame occupies latent index 0; we retain the three predicted future temporal tokens at indices 1–3 (corresponding to $\lfloor A / 4 \rfloor = 3$ future latent frames), yielding a tensor of shape (16, 3, 32, 40). The latent feature vector at timestep t is obtained by averaging over the spatial and temporal dimensions, producing a 16-dimensional representation that encodes action-conditioned predicted future dynamics.

<table><tr><td>Robot / Policy</td><td>Task</td><td>Language Instruction</td><td>#Ep.</td><td>SR</td><td>Avg Policy Calls</td><td>Avg Sim Steps</td></tr><tr><td rowspan="3">ReactorX / ACT</td><td>Banana</td><td>pick up banana and lion toy into basket</td><td>40</td><td>48%</td><td>12</td><td>1141</td></tr><tr><td>Arrange</td><td>put skin cream, praline, and bottle into the bas-</td><td>40</td><td>50%</td><td>12</td><td>1178</td></tr><tr><td>Lego</td><td>ket pick up lego jiggler box</td><td>40</td><td>8%</td><td>12</td><td>1146</td></tr><tr><td rowspan="3">ReactorX / π0.5</td><td>Banana</td><td>pick up banana and lion toy into basket</td><td>40</td><td>18%</td><td>14</td><td>1188</td></tr><tr><td>Lego</td><td>pick up lego block and cleaner into basket</td><td>40</td><td>18%</td><td>14</td><td>1190</td></tr><tr><td>Arrange</td><td>put skin cream, praline, and bottle into the bas- ket</td><td>40</td><td>22%</td><td>14</td><td>1190</td></tr><tr><td rowspan="3">ReactorX / SmolVLA</td><td>Banana</td><td>pick up banana and lion toy into basket</td><td>40</td><td>28%</td><td>14</td><td>1190</td></tr><tr><td>Lego</td><td>pick up lego block and cleaner into basket</td><td>40</td><td>30%</td><td>14</td><td>1192</td></tr><tr><td>Arrange</td><td>put skin cream, praline, and bottle into the bas- ket</td><td>40</td><td>65%</td><td>14</td><td>1192</td></tr><tr><td>Franka / GR00T N1.5</td><td>Pick 3 Toys pick up 3 toys</td><td></td><td>44</td><td>48% (21/44)</td><td>38</td><td>~1727</td></tr></table>

Table 11: Task descriptions and rollout statistics for real-world benchmarks. Each episode runs until task completion or a fixed time limit. For Franka, average steps are estimated as average policy calls × execution horizon (45 steps/call).

<table><tr><td>Representation model</td><td>MLP</td><td>LSTM</td><td>Transformer</td></tr><tr><td>Cosmos-Predict2.5-2B robot-AC</td><td> $0 . 8 5 \pm 0 . 0 2$ </td><td> $0 . 8 5 \pm 0 . 0 1$ </td><td> $0 . 8 4 \pm 0 . 0 2$ </td></tr><tr><td>V-JEPA 2-AC</td><td> $0 . 8 8 \pm 0 . 0 1$ </td><td> $0 . 8 6 \pm 0 . 0 2$ </td><td> $0 . 8 9 \pm 0 . 0 2$ </td></tr></table>

Table 12: World-model backbone comparison on LIBERO-Long. We compare ROC-AUC using action-conditioned representations from Cosmos-Predict2.5-2B AC and V-JEPA 2-AC across MLP, LSTM, and Transformer detectors. Values are reported as mean ± standard deviation across folds. Best results are shown in bold.

The results suggest that V-JEPA-style latent prediction provides stronger failure-detection features than diffusion-based video generation on LIBERO-Long.

A likely explanation is that failure detection does not require pixel-level details which are hard to predict, but requires representations for predictable aspects of a scene, exposing robot-object state and action-conditioned deviations from expected dynamics.

## 12.2 Hidden Latents versus Action-Conditioned Predicted Latents

The comparison between $z ^ { h }$ and $z ^ { p }$ tests whether action conditioning improves failure detection. Hidden latents $z ^ { h }$ primarily summarize the current visual observation, while predicted latents $z ^ { p }$ encode the world model’s action-conditioned expectation of how the scene should evolve. This distinction is important because many robot failures are not visually anomalous in isolation; they are mismatches between the intended action and the observed state transition.

<table><tr><td rowspan="2">Benchmark</td><td colspan="3">Hidden latent  $z ^ { \mathrm { h i d d e n } }$ </td><td colspan="3">Predicted latent  $z ^ { \mathrm { p r e d } }$ </td></tr><tr><td>MLP</td><td>LSTM</td><td>Transformer</td><td>MLP</td><td>LSTM</td><td>Transformer</td></tr><tr><td>LIBERO-Long</td><td> $0 . 7 7 \pm 0 . 0 2$ </td><td> $0 . 8 3 \pm 0 . 0 0$ </td><td> $0 . 8 5 \pm 0 . 0 2$ </td><td> $0 . 8 8 \pm 0 . 0 1$ </td><td> $0 . 8 6 \pm 0 . 0 2$ </td><td> $0 . 8 9 \pm 0 . 0 2$ </td></tr><tr><td>ManiSkill-Long</td><td> $0 . 7 4 \pm 0 . 0 2$ </td><td> $0 . 7 8 \pm 0 . 0 2$ </td><td> $0 . 8 1 \pm 0 . 0 2$ </td><td> $0 . 7 0 \pm 0 . 0 3$ </td><td> $0 . 7 6 \pm 0 . 0 0$ </td><td> $0 . 8 3 \pm 0 . 0 2$ </td></tr></table>

Table 13: Latent-feature ablation for V-JEPA 2-AC. We compare ROC-AUC using hidden latents before the action-conditioned layer and predicted latents after the action-conditioned layer across benchmarks. Values are reported as mean ± standard deviation across folds.

## 13 Qualitative Results

We present qualitative examples of Foresight (Transformer, predicted states) across all four benchmarks. Each figure shows ten uniformly sampled frames from the rollout (bottom strip) alongside the full failure score curve $s _ { t }$ and the calibrated functional conformal prediction threshold $\delta _ { t }$ (top panel). Frame borders are coloured green when $s _ { t } < \delta _ { t }$ (safe) and red once the alarm latches at the first crossing $t ^ { * } = \operatorname* { i n f } \{ t \colon s _ { t } \geq \delta _ { t } \}$ . For each benchmark we show one true negative (successful episode that remains below the threshold throughout) and one true positive (failing episode where Foresight raises an alarm before task termination). The conformal miscoverage levels α used are 0.02 (LIBERO-Long), 0.02 (ManiSkill-Long), 0.20 (BEHAVIOR-1K), and 0.10 (real-world), matching the values reported in the main results table.

![](images/927644679e3aefa0bc5bc84a5e58610b7271c1f09cce3cddd12ad0e913ee0bf5.jpg)

Figure 8: LIBERO-Long (True Negative) $( \alpha { = } 0 . 0 2 ,$ , Task 0). “Put both the alphabet soup and the tomato sauce in the basket.” The failure score $s _ { t }$ (blue) remains below the FCP threshold $\delta _ { t }$ (red dashed) throughout all inference steps; no alarm is raised and all frame borders are green.  
![](images/468006bdaa8af8a68f72ece7191444105c3144d260790f5fde7f8d2509e064f8.jpg)  
Figure 9: LIBERO-Long (True Positive) $( \alpha { = } 0 . 0 2 ,$ , Task 5). “Pick up the book and place it in the back compartment of the caddy.” Foresight raises an alarm before episode termination as the action-conditioned world model’s predicted states increasingly diverge from observed states. The robot failed the task because it dropped the book during the middle of execution.

![](images/853a9e1a706005190fe66462c9032a41d759d444078a7abf4f631b20826ef8a7.jpg)

![](images/684458b7324f3444d1c0de6f272f0506559b74dabd8d024f7391833beb0ad63e.jpg)  
Figure 10: ManiSkill-Long (True Negative) (α=0.02, Task 2: Cubes into Bowl). “Put three cubes into the bowl.”

![](images/33dbe48b2270744f7078fee58277899dd22535ef50c849266c93292cad4fb779.jpg)  
Figure 11: ManiSkill-Long (True Positive) (α=0.02, Task 3: Stack 3 Cubes ). “Stack 3 cubes together, starting with the red cube.” The robot failed to stack the red cube on the blue cube, leading to the final failure.

![](images/5761731a968234f61079cc6cd71788804fd3d2fbe12c47915b301d8f9bd43a95.jpg)  
Figure 12: BEHAVIOR-1K (True Negative) (α=0.20, Task 3: Setting Mousetraps). “Take four mousetraps from the bathroom cabinet and place at least two next to the same sink.”

![](images/00667b12309f3e6c8ced37e114e103783da7953cb0b8b221e7b899ac4f41ff73.jpg)  
Figure 13: BEHAVIOR-1K (True Positive) (α=0.20, Task 47: Cook Hot Dogs). “Take two hot dogs from the refrigerator and cook them in the microwave.” The robot fails during this task because it did not grasp the first hot dog.

![](images/8589ec7320821904140ec2a73d8fc45964094c0887579252ab2d37f6a8e70d24.jpg)  
Figure 14: Real-world (ReactorX / ACT) (True Negative) (α=0.10, Pick Banana and toy lion task). “Pick up banana and lion toy into basket.” No false alarm is raised, showing Foresight does not penalize successful executions.

![](images/3449f864ca91a983085e9f720b90847ca83e9bfdfe256df90cb0146ebfb63865.jpg)  
Figure 15: Real-world (ReactorX / ACT) (True Positive) (α=0.10, Pick Banana and toy lion task). “Pick up banana and lion toy into basket.” A failing real-robot episode from the same task. The robot failed to pick up the banana, leading to final task failure.

## 14 Runtime and Deployment

Foresight is intended for runtime monitoring, so deployment cost is an important practical consideration. All measurements are conducted on a single NVIDIA H200 GPU using CUDA timing events, averaged over 100 forward passes after 10 warm-up iterations.

## Inference Latency

Table 14 reports per-call inference latency for each stage of the Foresight pipeline and for the baselines. Foresight is invoked once per action-chunk boundary, so the relevant budget is one replan interval rather than one control step.

Table 14: Inference latency measured on a single NVIDIA H200 GPU (mean ± std over 100 runs). Foresight is composed of a frozen world-model backbone (V-JEPA 2-AC) and a lightweight failure detector head. SAFE baselines operate on features already produced by the policy backbone and therefore incur only the head cost. Latency is measured per replan step (every 16 control steps for π -FAST).
<table><tr><td>Method</td><td>Component</td><td>Params</td><td>Latency (ms)</td></tr><tr><td rowspan="7">Foresight</td><td>World-model encoder (ViT-G/16, 8 frames)</td><td>1,012M</td><td> $1 2 2 . 5 4 \pm 0 . 0 8$ </td></tr><tr><td>Action-conditioned predictor</td><td>305 M</td><td> $6 0 . 1 9 \pm 0 . 0 7$ </td></tr><tr><td>Subtotal: feature extraction</td><td>1,317M</td><td> $1 8 2 . 7 3 \pm 0 . 1 1$ </td></tr><tr><td>Failure detector: MLP</td><td>0.4M</td><td> $0 . 0 8 \pm 0 . 0 0$ </td></tr><tr><td>Failure detector: LSTM</td><td>3.4M</td><td> $0 . 1 6 \pm 0 . 0 0$ </td></tr><tr><td>Failure detector: Transformer</td><td>2.0M</td><td> $0 . 9 1 \pm 0 . 0 2$ </td></tr><tr><td>Total (MLP head)</td><td>1,317M</td><td>182.81 ± 0.00</td></tr><tr><td rowspan="4"></td><td>Total (LSTM head)</td><td>1,317M</td><td> $\mathbf { 1 8 2 . 8 9 \pm 0 . 0 0 }$ </td></tr><tr><td>Total (Transformer head)</td><td>1,317M</td><td> $\mathbf { 1 8 3 . 6 4 \pm 0 . 0 2 }$ </td></tr><tr><td>SAFE-MLP</td><td>0.4M</td><td> $0 . 0 8 \pm 0 . 0 0$ </td></tr><tr><td>SAFE-LSTM</td><td>3.4M</td><td> $0 . 1 6 \pm 0 . 0 0$ </td></tr><tr><td rowspan="2">Baselines</td><td>RND</td><td>211M</td><td> $9 . 6 8 \pm 0 . 0 4$ </td></tr><tr><td>FAIL-Detect</td><td>124M</td><td> $6 . 2 8 \pm 0 . 0 2$ </td></tr></table>

## Cost Decomposition and Deployment Implications

The world-model backbone (V-JEPA 2-AC) dominates the total monitoring cost, accounting for over 99% of Foresight’s inference time. The failure detector head itself is negligible regardless of architecture: the MLP, LSTM, and Transformer heads add less than 1 ms atop the 182.73 ms backbone cost. In comparison, SAFE baselines incur no backbone overhead, their policy encoder is already executed during normal action inference, so their effective marginal cost reduces to that of the head alone (<0.2 ms for MLP/LSTM, 6–10 ms for the diffusion-based RND and FAIL-Detect variants).

Despite this gap, Foresight’s absolute latency of ≈183 ms remains well within the deployment budget of one replan interval. For $\pi _ { 0 } { \mathrm { - F A S T } }$ , which executes a 16-step action chunk before replanning, Foresight is queried at the chunk boundary rather than at every control step, providing a temporal window of 16 control steps in which to complete inference.