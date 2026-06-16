# Retrieve, Don’t Retrain: Extending Vision-Language-Action Models to New Tasks at Test Time

Jeongeun Park NAVER AI Lab

Juhan Park Korea University

Taekyung Kim NAVER AI Lab

Sungjoon Choi Korea University

Dongyoon Han NAVER AI Lab

Sangdoo Yun NAVER AI Lab

recap-robot.github.io

Abstract: Extending a vision-language-action (VLA) policy to a new task typically requires task-specific teleoperated demonstrations and per-task fine-tuning, making adaptation costly in both data collection and compute. In this paper, we show that this target-side per-task adaptation cost can be replaced by retrieval. Our retrieval augmented policy is trained once on paired demonstrations from the target embodiment (query) and a cheaper embodiment (pool, e.g., human-hand video), then frozen. New tasks are added at deployment by appending pool-side demonstrations to a retrieval pool. The frozen policy conditions on retrieved trajectories at every control step, so new tasks are absorbed by indexing data rather than updating parameters. Fine-tuning is needed only to take on a new, unseen embodiment, not for each new task. We show that retrieval improves policies beyond a specific backbone, including standard VLA policies, but its effect is especially pronounced in Cosmos Policy, a video-generation-based world-action model (WAM). In this setting, retrieval supplies coarse task progression, while the WAM’s future-image objective provides an additional visual consistency signal that strengthens the retrieval-conditioned actions. On PushT, we study how retrieval provides a reusable high-level motion prior for cross-embodiment generalization to unseen goal angles, while on RoboTwin 2.0 our method outperforms cross-embodiment baselines on unseen tasks, and we additionally demonstrate the method on a real robot.

Keywords: Robot foundation models, World-action models, Retrieval-augmented policies, Vision-language-action models

## 1 Introduction

General-purpose robot policies [1, 2, 3, 4, 5, 6, 7, 8] aim to execute open-ended manipulation behaviors from natural-language instructions while generalizing across diverse environments, tasks, and embodiments. Yet a new embodiment still requires its own teleoperated demonstrations and per-task fine-tuning, so cost grows with each new task added. We argue that this per-task cost is avoidable: behavioral knowledge from a cheap, data-rich source (e.g., human-hand video demonstrations) can transfer to the target embodiment through a retrieval rather than retraining paradigm.

The cost of the previous approach is twofold. On the data side, target-embodiment demonstrations would be collected through teleoperation, which is slow, hardware-bound, and roughly 18× slower to acquire than equivalent human-hand demonstrations [9, 10]. On the compute side, modern visionlanguage-action (VLA) models and robot foundation models operate over high-dimensional visual and action sequences, so per-task fine-tuning of recent world-action models (WAM) [7, 11, 8] costs roughly 24 GPU-hours per task and continues to scale with model size, context length, and action horizon. Both costs compound with every new task introduced.

![](images/ed0521f632d6f411364926de3bf2f0e1e77e3008e541831edf0c37a6729461a1.jpg)  
Figure 1: RECAP overview. Instead of teleoperating each new task and fine-tuning the policy (top, ∼24 GPU-hours/task for Cosmos Policy [7]), ReCAP appends cheap human-hand demonstrations to a retrieval pool while keeping the policy frozen (bottom), 18× cheaper [9, 10], no additional training.

We propose RECAP (Retrieval-Conditioned Action Policy), which shifts adaptation from repeated optimization to retrieval over a reusable pool of source-embodiment demonstrations. The policy is trained once to bridge the gap between source and target embodiments and is then frozen; behavioral coverage expands by simply appending new demonstrations to the retrieval memory.

RECAP builds on a world-action model (WAM) [7, 8, 12, 11, 13, 14, 15], specifically Cosmos Policy [7]. We parameterize the action latents as a residual over retrieved trajectories: retrieval supplies the coarse high-level motion and task progression, while the policy learns only the embodimentspecific dynamics needed to execute the behavior on the target robot. Crucially, the WAM’s futureimage prediction objective enforces consistency between the retrieved trajectory and the predicted evolution of the scene, a visual alignment signal that becomes informative only when paired with retrieval in unseen tasks, and that we find especially beneficial for long-horizon behaviors where high-level motion structure dominates.

The main contribution of this paper is threefold: a paradigm that adapts a policy to new tasks entirely at test time, absorbing each new task by extending a retrieval pool with cheap pool-embodiment demonstrations while the policy stays frozen with no parameter updates; a retrieval-conditioned residual policy on a WAM (i.e., Cosmos Policy [7]) in which retrieval supplies the high-level motion so the policy learns only the embodiment-specific correction, reinforced by the WAM’s futureimage objective that is informative only when paired with retrieval; and consistent gains over crossembodiment baselines on PushT [16] (34.9% vs. 6.0% on seven unseen angles) and RoboTwin 2.0 [17] (31.5% vs. 26.0% on five unseen tasks), with a pool-progression study confirming monotonic coverage growth without parameter updates and a further real-robot validation.

## 2 Related Work

World action models. Recent VLA policies inherit either a pretrained language backbone with an added action head (OpenVLA [1], π0.5 [3], GR00T N1.6 [4]) or a pretrained video model that folds actions into the same generative process (DreamZero [8], Cosmos Policy [7], mimic-video [12], Fast-WAM [11]). We call the latter family world-action models (WAMs); their video backbone is pretrained on internet-scale data and already encodes semantics and physical dynamics, so the policy learns only control on top, with action and future-observation prediction emerging from one shared video generation. These WAMs train on target-robot demonstrations alone, leaving their dynamics priors unpaired with cheaper cross-embodiment supervision, which we address by building a WAM on Cosmos Policy that conditions on retrieved pool-embodiment trajectories at training and deployment (Section 4.1).

Retrieval and Cross-embodiment policy transfer. Retrieval-based imitation [18] adapts a policy with relevant demonstrations rather than training from scratch on each task. Within a single embodiment, FlowRetrieval [19] co-trains the policy on optical-flow-retrieved data, and STRAP [20] trains a task-specific specialist at deployment from DTW-matched sub-trajectories. To cut data cost, attention has turned to cheaper human demonstrations [21, 22, 23]: Hong et al. [24] uses a single human-hand demonstration to retrieve matching robot sub-trajectories and efficiently fine-tunes a policy on them for fast adaptation. These methods still pay for each new task with training, whether by co-training or a test-time fine-tune. Other methods reduce this training cost by using human data more directly. R+X [25] retrieves everyday human videos and executes them directly without training, but it stays in the human domain and does not learn an embodiment-specific correction. In-context methods such as MimicDroid [9] condition on a human prompt, which a user must handpick for each task. We instead learn the cross-embodiment gap once on paired (query, pool) data and freeze it; the user then grows the retrieval pool with new pool-embodiment demonstrations at test time, which broadens the policy’s task coverage without retraining.

## 3 Problem Formulation

Setting. We study a cross-embodiment imitation setting with two sides: a query side that we want to control autonomously at deployment, carrying a target embodiment (e.g., a robot arm), and a pool side carrying a pool embodiment that is easier to collect demonstrations from but is never deployed (e.g., a human hand). The two embodiments differ in geometry, contacts, and dynamics, and differ sharply in data cost, since a query demonstration requires a teleoperation rig and operator while a pool demonstration only needs a lightweight tracker on a human. Transfer between them rests on two assumptions: a shared state/action representation (we use SE(3) end-effector pose plus a gripper signal where applicable), and motions that are semantically similar at the trajectory level, so a coarse plan derived from a pool trajectory is informative for the query.

Train and test access. At training time we assume that the model is given paired demonstrations $\mathcal { D } _ { \mathrm { t r a i n } } ^ { \mathrm { q u e r y } }$ and $\mathcal { D } _ { \mathrm { t r a i n } } ^ { \mathrm { p o o l } }$ on a fixed task distribution, each a set of state-action pairs $\left\{ \left( s _ { t } , a _ { t } \right) \right\}$ collected on the corresponding embodiment. At test time the model faces new tasks outside this distribution; only pool demonstrations $\mathcal { D } _ { \mathrm { t e s t } } ^ { \mathrm { p o o l } }$ are provided, no additional query data is collected, and model parameters are not updated. The model is rolled out on the target embodiment, so the current query state $s _ { t } ^ { \mathrm { q u e r y } }$ is observed at every control step.

Retrieval-conditioned action prediction. Let $\mathcal { D } ^ { \mathrm { p o o l } }$ denote the active retrieval pool, $\mathcal { D } _ { \mathrm { t r a i n } } ^ { \mathrm { p o o l } }$ at training and $\mathcal { D } _ { \mathrm { t e s t } } ^ { \mathrm { p o o l } }$ at deployment. At each step t, we select t′ = arg mint′ $d ( s _ { t } ^ { \mathrm { q u e r y } } , s _ { t ^ { \prime } } ^ { \mathrm { p o o l } } )$ over $\mathcal { D } ^ { \mathrm { p o o l } }$ ， where d is a feature-space distance specified in Section 4.2. The retrieved chunk $( s _ { t ^ { \prime } : t ^ { \prime } + H } ^ { \mathrm { p o o l } } , a _ { t ^ { \prime } : t ^ { \prime } + H } ^ { \mathrm { p o o l } } )$ of action chunk length H steps together with $s _ { t } ^ { \mathrm { q u e r y } }$ is fed to the policy, which predicts the query action chunk $a _ { t : t + H } ^ { \mathrm { q u e r y } }$ . The same rule applies at training and deployment with only $\mathcal { D } ^ { \mathrm { p o o l } }$ changing; crucially, $\mathcal { D } _ { \mathrm { t e s t } } ^ { \mathrm { p o o l } }$ can be extended at any time without touching θ.

## 4 Proposed Method

We propose RECAP to adapt a policy to new tasks at test time without retraining, by retrieving relevant demonstrations rather than updating weights. The intuition is that the target and pool embodiments largely agree on what a task requires and differ mainly in how to execute it. A retrieved pool trajectory therefore supplies the shared high-level plan cheaply, leaving the policy to learn only the embodiment-specific correction, so adapting to a new task becomes a matter of indexing data rather than updating parameters, much as retrieval-augmented generation externalizes knowledge into a searchable store (Fig. 2). Section 4.1 details the backbone and its retrieval conditioning, and Section 4.2 specifies how the retrieved chunk is selected and extended at test time.

## 4.1 Retrieval-Augmented World Action Model

Because the policy stays frozen and conditions on an external pool, new tasks can be absorbed at test time by extending that pool rather than by retraining; we expect coverage to grow with the pool,

and cheap pool-embodiment data, such as human-hand video, to partly substitute for target-robot teleoperation.

Backbone and retrieval-conditioned input. The backbone is the Cosmos Policy formulation [7], which emits query-side actions and future image observations as one denoised video sequence.

We extend its conditioning with the retrieved pool-embodiment chunk:

$$
\begin{array} { r l } & { \pi _ { \boldsymbol { \theta } } \left( s _ { t } ^ { \mathrm { q u e r y } } , \ \left( s _ { t ^ { \prime } : t ^ { \prime } + H } ^ { \mathrm { p o o l } } , \ a _ { t ^ { \prime } : t ^ { \prime } + H } ^ { \mathrm { p o o l } } \right) \right) } \\ & { \qquad \longmapsto \ \hat { a } _ { t : t + H } ^ { \mathrm { q u e r y } } , \ \hat { s } _ { t + H } ^ { \mathrm { q u e r y } } . } \end{array}
$$

![](images/3dd25a01052b8b6ddd59cf171b0083221d11ca25d591adbf92a3417dea9d2977.jpg)

(1)

The retrieved chunk and the observed query frame are encoded into clean latent frames and prepended along the temporal axis as conditioning, while the query-side

Figure 2: RECAP framework. The current observation retrieves a matching state-action chunk from the pool database; the retrieved chunk and the current observation then condition a world action model that denoises the next action and next observation in one video sequence.

future actions and observations are denoised from noise (Fig. 2); the language instruction enters via cross-attention. The retrieved chunk thus extends the standard I2V conditioning, a single clean frame, to a clean state-action sub-sequence, with no architectural modification.

Joint training objective (World action model). Action and future-image latents are supervised jointly with a single flow-matching loss:

$$
\begin{array} { r } { \mathcal { L } ( \boldsymbol { \theta } ) \ = \ \lambda \mathcal { L } _ { \mathrm { a c t } } ( \hat { a } _ { t : t + H } ^ { \mathrm { q u e r y } } , a _ { t : t + H } ^ { \mathrm { q u e r y } } ) \ + \ \mathcal { L } _ { \mathrm { s t a t e } } ( \hat { s } _ { t + H } ^ { \mathrm { q u e r y } } , s _ { t + H } ^ { \mathrm { q u e r y } } ) \ } \end{array}\tag{2}
$$

Joint training yields actions aligned with the predicted next state, producing more grounded action outputs. For a standard VLA, we lack $\mathcal { L } _ { \mathrm { s t a t e } }$ and $\hat { s } _ { t + H } ^ { \mathrm { q u e r y } }$

Residual action parameterization. Because the retrieved pool action chunk $a _ { t ^ { \prime } : t ^ { \prime } + H } ^ { \mathrm { p o o l } }$ already encodes a coarse motion the target should execute, we let the action latents represent only the embodiment-specific correction [26, 27] on top:

$$
\hat { a } _ { t : t + H } ^ { \mathrm { q u e r y } } = a _ { t ^ { \prime } : t ^ { \prime } + H } ^ { \mathrm { p o o l } } + \Delta a _ { t : t + H } .\tag{3}
$$

This narrows what the action latents must encode to “how the query action differs from the pool’s,” the variation actually caused by the embodiment gap. This variation is weakly reflected in action labels but clearly visible in pixels, e.g., how contact happens, how the gripper closes. Residual focuses the action latents on this correction, and state prediction provides the dense visual signal to learn it.

## 4.2 Retrieval

At each control step t, the policy retrieves a pool-embodiment index $t ^ { \prime }$ from $\mathcal { D } ^ { \mathrm { p o o l } }$ whose surrounding chunk best matches the current query context. We first form a candidate set $\mathcal { C } _ { t } ^ { \mathrm { t r a j } }$ by taking the top-K trajectories closest to the query under a composite initial-frame descriptor ψ0, a language embedding of the goal, initial task-relevant object positions (via SAM 3 [28]), and initial proprioception. Within $\mathcal { C } _ { t } ^ { \mathrm { t r a j } }$ , the index distance d in Section 3 is a weighted sum of $L _ { 2 }$ distances over object pose, proprioception, and the upcoming action chunk (training only; dropped at inference), and a cosine distance over a DINOv3 [29] image feature.

At inference, new pool-embodiment demonstrations $\mathcal { D } _ { \mathrm { t e s t } } ^ { \mathrm { p o o l } }$ replace the active pool and are reindexed under $\psi _ { 0 }$ and the features above; retrieval re-runs every step, so $\mathcal { D } _ { \mathrm { t e s t } } ^ { \mathrm { p o o l } }$ can grow within a session.

![](images/7820dba758eb04c0ec0ad3abd911f79e6f4334cc68340b96cb2cbb53d0dee575.jpg)  
Figure 3: PushT cross-embodiment pool database setting. The training set pairs the triangle (target) and disc (pool) at $\pm 4 5 ^ { \circ }$ The test set is a pool database of disc-pusher demonstrations spanning all goal angles, which the frozen triangle policy retrieves from on the seven unseen angles.

## 5 Experiments

We evaluate the proposed RECAP policy in three challenging cross-embodiment settings. PushT variant [16] (§5.2) provides a controlled environment for analyzing how retrieval improves generalization and why retrieval-conditioned WAMs are effective. RoboTwin 2.0 [17] (§5.3) evaluates whether the same paradigm scales to multi-task dual-arm manipulation, while real-robot experiments (§5.4) test whether unseen tasks can be absorbed through retrieval alone using human-hand demonstrations without additional robot training. Across all setups, we study the same hypothesis: retrieval supplies coarse task progression, while the policy learns only the embodiment-specific dynamics needed to execute the behavior on the target robot.

## 5.1 Experiment Setup

PushT Environment. In the 2D PushT benchmark [16, 30] an agent pushes a T-shaped block to a goal pose; we make it cross-embodiment with two pushers of different contact dynamics, a triangle (target) and a disc (pool), and take the goal rotation angle as the task axis. Training uses 100 paired {triangle, disc} demonstrations at $\pm 4 5 ^ { \circ }$ ; the triangle is then evaluated on nine different angles ranging from $- 6 0 ^ { \circ }$ to $+ 6 0 ^ { \circ }$ in $1 5 ^ { \circ }$ steps, seven of them unseen. The test-time pool holds disc-pusher demonstrations at $5 ^ { \circ }$ resolution over $[ - 6 0 ^ { \circ } , + 6 0 ^ { \circ } ]$ , added without retraining.

RoboTwin Simulation Environment. On RoboTwin 2.0 [17], we take Aloha-Agilex [31] as the target and UR5 as the pool, training on five paired {target, pool} tasks and evaluating on five unseen ones (Table 1). A test-time pool-progression eval grows the pool through five levels (i.e., 11, 17, 23, 29, 35 tasks), each a strict superset of the previous, with the policy frozen throughout.

Real Robot Setup. On a physical robot, the pool embodiment is a human hand (video with wristpose tracked in VR) and the target is the teleoperated robot. We fine-tune on a single task, opencabinet, with 25 paired demonstrations. Then we freeze the policy and evaluate on three tasks: the seen open-cabinet and two held out from fine-tuning, place-bottle-in-plastic-box and close-cabinet (Fig. 8a). The only test-time exposure to the held-out behaviors is 10 human-hand demonstrations per task added to the pool.

## 5.2 PushT Experiments

In this section, we study the following aspects of retrieval-conditioned generalization through PushT experiments [16]: (1) whether expanding the retrieval pool at test time improves unseen-task coverage without retraining, (2) whether retrieval benefits more from a WAM backbone than from an action-only policy, and (3) whether the retrieved trajectory acts as a reusable high-level motion prior that the policy adapts to the target embodiment. PushT is a controlled testbed whose generalization axis (i.e., the goal angle) is one-dimensional and densely measurable, which enables exploring these questions.

Test-time pool progression. Expanding the retrieval pool at test time without parameter updates steadily recovers the unseen angles. We grow the pool with disc-pusher demonstrations at intermediate goal angles and track per-angle success across five snapshots (Fig. 4). Specifically, the unseen-angle average rises monotonically from 6.0% without retrieval to 34.9% at the full pool.

![](images/de7201cd60e6d74325af9f8d7d1ef42f61601212c338bad5806650fbaec732e9.jpg)  
angle ( )

![](images/d0a450f8eaed5e93fe887f32d5f88f69975a7e3599be80611698e31e2a71902a.jpg)  
angle ( )

![](images/0629d8bb265bd73b5a47b0a2327b90c9474be6b314629531aba70b9adc046228.jpg)  
angle ( )

![](images/666027b0948d6dc48b353a4af2117f943d3fed08212c416d9da3b94721e7659a.jpg)  
angle ( )

![](images/985692d1e6af245e9c62ad7d6e671cd6c66f5dac59c586f9058dfbf894900574.jpg)  
angle ( )

Figure 4: Test-time pool progression on PushT. The leftmost panel is the no-retrieval baseline with our full-pool curve overlaid (shaded gap). The other panels show per-angle success as the pool grows with no retraining, with the previous snapshot in gray and the incremental gain shaded.  
![](images/475e18c4d45dbf25bd5ceeecbf6d05f547059a48d464449403bf4455f9149b6e.jpg)  
(a) Backbone comparison.

![](images/3236bafd8e1ad2c72e0f6d2fcbcadba9eb1b6f868e4d5e49de04b555b32b4cd7.jpg)  
(b) Joint training (Cosmos).

![](images/9ec5db1d33de20c70cb4ed23569e8469fea6e962887237d8a412480a7f8d03cb.jpg)

![](images/3ee96f3e8a8836bdc59409d78c7060f00b4608c076a4e3320ae6f74ab9cecc14.jpg)  
(c) ROI ratio across layers.  
Figure 5: Comparative analyses of RECAP and baseline on PushT. (a) Unseen-angle success with and without retrieval on a $\pi _ { 0 . 5 }$ and a Cosmos (WAM) backbone; retrieval helps both, and the WAM benefits more. (b) The future-image objective improves unseen success only when paired with retrieval. (c) Action-slot attention across decoder layers, which peaks on the T-block and then on the predicted next position under retrieval but stays near uniform without it.

Notably, each angle reaches much of its final success before its matching pool angle is added, so the policy interpolates over neighboring demonstrations rather than memorizing the nearest one.

Retrieval and the role of the WAM objective. If retrieval already supplies the coarse motion plan, then the remaining learning problem is primarily adapting that plan to the target embodiment dynamics. We therefore hypothesize that a WAM should benefit more from retrieval than an actiononly policy, since its future-image objective encourages the retrieved trajectory to remain consistent with the predicted future scene, providing a stronger learning signal for the embodiment-specific dynamics adaptation. Figure 5a reveals that retrieval improves both backbones, raising the unseenangle success of the action-only $\pi _ { 0 . 5 }$ [3] from 6.6% to 25.1%. However, the gain is larger with a WAM backbone. In Fig. 5b, removing the image-prediction objective reduces our model to 27.4%, comparable to retrieval-augmented $\pi _ { 0 . 5 }$ (25.1%), while restoring it improves performance to 34.9%.

How the retrieval-conditioned policy works. The retrieved chunk acts as a coarse motion prior, while the policy adapts it to the target embodiment rather than planning from scratch. Decoder cross-attention analysis in Fig. 5c reveals a two-stage routing behavior: early layers attend to the manipulated object and retrieved trajectory, whereas later layers shift attention toward the policy’s own predicted next state, adapting the retrieved plan to the target embodiment dynamics. This structure does not emerge without retrieval, where the ROI ratio remains near 1.0 across layers, indicating near-uniform attention. Masking the action-to-retrieval attention further confirms that the retrieved trajectory causally influences the generated actions.

## 5.3 RoboTwin Simulation Experiments

In this section, we evaluate the proposed method in a multi-task, dual-arm manipulation simulation environment, RoboTwin 2.0 [17]. We compare against standard cross-embodiment baselines on seen and held-out unseen tasks, then test whether the test-time pool growth seen on PushT carries over to this multi-task regime.

Baseline comparisons. We compare against three cross-embodiment baselines that share our backbone but differ in how they incorporate the UR5 (pool) data. Baseline, Cosmos Policy [7] is trained on Aloha-Agilex (target) demonstrations alone, with no access to the pool. Retrieval Only executes the action sequence of the nearest pool demonstration without learning. Co-training is a common cross-embodiment baseline, also used in EgoBridge [21] and STRAP [20], that jointly trains a single policy on the union of target and pool trajectories. Table 1 shows that RECAP leads on both splits, at 43.5% seen and 31.5% unseen versus 32.5% and 26.0% for the strongest baseline. Replaying the nearest pool trajectory (i.e., Retrieval Only) is competitive only where that trajectory already approximates the target action. Otherwise, the learned residual on top of retrieval is what closes the gap. As Fig. 6 illustrates, the nearest UR5 trajectory collides and dislodges the object, while our policy produces the finer grip orientation the task needs.

Table 1: Quantitative analysis on RoboTwin. We report per-task success rate (%) on RoboTwin, with Aloha-Agilex as the target embodiment and UR5 as the retrieval pool. The left block shows seen tasks, and the right block shows unseen tasks.
<table><tr><td></td><td colspan="6">Seen tasks</td><td colspan="6">Unseen tasks</td></tr><tr><td>Method</td><td>PCB</td><td>OM</td><td>DB</td><td>MP</td><td>GR</td><td>Avg↑</td><td>MPP</td><td>PBS</td><td>CB</td><td>HM</td><td>LP</td><td>Avg↑</td></tr><tr><td>Baseline [7]</td><td>47.5</td><td>10.0</td><td>25.0</td><td>30.0</td><td>50.0</td><td>32.5</td><td>0.0</td><td>0.0</td><td>20.0</td><td>0.0</td><td>0.0</td><td>4.0</td></tr><tr><td>Retrieval Only</td><td>30.0</td><td>7.5</td><td>22.5</td><td>7.5</td><td>60.0</td><td>25.5</td><td>0.0</td><td>10.0</td><td>42.5</td><td>37.5</td><td>40.0</td><td>26.0</td></tr><tr><td>Co-training</td><td>0.0</td><td>5.0</td><td>7.5</td><td>50.0</td><td>72.5</td><td>27.0</td><td>0.0</td><td>0.0</td><td>40.0</td><td>0.0</td><td>10.0</td><td>10.0</td></tr><tr><td>RECAP (Ours)</td><td>60.0</td><td>12.5</td><td>40.0</td><td>35.0</td><td>70.0</td><td>43.5</td><td>5.0</td><td>12.5</td><td>47.5</td><td>47.5</td><td>45.0</td><td>31.5</td></tr></table>

PCB = Place Cans Plasticbox OM = Open Microwave DB = Pick Dual Bottles MP = Move Can Pot GR = Grab Roller MPP = Move Pillbottle Pad PBS = Place Bread Skillet CB = Click Bell HM = Hand-over Mic LP = Lift Pot

![](images/d873dc08abb6ae0f1179c57ff69ce7b9cfbd3434ce673c409e714a843914a98c.jpg)  
Figure 6: Qualitative comparison on the held-out hand-over-mic task. Baseline (top-left) and Co-training (bottom-left) fail to grasp the microphone; Retrieval Only (top-right) knocks it over (red box); Ours (bottom-right) grasps it successfully. Each inset shows the retrieved UR5 chunk that the policy conditions on.

Test-time pool progression. Figure 7 shows that growing the retrieval pool at test time with the policy frozen raises unseen-task success monotonically, from 9.0% to 31.5% at the full pool. Each increase coincides with a held-out task becoming retrievable, and once all five are in the pool, the frozen policy matches its supervised unseen-task average. Cheap pool-embodiment data at deployment can therefore stand in for new target-embodiment demonstrations on tasks unseen during fine-tuning.

![](images/a0149b6b59696c94a857dca11a5e237f13834307798d11a1a5312bf47095ed5a.jpg)  
Figure 7: Test-time pool progression on RoboTwin.

## 5.4 Real Robot Experiments

We test whether the protocol transfers to real-world robots, despite the large embodiment gap between the human-hand demonstrations in the retrieval pool and the target robot (Fig. 8a). On two held-out tasks, the no-retrieval baseline collapses to the trained open-cabinet motion regardless of the target task, reaching only 10% and 0%, while retrieval enables the frozen policy to follow the conditioned human trajectory and reach 80% and 30% on placing the bottle and closing the cabinet (Fig. 8b). This indicates that human-hand demonstrations in the pool can partly substitute for additional target-robot teleoperation, even across a substantial embodiment gap. The qualitative results are shown in Figure 9.

![](images/b4d88a7862c9ce5b5cbbd2dd5c85856cc65736c989c877709ca0360ef649bd83.jpg)  
(a) Query, Pool database setting.

![](images/7f61d74c415ba4c0f9020463383fd35afb9e59933c5fd10cbf1760de4a38a219.jpg)  
(b) Success rate.

Figure 8: Real-robot experiment. (a) The training-time database pairs the robot (query) with a human-hand pool; the test-time database adds human-hand demonstrations for the held-out tasks. (b) Per-task success rate over 10 rollouts, Baseline vs RECAP (Ours).  
Task: Close cabinet  
![](images/8e638464a63174eab271866fb78a7dfb5bfa75caad5a635512a9ee0e46fbdde2.jpg)  
Figure 9: Real-robot generalization to unseen tasks. We show rollouts on the two held-out tasks (three frames each; baseline left, ours right). Trained only on open-cabinet, the baseline replays that trajectory and fails to close the cabinet (top) or grasp the bottle (bottom), whereas our policy follows the commanded behavior by conditioning on retrieved human-hand chunks.

## 6 Discussions

Summary. We extend a world-action-model policy to new tasks without retraining. Trained once to condition on a retrieval pool and predict an embodiment-specific residual on the retrieved trajectory, the frozen policy absorbs a new task by adding cheap pool-embodiment demonstrations at deployment. Across a cross-embodiment PushT variant, RoboTwin [17], and a physical robot, this improves generalization to unseen angles and tasks over cross-embodiment baselines, with success growing as the pool expands, and our analysis ties the gain to the WAM’s future-image objective acting together with retrieval. Indexing cheaper pool-embodiment data at deployment can thus stand in for collecting new target-robot demonstrations.

Limitations and Future Work. Several constraints point to future work. The target and pool embodiments must share an end-effector action space, since the residual refines a retrieved chunk in a common low-level representation; structurally different action spaces (e.g., a dexterous hand versus a parallel gripper) would require an embodiment-agnostic interface or learned action translator. The pool must also contain trajectories rather than video alone, so video-only sources such as raw human or web video would first need to be lifted into a state-action representation. What constitutes an effective representation for cross-embodiment retrieval also remains an open question; our current descriptor combines language, object pose, proprioception, and visual features, but more scalable or embodiment-invariant representations may further improve transfer.

In addition, the residual formulation becomes less reliable when retrieved motions differ substantially in execution speed or temporal scale, particularly for larger chunks where errors can accumulate over time. Developing retrieval and adaptation mechanisms that remain robust under significant temporal or dynamical mismatch is an important direction for future work. Finally, scaling retrieval beyond curated trajectory pools to in-the-wild video sources such as raw YouTube video remains a promising step toward broadly reusable robot experience.

## References

[1] M. J. Kim, K. Pertsch, S. Karamcheti, T. Xiao, A. Balakrishna, S. Nair, R. Rafailov, E. P. Foster, P. R. Sanketi, Q. Vuong, T. Kollar, B. Burchfiel, R. Tedrake, D. Sadigh, S. Levine, P. Liang, and C. Finn. OpenVLA: An open-source vision-language-action model. In Proc. of the 8th Annual Conference on Robot Learning (CoRL), 2024. URL https://openreview. net/forum?id=ZMnD6QZAE6.

[2] J. Lee, J. Duan, H. Fang, Y. Deng, S. Liu, B. Li, B. Fang, J. Zhang, Y. R. Wang, S. Lee, et al. Molmoact: Action reasoning models that can reason in space. arXiv preprint arXiv:2508.07917, 2025.

[3] Physical Intelligence, K. Black, N. Brown, J. Darpinian, K. Dhabalia, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, et al. π0.5: a vision-language-action model with open-world generalization. arXiv preprint arXiv:2504.16054, 2025.

[4] J. Bjorck et al. Gr00t n: An open foundation model for generalist humanoid robots. arXiv preprint arXiv:2503.14734, 2025.

[5] J. Zheng, J. Li, Z. Wang, D. Liu, X. Kang, Y. Feng, Y. Zheng, J. Zou, Y. Chen, J. Zeng, T. Wang, Y.-Q. Zhang, J. Liu, and X. Zhan. X-VLA: Soft-prompted transformer as scalable cross-embodiment vision-language-action model. In Proc. of the Fourteenth International Conference on Learning Representations (ICLR), 2026. URL https://openreview.net/ forum?id=kt51kZH4aG.

[6] S. Liu, L. Wu, B. Li, H. Tan, H. Chen, Z. Wang, K. Xu, H. Su, and J. Zhu. RDT-1b: a diffusion foundation model for bimanual manipulation. In Proc. of the Thirteenth International Conference on Learning Representations (ICLR), 2025. URL https://openreview.net/ forum?id=yAzN4tz7oI.

[7] M. J. Kim, Y. Gao, T.-Y. Lin, Y.-C. Lin, Y. Ge, G. Lam, P. Liang, S. Song, M.-Y. Liu, C. Finn, and J. Gu. Cosmos policy: Fine-tuning video models for visuomotor control and planning. In Proc. of the Fourteenth International Conference on Learning Representations (ICLR), 2026. URL https://openreview.net/forum?id=wPEIStHxYH.

[8] S. Ye, Y. Ge, K. Zheng, S. Gao, S. Yu, G. Kurian, S. Indupuru, Y. L. Tan, C. Zhu, J. Xiang, et al. World action models are zero-shot policies. arXiv preprint arXiv:2602.15922, 2026.

[9] R. Shah, S. Liu, Q. Wang, Z. Jiang, S. Kumar, M. Seo, R. Mart´ın-Mart´ın, and Y. Zhu. Mimicdroid: In-context learning for humanoid robot manipulation from human play videos. arXiv preprint arXiv:2509.09769, 2025.

[10] C. Wang, L. Fan, J. Sun, R. Zhang, L. Fei-Fei, D. Xu, Y. Zhu, and A. Anandkumar. Mimicplay: Long-horizon imitation learning by watching human play. In Proc. of the 7th Annual Conference on Robot Learning (CoRL), 2023. URL https://openreview.net/forum?id= hRZ1YjDZmTo.

[11] T. Yuan, Z. Dong, Y. Liu, and H. Zhao. Fast-wam: Do world action models need test-time future imagination? arXiv preprint arXiv:2603.16666, 2026. URL https://arxiv.org/ abs/2603.16666.

[12] J. Pai, L. Achenbach, V. Montesinos, B. Forrai, O. Mees, and E. Nava. mimic-video: Videoaction models for generalizable robot control beyond vlas. arXiv preprint arXiv:2512.15692, 2025.

[13] L. Li, Q. Zhang, Y. Luo, S. Yang, R. Wang, F. Han, M. Yu, Z. Gao, N. Xue, X. Zhu, et al. Causal world modeling for robot control. arXiv preprint arXiv:2601.21998, 2026.

[14] S. Wang, J. Shi, Z. Fu, X. He, F. Liu, C. Yang, Y. Zhou, Z. Fei, J. Gong, J. Fu, et al. World action models: The next frontier in embodied ai. arXiv preprint arXiv:2605.12090, 2026.

[15] S. Li, Y. Gao, D. Sadigh, and S. Song. Unified video action model. In Proc. of the Robotics: Science and Systems (RSS), 2025, 2025.

[16] C. Chi, Z. Xu, S. Feng, E. Cousineau, Y. Du, B. Burchfiel, R. Tedrake, and S. Song. Diffusion policy: Visuomotor policy learning via action diffusion. The International Journal of Robotics Research, 44(10-11):1684–1704, 2025.

[17] T. Chen, Z. Chen, B. Chen, Z. Cai, Y. Liu, Z. Li, Q. Liang, X. Lin, Y. Ge, Z. Gu, et al. Robotwin 2.0: A scalable data generator and benchmark with strong domain randomization for robust bimanual robotic manipulation. arXiv preprint arXiv:2506.18088, 2025.

[18] M. Du, S. Nair, D. Sadigh, and C. Finn. Behavior retrieval: Few-shot imitation learning by querying unlabeled datasets. In Proc. of the Robotics: Science and Systems (RSS), 2024, 2024.

[19] L.-H. Lin, Y. Cui, A. Xie, T. Hua, and D. Sadigh. Flowretrieval: Flow-guided data retrieval for few-shot imitation learning. In Proc. of the 8th Annual Conference on Robot Learning (CoRL), 2024. URL https://openreview.net/forum?id=FHnVRmeqxf.

[20] M. Memmel, J. Berg, B. Chen, A. Gupta, and J. Francis. STRAP: Robot sub-trajectory retrieval for augmented policy learning. In Proc. of the Thirteenth International Conference on Learning Representations (ICLR), 2025. URL https://openreview.net/forum?id=4VHiptx7xe.

[21] R. Punamiya, D. Patel, P. Aphiwetsa, P. Kuppili, L. Y. Zhu, S. Kareer, J. Hoffman, and D. Xu. Egobridge: Domain adaptation for generalizable imitation from egocentric human data. In Proc. of the Thirty-ninth Annual Conference on Neural Information Processing Systems (NeurIPS), 2026. URL https://openreview.net/forum?id=FGMBxzpgis.

[22] V. Jain, M. Attarian, N. J. Joshi, A. Wahid, D. Driess, Q. Vuong, P. R. Sanketi, P. Sermanet, S. Welker, C. Chan, I. Gilitschenski, Y. Bisk, and D. Dwibedi. Vid2robot: End-to-end videoconditioned policy learning with cross-attention transformers. In Proc. of the Robotics: Science and Systems (RSS), 2024, 2024.

[23] A. Bahety, P. Mandikal, B. Abbatematteo, and R. Mart´ın-Mart´ın. Screwmimic: Bimanual imitation from human videos with screw space projection. In Proc. of the Robotics: Science and Systems (RSS), 2024, 2024.

[24] M. Hong, A. Liang, K. Kim, H. Rajaprakash, J. Thomason, E. Bıyık, and J. Zhang. Hand me the data: Fast robot adaptation via hand path retrieval. arXiv preprint arXiv:2505.20455, 2025.

[25] G. Papagiannis, N. Di Palo, P. Vitiello, and E. Johns. R+ x: Retrieval and execution from everyday human videos. In Proc. of the 2025 IEEE International Conference on Robotics and Automation (ICRA), pages 8284–8290. IEEE, 2025.

[26] S. Sha, Y. Wang, B. Huang, A. Loquercio, and Y. Li. Efficient and reliable teleoperation through real-to-sim-to-real shared autonomy. arXiv preprint arXiv:2603.17016, 2026.

[27] C. Schaff and M. R. Walter. Residual policy learning for shared autonomy. In Proc. of the Robotics: Science and Systems (RSS), 2020, 2020.

[28] N. Carion, L. Gustafson, Y.-T. Hu, S. Debnath, R. Hu, D. Suris, C. Ryali, K. V. Alwala, H. Khedr, A. Huang, J. Lei, T. Ma, B. Guo, A. Kalla, M. Marks, J. Greer, M. Wang, P. Sun, R. Radle, T. Afouras, E. Mavroudi, K. Xu, T.-H. Wu, Y. Zhou, L. Momeni, R. Hazra, S. Ding, ¨ S. Vaze, F. Porcher, F. Li, S. Li, A. Kamath, H. K. Cheng, P. Dollar, N. Ravi, K. Saenko, ´ P. Zhang, and C. Feichtenhofer. Sam 3: Segment anything with concepts, 2025. URL https: //arxiv.org/abs/2511.16719.

[29] O. Simeoni, H. V. Vo, M. Seitzer, F. Baldassarre, M. Oquab, C. Jose, V. Khalidov, ´ M. Szafraniec, S. Yi, M. Ramamonjisoa, et al. Dinov3. arXiv preprint arXiv:2508.10104, 2025.

[30] R. Cadene, S. Alibert, A. Soare, Q. Gallouedec, A. Zouitine, S. Palma, P. Kooijmans, M. Aractingi, M. Shukor, D. Aubakirova, M. Russi, F. Capuano, C. Pascal, J. Choghari, J. Moss, and T. Wolf. Lerobot: State-of-the-art machine learning for real-world robotics in pytorch. https://github.com/huggingface/lerobot, 2024.

[31] T. Z. Zhao, V. Kumar, S. Levine, and C. Finn. Learning fine-grained bimanual manipulation with low-cost hardware. arXiv preprint arXiv:2304.13705, 2023.

## Appendix

This appendix collects the additional results, analyses, and implementation details that support the main paper. Section A reports the full PushT baseline comparison across all nine goal angles and the complete $2 \times 2$ action-parameterization × next-state ablation. Section B expands the decoder analysis of Section 5.2, giving the full attention probe, the no-retrieval comparison, and the causal masking interventions that the main text only summarizes. Section C builds on that same analysis to examine the failure rollouts, separating them from matched successes and relating the attention differences to behavioral failure clusters. Section D details the RoboTwin 2.0 train and test tasks, the excluded query episodes, and the progressive retrieval pool. Section E spells out the full two-stage retrieval rule, trajectory prefilter then subframe scoring at training and inference, that the main text abbreviates. Section F lists all backbone, retrieval, and training hyperparameters.

## A Additional PushT Results

Table 2: PushT generalization across goal angles (success rate, %, higher is better). Training covers only $\pm 4 5 ^ { \circ }$ (shaded Seen columns); the other seven angles are unseen at training time. Bold marks the per-column best.
<table><tr><td></td><td colspan="9">Goal angle ()</td><td colspan="2"> $\mathbf { A v g } \uparrow$ </td></tr><tr><td>Method</td><td></td><td>-60-45-30</td><td></td><td>-15</td><td>0</td><td>+15</td><td>+30</td><td>+45</td><td>+60</td><td>Seen</td><td>Unseen</td></tr><tr><td>Cosmos Policy</td><td>6.0</td><td>20.0</td><td>0.0</td><td>0.0</td><td>2.0</td><td>2.0</td><td>14.0</td><td>46.0</td><td>18.0</td><td>33.0</td><td>6.0</td></tr><tr><td>Retrieval Only</td><td>24.0</td><td>10.0</td><td>16.0</td><td>10.0</td><td>14.0</td><td>12.0</td><td>18.0</td><td>22.0</td><td>24.0</td><td>16.0</td><td>16.9</td></tr><tr><td>Co-train (all)</td><td></td><td>16.022.0</td><td>20.0</td><td>18.0</td><td>20.0</td><td>20.0</td><td>20.0</td><td>24.0</td><td>20.0</td><td>23.0</td><td>19.1</td></tr><tr><td>Ours</td><td>28.0</td><td>40.0</td><td>44.0</td><td>26.0</td><td>28.0</td><td>36.0</td><td>48.0</td><td>60.0</td><td>34.0</td><td>50.0</td><td>34.9</td></tr></table>

Comparison with prior cross-embodiment recipes. Table 2 reports a comparison against three cross-embodiment baselines that share our backbone but differ in how they incorporate the discpusher (pool) data. Cosmos Policy is trained on the triangle (target) demonstrations alone, with no access to the pool. Retrieval Only executes the action sequence of the nearest pool demonstration without learning. Co-train (all) jointly trains a single policy on the union of target and pool trajectories. At the full retrieval pool, our method generalizes beyond the $\pm 4 5 ^ { \circ }$ training band and achieves the best success on both splits, 34.9% on unseen angles and 50.0% on seen. Without retrieval the same backbone (Cosmos Policy) reaches only 6.0% on unseen; treating the pool as undifferentiated training data (Co-train) is the strongest baseline at 19.1% unseen but still well short of ours, indicating that pool data is necessary but not sufficient.

Table 3: Full action-parameterization × next-state ablation on PushT (avg unseen-angle success, %). Rows toggle the auxiliary next-state objective; columns toggle the action parameterization. The last row reports the non-retrieval baseline at the same backbone.
<table><tr><td>Next-state pred.</td><td>Absolute</td><td>Residual</td><td> $\Delta _ { \mathrm { R e s - A b s } }$ </td></tr><tr><td>×</td><td>26.3</td><td>27.4</td><td>+1.1</td></tr><tr><td></td><td>27.4</td><td>34.9</td><td>+7.4</td></tr><tr><td> $\Delta _ { \checkmark - \times }$ </td><td>+1.1</td><td>+7.4</td><td></td></tr><tr><td colspan="4">No-retrieval baseline:  $8 . 3  6 . 0$   $( \Delta _ { \checkmark } - \times = - 2 . 3 )$ </td></tr></table>

Full action-parameterization × next-state ablation. Fig. 5b in the main body ablates the nextstate objective while fixing the action parameterization to residual. For completeness, Table 3 reports the full $2 \times 2$ ablation crossing action parameterization (absolute vs. residual over the retrieved trajectory) with the auxiliary next-state objective, together with the non-retrieval baseline. Residual outperforms absolute at both settings of the next-state objective. Next-state prediction has negligible effect under absolute parameterization (+1.1 on unseen) but contributes +7.4 under residual; the same auxiliary loss is detrimental in the non-retrieval regime (−2.3). Together, these results indicate that the benefit of next-state supervision is specific to the WAM × retrieval interaction.

Table 4: Decoder ROI peaks at L10 (intake) and L15 (commit). Action-slot attention to each input group, pooled over the 50 unseen-angle rollouts. Ratio > 1 means concentration above a uniform spatial baseline. “ret-1” / “ret-end” denote the first and last frame of the retrieved chunk.
<table><tr><td>Slot</td><td>Dominant ROI</td><td>Layer</td><td>Ratio</td></tr><tr><td>Primary view</td><td>T-block</td><td>L10</td><td>2.1</td></tr><tr><td>Retrieved frame, ret-1</td><td>T-block</td><td>L10</td><td>5.1</td></tr><tr><td>Retrieved frame,ret-end</td><td>T-block (next)</td><td>L10</td><td>5.4</td></tr><tr><td>Primary view</td><td>Predicted end</td><td>L15</td><td>3.9</td></tr><tr><td>Retrieved frame,ret-end</td><td>Predicted end</td><td>L15</td><td>3.0</td></tr><tr><td>Retrieved frame, ret-1</td><td>(release)</td><td>L15</td><td>~1.0</td></tr></table>

## B PushT Mechanism Analysis

A retrieval-conditioned policy turns a retrieved chunk into an action by routing its decoder computation along two attention axes, and this section establishes those axes, contrasts them against the no-retrieval baseline, and verifies them causally. The first axis is intake at layer 10 (L10), where the action slot reads in the retrieved task region, namely the current and goal T-block poses carried by the retrieved chunk. The second is commit at layer 15 (L15), where the slot turns to the policy’s own predicted end-of-chunk pose and settles on the action it will execute. We first define the attention probe and its ROI-ratio metric (Section 5.2 introduced the summary view); we then show that these two peaks are the only structured attention in the decoder (Table 4), that they disappear when the same backbone is trained without retrieval, and that masking either peak degrades success. Section C carries the same two axes into the failure regime and shows how each one breaks down.

Probe protocol. We localize what the action slot looks at by reading decoder cross-attention and normalizing it against a uniform spatial baseline, so that a value above one means the slot concentrates on a region rather than spreading evenly. We probe the (residual, next-state-on) PushT policy at iteration 7000 across the nine-angle grid of Table 2. For each rollout we record decoder crossattention from the action slot to four input groups: the primary view, the retrieved chunk frames (its first and last frame, denoted “ret-1” and “ret-end”), the retrieved proprioception, and the retrieved action. Three image regions of interest (ROIs) are defined on the primary view and on each retrieved frame: T BLOCK (the T-block bounding box at the current step), T BLOCK NEXT (the T-block at the step the retrieved chunk ends at), and PREDICTED END (the policy’s predicted end-of-chunk pose, projected back into the image). For a given (slot, layer), let $A _ { p }$ denote the softmax-normalized attention weight on image patch $\begin{array} { r } { p \left( \operatorname { s o } \sum _ { p \in I } A _ { p } = 1 \right) } \end{array}$ , and let $R \subset I$ be the patches inside a given ROI. We define the ROI ratio as the mean attention inside R normalized by the per-patch uniform baseline:

$$
{ \mathrm { R O I ~ r a t i o } } = { \frac { { \frac { 1 } { | R | } } \sum _ { p \in R } A _ { p } } { { \frac { 1 } { | I | } } \sum _ { p \in I } A _ { p } } } = { \frac { | I | } { | R | } } \sum _ { p \in R } A _ { p } .\tag{4}
$$

A ratio of 1 corresponds to uniform attention; a ratio of k means the action slot attends to that ROI k times more strongly than uniform. The analysis uses three data subsets: (i) 50 rollouts (5 success + 5 failure × 5 angles) for the ours-vs-baseline comparison, (ii) 70 rollouts (5 success + 5 failure × 7 angles) for failure analysis, and (iii) 21 ablation runs (7 angles × 3 interventions).

Two-stage routing: L10 intake, L15 commit. The decoder concentrates the action slot’s attention at exactly two layers, and these two peaks are precisely the intake and commit axes (Table 4, traced along a representative rollout in Fig. 10). At L10, the slot attends to the task region carried by the retrieved chunk: it places 5.1× uniform attention on the current T-block in the first retrieved frame (ret-1) and 5.4× on the T-block pose at the end of the retrieved chunk (ret-end). It thus reads the retrieved trajectory temporally, with the earlier frame anchoring where the block is now and the later frame anchoring where the retrieved plan will take it. At L15, the slot drops the retrieved frames and instead concentrates on the policy’s own predicted end-of-chunk pose, at 3.9× uniform from the primary view and 3.0× from ret-end, committing to the action it will execute rather than continuing to read evidence. The surrounding layers do no selective routing: encoder-side layers (L0–L5) and late layers (L20–L25) stay near uniform (ratio ≈ 1.0), so intake and commit are two sharp, wellseparated stages rather than a gradual blend. The policy therefore first reads the retrieved plan at L10 and then commits to its own correction of it at L15, which is the decoder-level signature of the residual-over-retrieval behavior.

![](images/2f7665cb86d5cf7ac81f32974ac95e7eaa764096d9041b050ed35c61cba8339f.jpg)  
Figure 10: Action-slot attention along a representative rollout. Top row: scene at six timesteps. Subsequent rows: action-slot attention overlaid on the primary view at L10 (intake) and L15 (commit), and on the retrieved frame at L10. The primary L10 row attends to the current T-block; the primary L15 row shifts to the predicted end-of-chunk pose; the retrieved L10 row attends to the retrieved chunk’s T-block region, consistent with the per-layer summary in Table 4.

Table 5: Ours vs no-retrieval baseline at the same backbone. Peak ROI ratios pooled over the five unseen angles common to both runs. The no-retrieval baseline is the Cosmos Policy row of Table 2; ROIs, layers, and rollout subsets are defined identically.
<table><tr><td>Metric</td><td>Ours</td><td>Baseline (no retrieval)</td></tr><tr><td>L15 predicted-end peak</td><td>~3.8 (sharp)</td><td>~1.0 (fat)</td></tr><tr><td>L10 task-ROI peak</td><td>2.0-2.5</td><td>1.0-1.2</td></tr><tr><td>L10- →L15 transition</td><td>present</td><td>absent</td></tr></table>

Both axes require retrieval. Neither axis appears when the same backbone is trained without retrieval, so the two-stage routing is a product of retrieval-augmented training rather than of the video backbone itself. Matched on the five unseen angles common to both runs (Table 5), the no-retrieval Cosmos Policy keeps every ROI ratio near 1.0 at every decoder layer: the sharp L15 predicted-end peak (≈ 3.8 for ours) flattens to ≈ 1.0, the L10 task peak (2.0–2.5) flattens to 1.0–1.2, and the L10→L15 transition is absent altogether. Because the backbone, pretraining, and architecture are identical across the two runs, the only change that introduces the peaks is conditioning on retrieved chunks. The functional contribution of retrieval is therefore not merely to supply extra context but to induce the L15 commitment step in which the policy acts on its own prediction.

Table 6: Causal ablation across seven unseen angles. “L10 block” masks the action-to-retrieval cross-attention at decoder layer 10 (similarly for L15); applied separately to the five-success rollouts (5S) and the five-failure rollouts (5F) at each angle. The baseline column reproduces the unseenangle success of Table 2.
<table><tr><td>Angle(°）</td><td>Baseline</td><td>L10 block, 5S</td><td>L15 block, 5S</td><td>L10 block,5F (rescue)</td></tr><tr><td>-60</td><td>28%</td><td> $1 0 0 \to 6 0 \% ( - 4 0 )$ </td><td>100→60%(-40)</td><td>0→40%</td></tr><tr><td>-30</td><td>44%</td><td> $1 0 0 \to 8 0 \% ( - 2 0 )$ </td><td>100→80%(-20)</td><td>0→20%</td></tr><tr><td>-15</td><td>26%</td><td> $1 0 0 \to 8 0 \% ( - 2 0 )$ </td><td> $1 0 0 \to 4 0 \% ( - 6 0 )$ </td><td>0→20%</td></tr><tr><td>0</td><td>28%</td><td> $1 0 0 \to 2 0 \% ( - 8 0 )$ </td><td> $1 0 0 \to 4 0 \% ( - 6 0 )$ </td><td>0→20%</td></tr><tr><td>+15</td><td>36%</td><td> $1 0 0 \to 6 0 \% ( - 4 0 )$ </td><td> $1 0 0 \to 8 0 \% ( - 2 0 )$ </td><td>0→0%</td></tr><tr><td>+30</td><td>48%</td><td> $1 0 0 \to 8 0 \% ( - 2 0 )$ </td><td> $1 0 0 \to 6 0 \% ( - 4 0 )$ </td><td>0→20%</td></tr><tr><td>+60</td><td>34%</td><td> $1 0 0 \to 4 0 \% ( - 6 0 )$ </td><td> $1 0 0 \to 4 0 \% ( - 6 0 )$ </td><td>0→40%</td></tr></table>

![](images/1049ec8ca49d48e874152c8a679a11ba32db02bd4adfa0b4c5e1be4b587d4eca.jpg)

![](images/8383ae45b998687096bc1a889cf573fd30d63cbfee957ef0bc9945f7bb268c7f.jpg)  
Figure 11: Two failure modes across all seven unseen angles. (a) L15 own-prediction commit, success vs failure: success bars exceed failure bars at every angle, so weakened L15 commit is a universal failure signature, with the strongest weakening at 0◦. (b) L10 retrieval-intake gap, failure − success: angles close to the seen training band $( \pm 3 0 ^ { \circ } , \dot { + } 1 5 ^ { \circ } )$ show negative gaps (under-anchoring: failures under-use retrieval), while angles far from it $( 0 ^ { \circ } , \pm 6 0 ^ { \circ } )$ show positive gaps (over-anchoring: failures over-rely on retrieval); $0 ^ { \circ }$ is the heaviest over-anchoring.

Both axes are causally necessary. Masking the action-to-retrieval cross-attention at a single decoder layer confirms that both axes carry the behavior causally rather than merely correlating with it (Table 6). We set the action→{ret-1, ret-end, retrieved proprio, retrieved action} attention to −∞ before softmax, independently at L10 and at L15, across all seven angles. Blocking L10 intake on the five-success rollouts drops success by 20–80 percentage points, so reading the retrieved plan is required to succeed; blocking L15 commit drops it by 20–60 pp, so retrieved evidence alone is not enough without the commitment step. The same L10 mask is equally telling on failures: applied to the five-failure rollouts it recovers nontrivial success on 6 of 7 angles, from 0% to 20–40%, with the largest rescue at $0 ^ { \circ }$ and $\pm 6 0 ^ { \circ }$ , the very angles that over-anchor on retrieval (Table 8). Excessive L10 intake is therefore not a benign side effect but a direct cause of failure, which sets up the failure analysis in Section C.

## C PushT Failure Case Analysis

This section follows the two axes from Section B into the failure regime and finds that each one breaks in its own way: L15 commit weakens on essentially every failure, whereas L10 intake is distorted in a direction that depends on the goal angle. For each unseen angle we split the ten probe rollouts into a five-success and a five-failure subset and compare the axis-specific attention measurement between them (Fig. 11), which isolates one failure signature per axis and keeps the diagnosis aligned with the mechanism established above.

Table 7: L15 commit weakening on failures. Action-slot attention from the last retrieved frame to the policy’s predicted end-of-chunk pose at L15, mean over 5 episodes per cell.
<table><tr><td>Angle (°)</td><td>Success</td><td>Failure</td><td>△ (succ - fail)</td></tr><tr><td>-60</td><td>3.12</td><td>2.43</td><td>+0.69</td></tr><tr><td>-30</td><td>2.63</td><td>1.85</td><td>+0.78</td></tr><tr><td>-15</td><td>2.95</td><td>2.09</td><td>+0.86</td></tr><tr><td>0</td><td>2.99</td><td>1.97</td><td>+1.01</td></tr><tr><td>+15</td><td>2.92</td><td>2.50</td><td>+0.42</td></tr><tr><td>+30</td><td>3.01</td><td>2.87</td><td>+0.14</td></tr><tr><td>+60</td><td>3.41</td><td>3.28</td><td>+0.14</td></tr></table>

Table 8: L10 anchoring spectrum on failures. Action-slot attention from the first retrieved frame to the current T-block region at L10, mean over 5 episodes per cell. Negative $\Delta$ (fail − succ) is under-anchoring; positive is over-anchoring.
<table><tr><td>Angle(°)</td><td>Success</td><td>Failure</td><td> $\Delta { \ : } ( \mathrm { f a i l - s u c c } )$ </td><td>Zone</td></tr><tr><td>+15</td><td>5.33</td><td>4.85</td><td>-0.47</td><td>Under</td></tr><tr><td>-30</td><td>5.58</td><td>5.19</td><td>-0.39</td><td>Under</td></tr><tr><td>+30</td><td>5.73</td><td>5.58</td><td>-0.15</td><td>Under</td></tr><tr><td>-15</td><td>4.89</td><td>5.13</td><td>+0.24</td><td>Mild over</td></tr><tr><td>-60</td><td>5.02</td><td>6.03</td><td>+1.02</td><td>Moderate over</td></tr><tr><td>+60</td><td>4.32</td><td>5.61</td><td>+1.28</td><td>Moderate over</td></tr><tr><td>0</td><td>4.32</td><td>6.62</td><td>+2.31</td><td>Heavy over</td></tr></table>

Axis 1 (L15 commit): weakened on every failure. The L15 commit axis weakens on failure at all seven unseen angles, which makes it the universal, condition-independent failure signature. At every angle the action slot attends less to its predicted end-of-chunk pose on failure than on matched success rollouts (Fig. 11a, Table 7): the success-minus-failure gap is positive everywhere, growing from +0.14 at the easy $+ 3 0 ^ { \circ }$ and $+ 6 0 ^ { \circ }$ angles $\mathrm { t o } + 1 . 0 1$ at the hardest angle, $0 ^ { \circ }$ . Because L15 commit is the step at which the policy converts retrieved evidence and the current observation into the action it will execute, a weaker L15 signal means the policy has failed to commit to a correct action under that unseen condition, independently of how well it read the retrieval at L10. Crucially, the sign of the effect never flips across angles, so L15 commit behaves as a single axis that can be addressed condition-independently.

Axis 2 (L10 intake): an under-/over-anchoring spectrum. The L10 intake axis fails in opposite directions depending on how far the goal angle sits from the seen training band, so unlike L15 it is condition-specific in both magnitude and sign. Near a seen angle $( \pm 3 0 ^ { \circ } , + 1 5 ^ { \circ } )$ , failures underanchor: they take in less retrieval than matched successes (gaps $\mathrm { o f ~ - 0 . 1 5 ~ t o ~ - 0 . 4 7 }$ in Table 8, Fig. 11b), because the rollout resembles a condition the policy already handles and the failure mode is to neglect the retrieved evidence. Far from the seen band $( 0 ^ { \circ } , \pm 6 0 ^ { \circ } )$ , failures over-anchor: they take in more retrieval (gaps up $\mathrm { { t o \ t 2 . 3 1 \ a t \ 0 ^ { \circ } } }$ , over twice the next largest), because the unfamiliar condition needs the retrieved plan more and the failure mode is to over-rely on it, staying at intake and never committing at L15. The crossover from under- to over-anchoring tracks distance from $\pm 4 5 ^ { \circ }$ monotonically, so a single axis explains both ends of the spectrum, with $0 ^ { \circ }$ as the extreme over-anchoring case and the same angle at which L15 commit is weakest.

A behavior-level split mirrors the attention axes. Independently of attention, every failure rollout falls into one of two behavioral clusters, and the dominant one is the behavioral counterpart of the attention failures above. We label rollouts by coverage, the fraction of T-block area moved toward the goal: over-anchored failures $( \mathrm { c o v e r a g e } \geq 0 . 1 0 )$ move the block but never complete the push, while non-engaging failures (coverage $< 0 . 1 0 )$ leave the policy largely inactive. Across the seven unseen angles the split is 25 over-anchored to 10 non-engaging (71% vs. 29%); both clusters occur at every angle, and over-anchored dominates except at $- 3 0 ^ { \circ } \ ( 2 \ \mathrm { v s . \ 3 ) }$ . The over-anchored cluster is what L10 over-anchoring and weakened L15 commit look like in behavior, namely a policy that keeps adjusting but never decisively executes, whereas the non-engaging cluster reflects a degenerate collapse of the policy’s own prediction that reshaping retrieval attention cannot fix.

Implications for future work. The two axes call for different fixes because one fails universally and the other fails condition-specifically. L15 commit weakens in the same direction at every angle, so it can be targeted directly, for example by supervising the action-to-predicted-end attention at training time to strengthen commitment under unseen conditions. L10 intake fails in opposite directions across angles, so it instead needs adaptive control, for example an attention temperature or gate that curbs over-anchoring at hard angles while preserving engagement at easy ones. Neither remedy touches the non-engaging cluster, which lies outside both axes and is better treated as a separate robustness problem for the policy’s own-prediction pathway.

## D RoboTwin Setup Details

Train and test tasks. The policy is fine-tuned on five Aloha-Agilex (target) tasks paired with UR5 (pool) retrievals: Place Cans Plasticbox (50 paired episodes), Move Can Pot (48), Open Microwave (49), Grab Roller (49), and Pick Dual Bottles (50), for 246 paired training episodes in total. Crosstask evaluation uses five held-out tasks unseen during fine-tuning: Move Pillbottle Pad, Lift Pot, Click Bell, Hand-over Mic, and Place Bread Skillet. A representative paired observation is shown in Fig. 6 of the main body.

Excluded query episodes. A small number of query episodes are removed from the training set because their cross-embodiment retrieval is catastrophically mis-aligned and would inject systematically wrong supervision: Move Can Pot episodes 32 and 45 (mid-task rotation mismatch with visual and quaternion-cosine outliers), Open Microwave episode 34 (75-frame plateau with the retrieval stuck), and Grab Roller episode 38 (quaternion hemisphere wrap-around).

Progressive retrieval pool. The test-time pool-progression eval grows the retrievable pool through five strictly nested levels, from 11 to 35 UR5 tasks, with the policy frozen throughout (Table 9). The base of every level is the five training tasks (under the ur5 clean 50 and ur5 randomized 500 splits); each level then appends six further UR5 tasks in a fixed order, and exactly one of the five held-out evaluation tasks (in bold) becomes retrievable at each level, which is what drives the perlevel success increase in Fig. 7.

Retrieval scoring weights. Proprioception is encoded as a sign-flip-invariant 20-D representation (xyz position + 6-D rotation + gripper, with sign-flips on the rotation hemisphere collapsed). Position is up-weighted $( w _ { \mathrm { p o s } } = 4 . 0 )$ against the 6-D rotation component. The subframe-matching cost weights are: proprio 1.0, proprio-history (8-step window) 0.05, action chunk 0.1, delta-action 0.1, image 0.1, time 0.0. The trajectory-level prefilter keeps the top-5 candidate trajectories by initial-scene similarity with a SAM3 width-height term of weight 1.0.

## E Retrieval Rule Details

The main text (Section 4.2) states the retrieval rule in abbreviated form. This section gives the full procedure. Recall from Section 3 that at each control step t the policy selects a pool subframe t′ whose surrounding chunk conditions the prediction. Searching every subframe of every pool trajectory under the full feature distance at every step does not scale, so we factor the rule into two stages: a trajectory-level prefilter (Stage 1) that narrows $\mathcal { D } ^ { \mathrm { p o o l } }$ to a small candidate set, and a subframe-level match (Stage 2) that picks the moment within those candidates whose local state best matches the query. The same two stages run at training and at deployment; only the active pool Dpool and the subframe cost differ between the two, as detailed below.

Table 9: Progressive retrieval pool composition (RoboTwin). Each level is a strict superset of the previous one and appends six UR5 tasks; the held-out evaluation task that becomes retrievable at that level is in bold. The base is the five training tasks under the ur5 clean 50 and ur5 randomized 500 splits.
<table><tr><td>Level (size)</td><td>UR5 tasks added (held-out eval task in bold)</td></tr><tr><td>Base (5)</td><td>Place Cans Plasticbox,Move Can Pot, Open Microwave,Grab Roller, Pick Dual Bottles</td></tr><tr><td>Pool 2 (11)</td><td>Lift Pot, Place Bread Basket,Beat Block Hammer, Move Stapler Pad, Press Sta- pler, Click Alarmclock</td></tr><tr><td>Pool 3 (17)</td><td>Hand-over Mic,Dump Bin Bigbin,Adjust Bottle,Open Laptop,Place Mouse Pad, Shake Bottle</td></tr><tr><td>Pool 4 (23)</td><td>Move Pillbotte Pad, Place Burger Fries, Place Phone Stand,Place Dual Shoes, Place Object Basket,Move Playingcard Away</td></tr><tr><td>Pool 5 (29)</td><td>Place Bread Skillet, Place Container Plate,Place Fan, Place Object Scale, Hang- ing Mug,Place Can Basket</td></tr><tr><td>Pool 6 (35)</td><td>Click Bell, Stack Bowls,Hand-over Block,Turn Switch, Stamp Seal,Place Empty Cup,Rotate QR Code</td></tr></table>

Stage 1: initial-scene trajectory retrieval. The first stage narrows the pool to trajectories whose initial scene matches the query’s. We compare only the first frame of each pool trajectory $\tau \in \mathcal { D } ^ { \mathrm { p o o l } }$ against the query’s first frame on three signals: a language embedding of the goal instruction, the initial positions of task-relevant objects extracted via SAM3 [28], and the initial proprioception. Writing $\psi _ { 0 } ( \cdot )$ for this composite initial-scene descriptor, we keep the top-K closest trajectories,

$$
\mathcal { C } _ { t } ^ { \mathrm { t r a j } } = \underset { \tau \in \mathcal { D } ^ { \mathrm { p o o l } } } { \mathrm { T o p - K } } \Big ( - \big \| \psi _ { 0 } ( \mathrm { q u e r y } ) - \psi _ { 0 } ( \tau ) \big \| ^ { 2 } \Big ) ,\tag{5}
$$

where Top-K returns the K trajectories with the smallest distance. This stage plays two roles. It speeds up inference, since Stage 2 runs at every control step and shrinking its search set from the entire pool to $K$ trajectories is what makes step-wise retrieval feasible on large pools. It also improves precision, since trajectories with the wrong instruction, object layout, or initial arm pose are filtered out before any subframe is scored, so Stage 2 does not spend its budget comparing the query against semantically irrelevant candidates whose local frames may happen to look similar by accident.

Stage 2: subframe retrieval (training). Within $\mathcal { C } _ { t } ^ { \mathrm { t r a j } }$ , we locate the subframe whose local state best matches the query’s. A subframe is described by four heterogeneous features: the task-relevant object pose $\phi _ { \mathrm { o b j } }$ (T-block position in PushT, SAM3-anchored object poses in RoboTwin), a DINO image feature $\phi _ { \mathrm { v i s } }$ [29] of the current scene, the proprioception $\phi _ { \mathrm { p r o p } } .$ , and (at training only) the upcoming action chunk $\phi _ { \mathrm { a c t } }$ . These features live on different scales and admit different natural distances, so we score them with a weighted combination rather than a single Euclidean norm,

$$
\begin{array} { r l } { d _ { \mathrm { t r } } \big ( t , t ^ { \prime } \big ) } & { = w _ { \mathrm { o b j } } d _ { L _ { 2 } } \big ( \phi _ { \mathrm { o b j } } ^ { ( t ) } , \phi _ { \mathrm { o b j } } ^ { ( t ^ { \prime } ) } \big ) + w _ { \mathrm { p r o p } } d _ { L _ { 2 } } \big ( \phi _ { \mathrm { p r o p } } ^ { ( t ) } , \phi _ { \mathrm { p r o p } } ^ { ( t ^ { \prime } ) } \big ) } \\ & { + w _ { \mathrm { v i s } } d _ { \mathrm { c o s } } \big ( \phi _ { \mathrm { v i s } } ^ { ( t ) } , \phi _ { \mathrm { v i s } } ^ { ( t ^ { \prime } ) } \big ) + w _ { \mathrm { a c t } } d _ { L _ { 2 } } \big ( \phi _ { \mathrm { a c t } } ^ { ( t ) } , \phi _ { \mathrm { a c t } } ^ { ( t ^ { \prime } ) } \big ) , } \end{array}\tag{6}
$$

where $\phi _ { \bullet } ^ { ( t ) }$ abbreviates the corresponding feature of the query subframe at time t (analogously for the pool side at $t ^ { \prime } )$ . The retrieved index is

$$
t ^ { \prime } = \arg \operatorname* { m i n } _ { t ^ { \prime } } ~ d _ { \mathrm { t r } } ( t , t ^ { \prime } ) , \qquad ( s _ { t ^ { \prime } } ^ { \mathrm { p o o l } } , a _ { t ^ { \prime } } ^ { \mathrm { p o o l } } ) \in \mathcal { C } _ { t } ^ { \mathrm { t r a j } } ,\tag{7}
$$

and the chunk $( s _ { t ^ { \prime } : t ^ { \prime } + H } ^ { \mathrm { p o o l } } , a _ { t ^ { \prime } : t ^ { \prime } + H - 1 } ^ { \mathrm { p o o l } } )$ starting at $t ^ { \prime }$ becomes the retrieval conditioning of Eq. (1).

Distance choices. We use squared $L _ { 2 }$ distance $d _ { L _ { 2 } }$ for the low-dimensional geometric features (object pose $\phi _ { \mathrm { o b j } } ,$ proprioception $\phi _ { \mathrm { p r o p } } ,$ action chunk $\phi _ { \mathrm { a c t } } )$ and cosine distance $d _ { \mathrm { c o s } }$ for the highdimensional DINO embedding $\phi _ { \mathrm { v i s } }$ . The visual weight $w _ { \mathrm { v i s } }$ is kept small in practice, so DINO acts as a soft visual sanity check rather than dominating the geometric components, which carry the metric signal the retrieved chunk is meant to align with.

Algorithm 1 Test-time inference with retrieval (one task episode).   
Require: trained WAM $\pi _ { \boldsymbol { \theta } } ;$ pool $\mathcal { D } _ { \mathrm { t e s t } } ^ { \mathrm { p o o l } }$ pre-indexed for trajectory retrieval (Eq. (5)); per-component   
features $( \phi _ { \mathrm { o b j } } , \phi _ { \mathrm { p r o p } } , \phi _ { \mathrm { v i s } } ) ;$ ; horizon H; action stride $K \leq H .$   
1: $t \gets 0$   
2: $\mathcal { C } ^ { \mathrm { t r a j } } \gets$ Top-K trajectories in $\mathcal { D } _ { \mathrm { t e s t } } ^ { \mathrm { p o o l } }$ under $\psi _ { 0 }$ distance to the query (Eq. (5))   
3: while episode not terminated do   
4: observe query state $s _ { t } ^ { \mathrm { q u e r y } }$   
5: subframe retrieval: t′ ← arg mint′ $d _ { \mathrm { i n f } } ( t , t ^ { \prime } )$ over $\mathcal { C } ^ { \mathrm { t r a j } }$ ▷ inference cost Eq. (8)   
6: fetch retrieved chunk $\mathbf { r } _ { t } \gets ( s _ { t ^ { \prime } : t ^ { \prime } + H } ^ { \mathrm { p o o l } } , a _ { t ^ { \prime } : t ^ { \prime } + H - 1 } ^ { \mathrm { p o o l } } )$   
7: predict: $\hat { a } _ { t : t + H - 1 } ^ { \mathrm { q u e r y } } \gets \pi _ { \theta } \big ( s _ { t } ^ { \mathrm { q u e r y } } , \ : \dot { \mathbf { r } } _ { t } \big )$ ▷ residual: $\hat { a } = a ^ { \mathrm { p o o l } } + \Delta a$ (Eq. (3))   
8: execute first K actions $\hat { a } _ { t : t + K - 1 } ^ { \mathrm { q u e r y } }$   
9: $t \gets t + K$   
10: end while

Role of the action term at training. The action term $\phi _ { \mathrm { a c t } }$ is the one feature available at training but not at deployment, so its role is worth stating explicitly. Including it in Eq. (6) means the retrieved chunk matches not only the local state but also what the demonstration was about to do, which gives the WAM a tightly aligned training signal: the retrieved chunk is what a trajectory close to the query’s own actually executed next. The residual in Eq. (3) therefore has to encode only a small embodiment-specific correction, and the next-state objective supplies the dense signal to learn it.

Stage 2: subframe retrieval (inference). At deployment the future query action $a _ { t : t + H - 1 } ^ { \mathrm { q u e r y } }$ is unknown, since it is exactly what the WAM is predicting, so the subframe cost drops the action term while keeping the Stage 1 trajectory filter $\operatorname { E q . } \left( 5 \right)$ unchanged,

$$
d _ { \mathrm { i n f } } ( t , t ^ { \prime } ) = w _ { \mathrm { o b j } } d _ { L _ { 2 } } \big ( \phi _ { \mathrm { o b j } } ^ { ( t ) } , \phi _ { \mathrm { o b j } } ^ { ( t ^ { \prime } ) } \big ) + w _ { \mathrm { p r o p } } d _ { L _ { 2 } } \big ( \phi _ { \mathrm { p r o p } } ^ { ( t ) } , \phi _ { \mathrm { p r o p } } ^ { ( t ^ { \prime } ) } \big ) + w _ { \mathrm { v i s } } d _ { \mathrm { c o s } } \big ( \phi _ { \mathrm { v i s } } ^ { ( t ) } , \phi _ { \mathrm { v i s } } ^ { ( t ^ { \prime } ) } \big ) ,\tag{8}
$$

with the retrieved index $t ^ { \prime } = \arg \operatorname* { m i n } _ { t ^ { \prime } } d _ { \operatorname* { i n f } } ( t , t ^ { \prime } )$ over $\mathcal { C } _ { t } ^ { \mathrm { t r a j } }$ . The model is trained against the richer state-and-action match of Eq. (6), so it absorbs the train-to-inference mismatch into the action latents rather than requiring the inference-time match to reproduce the action term.

Pool extension at deployment. The operating mode this rule supports is that, at deployment, the practitioner may bring new pool-embodiment demonstrations for tasks the model has never seen during training. These demonstrations replace the active pool, ${ \mathcal { D } } ^ { \mathrm { p o o l } }  { \mathcal { D } } _ { \mathrm { t e s t } } ^ { \mathrm { p o o l } }$ , and are reindexed under $\psi _ { 0 }$ for Stage 1 Eq. (5) and under $( \phi _ { \mathrm { o b j } } , \phi _ { \mathrm { p r o p } } , \phi _ { \mathrm { v i s } } )$ for Stage $2 \operatorname { E q . } \left( 8 \right)$ . This pool extension touches no model parameters, so new tasks are added by indexing new data, not by retraining (the property tested in Figs. 4 and 7). Because both stages re-run against the current index, the practitioner can also grow $\mathcal { \bar { D } } _ { \mathrm { t e s t } } ^ { \mathrm { { p o o l } } }$ within a session and see coverage expand without restart.

Per-feature weights. The feature weights $( w _ { \mathrm { o b j } } , w _ { \mathrm { p r o p } } , w _ { \mathrm { v i s } } , w _ { \mathrm { a c t } } )$ and the proprioception encoding are benchmark-specific. The RoboTwin settings, including the sign-flip-invariant 20-D proprioception representation, the per-component subframe weights, and the top-5 trajectory prefilter, are listed in Section D. On PushT, $\phi _ { \mathrm { o b j } }$ is the T-block pose, the geometric terms dominate, and $w _ { \mathrm { v i s } }$ is set low so that DINO only breaks ties between geometrically comparable subframes.

Inference loop. Algorithm 1 assembles the two stages into the full test-time loop for one episode. Stage 1 runs once at the start, fixing the candidate set $\mathcal { C } ^ { \mathrm { t r a j } }$ from the initial scene, while Stage 2 re-runs every control step: the policy observes the current query state, selects the matching pool subframe $t ^ { \prime }$ under the inference cost Eq. (8), fetches the chunk starting at $t ^ { \prime } ,$ , and predicts the residual action chunk Eq. (3). It then executes the first K of the H predicted actions and advances by K, so the model re-retrieves and re-plans as the scene evolves rather than committing to a single retrieved trajectory for the whole episode.

Table 10: Hyperparameters. Backbone, and the RoboTwin 2.0 retrieval and training settings. The Stage 2 weights are the per-component subframe-matching weights of Eq. (6), and the proprioception encoding feeds $\underline { { \phi _ { \mathrm { p r o p } } } } .$
<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td colspan="2">Backbone</td></tr><tr><td>Base video model</td><td>Cosmos Predict2.5 (2B)</td></tr><tr><td>Output Action parameterization</td><td>joint action + future image (WAM), flow matching residual over the retrieved chunk</td></tr><tr><td colspan="2"></td></tr><tr><td>Retrieval, Stage 1 (trajectory prefilter) Top-K trajectories</td><td>5</td></tr><tr><td>SAM3 width-height term weight</td><td>1.0</td></tr><tr><td colspan="2">Retrieval, Stage 2 (subframe scoring weights)</td></tr><tr><td>Proprioception  $w _ { \mathrm { p r o p } }$ </td><td>1.0</td></tr><tr><td>Proprioception history (8-step window)</td><td>0.05</td></tr><tr><td> Image (DINO) Wvis</td><td>0.1</td></tr><tr><td>Action chunk  $w _ { \mathrm { a c t } }$ </td><td>0.1</td></tr><tr><td>Delta action</td><td>0.1</td></tr><tr><td>Time</td><td>0.0</td></tr><tr><td colspan="2">Retrieval, proprioception encoding</td></tr><tr><td>Representation</td><td>20-D,6-D rotation (sign-flip invariant)</td></tr><tr><td>Position weight  $w _ { \mathrm { p o s } }$ </td><td>4.0</td></tr><tr><td colspan="2">Training</td></tr><tr><td>Retrieval pairs per sample (top-k)</td><td>3 (weighted, top-N = 5 prefilter)</td></tr><tr><td>Horizon H (chunk size)</td><td>16 frames</td></tr><tr><td>Action stride K</td><td>1</td></tr><tr><td>Action-loss weight 入</td><td>16</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Learning rate</td><td> $1 \times 1 0 ^ { - 4 }$  (cosine, 2000-step warm-up)</td></tr><tr><td>Batch size</td><td>16/GPU × 4 grad.accum. ×8 GPUs = 512</td></tr><tr><td>Training iterations</td><td>20,000</td></tr></table>

## F Hyperparameters

This section collects the backbone, retrieval, and training hyperparameters in one place (Table 10). The retrieval values are the RoboTwin configuration; on PushT the object feature $\phi _ { \mathrm { o b j } }$ is the T-block pose, the geometric terms dominate, and the visual weight is kept low, as described qualitatively in Section E.