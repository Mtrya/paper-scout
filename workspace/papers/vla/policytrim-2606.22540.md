# PolicyTrim: Boosting Intrinsic Policy Eficiency of Vision-Language-Action Models

Xianghui Wang<sup>1⋆</sup>, Feng Chen<sup>2∗</sup>, Wenbo Zhang<sup>2</sup>, Hua Yan<sup>1</sup>, Zixuan Wang<sup>1†</sup>, Changsheng Li<sup>3</sup>, Yinjie Lei<sup>1‡</sup>

<sup>1</sup>Sichuan University <sup>2</sup>Adelaide University <sup>3</sup>Beijing Institute of Technology wangxianghui811421084@gmail.com, yinjie@scu.edu.cn

Abstract. Vision-Language-Action (VLA) models provide a unified paradigm for robotic manipulation, yet their real-world deployment is often bottlenecked by execution eficiency. While existing eforts predominantly focus on compute-centric eficiency to reduce per-step inference latency, the intrinsic policy eficiency of these models remains largely unexplored. Policy eficiency is fundamentally afected by two factors, namely the efective executable length of predicted action chunks and the total physical steps required to complete a task. These two factors jointly determine the total number of forward inference calls during execution. We observe that current VLA policies struggle with planning unreliability and action redundancy, sufering from severe prediction degradation at the tail of action chunks and tending to generate unnecessarily redundant physical steps. To address this, we propose PolicyTrim, a reinforcement learning-based post-training framework that extends the reliable action chunk length and reduces redundant physical steps. For reliable chunk extension, we employ a dynamic exploration strategy that explicitly rewards the successful completion of longer executable lengths, progressively pushing the trustworthy prediction horizon to its empirical limit. For step eficiency, we design a redundancy-aware reward that directly favors successful task completions with fewer steps while penalizing unreproducible shortcuts, efectively eliminating redundant physical actions. Extensive experiments across three benchmarks and three VLA models demonstrate that PolicyTrim improves action chunk utilization by 3× and reduces physical execution steps by 51.4%. Ultimately, our framework delivers up to a 5.83× end-to-end deployment speedup without compromising task success rates. Project Page: https://inceptionwang.github.io/PolicyTrim/

## 1 Introduction

Vision-Language-Action (VLA) models integrate visual perception, language understanding, and action generation into a single end-to-end framework, establishing a scalable paradigm for general-purpose robotic manipulation [2–4,10–12,19,

![](images/1b582764ea258d9b03ccf1b364f6e2e03b9fadf5f467fa5c6e3f29104847b324.jpg)

![](images/79dbfa88ab0b9f0d9071d508f8132e763ff449de08222c32ace3bcf8d894c1f1.jpg)  
(a) Step Count Across Tasks (Left: Baseline, Right: Ours)

![](images/f49272b1b7de836424f613de424dc283c87174509b518486320416a0ba6302ba.jpg)  
(b) Impact of Execution Length

![](images/98bb015bbf2867cac8871134bd2400ec851cb6143f26cfdac35e939189ed3f9c.jpg)  
(c) Visualization of Redundant Actions Induced by Extended Execution Length  
Fig. 1: Intrinsic policy ineficiency in deployed VLA models manifests along two dimensions. (a) Repeated rollouts on identical tasks reveal substantial variance in step counts, indicating concise execution paths exist but emerge only by chance. (b) Forcing longer action chunk execution simultaneously degrades success rates and inflates physical steps, confirming that unreliable tail predictions are a key factor. (c) A visualization of how tail prediction errors trigger misalignment and grasp failures, compelling the robot into redundant corrective actions before eventual task completion.

42]. To reduce the computational overhead of large vision-language backbones, the community has pursued compute-centric remedies such as visual token pruning [24], quantization, and KV-cache reuse [20,26,48,52,60], all aimed at reducing per-inference latency. However, the policy eficiency bottleneck of the models is largely unexplored, governed by the efective executable length of predicted action chunks and the total physical steps required to complete a task. These two factors jointly determine the total number of forward inference calls during execution. Empirically, current VLA models face planning unreliability and action redundancy, exhibiting severe prediction degradation at the tail of action chunks and tending to generate redundant execution steps. As illustrated in Fig. 1(b), when the model executes longer action chunks per inference, we observe a simultaneous decline in task success rates and an increase in the average physical steps required to complete the task. We believe this phenomenon stems from the degrading quality of tail predictions within action chunks [17, 44], resulting in an accumulation of physical errors that forces the robot to take redundant corrective actions. Furthermore, Fig. 1(a) shows that rolling out a trained model multiple times on the same task reveals substantial variance in successful execution lengths. This indicates that more compact and eficient execution paths are physically reachable but currently emerge only by chance, leaving ample room for optimizing the model’s step eficiency. Consequently, intrinsic policy eficiency remains the primary bottleneck for deployed VLA systems.

In this paper, we propose PolicyTrim, a two-stage RL-based post-training framework that enhances the policy eficiency of VLA models through reliable chunk extension and redundant step reduction [36]. For reliable chunk extension, PolicyTrim diversifies the execution window lengths within a sampled group by assigning a fixed window size to each individual trajectory. This mechanism serves as a progressive reliability sweep to probe whether tail predictions at each chunk position remain trustworthy throughout actual task execution. Rollouts that successfully complete the task using longer action chunks receive higher rewards, progressively pushing the reliable planning frontier toward the empirical limit. Additionally, to achieve redundant step reduction, we design a step-saving reward that grants higher values to successful rollouts reaching the goal in fewer physical steps while a stability regularizer explicitly prevents the policy from collapsing into unreproducible shortcuts. Through these two stages, PolicyTrim improves overall policy eficiency without requiring architectural modifications or additional expert data. The main contributions of this work are summarized as follows:

– We identify policy eficiency as a critical yet overlooked deployment bottleneck for VLA models and distinguish it from pure computational eficiency by highlighting the dual challenges of unreliable tail predictions in action chunks and redundant physical execution steps.

– We propose PolicyTrim, a post-training framework that extends the reliable planning horizon and reduces step redundancy, without architectural changes or extra demonstrations.

– Experiments across multiple benchmarks and models show that PolicyTrim cuts physical execution steps in half, triples action chunk utilization, and delivers up to 5.83× end-to-end speedup without sacrificing success rate.

## 2 Related Work

## 2.1 Vision-Language-Action Models

Vision-Language-Action (VLA) models unify visual perception, language understanding, and action generation into a single end-to-end framework for robotic manipulation [22,23,29,35,50]. Representative architectures span autoregressive VLAs such as OpenVLA [19] and RT-2 [4], as well as difusion-based policies such as π [2], π [1], and GR00T [31]. To handle high-frequency continuous control, modern VLAs widely adopt action chunking [9, 18, 59], predicting a sequence of future actions at each decision step. However, the quality of predicted actions tends to degrade toward the tail of each chunk, and policies trained via imitation learning often exhibit unnecessarily redundant execution trajectories, both of which result in ineficient real-world deployment. While these models demonstrate strong task competence, their real-world deployment eficiency remains a critical concern. The end-to-end execution time of a VLA system is jointly determined by two factors: the per-step inference latency and the number of inference calls required to complete a task. The former is governed by computational eficiency, while the latter reflects the intrinsic policy eficiency of the deployed model. Despite growing attention to deployment eficiency, existing eforts have focused almost exclusively on the computational side, leaving policy eficiency largely unexplored.

## 2.2 Eficient Vision-Language-Action Models

Current eficiency methods target per-inference computational cost while treating the learned policy as fixed [56]. Visual token pruning [16, 24, 43] and action tokenization compression [32,47] reduce input and output overhead respectively. Speculative decoding [46] and early-exit decoding [38, 39] accelerate autoregressive generation, while KV-cache [20, 51, 52] reuse further reduces decoding latency. Model quantization [45, 53, 57, 58] and lightweight designs [5, 15, 37, 49] further lower hardware demands. By strictly preserving the original policy distribution, these methods inevitably inherit its intrinsic ineficiencies [54], such as unreliable tail predictions within action chunks and redundant physical execution steps, which remain unaddressed regardless of how aggressively inference is accelerated. Critically, when a policy requires an excessive number of inference calls to complete a task, gains from per-step acceleration are fundamentally bounded, as the total execution time is jointly determined by both latency and call frequency. In contrast, improving policy eficiency directly reduces the number of inference calls required, ofering a complementary and multiplicative source of speedup. Since the two axes are fully decoupled, advances in either dimension can be stacked to yield compounded gains beyond what either approach achieves independently. Policy eficiency therefore constitutes an orthogonal axis to computational eficiency, one that requires post-training intervention rather than architectural or hardware-level optimization alone.

## 2.3 Reinforcement Learning for VLA

Reinforcement learning has emerged as a post-training paradigm to push VLA policies beyond their initial capabilities [8, 14, 21, 28]. Early eforts applied PPO [34], but the actor-critic architecture introduces prohibitive memory overhead when scaling to large VLA backbones [13]. GRPO [36] eliminates the value model by computing group-relative advantages, making RL post-training practical for large-scale VLAs. However, existing GRPO approaches for VLAs universally rely on binary success rewards [6, 14, 21, 28], which create two fundamental limitations. First, once the policy achieves a high success rate, reward variance within the sampled group collapses, causing advantage estimates to lose discriminative power and learning to stagnate. Second, binary rewards provide no signal to distinguish shorter completions from longer ones, nor to incentivize extending the reliable prediction horizon beyond the conservative default, leaving execution eficiency entirely unoptimized. PolicyTrim addresses both limitations through a step-saving reward that maintains meaningful optimization signal even at high success rates and a progressive chunk exploration mechanism that actively pushes the trustworthy prediction frontier toward the architectural limit.

![](images/d361be3ca1345980be988859ddebf1f80c04237e277f50cd75bfd6b196273149.jpg)  
Fig. 2: Overview of PolicyTrim. PolicyTrim is a two-stage RL post-training framework that enhances intrinsic policy eficiency of VLA models. The first stage progressively extends the reliable action chunk horizon by rewarding successful execution of longer chunks. The second stage eliminates redundant physical steps via a step-saving reward coupled with group-anchored stability regularization. Together, the two stages jointly reduce the total number of forward inference calls required to complete a task.

## 3 Method

## 3.1 Overview

An overview of PolicyTrim is illustrated in Fig. 2. We propose a two-stage posttraining framework that extends the executable action horizon per inference and reduces the number of steps required to complete a task for VLA models. At an arbitrary decision step t, the policy $\pi _ { \theta }$ processes the current visual observation $o _ { t }$ and language instruction l to predict a sequence of future actions $a _ { t : t + H }$ in parallel, where H denotes the maximum chunk capacity. Since prediction quality degrades toward the tail of action chunks, the system is typically constrained to executing only a truncated subset of actions to avoid the accumulation of physical errors. This subset defines an execution window $h = \lfloor \gamma H \rfloor$ before the model acquires a new observation for the next planning cycle, where $\gamma \in ( 0 , 1 ]$ represents the acceptance ratio. While this conservative execution strategy helps maintain physical stability, it handicaps policy eficiency by discarding potentially reliable predictions and necessitating frequent inference calls.

This paper introduces PolicyTrim, an RL-based post-training framework designed to enhance policy eficiency. As illustrated in Fig. 2, the framework decouples this enhancement objective into two progressive learning stages targeting reliable action chunk extension and redundancy-aware step reduction respectively. To facilitate reliable action chunk extension, the initial stage diversifies the execution lengths within a sampled group by assigning a diferent acceptance ratio to each individual trajectory. This mechanism implements a progressive reliability sweep that probes the empirical boundaries of trustworthy predictions. By rewarding successful trajectories proportionally to their chunk utilization, the model learns to yield more reliable action predictions further into the chunk tail and progressively extends the execution window. Building upon these extended chunks, a complementary redundancy-aware mechanism subsequently drives the second stage of physical step reduction. This phase introduces a step-saving reward coupled with a per-trajectory stability penalty. This method explicitly incentivizes the policy to prune redundant intermediate actions and converge toward the most concise physical execution trajectory while preventing catastrophic collapse onto irreproducible shortcuts. Ultimately, this sequential refinement of expanding reliable action chunks and reducing total steps naturally leads to a significant reduction in the required inference calls.

## 3.2 Reliable Action Chunk Extension

To enable reliable predictions over extended action horizons, the proposed method avoids rigidly forcing the model to execute the maximum length from scratch. It is instead framed as a progressive extension process where the framework dynamically probes the empirical reliability of various chunk positions by assigning execution windows of varying lengths across sampled trajectories. The policy is explicitly incentivized to overcome the inherent tail degradation by assigning higher rewards to those successful rollouts that manage to sustain longer execution length. Ultimately, this progressive refinement paradigm seamlessly pushes the trustworthy prediction frontier toward the maximum usable action chunk length H supported by the underlying model architecture.

Dynamic Execution Horizon Exploration. To probe the empirical reliability boundary without additional rollouts, each trajectory within the group is assigned a distinct execution window, collectively spanning multiple prediction horizons within a single group. Formally, we define a discrete set of acceptance ratios $\boldsymbol { \Gamma } = \{ \gamma _ { 1 } , \gamma _ { 2 } , . . . , \gamma _ { M } \}$ with $0 < \gamma _ { m } \leq 1$ , and assign each trajectory $\tau _ { i }$ its own ratio $\gamma _ { i }$ , yielding an execution window $h _ { i } = \lfloor \gamma _ { i } H \rfloor$ . This per-trajectory assignment turns each sampled group into a reliability sweep, probing prediction trustworthiness across short, intermediate, and long chunk positions in parallel. Reliable Horizon Reward. The reward for chunk extension is composed of a task completion reward and a horizon reward. The task completion reward $R _ { s u c c } ( \tau _ { i } )$ assigns 1 to a successful trajectory and 0 otherwise. The horizon reward is scaled by the assigned acceptance ratio:

$$
R _ { h o r i z o n } ( \tau _ { i } ) = \beta \cdot \gamma _ { i } ,\tag{1}
$$

where a larger $\gamma _ { i }$ indicates that the model sustained accurate predictions over a longer execution window without intermediate re-observation. However, a naive combination of the two becomes horizon-biased when no trajectory in the group succeeds, because the task completion signal collapses to a constant while the horizon reward still pushes the policy toward longer but potentially erroneous execution. To incentivize horizon extension without collapsing to unreliable longchunk behaviors, we activate the horizon reward only when the group contains at least one successful trajectory. Concretely, the integrated reward for chunk extension is formulated as:

$$
R _ { e x t } ( \tau _ { i } ) = \mathcal { T } _ { s u c c } ^ { ( i ) } \cdot ( R _ { s u c c } ( \tau _ { i } ) + R _ { h o r i z o n } ( \tau _ { i } ) ) ,\tag{2}
$$

where $\mathcal { T } _ { s u c c } ^ { ( i ) } \in \{ 0 , 1 \}$ is a binary indicator that equals 1 only if trajectory $\tau _ { i }$ successfully completes the task.

Group-Relative Policy Update. We optimize the aforementioned objective utilizing Group Relative Policy Optimization (GRPO). For a group of G trajectories sampled from the current policy $\pi _ { \theta }$ under the identical initial state $o _ { t }$ and language instruction l, we compute a group-normalized advantage for each trajectory based on the current objective $R _ { e x t } ( \tau _ { i } )$

$$
A _ { i } = \frac { R _ { e x t } ( \tau _ { i } ) - \mu _ { R } } { \sigma _ { R } + \epsilon } ,\tag{3}
$$

where $\mu _ { R }$ and $\sigma _ { R }$ are the mean and standard deviation of $R _ { e x t } ( \tau _ { i } )$ within the group. The advantage computation then directly contrasts these varying execution lengths under the same task instance, enabling the policy to learn which chunk positions still yield trustworthy actions and where prediction quality begins to degrade. Subsequently, the policy is updated via a clipped surrogate objective with a Kullback-Leibler (KL) penalty to the reference policy $\pi _ { \boldsymbol { r e f } } .$

$$
L ( \theta ) = \mathbb { E } _ { i } \left[ \operatorname* { m i n } \left( r _ { i } ( \theta ) A _ { i } , \mathrm { c l i p } ( r _ { i } ( \theta ) , 1 - \epsilon , 1 + \epsilon ) A _ { i } \right) \right] - \beta _ { K L } D _ { K L } ( \pi _ { \theta } | | \pi _ { r e f } ) ,\tag{4}
$$

where $r _ { i } ( \theta )$ denotes the importance sampling ratio between the updated policy and the old policy distribution, and ϵ defines the clipping range that constrains $r _ { i } ( \theta )$ within $( 1 - \epsilon , 1 + \epsilon )$ to prevent excessively large policy updates. The KL divergence $D _ { K L } ( \pi _ { \theta } | | \pi _ { r e f } )$ is computed per-token across the generated action chunk to prevent substantial deviation from the manipulation priors established during pre-training.

## 3.3 Redundancy-Aware Step Reduction

To explicitly encourage concise execution while preserving task correctness, we introduce a step-saving reward for successful trajectories, complemented by a group-consistent step regularization to prevent exploitation of fragile shortcuts. Step-Saving Reward. We define a step budget $S _ { b a s e }$ based on the initial policy’s performance statistics. For a successful trajectory completing the task in $S ( \tau _ { i } )$ steps, the step-saving reward is defined as:

$$
R _ { s t e p } ( \tau _ { i } ) = \frac { \operatorname* { m a x } ( 0 , S _ { b a s e } - S ( \tau _ { i } ) ) } { S _ { b a s e } } .\tag{5}
$$

Trajectories that complete the task in fewer steps receive proportionally higher reward, directly incentivizing the policy to converge toward more concise execution trajectories.

Group-Anchored Regularization. Naively minimizing execution steps without constraints risks policy collapse. Due to high initial variance in step counts, occasional short trajectories may receive disproportionately large rewards, causing the policy to exploit shortcuts that are not reliably reproducible. To address this, we introduce a group-anchored stability penalty that regularizes the step distribution within each sampled group:

$$
P _ { s t a b } ( \tau _ { i } ) = \lambda _ { s t a b } \cdot \mathrm { t a n h } \Bigg ( \frac { | S ( \tau _ { i } ) - \mu _ { g r o u p } | } { \operatorname* { m a x } ( \sigma _ { g r o u p } , \sigma _ { f l o o r } ) } \Bigg ) ,\tag{6}
$$

where $\mu _ { g r o u p }$ and $\sigma _ { g r o u p }$ are the mean and standard deviation of $S ( \tau )$ computed over successful trajectories in the group, serving as a group consensus anchor. The tanh function smoothly maps the normalized deviation to a bounded penalty in [0, 1], ensuring that trajectories deviating substantially from the group consensus incur progressively larger penalties while avoiding unbounded penalization. This design discourages the policy from exploiting fragile step-count outliers, guiding it to shift smoothly toward higher eficiency rather than collapsing onto irreproducible shortcuts. Meanwhile, as the policy converges to a consistent execution regime, $\sigma _ { g r o u p }$ may become very small, causing even minor step deviations to incur disproportionately large penalties. To address this, we introduce a floor parameter $\sigma _ { f l o o r }$ to bound the denominator, preventing such over-penalization and ensuring that subsequent policy updates remain well-conditioned once a stable regime is reached.

Joint Step-Performance Reward. The final reward for step reduction integrates the task completion reward $R _ { s u c c }$ , the step-saving reward $R _ { s t e p }$ , and the stability penalty $P _ { s t a b }$

$$
R _ { e f f } ( \tau _ { i } ) = \mathcal { T } _ { s u c c } ^ { ( i ) } \cdot ( R _ { s u c c } ( \tau _ { i } ) + R _ { s t e p } ( \tau _ { i } ) - P _ { s t a b } ( \tau _ { i } ) ) ,\tag{7}
$$

where the binary indicator $\mathcal { T } _ { s u c c } ^ { ( i ) }$ ensures that failed rollouts strictly receive a reward of zero, preventing step-saving incentives from reinforcing unsuccessful behaviors. We then utilize these rewards to calculate the group-relative advantage using Eq. (3) and execute the final policy update following Eq. (4).

Overall, inspired by the empirical insights from Fig. 1, we decouple the optimization into two sequential stages. Reliable chunk extension first establishes a wider trustworthy prediction horizon. However, committing to longer action chunks alone provides no mechanism to constrain the total number of physical steps, and may even inflate it through compounded tail prediction errors. Conversely, jointly optimizing both objectives simultaneously conflates two fundamentally distinct sub-goals, potentially entangling the reward signal and complicating the optimization landscape. We therefore apply redundancy-aware step reduction as a subsequent stage, building upon the extended chunk horizon to compress the physical execution path and systematically address the dual bottlenecks of VLA models.

## 4 Experiment

## 4.1 Experimental Setup

Benchmarks. We evaluate on three diverse benchmarks including LIBERO [25], ManiSkill [41], Meta-World [30] and further validate its sim-to-real transfer on a physical robot platform. Reported metrics include average success rate, average physical steps, average action chunk execution length, end-to-end execution speedup, and wall-clock execution time for real-world deployment.

• LIBERO is a tabletop manipulation benchmark comprising four subsets of increasing dificulty. The Spatial and Object subsets evaluate spatial reasoning and object generalization, the Goal subset introduces goal-conditioned reasoning, and the Long subset requires multi-stage manipulation over extended horizons, making it well-suited for assessing step eficiency under prolonged execution.

• ManiSkill is a high-fidelity simulation platform ofering physics-rich continuous control tasks. Following the experimental setup in [55], we adopt its diverse pick-and-place task combinations to evaluate policy eficiency and cross-task generalization in precise manipulation scenarios.

• Meta-World covers a wide range of manipulation tasks beyond pick-andplace, encompassing diverse end-efector motions and interaction modes. We adopt the MT50 suite to assess policy eficiency across heterogeneous action spaces and manipulation dynamics.

• Real-world Deployment uses an Agilex Piper arm equipped with two Intel RealSense D435i cameras. We evaluate three tabletop manipulation tasks, FlipMug, HangMug, and TapeBox.

VLA Models. To verify cross-architecture generalization, we apply PolicyTrim to three VLA models spanning distinct architectural paradigms. $\pi _ { 0 . 5 }$ is a conditional difusion policy that generates action chunks through iterative flowmatching denoising, conditioned on vision-language features from a pretrained VLM backbone. Its stochastic decoding naturally captures complex action distributions. OpenVLA-OFT builds upon a 7B-parameter vision-language model and replaces autoregressive token generation with parallel chunk decoding via placeholder action tokens and bidirectional attention, enabling single-forward-pass prediction of entire action sequences. GR00T is a generalist robot transformer that adopts a dual-system design, pairing a slow vision-language reasoning module with a fast difusion-based action generation head.

Implementation Details. All experiments are built upon the RLinf framework [55]. We use a group size of $G = 8$ trajectories for each task in every iteration. For all VLA models, the maximum chunk capacity H predicted at each step is initialized to match or exceed the original settings of the respective checkpoints. For reliable chunk extension, the sampling set of the acceptance ratio is $\boldsymbol { \Gamma } = \{ \gamma _ { 1 } , \gamma _ { 2 } , . . . , \gamma _ { M } \}$ with $0 < \gamma _ { m } \leq 1$ and $M = 3$ by default; $\gamma _ { 1 }$ is set such that $\gamma _ { 1 } \cdot H$ equals the model’s original default execution length, and the remaining ratios are uniformly spaced up to 1. For step eficiency, the step budget $S _ { b a s e }$ is set to approximately 1.3 times the average successful steps of the initial baseline policy. All experiments are conducted on 8 A100 GPUs with 64 parallel simulation environments.

Table 1: Evaluation of $\pi _ { 0 . 5 }$ , OpenVLA-OFT, and GR00T on the four subsets of the LIBERO benchmark. We report average success rate (SR), average physical steps $\left( S _ { \mathrm { t o t a l } } \right)$ , average action chunk execution length $\left( h _ { \mathrm { c h u n k } } \right)$ , and end-to-end execution Speedup (Spd).
<table><tr><td rowspan="2">Task</td><td rowspan="2">Method</td><td colspan="4"> $\pi _ { 0 . 5 }$ </td><td colspan="4">OpenVLA-OFT</td><td colspan="4">GR00T</td></tr><tr><td>SR</td><td> $S _ { \mathrm { t o t a l } }$ </td><td> $h _ { \mathrm { c h u n k } }$ </td><td>Spd↑ SR</td><td></td><td> $S _ { \mathrm { t o t a l } }$   $h _ { \mathrm { c h u n k } }$ </td><td></td><td>Spd↑ SR</td><td></td><td> $S _ { \mathrm { t o t a l } }$ </td><td> $h _ { \mathrm { c h u n k } }$ </td><td>Spd↑</td></tr><tr><td>Spatial</td><td>Baseline PolicyTrim</td><td>97.8</td><td>97.8 108.3</td><td>5</td><td>1.0</td><td></td><td>98.6 111.2 8</td><td>1.0</td><td></td><td>91.4 67.2</td><td></td><td>5 10</td><td>1.0 2.37</td></tr><tr><td>Object</td><td>Baseline</td><td></td><td>59.8 99.1 125.0</td><td>15 5</td><td>5.43 1.0</td><td>98.8 62.1 98.5 135.2</td><td>8 8</td><td>1.79 1.0</td><td>92.0 95.0</td><td>56.6 71.3</td><td></td><td>5</td><td>1.0</td></tr><tr><td>Goal</td><td>PolicyTrim 98.5 64.3 Baseline</td><td></td><td>98.7 110.6</td><td>15 5</td><td>5.83 1.0</td><td>98.5 68.8 97.7 118.6</td><td>8 8</td><td>1.97 1.0</td><td>95.3 84.2</td><td>65.5 63.3</td><td>5</td><td>10</td><td>2.18 1.0</td></tr><tr><td></td><td>PolicyTrim 98.8 63.5</td><td></td><td></td><td>15</td><td>5.23</td><td>98.066.9</td><td>8</td><td>1.77</td><td>86.3</td><td>60.8</td><td></td><td>10</td><td>2.08</td></tr><tr><td>Long</td><td>Baseline PolicyTrim 93.3 171.8</td><td></td><td>93.0 249.8</td><td>5 10</td><td>1.0 2.91</td><td>93.1 178.3</td><td>92.9 249.3 8 8</td><td>1.0</td><td>1.4089.2 165.9</td><td>86.1 177.9</td><td></td><td>5 10</td><td>1.0 2.14</td></tr></table>

Table 2: Evaluation on ManiSkill and Meta-World. We report average success rate (SR), average physical steps $\left( S _ { \mathrm { t o t a l } } \right)$ , average action chunk execution length $\left( \boldsymbol { h } _ { \mathrm { c h u n k } } \right)$ and end-to-end execution Speedup (Spd).
<table><tr><td rowspan="2">Benchmark Method</td><td rowspan="2"></td><td colspan="3">π0.5</td><td colspan="3">OpenVLA-OFT</td></tr><tr><td>SR  $S _ { \mathrm { t o t a l } }$ </td><td> $h _ { \mathrm { c h u n k } }$ </td><td>Spd↑ SR</td><td> $S _ { \mathrm { t o t a l } }$ </td><td> $h _ { \mathrm { c h u n k } }$ </td><td>Spd↑</td></tr><tr><td>ManiSkill</td><td>Baseline PolicyTrim 89.8 38.3</td><td>88.145.2</td><td>5 10</td><td>1.0</td><td>60.653.1 2.36 63.2 46.7</td><td>8 8</td><td>1.0 1.14</td></tr><tr><td rowspan="2">Meta-World</td><td>Baseline</td><td>65.1 66.3</td><td>5</td><td>1.0</td><td></td><td></td><td></td></tr><tr><td>PolicyTrim 65.4 52.6</td><td></td><td>10</td><td>2.52</td><td></td><td></td><td></td></tr></table>

## 4.2 Main Results

Results on LIBERO. Table 1 presents a comprehensive quantitative evaluation on the four LIBERO subsets using $\pi _ { 0 . 5 }$ , OpenVLA-OFT, and GR00T. Across all three models and subsets, PolicyTrim consistently reduces physical execution steps while maintaining comparable task success rates, with the magnitude of reduction varying by architecture and task complexity. The most pronounced reduction is observed for $\pi _ { 0 . 5 }$ on the Object subset, where the step count drops to as low as 51.4% of the baseline. These results suggest that our eficiency gains arise from a fundamentally improved execution strategy rather than any sacrifice in task competence. For action chunk extension, we observe an architectural divergence among the evaluated models. $\pi _ { 0 . 5 }$ and GR00T both adopt difusionbased action decoding, which naturally accommodates extended chunk horizons; PolicyTrim successfully increases the reliable action chunk execution length for both architectures. OpenVLA-OFT, by contrast, employs a parallel decoding scheme with placeholder action tokens and bidirectional attention, where even a marginal increase in chunk length causes severe accuracy degradation, rendering chunk extension efectively untrainable for this architecture. We therefore apply only the step reduction stage to OpenVLA-OFT, and the reported chunk length improvements are specific to $\pi _ { 0 . 5 }$ and GR00T.

![](images/d10d35543a86ddc0c0a070bb15886f545382a278edfa5aa6c0dfcafc2dc55350.jpg)  
Fig. 3: Qualitative comparison on randomly sampled LIBERO tasks. Under identical configurations, the baseline incurs redundant physical actions, whereas PolicyTrim achieves task completion in roughly half the steps.

Results on ManiSkill and Meta-World. Table 2 reports the performance on ManiSkill and Meta-World. PolicyTrim consistently improves both success rates and step eficiency across benchmarks, achieving up to 2.52× end-to-end speedup on Meta-World and 2.36× on ManiSkill with $\pi _ { 0 . 5 }$ . The consistent improvements across diverse physical simulators, action spaces, and model architectures demonstrate that PolicyTrim generalizes well beyond the LIBERO benchmark, confirming that policy eficiency optimization yields substantial and reliable gains across varied deployment scenarios.

Architectural Generality. Table 3 reports the cross-architecture results on both parallel-decoding and autoregressive VLA models. Beyond the standard

Table 3: Cross-architecture results. We report success rate (SR), average physical steps, action horizon h, and end-to-end speedup.
<table><tr><td>Model</td><td>Method</td><td>SR</td><td>Step</td><td>h</td><td>Spd↑</td></tr><tr><td>OpenVLA-OFT</td><td>Baseline</td><td>98.6</td><td>111.2</td><td>8</td><td>1.00×</td></tr><tr><td>OpenVLA-OFT</td><td>S1+S2</td><td>98.8</td><td>65.4</td><td>14</td><td>2.97×</td></tr><tr><td>OpenVLA</td><td>Baseline</td><td>84.7</td><td>113.5</td><td></td><td>1.00×</td></tr><tr><td>OpenVLA</td><td>S2</td><td>87.0</td><td>80.6</td><td></td><td>1.41×</td></tr></table>

Table 4: Real-world deployment results. Standard uses a fixed target pose, while Dynamic perturbs the target during grasping. Values under Standard and Dynamic are success rates in %, and Time is measured in seconds.
<table><tr><td></td><td colspan="3">Standard</td><td colspan="2"></td><td colspan="3">Dynamic |Time(Standard)</td></tr><tr><td>Method</td><td>Flip Hang Tape Flip Tape|Flip Hang Tape</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Baseline</td><td>70</td><td>60</td><td>95</td><td>70</td><td>65</td><td>14.6</td><td>15.6</td><td>17.5</td></tr><tr><td>PolicyTrim</td><td>75</td><td>65</td><td>95</td><td>70</td><td>70</td><td>7.6</td><td>8.7</td><td>9.4</td></tr></table>

OpenVLA-OFT setting in Table 1, we re-pretrain OpenVLA-OFT with a larger action chunk capacity of $h = 1 6$ , since the original parallel decoder uses a fixed horizon of $h = 8$ and already executes all predicted actions, leaving no room for Stage 1 extension. With this re-pretrained OpenVLA-OFT backbone, the full two-stage PolicyTrim pipeline achieves a 2.97× speedup while preserving task success. We further evaluate the autoregressive OpenVLA model, where Stage 1 is not directly applicable because action chunks are not generated through the same fixed-horizon parallel decoding mechanism. Applying Stage 2 alone still yields a 1.41× speedup and improves the success rate from 84.7% to 87.0%. These results demonstrate that PolicyTrim provides consistent gains across different VLA decoding architectures, and that its redundancy-aware optimization is not tied to a specific action decoding form.

Real-World Deployment. Table 4 reports the real-world deployment results of π on three tabletop manipulation tasks, FlipMug, HangMug, and TapeBox. PolicyTrim maintains or improves the success rate across all tasks, under both the standard setting with a fixed target pose and the dynamic setting where the target is randomly perturbed during grasping, while substantially reducing wallclock execution time. On average, PolicyTrim achieves a 1.86× speedup over the baseline under the standard real-world setting. These results are obtained on an Agilex Piper arm equipped with two Intel RealSense D435i cameras, where PolicyTrim is first RL post-trained in simulation and then adapted to the real world through supervised fine-tuning on a small number of real demonstrations. The consistent reduction in real execution time demonstrates that PolicyTrim’s eficiency gains transfer from simulation to physical deployment.

Table 5: Ablation study of diferent components on LIBERO-Spatial benchmarks.
<table><tr><td>Reliable Chunk Extension</td><td>Step-Saving Reward</td><td>Group-Anchored Regularization</td><td>SR</td><td> $S _ { \mathrm { t o t a l } }$ </td><td> $h _ { \mathrm { c h u n k } }$ </td><td>Spd↑</td></tr><tr><td>X</td><td>X</td><td>X</td><td>97.8</td><td>108.3</td><td>5</td><td>1.0</td></tr><tr><td>√</td><td>X</td><td>X</td><td>97.2</td><td>113.8</td><td>15</td><td>2.86</td></tr><tr><td>X</td><td>√</td><td>X</td><td>93.7</td><td>81.7</td><td>5</td><td>1.32</td></tr><tr><td>X</td><td>√</td><td>√</td><td>97.5</td><td>61.6</td><td>5</td><td>1.75</td></tr><tr><td>√</td><td>√</td><td>√</td><td>98.3</td><td>59.8</td><td>15</td><td>5.43</td></tr></table>

Table 6: Ablation of Dynamic Execution Horizon Exploration on LIBERO-Object using π<sub>0.5</sub> with $H = 2 0$ . Fixed-γ variants replace diverse ratio sampling with a single acceptance ratio.
<table><tr><td>Method</td><td>SR</td><td> $S _ { \mathrm { t o t a l } }$ </td><td> $h _ { \mathrm { c h u n k } }$ </td><td>Spd↑</td></tr><tr><td>Baseline</td><td>99.1</td><td>125.0</td><td>5</td><td>1.00</td></tr><tr><td>Fixed  $\gamma { = } 0 . 2 5$ </td><td>99.2</td><td>127.1</td><td>5</td><td>0.98</td></tr><tr><td>Fixed  $\gamma { = } 0 . 5 0$ </td><td>98.8</td><td>125.2</td><td>10</td><td>1.99</td></tr><tr><td>Fixed  $\gamma { = } 0 . 7 5$ </td><td>95.8</td><td>130.1</td><td>15</td><td>2.88</td></tr><tr><td>Fixed  $\gamma = 1 . 0$ </td><td>94.4</td><td>131.8</td><td>20</td><td>3.79</td></tr><tr><td>Dynamic Horizon Exploration</td><td>98.8</td><td>127.4</td><td>15</td><td>2.94</td></tr></table>

Qualitative Analysis. Fig. 3 provides a qualitative comparison between the baseline and PolicyTrim on tasks randomly sampled from LIBERO under identical configurations. The baseline consistently exhibits redundant corrective motions before task completion, whereas PolicyTrim executes smooth and direct trajectories toward the goal. This behavioral contrast is especially pronounced in tasks involving precise placement, where the baseline displays noticeable hesitation and jittering near the target, substantially inflating the total step count. PolicyTrim avoids such corrective detours entirely, reducing physical steps by nearly half.

## 4.3 Ablation Study

In Table 5, we conduct ablation studies on the LIBERO-Spatial subset using the $\pi _ { 0 . 5 }$ model to isolate the contribution of each PolicyTrim component. All ablated configurations share the same initialization checkpoint and hyperparameters, with only the target component removed.

Efect of Reliable Action Chunk Extension. Adding Chunk Extension alone triples the average execution window from $h _ { \mathrm { c h u n k } } = 5$ to 15, reducing inference call frequency and yielding a 2.86× speedup. However, total physical steps simultaneously increase from 108.3 to 113.8, confirming that committing to longer action chunks amplifies the downstream impact of residual tail prediction errors and compels the robot to take additional corrective actions. This validates the necessity of an explicit step reduction stage following chunk extension, as longer chunks alone do not translate to more concise physical execution.

Efect of Step-Saving Reward. Applying the Step-Saving Eficiency Reward alone successfully reduces $S _ { \mathrm { t o t a l } }$ by 24.6% from 108.3 to 81.7, demonstrating its efectiveness in driving the policy toward more concise execution. However, this comes at the cost of an unacceptable degradation in task competence, with the success rate dropping sharply from 97.8% to 93.7%. Without Group-Anchored Regularization, the policy exploits unstable short trajectories that occur by chance but are not reliably reproducible under the current policy distribution, efectively trading task success for superficial step savings. This reveals that the step reduction objective alone is insuficient, as the policy can find shorter paths but collapses onto fragile shortcuts that fail to generalize.

Efect of Group-Anchored Regularization. When Group-Anchored Regularization is added on top of the Step-Saving Reward, a striking improvement emerges. The success rate recovers from 93.7% to 97.5% while $S _ { \mathrm { t o t a l } }$ is further reduced from 81.7 to 61.6. As shown in Fig. 4, without Group-Anchored Regularization, the reward collapses around step 125 as the policy exploits fragile short trajectories that cannot be reliably reproduced, caus-

![](images/c21a8c552a82b7c0915519dcbd42f263dec8c3347e9b22f66b634cdedaf25819.jpg)

![](images/9e7ae8500d5b75fe274103362e62e32c413e48db28491cb5a359997cf21612ac.jpg)  
Fig. 4: Training reward curves without (Left) and with (Right) Group-Anchored Regularization on LIBERO-Spatial (π<sub>0.5</sub>).

ing training to destabilize. Incorporating Group-Anchored Regularization instead yields a steadily increasing reward throughout training. The fact that a regularization term simultaneously improves both task competence and execution eficiency reveals its essential role in steering the policy away from fragile shortcuts and toward genuinely concise and reproducible execution paths.

Efect of Dynamic Execution Horizon Exploration. As reported in Table 6, fixing the acceptance ratio to a single value exposes a clear trade-of: as $\gamma$ increases, the execution window grows at the cost of progressive SR degradation, with Fixed $\gamma { = } 0 . 7 5$ and $\gamma = 1 . 0$ dropping to 95.8% and 94.4% respectively. Dynamic Execution Horizon Exploration resolves this tension by simultaneously probing multiple horizons within each group, achieving $h _ { \mathrm { c h u n k } } { = } 1 5$ comparable to Fixed $\gamma { = } 0 . 7 5$ while preserving an SR of 98.8% nearly identical to the baseline, underscoring its critical role in stable and efective chunk extension.

## 4.4 Discussion

What Causes Policy Ineficiency in VLA Policies? The root cause likely lies in the training paradigms themselves. Imitation learning [12, 27, 33] optimizes the policy to reproduce demonstrated behaviors without any explicit signal favoring execution eficiency. Moreover, prediction errors accumulate along action chunks due to distribution shift, causing the policy to take redundant corrective actions that further inflate the total execution steps. Standard RL post-training [7, 40] with binary success rewards also provides no explicit incentive for execution efficiency. Moreover, as the policy matures and success rates rise, reward variance within each sampled group tends to collapse, gradually diminishing the discriminative power of advantage estimates.

Table 7: Combining PolicyTrim with VLA-Cache on the four LIBERO subsets using OpenVLA-OFT. PolicyTrim and VLA-Cache target orthogonal eficiency axes and yield compounded speedups.
<table><tr><td rowspan="2">Method</td><td colspan="4">Spatial</td><td colspan="4">Object</td></tr><tr><td>SR</td><td> $S _ { \mathrm { t o t a l } }$ </td><td> $h _ { \mathrm { c h u n k } }$ </td><td>Spd↑</td><td>SR</td><td> $S _ { \mathrm { t o t a l } }$ </td><td> $h _ { \mathrm { c h u n k } }$ </td><td>Spd↑</td></tr><tr><td>Baseline</td><td>97.8</td><td>113.1</td><td>8</td><td>1.0</td><td>97.6</td><td>138.8</td><td>8</td><td>1.0</td></tr><tr><td>VLA-Cache</td><td>98.3</td><td></td><td>8</td><td>1.26</td><td>97.5</td><td></td><td>8</td><td>1.26</td></tr><tr><td>+PolicyTrim</td><td>98.8</td><td>63.2</td><td>8</td><td>2.26</td><td>98.5</td><td>70.5</td><td>8</td><td>2.48</td></tr><tr><td rowspan="2">Method</td><td colspan="4">Goal</td><td colspan="4">Long</td></tr><tr><td>SR</td><td> $S _ { \mathrm { t o t a l } }$ </td><td> $h _ { \mathrm { c h u n k } }$ </td><td>Spd↑</td><td>SR</td><td> $S _ { \mathrm { t o t a l } }$ </td><td> $h _ { \mathrm { c h u n k } }$ </td><td>Spd↑</td></tr><tr><td>Baseline</td><td>97.6</td><td>115.7</td><td>8</td><td>1.0</td><td>94.2</td><td>256.3</td><td>8</td><td>1.0</td></tr><tr><td>VLA-Cache</td><td>98.3</td><td></td><td>8</td><td>1.26</td><td>95.4</td><td></td><td>8</td><td>1.26</td></tr><tr><td>+PolicyTrim</td><td>98.5</td><td>65.4</td><td>8</td><td>2.23</td><td>95.4</td><td>183.1</td><td>8</td><td>1.76</td></tr></table>

Orthogonality with Compute-centric Methods. Policy eficiency and computational eficiency are orthogonal and jointly exploitable. While compute-centric methods reduce per-step inference latency, PolicyTrim targets the total number of forward inference calls, a dimension existing acceleration techniques leave entirely unaddressed. As demonstrated in Table 7, integrating PolicyTrim with VLA-Cache yields speedups of up to 2.48× on LIBERO-Object, well beyond the 1.26× achievable by VLA-Cache alone, confirming that the two approaches together constitute a more complete path toward practical deployment.

## 5 Conclusion

In this work, we identified policy eficiency as a critical yet overlooked bottleneck for deploying VLA models, and proposed PolicyTrim, a two-stage RL post-training framework that addresses this without architectural modifications or additional demonstrations. The first stage employs a dynamic horizon exploration mechanism to progressively push the trustworthy prediction frontier toward its empirical limit, while the second introduces a redundancy-aware reward coupled with group-anchored stability regularization to drive the policy toward genuinely concise execution without collapsing onto irreproducible shortcuts. Extensive experiments across diverse benchmarks and backbones demonstrate that

PolicyTrim significantly reduces inference frequency while maintaining strong task competence. As an orthogonal complement to compute-centric acceleration, PolicyTrim can be seamlessly combined with existing inference optimization techniques for compounded eficiency gains, pointing toward a more holistic path to practical and scalable robotic deployments. While PolicyTrim provides a robust execution framework, future work may explore integrating PolicyTrim with on-device continuous learning to further adapt execution eficiency to dynamic and unseen environments.

## Acknowledgment

This work is supported by the National Natural Science Foundation of China (U23B2013, U2441242 and 62276176). This work was also partly supported by the SICHUAN Provincial Natural Science Foundation (No. 2024NSFJQ0023).

## References

1. Black, K., Brown, N., Darpinian, J., Dhabalia, K., Driess, D., Esmail, A., Equi, M.R., Finn, C., Fusai, N., Galliker, M.Y., Ghosh, D., Groom, L., Hausman, K., Ichter, B., Jakubczak, S., Jones, T., Ke, L., LeBlanc, D., Levine, S., Li-Bell, A., Mothukuri, M., Nair, S., Pertsch, K., Ren, A.Z., Shi, L.X., Smith, L., Springenberg, J.T., Stachowicz, K., Tanner, J., Vuong, Q., Walke, H., Walling, A., Wang, H., Yu, L., Zhilinsky, U.: π : a Vision-Language-Action Model with Open-World Generalization. In: Lim, J., Song, S., Park, H.W. (eds.) Proceedings of The 9th Conference on Robot Learning. Proceedings of Machine Learning Research, vol. 305, pp. 17–40. PMLR (27–30 Sep 2025) 3

2. Black, K., Brown, N., Driess, D., Esmail, A., Equi, M.R., Finn, C., Fusai, N., Groom, L., Hausman, K., Ichter, B., Jakubczak, S., Jones, T., Ke, L., Levine, S., Li-Bell, A., Mothukuri, M., Nair, S., Pertsch, K., Shi, L.X., Smith, L., Tanner, J., Vuong, Q., Walling, A., Wang, H., Zhilinsky, U.: π<sub>0</sub>: A Vision-Language-Action Flow Model for General Robot Control. In: Proceedings of Robotics: Science and Systems. Los Angeles, CA, USA (June 2025). https://doi.org/10.15607/RSS. 2025.XXI.010 1, 3

3. Brohan, A., Brown, N., Carbajal, J., et al.: RT-1: Robotics transformer for realworld control at scale (2022), arXiv preprint arXiv:2212.06817 1

4. Brohan, A., et al.: RT-2: Vision-language-action models transfer web knowledge to robotic control. In: Conference on Robot Learning (CoRL) (2023), arXiv preprint arXiv:2307.15818 1, 3

5. Budzianowski, P., Maa, W., Freed, M., Mo, J., Hsiao, W., Xie, A., et al.: Edgevla: Eficient vision-language-action models (2025), arXiv preprint arXiv:2507.14049 4

6. Chen, F., He, Y., Lin, L., Gou, C., Liu, J., Zhuang, B., Wu, Q.: Sparsity forcing: Reinforcing token sparsity of mllms. arXiv preprint arXiv:2504.18579 (2025) 4

7. Chen, K., Liu, Z., Zhang, T., Guo, Z., Xu, S., Lin, H., Zang, H., Zhang, Q., Yu, Z., Fan, G., et al.: π<sub>RL</sub>: Online RL Fine-tuning for Flow-based Vision-Language-Action Models. arXiv preprint arXiv:2510.25889 (2025) 15

8. Chen, Y., Tian, S., Liu, S., Zhou, Y., Li, H., Zhao, D.: Conrft: A reinforced finetuning method for vla models via consistency policy. In: Proceedings of Robotics: Science and Systems. Los Angeles, CA, USA (June 2025). https://doi.org/10. 15607/RSS.2025.XXI.019 4

9. Chi, C., Xu, Z., Feng, S., Cousineau, E., Du, Y., Burchfiel, B., Tedrake, R., Song, S.: Difusion policy: Visuomotor policy learning via action difusion. In: Robotics: Science and Systems (RSS) (2023), arXiv preprint arXiv:2303.04137 3

10. Collaboration, O.X.E., O’Neill, A., et al.: Open x-embodiment: Robotic learning datasets and rt-x models (2023), arXiv preprint arXiv:2310.08864 1

11. Driess, D., Xia, F., Sajjadi, M.S.M., Lynch, C., Chowdhery, A., Ichter, B., Wahid, A., Tompson, J., Vuong, Q., Yu, T., et al.: Palm-e: An embodied multimodal language model. In: International Conference on Machine Learning (ICML). pp. 8469–8488. PMLR (2023) 1

12. Ghosh, D., Walke, H.R., Pertsch, K., et al.: Octo: An open-source generalist robot policy. In: Proceedings of Robotics: Science and Systems. Delft, Netherlands (July 2024). https://doi.org/10.15607/RSS.2024.XX.090 1, 14

13. Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W.: Lora: Low-rank adaptation of large language models. In: International Conference on Learning Representations (ICLR) (2022) 4

14. Huang, D., Fang, Z., Zhang, T., Li, Y., Zhao, L., Xia, C.: Co-rft: Eficient finetuning of vision-language-action models through chunked ofline reinforcement learning (2025), arXiv preprint arXiv:2508.02219 4

15. Hung, C.Y., Sun, Q., Hong, P., Zadeh, A., Li, C., Tan, U., Majumder, N., Poria, S., et al.: Nora: A small open-sourced generalist vision language action model for embodied tasks. arXiv preprint arXiv:2504.19854 (2025) 4

16. Jiang, T., Jiang, X., Ma, Y., Wen, X., Li, B., Zhan, K., Jia, P., Liu, Y., Sun, S., Lang, X.: The better you learn, the smarter you prune: Towards eficient vision-language-action models via diferentiable token pruning. arXiv preprint arXiv:2509.12594 (2025) 4

17. Jing, D., Wang, G., Liu, J., Tang, W., Sun, Z., Yao, Y., Wei, Z., Liu, Y., Lu, Z., Ding, M.: Mixture of horizons in action chunking (2025), arXiv preprint arXiv:2511.19433 2

18. Kim, M.J., Finn, C., Liang, P.: Fine-tuning vision-language-action models: Optimizing speed and success. In: Proceedings of Robotics: Science and Systems. Los Angeles, CA, USA (June 2025). https://doi.org/10.15607/RSS.2025.XXI.017 3

19. Kim, M.J., Pertsch, K., Karamcheti, S., Xiao, T., Balakrishna, A., Nair, S., Rafailov, R., Foster, E.P., Sanketi, P.R., Vuong, Q., Kollar, T., Burchfiel, B., Tedrake, R., Sadigh, D., Levine, S., Liang, P., Finn, C.: OpenVLA: An opensource vision-language-action model. In: Agrawal, P., Kroemer, O., Burgard, W. (eds.) Proceedings of The 8th Conference on Robot Learning. Proceedings of Machine Learning Research, vol. 270, pp. 2679–2713. PMLR (06–09 Nov 2025) 1, 3

20. Koo, J., Cho, T., Kang, H., Pyo, E., Oh, T.G., Kim, T., Choi, A.J.: RetoVLA: Reusing register tokens for spatial reasoning in vision-language-action models (2025), arXiv preprint arXiv:2509.21243 2, 4

21. Li, H., Zuo, Y., Yu, J., Zhang, Y., et al.: Simplevla-rl: Scaling vla training via reinforcement learning (2025), arXiv preprint arXiv:2509.09674 4

22. Li, Q., Liang, Y., Wang, Z., et al.: Cogact: A foundational vision-language-action model for synergizing cognition and action in robotic manipulation (2024), arXiv preprint arXiv:2411.19650 3

23. Li, X., Liu, M., Zhang, H., et al.: Vision-language foundation models as efective robot imitators. In: International Conference on Learning Representations (ICLR) (2024) 3

24. Li, Y., Meng, Y., Sun, Z., Ji, K., Tang, C., Fan, J., Ma, X., Xia, S., Wang, Z., Zhu, W.: Sp-vla: A joint model scheduling and token pruning approach for vla model acceleration (2025), arXiv preprint arXiv:2506.12723 2, 4

25. Liu, B., et al.: LIBERO: Benchmarking knowledge transfer for lifelong robot learning. In: NeurIPS Datasets and Benchmarks Track (2023), arXiv preprint arXiv:2306.03310 9

26. Liu, J., Chen, H., An, P., Liu, Z., Zhang, R., Gu, C., Li, X., Guo, Z., Chen, S., Liu, M., et al.: Hybridvla: Collaborative difusion and autoregression in a unified vision-language-action model. arXiv preprint arXiv:2503.10631 (2025) 2

27. Liu, S., Wu, L., Li, B., Tan, H., Chen, H., Wang, Z., Xu, K., Su, H., Zhu, J.: RDT-1B: a difusion foundation model for bimanual manipulation. In: International Conference on Learning Representations (ICLR) (2025) 14

28. Lu, G., Guo, W., Zhang, C., Zhou, Y., Jiang, H., Gao, Z., Tang, Y., Wang, Z.: Vlarl: Towards masterful and general robotic manipulation with scalable reinforcement learning (2025), arXiv preprint arXiv:2505.18719 4

29. Ma, Y.J., Song, Z., Zhuang, Y., Hao, J., King, I.: A survey on vision-languageaction models for embodied ai (2024), arXiv preprint arXiv:2405.14093 3

30. McLean, R., Chatzaroulas, E., McCutcheon, L., Röder, F., Yu, T., He, Z., Zentner, K., Julian, R., Terry, J.K., Woungang, I., Farsad, N., Castro, P.S.: Meta-world+: An improved, standardized, RL benchmark. In: The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track (2025) 9

31. NVIDIA, et al.: GR00T N1: An open foundation model for generalist humanoid robots (2025), arXiv preprint arXiv:2503.14734 3

32. Pertsch, K., Stachowicz, K., Ichter, B., Driess, D., Nair, S., Vuong, Q., Mees, O., Finn, C., Levine, S.: FAST: Eficient action tokenization for vision-language-action models. In: Proceedings of Robotics: Science and Systems. Los Angeles, CA, USA (June 2025). https://doi.org/10.15607/RSS.2025.XXI.012 4

33. Qu, D., Song, H., Chen, Q., Yao, Y., Ye, X., Gu, J., Wang, Z., Ding, Y., Zhao, B., Wang, D., Li, X.: Spatialvla: Exploring spatial representations for visual-languageaction models. In: Proceedings of Robotics: Science and Systems. Los Angeles, CA, USA (June 2025). https://doi.org/10.15607/RSS.2025.XXI.011 14

34. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., Klimov, O.: Proximal policy optimization algorithms (2017), arXiv preprint arXiv:1707.06347 4

35. Shao, R., Li, W., Zhang, L., Zhang, R., Liu, Z., Chen, R., Nie, L.: Large vlm-based vision-language-action models for robotic manipulation: A survey (2025), arXiv preprint arXiv:2508.13073 3

36. Shao, Z., Wang, P., Zhu, Q., et al.: Deepseekmath: Pushing the limits of mathematical reasoning in open language models (2024), arXiv preprint arXiv:2402.03300 3, 4

37. Shukor, M., Aubakirova, D., Capuano, F., Kooijmans, P., Palma, S., et al.: Smolvla: A vision-language-action model for afordable and eficient robotics (2025), arXiv preprint arXiv:2506.01844 4

38. Song, W., Chen, J., Ding, P., Huang, Y., Zhao, H., Wang, D., Li, H.: CEED-VLA: Consistency vision-language-action model with early-exit decoding (2025), arXiv preprint arXiv:2506.13725 4

39. Song, W., Chen, J., Ding, P., Zhao, H., Zhao, W., Zhong, Z., Ge, Z., Ma, J., Li, H.: PD-VLA: Accelerating vision-language-action model integrated with action chunking via parallel decoding (2025), arXiv preprint arXiv:2503.02310 4

40. Tan, S., Dou, K., Zhao, Y., Krähenbühl, P.: Interactive post-training for visionlanguage-action models (2025) 15

41. Tao, S., Xiang, F., Shukla, A., Qin, Y., Hinrichsen, X., Yuan, X., Bao, C., Lin, X., Liu, Y., Chan, T.k., Gao, Y., Li, X., Mu, T., Xiao, N., Gurha, A., Viswesh, N.R., Choi, Y.W., Chen, Y.R., Huang, Z., Calandra, R., Chen, R., Luo, S., Su, H.: Maniskill3: Gpu parallelized robotics simulation and rendering for generalizable embodied ai. Robotics: Science and Systems (2025), arXiv preprint arXiv:2410.00425 9

42. Team, R., et al.: RDT-1B: A difusion foundation model for bimanual manipulation. In: International Conference on Learning Representations (ICLR) (2025), arXiv preprint arXiv:2410.07864 1

43. Wang, H., Xu, J., Pan, J., Zhou, Y., Dai, G.: Specprune-vla: Accelerating visionlanguage-action models via action-aware self-speculative pruning. arXiv preprint arXiv:2509.05614 (2025) 4

44. Wang, H., et al.: Vla knows its limits (2026), arXiv preprint arXiv:2602.21445 2

45. Wang, H., Xiong, C., Wang, R., Chen, X.: Bitvla: 1-bit vision-language-action models for robotics manipulation (2025), arXiv preprint arXiv:2506.07530 4

46. Wang, S., Yu, R., Yuan, Z., Yu, C., Gao, F., Wang, Y., Wong, D.F.: Spec-vla: Speculative decoding for vision-language-action models with relaxed acceptance (2025), arXiv preprint arXiv:2507.22424 4

47. Wang, Y., Zhu, H., Liu, M., Yang, J., Fang, H.S., He, T.: VQ-VLA: Improving vision-language-action models via scaling vector-quantized action tokenizers. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 11089–11099 (October 2025) 4

48. Wen, J., Zhu, M., Liu, J., Liu, Z., Yang, Y., Zhang, L., Zhang, S., Zhu, Y., Xu, Y.: dvla: Difusion vision-language-action model with multimodal chain-of-thought. arXiv preprint arXiv:2509.25681 (2025) 2

49. Wen, J., Zhu, Y., Li, J., Zhu, M., Tang, Z., Wu, K., Xu, Z., Liu, N., Cheng, R., Shen, C., et al.: Tinyvla: Towards fast, data-eficient vision-language-action models for robotic manipulation. IEEE Robotics and Automation Letters (2025) 4

50. Wu, H., Jing, Y., Cheang, C., et al.: GR-1: Unleashing large-scale video generative pre-training for visual robot manipulation (2023), arXiv preprint arXiv:2312.13139 3

51. Xu, S., Wang, Y., Xia, C., Zhu, D., Huang, T., Xu, C.: Vla-cache: Eficient visionlanguage-action manipulation via adaptive token caching (2025), arXiv preprint arXiv:2502.02175 4

52. Xu, W., Zhuang, L.: Kv-eficient vla: A method of speed up vision language model with rnn-gated chunked kv cache (2025), arXiv preprint arXiv:2509.21354 2, 4

53. Xu, Y., Yang, Y., Fan, Z., Liu, Y., Li, Y., Li, B., Zhang, Z.: Qvla: Not all channels are equal in vision-language-action model’s quantization (2026), arXiv preprint arXiv:2602.03782 4

54. Yang, Y., Wang, Y., Wen, Z., Liu, Z., Zou, C., Zhang, Z., Wen, C., Zhang, L.: Eficientvla: Training-free acceleration and compression for vision-language-action models (2025), arXiv preprint arXiv:2506.10100 4

55. Yu, C., Wang, Y., Guo, Z., Lin, H., Xu, S., Zang, H., Zhang, Q., Wu, Y., Zhu, C., Hu, J., et al.: Rlinf: Flexible and eficient large-scale reinforcement learning via macro-to-micro flow transformation. arXiv preprint arXiv:2509.15965 (2025) 9

56. Yu, Z., Wang, B., Zeng, P., Zhang, H., Zhang, J., Gao, L., Song, J., Sebe, N., Shen, H.T.: A survey on eficient vision-language-action models (2025), arXiv preprint arXiv:2510.24795 4

57. Zhang, J., Hsieh, Y., Wang, Z., Lin, H., Wang, X., Wang, Z., Lei, Y., Zhang, M.: Quantvla: Scale-calibrated post-training quantization for vision-language-action models (2026), arXiv preprint arXiv:2602.20309 4

58. Zhang, Y., et al.: Sqap-vla: Saliency-aware quantization and pruning for visionlanguage-action models (2025), arXiv preprint arXiv:2512.08813 4

59. Zhao, T.Z., et al.: Learning fine-grained bimanual manipulation with low-cost hardware. In: Robotics: Science and Systems (RSS) (2023), introduces Action Chunking with Transformers (ACT) 3

60. Zheng, W., Li, B., Xu, B., Feng, E., Gu, J., Chen, H.: Leveraging os-level primitives for robotic action management. arXiv preprint (2025), arXiv preprint arXiv:2508.10259 2

## A PolicyTrim Training Algorithm

## A.1 PolicyTrim Training Algorithm

We summarize the two-stage optimization procedure of PolicyTrim below. PolicyTrim improves intrinsic policy eficiency through two sequential GRPO-based post-training stages.

Algorithm 1 PolicyTrim Post-Training   
Require: Policy $\pi \theta$ , reference policy $\pi _ { \mathrm { r e f } }  \pi _ { \theta }$ , task distribution $\tau$   
Require: Acceptance ratios $T ,$ max chunk length $H ,$ , group size $G ,$ step budget $S _ { \mathrm { b a s e } }$   
Require: Success reward r , coeficients $\beta _ { \mathrm { h o r } } , \lambda _ { \mathrm { s t a b } } , \beta _ { \mathrm { K L } }$ , constants σ<sub>floor</sub>, ϵ<sub>adv</sub>, ϵ<sub>clip</sub>   
GR ${ \ ` P O  – U p d a t e ( \pi _ { \theta } , \{ R _ { i } \} _ { i = 1 } ^ { G } ) }$   
$A _ { i }  ( R _ { i } - \mu _ { R } ) / ( \sigma _ { R } + \epsilon _ { \mathrm { a d v } } )$ , with $\mu _ { R } , \sigma _ { R }$ computed over $\{ R _ { i } \} _ { i = 1 } ^ { G }$   
Update π with clipped KL-regularized GRPO using $\{ A _ { i } \} _ { i = 1 } ^ { G }$   
Stage 1: Reliable Action Chunk Extension   
1: for each iteration do   
2: Sample $( l , o _ { 0 } ) \sim \tau$   
3: for $i = 1$ to G do   
4: Sample $\gamma _ { i } \sim I ,$ set $h _ { i } \gets \lfloor \gamma _ { i } H \rfloor$ , and roll out $\tau _ { i }$   
5: $R _ { i }  \mathcal { T } _ { \mathrm { s u c c } } ^ { ( i ) } ( r _ { \mathrm { s u c c } } + \beta _ { \mathrm { h o r } } \gamma _ { i } )$   
6: end for   
7: $G R P O \ – U p d a t e ( \pi _ { \theta } , \{ R _ { i } \} _ { i = 1 } ^ { G } )$   
8: end for   
9: $\pi _ { \mathrm { r e f } }  \pi _ { \theta }$   
Stage 2: Redundancy-Aware Step Reduction   
10: for each iteration do   
11: Sample $( l , o _ { 0 } ) \sim \tau$ and set $h \gets \lfloor \operatorname* { m a x } ( T ) H \rfloor$   
12: for $i = 1$ to G do   
13: Roll out $\tau _ { i }$ with window h, and record $\mathcal { T } _ { \mathrm { s u c c } } ^ { ( i ) }$ and $S _ { i }$   
14: end for   
15: Compute $\mu _ { \mathrm { g r o u p } } , \sigma _ { \mathrm { g r o u p } }$ over successful $\{ S _ { i } \}$   
16: for $i = 1$ to G do   
17: $\begin{array} { r l } & { R _ { i } \gets Z _ { \mathrm { s u c c } } ^ { ( i ) } \biggl ( r _ { \mathrm { s u c c } } + \frac { \operatorname* { m a x } ( 0 , S _ { \mathrm { b a s e } } - S _ { i } ) } { S _ { \mathrm { b a s e } } } - \lambda _ { \mathrm { s t a b } } \operatorname { t a n h } \bigl ( \frac { | S _ { i } - \mu _ { \mathrm { g r o u p } } | } { \operatorname* { m a x } ( \sigma _ { \mathrm { g r o u p } } , \sigma _ { \mathrm { f l o o r } } ) } \bigr ) \biggr ) } \end{array}$   
18: end for   
19: $G R P O \ – U p d a t e ( \pi _ { \theta } , \{ R _ { i } \} _ { i = 1 } ^ { G } )$   
20: end for   
21: return $\pi \theta$

Across all stages, each reward term is rescaled to the range of $r _ { \mathrm { s u c c } }$ . In Stage 2, the combined reward is renormalized after aggregation to preserve this range. Rewards are then standardized within each sampled group to compute grouprelative advantages, and the policy is optimized with a clipped KL-regularized GRPO objective.

## B Implementation Details

All experiments used critic-free GRPO. We applied group-relative reward normalization and updated the policy directly from rollout returns, without a critic or separate reward model. Unless otherwise noted, training was conducted on a single node, using a GRPO group size of 8 throughout.

PolicyTrim has two sequential stages. Stage 1 extends reliable action chunks. Stage 2 reduces redundant steps using a step-saving reward and group-anchored stability regularization. Unless otherwise specified, their default coeficients were 0.8 and 0.2, respectively. Pi0.5 and GR00T used both stages, whereas OpenVLA-OFT used only Stage 2 due to backbone constraints. Each stage was trained for up to 500 epochs.

## B.1 Training Hyperparameters

Pi0.5. The global batch size was 2048 on LIBERO. The environment horizon was 160 for Spatial and Object, 240 for Goal, and 400 for LIBERO-10. The prediction horizon was fixed to 15 on LIBERO. The action horizon was set to [5, 10, 15] for Spatial, Goal, and Object, and to [10, 15] for LIBERO-10. On ManiSkill, the global batch size was 5120, the environment horizon was 80, the prediction horizon was 10, and the action horizon was [5, 10]. On MetaWorld, the global batch size was 2048, the environment horizon was 100, the prediction horizon was 10, and the action horizon was [5, 10].

OpenVLA-OFT. The global batch size was 16384 on LIBERO. The environment horizon was 256 for Spatial, Object, and Goal, and 512 for LIBERO-10. The prediction horizon was fixed to 8, with action horizon [8]. On ManiSkill, the global batch size was 640, the environment horizon was 80, and both the prediction horizon and action horizon were fixed to 8.

GR00T. The global batch size was 1024 throughout. On LIBERO-Spatial, the environment horizon was 160, with prediction horizon 10 and action horizon [5, 10]. On LIBERO-Object, Goal, and LIBERO-10, the prediction horizon was fixed to 10 and the action horizon to [10], with environment horizons of 160, 240, and 320, respectively.

## B.2 Backbone-Specific Settings

Pi0.5. Pi0.5 used the OpenPI backbone with FSDP in no-shard mode and without gradient checkpointing. All runs used three denoising steps. The action head used Flow-SDE, and the learning rate was $5 \times 1 0 ^ { - 6 }$

OpenVLA-OFT. OpenVLA-OFT used the implementation in bfloat16 precision with full-parameter fine-tuning. All checked-in GRPO configurations enabled gradient checkpointing. The learning rate was $2 \times 1 0 ^ { - 5 }$

GR00T. GR00T used bfloat16 precision and four denoising steps. Gradient checkpointing was disabled. The learning rate was $5 \times 1 0 ^ { - 6 }$

## C Additional Results

## C.1 Qualitative Results

As shown in Fig. 5, PolicyTrim consistently reaches the target state with more compact execution trajectories across both backbones. On GR00T, the optimized policy preserves reliable long-horizon behavior while reducing unnecessary motion before successful placement. On OpenVLA-OFT, PolicyTrim produces noticeably shorter and cleaner trajectories, especially in tasks that otherwise involve hesitation or redundant adjustments. These examples qualitatively support our main claim that PolicyTrim improves intrinsic policy eficiency by simultaneously enlarging the reliable action horizon and reducing redundant physical steps.

![](images/55de9428aa8f252baa3363280c3374150ad7f8a5155a47ec1a4cf4c588f069b1.jpg)  
Fig. 5: Qualitative comparisons on GR00T and OpenVLA-OFT. For each instruction, we compare execution snapshots of the baseline policy and PolicyTrim. Across both backbones, PolicyTrim reaches the goal with more compact trajectories and fewer redundant physical steps. Green boxes highlight salient late-stage frames where task completion occurs.

Table 8: Ablation on group size G for $\pi _ { 0 . 5 }$ on the four LIBERO subsets. We report success rate (SR), average physical steps $\left( S _ { \mathrm { t o t a l } } \right)$ , average action chunk execution length $\left( \boldsymbol { h } _ { \mathrm { c h u n k } } \right)$ , and end-to-end execution speedup (Spd).
<table><tr><td rowspan="3">G</td><td colspan="3">Spatial</td><td colspan="4">Object</td><td colspan="4">Goal</td><td colspan="4">Long</td></tr><tr><td>SR</td><td> $S _ { \mathrm { t o t a l } } \ h _ { \mathrm { c h u n k } }$ </td><td></td><td>Spd↑ SR</td><td></td><td> $S _ { \mathrm { t o t a l } } \ h _ { \mathrm { c h u n k } }$ </td><td></td><td>Spd↑ SR</td><td> $S _ { \mathrm { t o t a l } }$ </td><td> $h _ { \mathrm { c h u n k } }$ </td><td></td><td>Spd↑ SR</td><td> $S _ { \mathrm { t o t a l } } \ h _ { \mathrm { c h u n k } }$ </td><td></td><td>Spd↑</td></tr><tr><td>8 97.8 59.8</td><td></td><td>15</td><td></td><td>5.43 98.5</td><td>64.3</td><td>15</td><td></td><td>5.83 98.8 63.5</td><td></td><td>15</td><td></td><td></td><td>5.23 93.3 171.8</td><td>10</td><td>2.91</td></tr><tr><td>10 97.4 60.5</td><td></td><td>15</td><td></td><td>5.37 98.1</td><td>65.0</td><td>15</td><td>5.77 98.4 64.2</td><td></td><td></td><td>15</td><td></td><td>5.17 92.9 172.6</td><td></td><td>10</td><td>2.89</td></tr><tr><td>12 97.9 59.4</td><td></td><td>15</td><td>5.47</td><td>98.6</td><td>63.9</td><td>15</td><td>5.87 98.9</td><td></td><td>63.1</td><td>15</td><td></td><td>5.26 93.5 171.2</td><td></td><td>10</td><td>2.92</td></tr><tr><td>14 97.660.1</td><td></td><td>15</td><td>5.41</td><td>98.3</td><td>64.7</td><td>15</td><td>5.80 98.6 63.8</td><td></td><td></td><td>15</td><td></td><td>5.2093.1 172.1</td><td></td><td>10</td><td>2.90</td></tr><tr><td>16 97.3 60.7</td><td></td><td>15</td><td>5.35</td><td>98.0</td><td>65.1</td><td>15</td><td>5.76 98.2 64.4</td><td></td><td></td><td>15</td><td></td><td>5.15 92.7 172.7</td><td></td><td>10</td><td>2.89</td></tr></table>

Table 9: Ablation study of diferent PolicyTrim components on the four LIBERO subsets using π .
<table><tr><td rowspan="3">Reliable Chunk Step-Saving Group-Anchored Extension</td><td rowspan="3">Reward</td><td rowspan="3">Regularization</td><td colspan="3">Spatial</td><td colspan="4">Object</td></tr><tr><td>SR  $S _ { \mathrm { t o t a l } }$ </td><td></td><td> $h _ { \mathrm { c h u n k } }$ </td><td>Spd↑ SR</td><td> $S _ { \mathrm { t o t a l } }$ </td><td> $h _ { \mathrm { c h u n k } }$ </td><td>Spd↑</td></tr><tr><td>X</td><td>X</td><td>X</td><td>97.8 108.3</td><td>5</td><td></td><td></td><td>1.00 99.1 125.0</td><td>5</td><td>1.00</td></tr><tr><td>√</td><td>X</td><td>X</td><td>97.4 111.6</td><td>15</td><td></td><td></td><td>2.91 98.8 128.6</td><td>15</td><td>2.92</td></tr><tr><td>X</td><td>√</td><td>X</td><td>95.1 86.9</td><td>5</td><td></td><td></td><td>1.25 96.7 99.8</td><td>5</td><td>1.25</td></tr><tr><td>X</td><td>√</td><td>√</td><td>97.1 65.5</td><td>5</td><td></td><td></td><td>1.65 98.4 72.6</td><td>5</td><td>1.72</td></tr><tr><td>√</td><td>√</td><td>√</td><td>97.859.8</td><td>15</td><td></td><td></td><td>5.43 98.5 64.3</td><td>15</td><td>5.83</td></tr><tr><td>Reliable Chunk Step-Saving Group-Anchored</td><td></td><td></td><td colspan="3">Goal</td><td colspan="4">Long</td></tr><tr><td>Extension</td><td>Reward</td><td>Regularization</td><td>SR Stotal hchunk Spd↑ SR Stotal hchunk Spd↑</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>X</td><td>X</td><td>X</td><td>98.7 110.6</td><td></td><td>5</td><td></td><td>1.0093.0 249.8</td><td>5</td><td>1.00</td></tr><tr><td>√</td><td>X</td><td>X</td><td>98.3 113.9</td><td></td><td>15</td><td></td><td>2.91 92.6 253.4</td><td>10</td><td>1.97</td></tr><tr><td>X</td><td>√</td><td>X</td><td>95.888.9</td><td></td><td>5</td><td></td><td>1.2489.8 205.2</td><td>5</td><td>1.22</td></tr><tr><td>X</td><td>√</td><td>√</td><td>97.7 68.9</td><td></td><td>5</td><td></td><td>1.61 91.9 191.3</td><td>5</td><td>1.31</td></tr><tr><td>√</td><td>√</td><td>√</td><td>98.8 63.5</td><td></td><td>15</td><td></td><td>5.23 93.3 171.8</td><td>10</td><td>2.91</td></tr></table>

## C.2 Ablation on Group Size

The results show that PolicyTrim is largely insensitive to the choice of group size G. As G varies from 8 to 16, all four LIBERO subsets exhibit only minor fluctuations in success rate, total execution steps, and end-to-end speedup, without any consistent trend. These results suggest that the default choice $G = 8$ is already suficient in practice.

## C.3 Ablation on Components

In Table 9, we extend the component ablation study to all four LIBERO subsets using the $\pi _ { 0 . 5 }$ model. The results show a consistent pattern across all subsets. Reliable Action Chunk Extension alone mainly improves execution speed by enlarging the action horizon, but slightly increases the number of physical steps and can also lead to a small drop in success rate. In contrast, the Step-Saving Reward alone reduces $S _ { \mathrm { t o t a l } }$ substantially, but this comes at the cost of degraded task success, indicating that shorter trajectories alone are not necessarily reliable. Adding Group-Anchored Regularization largely recovers task competence while further improving execution eficiency, showing its role in stabilizing concise behaviors. Finally, enabling all three components together gives the best overall trade-of, consistently achieving the highest end-to-end speedup while preserving strong task performance across Spatial, Object, Goal, and Long.

## C.4 Real-World Deployment

We deploy PolicyTrim on a real robot platform and evaluate it on three tabletop manipulation tasks: FlipMug, HangMug, and TapeBox. Figure 6 shows the real-world execution visualization on the FlipMug task. More real-world visualizations are available on the project GitHub page.

![](images/8ad93b27c97a28302d0ee98f3bf8d9e378e0547a1f9c20ff6ebef75a8e5f8363.jpg)  
Fig. 6: Real-world execution visualization on the FlipMug task.

## C.5 Robustness under Visual Perturbations

We further evaluate PolicyTrim under visual distribution shifts in simulation. Specifically, we test on LIBERO-Spatial with two visual perturbations: Gaussian blur with kernel size k = 13 and 50% random occlusion. These perturbations evaluate whether the learned eficient execution strategy remains robust when visual observations become degraded or partially missing.

Table 10: Simulation robustness results on LIBERO-Spatial under visual perturbations. We report SR / Step, where SR is success rate in % and Step is the average number of physical execution steps.
<table><tr><td>Method</td><td>Gaussian Blur Random Occlusion</td></tr><tr><td>Baseline</td><td>79.6 128.9 83.2 120.3</td></tr><tr><td>PolicyTrim 88.4 / 77.6</td><td>90.6 / 75.9</td></tr></table>

As shown in Table 10, PolicyTrim consistently outperforms the baseline under both perturbations. Besides improving success rate, it also substantially reduces the number of physical steps, indicating that PolicyTrim does not merely overfit to clean visual observations but learns a more robust and concise execution strategy.

## C.6 Post-Training Cost

PolicyTrim is built on top of the RLinf framework and follows its rollout and environment-interaction settings for each task. The total post-training cost is approximately 68 hours wall-clock time on 8×H100 GPUs, compared with approximately 78 hours for the corresponding RLinf setting. The reduced training time is mainly because the extended reliable action chunk horizon reduces the number of inference calls required per rollout. Since the learned policy-eficiency improvement generalizes across tasks and can be reused after post-training, this one-time training cost is justified by the resulting deployment-time speedup.

## C.7 Horizon-Sweep Baseline

To verify that PolicyTrim’s improvement does not simply come from executing longer action chunks, we compare against fixed-horizon execution baselines. For π , we evaluate diferent fixed execution horizons h and report the resulting success rate and physical step count.

Table 11: Horizon-sweep baseline for $\pi _ { 0 . 5 }$ . Fixed larger horizons degrade success rate, while PolicyTrim learns to extend the reliable horizon through RL post-training.

<table><tr><td>Method</td><td>SR Step</td></tr><tr><td>Fixed h = 5</td><td>97.8 108.3</td></tr><tr><td>Fixed h = 10</td><td>97.2 109.5</td></tr><tr><td>Fixed h = 15 Fixed</td><td>94.1 110.2</td></tr><tr><td> $h = 2 0$ </td><td>93.1 117.5</td></tr><tr><td>Stage 1 only Full PolicyTrim 97.8 59.8</td><td>97.2 111.8</td></tr></table>

As shown in Table 11, naively increasing the fixed execution horizon causes the success rate to drop from 97.8% to 93.1%. In contrast, Stage 1 maintains a high success rate under an extended action horizon, confirming that the gain comes from RL-based reliable chunk extension rather than from simply forcing the model to execute longer chunks. The full PolicyTrim pipeline further reduces the physical step count from 111.8 to 59.8 through Stage 2 step reduction.

## C.8 Hyperparameter Sensitivity

We evaluate the sensitivity of PolicyTrim to three key hyperparameters: the stepbudget multiplier α, the GRPO group size G, and the stability regularization coeficient $\lambda _ { \mathrm { s t a b } }$ . All experiments are conducted on LIBERO-Spatial using π<sub>0.5</sub>. Table 12: Hyperparameter sensitivity on LIBERO-Spatial. We report SR / Step. Default values are shown in bold.

<table><tr><td>Value 1.1 1.2 1.3 1.5 2.0 α SR/Step 97.2/61.2 97.5/60.4 97.8/59.8 97.9/60.5 97.6/61.8</td></tr><tr><td>Value 0.05 0.10 0.15 0.20 0.30  $\lambda _ { \mathrm { s t a b } }$  SR/Step 97.1/58.9 97.5/59.3 97.8/59.8 97.7/60.4 97.2/61.2</td></tr></table>

The results in Table 12 show that PolicyTrim is robust to these hyperparameters. Across all tested settings, the success rate varies within a narrow range and the physical step count remains stable. This indicates that the default configuration works well without requiring careful per-task tuning.

## D Failure Case Analysis

We provide several representative failure cases for qualitative analysis. These examples highlight scenarios where the policy does not fully complete the task or requires additional corrections during execution.

![](images/47217ccfed7c7a4e1f1c5b07aa039d7b13650ecf56d374b2a9b6ae920973d6d1.jpg)  
Fig. 7: Failure case without group-anchored stability regularization. The policy approaches the bowl with insuficient clearance, causing a collision and task failure.

In this failure case, removing the group-anchored stability regularization causes the policy to overemphasize step-saving behavior and pursue execution speed too aggressively. To grasp the bowl on the wooden cabinet more quickly, the robot follows a shorter but unsafe trajectory, without lifting the end-efector high enough before approaching the target. As a result, the gripper collides with the bowl, knocking it out of the reachable workspace and leading to task failure. This example highlights the importance of group-anchored stability regularization in preventing overly aggressive eficiency optimization and preserving safer interaction behavior.