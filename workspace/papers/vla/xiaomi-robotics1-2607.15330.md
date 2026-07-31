# Xiaomi-Robotics-1: Scaling Vision-Language-Action Models with over 100K Hours of Real-World Trajectories

Xiaomi Robotics¹

## Abstract

We present Xiaomi-Robotics-1, a foundational vision-language-action (VLA) model capable of (1) following diverse language instructions to perform a wide range of mobile manipulation tasks in unseen environments out-of-the-box, and (2) efficiently adapting to novel downstream tasks with minimal fine-tuning data. We propose a two-stage training recipe consisting of pre-training and post-training. During pre-training, we imbue the model with broad and generalizable actiongeneration capabilities by training on over 100k hours of real-world manipulation trajectories, collected via UMI devices across a massive scale of environments and tasks. Crucially, we develop a scalable auto-labeling pipeline that annotates trajectory clips with natural languages describing scene state transitions, providing rich and precise conditioning for action learning. During post-training, we aim to align these capabilities with robot embodiments and imperative task instructions that humans naturally use to prompt robots, effectively mapping descriptive state transition understanding into actionable task prompts. Extensive experiments demonstrate strong scaling behavior. Xiaomi-Robotics-1 consistently improves with increased data scales and model sizes during pre-training. This scaling behavior directly transfers to post-training, where a stronger pre-training model yields better out-of-the-box performance in real-robot evaluations within unseen environments. Furthermore, Xiaomi-Robotics-1 serves as a strong robot foundation policy that can be efficiently fine-tuned on complex, dexterous tasks with high data efficiency. Across multiple simulation benchmarks, Xiaomi-Robotics-1 outperforms state-of-the-art methods. Notably, it establishes a new state-of-the-art with a 57.4% success rate on RoboCasa365, surpassing the previous best of 46.6%. Furthermore, it achieves an average score of 20.07 on RoboDojo, significantly outperforming the prior state-of-the-art (13.07). Code and model checkpoints will be released. Project page: https://robotics.xiaomi.com/xiaomi-robotics-1.html

## 1 Introduction

The remarkable capabilities of modern large models are fundamentally driven by scale, where massive and diverse training corpora have underpinned unprecedented leaps in performance for both large language models |7, 22, 27, 45| and vision-language models |1, 14, 62, 63]. Recent work on vision-language-action (VLA) models |4, 5, 24, 25, 54, 71, 76| and world-action models (WAM) 36, 81, 83| has produced increasingly promising results in robot manipulation, with early evidence that policies become more capable and generalizable as the training data grows in scale and diversity. Following the same scaling trajectory of large models is therefore a natural and appealing direction for robotics. However, robotics is hindered by a unique bottleneck of data. The dominant data collection paradigm, real-robot teleoperation, is slow, costly, and hardware-bound, making it difficult to scale. Furthermore, teleoperated data tends to be highly redundant, concentrated on a narrow slice of tasks and environments, limiting the diversity of the data.

![](images/d4e1db0fa1b467b05991e0429471f7815a14df8030079675d1b8a5e304a67ef1.jpg)  
Figure 1 Overview. Xiaomi-Robotics-1 is pre-trained on over 100k hours of real-world UMI trajectories with auto-labeled state-transition language prompts. It is then aligned to robot embodiments and imperative instruction prompts via cross-embodiment post-training. Xiaomi-Robotics-1 scales effectively with data and model size. It is able to perform multiple tasks in unseen environment out-of-the-box and learn new tasks efficiently.

We present Xiaomi-Robotics-1 (Fig. 1), a foundational vision-language-action (VLA) model trained on a massive scale of real-world manipulation trajectories. Drawing inspiration from the training paradigms of large language models, we propose a two-stage training recipe comprising pre-training and post-training. During pre-training, we endow the model with robust and generalizable action-generation capabilities by leveraging data sources that scale readily in both volume and diversity. Specifically, we curate a dataset of over 100k hours of real-world manipulation trajectories with UMI devices [17], spanning a wide range of environments and tasks. Traditional trajectory labeling typically requires manual segmentation by task semantics and language annotations—a labor-intensive process that becomes prohibitive at this scale. To address this challenge, we develop a scalable auto-labeling pipeline that leverages a pre-trained vision-language model (VLM) [70] to annotate fixed-length trajectory segments with language descriptions detailing scene state transitions. These annotations provide precise and sufficient semantic supervision. Trained on these data, the model learns to generate actions that transform the scene from its current state to the language-specified target state (Fig. 6). In the post-training phase, we utilize over 10k hours of cross-embodiment data to align the strong action-generation capabilities acquired during pre-training. This stage bridges two gaps: adapting the model from generating actions for UMI grippers to actions for robot embodiments, and transitioning from state-transition prompts to imperative instructions typically used by humans to prompt robots. After post-training, Xiaomi-Robotics-1 is able to follow instructions and perform a wide range of tasks in unseen environments. Furthermore, it serves as a strong robot foundation policy that can be efficiently fine-tuned to learn new tasks.

We perform extensive experiments to study the scaling properties of Xiaomi-Robotics-1. Results show that Xiaomi-Robotics-1 scales effectively during the pre-training phase, achieving lower validation action errors as data and model scale up. Moreover, the scaling behavior observed in pre-training directly transfers to post-training, where stronger pre-training models yield better post-training success rates in out-of-the-box real-robot evaluation in unseen environments. These results are encouraging, as they indicate that we are able to continue improving performance as we further scale data and model size. In addition, we fine-tune the model on multiple complex dexterous tasks with minimal data. Xiaomi-Robotics-1 achieves an average success rate of 75% across four challenging tasks given less than 10 hours of data per task on average, outperforming π0.5 [5] which obtains 40%. In addition, we evaluate Xiaomi-Robotics-1 on four challenging simulation benchmarks, i.e., RoboCasa [52], RoboCasa365 [53], VLABench [87], and RoboDojo [12]. Xiaomi-Robotics-1 achieves state-of-the-art results across all four benchmarks. Notably, it sets a new state-of-the-art with a 57.6% success rate on RoboCasa365, a substantial leap from the previous best of 46.6%. On RoboDojo, it delivers an average score of 20.07, significantly outperforming the prior state-of-the-art of 13.07. Finally, Xiaomi-Robotics-1 enables the robot to autonomously accomplish a long-horizon, room-level mobile manipulation task of suitcase packing that spans over 10 minutes (see the project page for the video). Code and model checkpoints will be released. Project page: https://robotics.xiaomi.com/xiaomi-robotics-1.html

![](images/b596300657e1a7a92761ceb6d7e5229bfc9dc04fd9351f705bbbcb79d66d91df.jpg)  
Figure 2 Model Architecture. Xiaomi-Robotics-1 adopts a Mixture-of-Transformers [44] architecture that couples a pre-trained VLM with a DiT. The VLM encodes the observation and language instruction, and additionally predicts action chunks via Choice Policies 59] to accelerate training convergence. Conditioned on the robot state and the VLM's KV cache of the observation and language tokens, the DiT generates the action chunk via flow matching. Note that the action-related tokens from the VLM are excluded from the DiT's attention computation.

## 2 Xiaomi-Robotics-1

Xiaomi-Robotics-1 is an end-to-end vision-language-action (VLA) model trained at scale on heterogeneous data sources, including UMI trajectories, cross-embodiment robot trajectories, and vision-language data Given an observation ot and a language instruction l, the model πθ is trained to predict an action chunk $\mathbf { a } _ { t : t + H }$ by maximizing the log-likelihood over the training dataset D:

$$
\operatorname* { m a x } _ { \theta } \mathbb { E } _ { ( \mathbf { o } _ { t } , l , \mathbf { a } _ { t : t + H } ) \sim \mathcal { D } } \log \pi _ { \theta } ( \mathbf { a } _ { t : t + H } \mid \mathbf { o } _ { t } , l )
$$

We adopt a two-stage training recipe consisting of pre-training and post-training. Pre-training leverages a scalable non-robot dataset with rich open-world diversity to endow the model with broad and generalizable representations for action generation. Post-training then aligns these representations to robot embodiments and instruction-conditioned action generation, using a high-quality set of cross-embodiment data. In the following sections, we describe the details of model architecture, data curation, and training recipe.

## 2.1 Model

As illustrated in Fig. 2, Xiaomi-Robotics-1 adopts a Mixture-of-Transformers (MoT) [44] architecture consisting of a pre-trained vision-language model (VLM) (i.e., Qwen3-VL [3]) and a diffusion transformer (DiT) |57]. The DiT matches the VLM in the number of layers but employs a smaller hidden size for faster inference speed. The model parameters for different scaling variants of Xiaomi-Robotics-1 are detailed in

Table 1 Model configurations for different scaling variants of Xiaomi-Robotics-1.
<table><tr><td rowspan="2">Model</td><td rowspan="2"> $\#$  Layers</td><td colspan="2">VLM</td><td colspan="2">DiT</td><td rowspan="2">Total Params</td></tr><tr><td>Hidden Size</td><td>Params</td><td>Hidden Size</td><td>Params</td></tr><tr><td>Xiaomi-Robotics-1-2B</td><td>28</td><td>2048</td><td>2.1B</td><td>1024</td><td>470M</td><td>2.6B</td></tr><tr><td>Xiaomi-Robotics-1-5B</td><td>36</td><td>2560</td><td>4.4B</td><td>1024</td><td>604M</td><td>5.1B</td></tr><tr><td>Xiaomi-Robotics-1-10B</td><td>36</td><td>4096</td><td>8.8B</td><td>2048</td><td>1.5B</td><td>10.5B</td></tr></table>

Tab. 1. The VLM takes the current observation $\mathbf { o } _ { t }$ and language instruction l as inputs. Conditioned on the robot proprioceptive state $\mathbf { s } _ { t }$ and the KV cache produced by the VLM, the DiT generates the action chunk via flow-matching [49]:

$$
\begin{array} { r } { L _ { \mathrm { F l o w } } ( \theta ) = | | \mathbf { v } _ { \theta } ( \mathbf { o } _ { t } , l , \mathbf { s } _ { t } , \tilde { \mathbf { a } } _ { t : t + H } ^ { \tau } , \tau ) - \mathbf { u } ( \tilde { \mathbf { a } } _ { t : t + H } ^ { \tau } , \mathbf { a } _ { t : t + H } , \tau ) | | _ { 2 } ^ { 2 } } \end{array}
$$

τ is the flow-matching timestep. $\tilde { \mathbf { a } } _ { t : t + H } ^ { \tau } = \tau \mathbf { a } _ { t : t + H } + ( 1 - \tau ) \boldsymbol { \epsilon }$ is the noisy action where $\mathbf { \epsilon } \gets \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ . Following [4], we sample timestep τ from a Beta distribution, placing more weight on noisier timesteps during training:

$$
u \sim \mathrm { B e t a } ( 1 . 5 , 1 ) , \quad \tau = ( 1 - u ) * 0 . 9 9 9 \in [ 0 , 0 . 9 9 9 ]
$$

Similar to [8], we leverage adaptive normalization layers (adaLN) [57] to inject the flow-matching timestep condition into the DiT for action generation. During inference, we initialize the predicted action chunk from a random noise $\mathbf { a } _ { t : t + H } ^ { \tau = 0 } \sim \mathcal { N } ( \mathbf { 0 } , I )$ . The clean action chunk is recovered via a 5-step Euler integration, $\begin{array} { r } { \mathbf { a } _ { t : t + H } ^ { \tau + \Delta \tau } = \mathbf { a } _ { t : t + H } ^ { \tau } + \Delta \tau \cdot \mathbf { v } _ { \theta } ( \mathbf { o } _ { t } , l , \mathbf { s } _ { t } , \mathbf { a } _ { t : t + H } ^ { \tau } , \tau ) } \end{array}$ , where the step size is set to $\Delta \tau = 0 . 2$

To accelerate convergence [58], we introduce an auxiliary action-generation supervision on the VLM. Specifically we leverage Choice Policies [59] to enable action generation directly within the VLM framework [8]. We encode the robot state into a token using a multi-layer perceptron (MLP) and append it, along with the action and score query tokens, to the end of the vision-language token sequence. The outputs corresponding to the action and score query tokens predict K candidate action chunks and their associated K scores, respectively. We adopt a winner-takes-all paradigm as in [59], where only the candidate with the smallest $L _ { 1 }$ loss is included in action loss computation:

$$
L _ { \mathrm { R e g r e s s i o n } } ( \theta ) = \lvert \lvert \hat { \mathbf { a } } _ { t : t + H } ^ { * } - \mathbf { a } _ { t : t + H } \rvert \rvert _ { 1 } + \sum _ { k } ^ { K } \lvert \lvert \hat { s } _ { k } - s _ { k } \rvert \rvert _ { 2 } ^ { 2 }
$$

Let $\hat { \mathbf { a } } _ { t : t + H } ^ { k }$ denotes the k-th predicted candidate action chunk, then $\hat { \mathbf { a } } _ { t : t + H } ^ { * }$ is the one with the smallest $L _ { 1 }$ distance to the ground truth $\mathbf { a } _ { t : t + H } . \hat { s } _ { k }$ is the predicted score for the k-th candidate, and its regression target $s _ { k }$ is defined as $s _ { k } = | | \hat { \mathbf { a } } _ { t : t + H } ^ { k } - \mathbf { a } _ { t : t + H } | | _ { 1 }$ . That is, the $L _ { 1 }$ distances between the $K$ predicted action chunks and the ground-truth action chunk serve as the target labels for score prediction.

Applying action-generation supervision directly on the VLM steers its representations toward features that better support action generation, thereby making the DiT learning more effective. However, we empirically observe that letting the DiT tokens attend to the KV cache of these action-related tokens degrades performance. We hypothesize that this arises from a shortcut in which the DiT simply copies the actions generated by the VLM rather than effectively grounding its own generation in the visual and textual context. To mitigate this issue, we exclude these action-related tokens from the DiT's attention computation, constraining the DiT tokens to attend solely to the representations of the language instruction and visual observations.

## 2.2 Training & Data

## 2.2.1 Pre-training

During pre-training, our primary objective is to endow the model with broad and generalizable representations that transfer across diverse manipulation scenarios. To this end, we curate a dataset of over 100,000 hours of real-world manipulation trajectories, captured with Universal Manipulation Interface (UMI) handheld grippers [17] and egocentric cameras (Fig. 3). The dataset spans a diverse array of tasks collected across a massive scale of environments, including households, commercial premises, industrial sites, offices, and outdoor spaces. Traditional robot trajectory annotation requires manually segmenting trajectories according to task semantics and labeling each segment with a language instruction—a labor-intensive process that becomes prohibitive at this scale.

![](images/55a6d044dd1fe8bd9266d396af487f073ca69200fabe0f40bfa94332669c35a0.jpg)

![](images/f19245e51e2d1e5619181f32c64a7cf87856c4630758e45636f1bfec9425e913.jpg)

![](images/b5093c08496c0937ab37d73fa2534f8abed44bccf123cea35b2dea8ead696063.jpg)

![](images/5f4f8910f56aa757d2b3b71b50d28d7717c89b717eb18f8693f7aa2b9b54338d.jpg)  
Figure 3 Pre-training Dataset. The pre-training dataset of Xiaomi-Robotics-1 contains over 100k hours of real-world manipulation trajectories collected with UMI devices.

To scale language annotation, we develop an auto-labeling pipeline that first divides each trajectory into equal-length segments and leverage Qwen3.5-27B [70] to caption the state transitions of both the grippers and the interacting objects in the scene within each segment (see Fig. 11 for examples). To accelerate the annotation process, we develop a producer-consumer pipeline that decouples clip segmentations from caption labeling: while CPU worker threads cut per-segment clips into an in-memory filesystem, client threads keep hundreds of captioning requests in flight. This highly effcient pipeline allows us to label the entire corpus of over 100k hours in roughly two weeks. Trained on this dataset, the model learns to generate actions that drive the scene from the state in the current observation to the target state described by the language annotation.

The model is optimized to predict actions by jointly minimizing the flow-matching loss $\boldsymbol { L } _ { \mathrm { F l o w } }$ of the DiT and the regression loss $L _ { \mathrm { { R e g r e s s i o n } } }$ of the VLM choice policy. To preserve the vision-language capabilities of the pre-trained VLM, we further co-train the model on a high-quality vision-language dataset curated in our previous work [8] under the next-token prediction objective LNTp. The overall training objective is formulated as:

$$
L = L _ { \mathrm { F l o w } } + L _ { \mathrm { R e g r e s s i o n } } + \lambda L _ { \mathrm { N T P } }\tag{1}
$$

where λ is set to 0.1 in our experiments. Vision-language data and UMI trajectories are sampled at a ratio of 1:9. To maximize training throughput, we pack all vision-language tokens within a batch into a single sequence for a VLM forward pass. Since the VLM is computationally more expensive than the DiT, we

![](images/fb231d58b98092d9902f37d12aec15b5c269fe878af64047a0abfc19ced311f9.jpg)  
Figure 4 Post-training Dataset. The post-training dataset of Xiaomi-Robotics-1 comprises about 10k hours of cross-embodiment trajectories, including over 7.2k hours of in-house robot data collected with mobile manipulators and dual-arm robots, over 1k hours of instruction-labeled UMI data, and open-source robot datasets.

amortize its cost by sampling four flow-matching timesteps per sample. The resulting four DiT inputs are similarly packed and processed in one DiT pass, conditioned on the corresponding unpacked VLM KV cache.

## 2.2.2 Post-training

The goal of post-training is twofold. First, we transfer the action-generation capabilities of UMI grippers acquired during pre-training to robot embodiments. Second, we shift the language conditioning from the state-transition descriptions used in pre-training to the imperative instructions humans typically issue when prompting robots to perform tasks.

We curate the post-training dataset with cross-embodiment manipulation trajectories collected using UMI devices, static robot arms, and mobile manipulators. Specifically, we collect over 7,200 hours of robot data using mobile manipulators and dual-arm robots across a diverse range of household environments and tasks (Fig. 4). We leverage Qwen3.5 |70| to annotate human-segmented video clips with language instructions. In addition, we incorporate over 1,000 hours of human-annotated UMI data labeled with both temporal segments and language instructions. Unlike the state-transition descriptions used in pre-training, these language instructions closely mirror how humans prompt robots to perform tasks, directly matching our alignment objective in the post-training phase (see Fig. 11 and 12 for comparison). Finally, we include open-source robot datasets, including Bridge V2 [74], RT-1 [6], and DROID [28]. We filter out idle segments within trajectories to prevent the model from learning uninformative or noisy signals. In total, our post-training dataset comprises about 10,000 hours of trajectory data.

For arm actions, we adopt relative delta end-effector (EE) poses with respect to the current state:

$$
\begin{array} { r } { \mathbf { a } _ { t + i } = ( \mathbf { \Pi } _ { \mathrm { B a s e } } ^ { \mathrm { E E } } T _ { t } ) ^ { - 1 } \mathbf { \Pi } _ { \mathrm { B a s e } } ^ { \mathrm { E E } } \hat { T } _ { t + i } } \end{array}
$$

where $\mathrm { E E } _ { T _ { t } }$ denotes the pose of the end-effector with respect to the base at the current timestep t, and $\operatorname { E E } _ { \hat { m } }$ $\mathrm { B a s e } ^ { T _ { t + i } }$ represents the target end-effector pose at timestep $t + i .$ To align the arm action spaces across different embodiments, we unify the orientation of the end-effector frames across all robot data and UMI data in both the pre-training and post-training datasets. Consequently, similar arm motions (e.g., moving forward or backward with respect to the end-effector frame) yield consistent action values regardless of the underlying hardware platform. For mobile robot data, we represent the base and waist actions using the base velocity and the relative delta of the waist position, respectively. To accommodate heterogeneous embodiments, we adopt a unified action vector for all trajectory data. Although the arm actions are aligned across embodiments, the action spaces of different robots still differ in dimensionality. We mask out the dimensions corresponding to missing action components during loss computation.

![](images/14d68638c6e108b75c023bc0c2a2e7a991c403734a2ab59c58528dfcc99cad1e.jpg)

![](images/304fedfbf98275609cbb2cbf2d97acc7b1d54974b9f00f6f76068c78313efb27.jpg)  
Figure 5 Scaling of Pre-training. We show the validation action errors (MSE) from the data-scaling and model-scaling pre-training experiments. We terminate the training for 12.5% and 25% data in the data-scaling experiment early as the validation loss indicates overfitting.

We train the model with the same objective as in pre-training (Eq. 1). Vision-language data, open-source robot data, instruction-labeled UMI data, and our in-house robot data are sampled at a ratio of 0.5:0.5:0.5:8.5. After post-training, the model can be prompted with language instructions to perform a wide range of tasks in unseen environments out-of-the-box. In addition, it can efficiently adapt to novel downstream tasks with minimal amount of data.

## 3 Experiments

We design Xiaomi-Robotics-1 with scaling in mind. In this section, we investigate its scaling properties through extensive experiments. Specifically, we design experiments to answer the following questions:

• Does Xiaomi-Robotics-1 scale effectively with increasing data scale and model size during pre-training?

• Does a stronger pre-trained model translate to better post-training performance when evaluated out-ofthe-box in novel environments?

• Can Xiaomi-Robotics-1 adapt to challenging new tasks with a minimal amount of data?

• How does Xiaomi-Robotics-1 compare to other robot foundation models in real-robot experiments and simulation benchmarks?

## 3.1 Pre-training: Data and Model Scaling

Data Scaling. We perform data-scaling experiments with Xiaomi-Robotics-1-5B. Due to compute budget limit, we pre-train the model on 12.5%, 25%, 50%, and 100% of about 20k hours of UMI data, respectively. Each model is evaluated on a held-out validation set. We use the mean-squared error (MSE) between the action predicted by flow-matching and the ground truth as the evaluation metric. As shown in Fig. 5, Xiaomi-Robotics-1 attains lower validation action errors with the increase of data scale. With 12.5% and 25% of data, the validation action errors first decrease and then increase during training, indicating overfitting. In contrast, the 50% and 100% data settings yield a monotonic decrease in loss, with the 20k setting exhibiting a steeper descent. We show qualitative results of action prediction on validation data in Fig. 6.

Model Scaling. We perform model-scaling experiments on three size variants of Xiaomi-Robotics-1 (2B, 5B, and 10B) as specified in Tab. 1. All three models are trained on the same 20k hours of data as in the data-scaling experiments and then evaluated on the same held-out validation set. As illustrated in Fig. 5, Xiaomi-Robotics-1 exhibits consistent improvements in action prediction precision as the model size scales up. However, the performance gap among different model sizes are less pronounced than those observed across different data scales. This result suggests that model capacity at the billions-parameter scale may already be sufficient to capture the current dataset's distribution, thereby making data volume the primary bottleneck for further generalization. These findings do not diminish the value of model scaling, but rather highlight the critical importance of prioritizing the collection of large-scale, diverse datasets to unlock further performance gains.

![](images/e4faae12d8fb8c30d5fd222246de8f4faf624c3dd7b68f2215b5d2967faec19c.jpg)  
Figure 6 Qualitative Results of Pre-training. After pre-training, Xiaomi-Robotics-1 is able to predict action trajectories for UMI grippers on a held-out validation set according to the language description of state transitions.

## 3.2 Post-training: Out-of-the-Box Evaluation in Novel Environments

In this section, we perform post-training experiments on the cross-embodiment post-training dataset and study its out-of-the-box performance in novel environments that are unseen during training. In particular, we are interested in understanding whether the data scaling and model scaling properties from pre-training can transfer to post-training. To mitigate overfitting, for the in-house robot data, we sample a diverse subset from the whole dataset for post-training. Models are evaluated out-of-the-box in unseen environments after post-training without any per-task or per-environment fine-tuning. Specifically, we evaluate on 4 tasks (Fig. 7), i.e., shoe storage, bag packing, table organization, sofa tidying. These tasks are seen in the post-training dataset but the environments and object instances during evaluation are unseen.

## 3.2.1 Effectiveness of Scaling Pre-training Data

We first examine whether the benefits of scaling pre-training data transfer to post-training with the 5B variant of Xiaomi-Robotics-1. Using an identical training recipe, we post-train models initialized from checkpoints pre-trained on 12.5%, 25%, 50%, and 100% of 20k pre-training data (Sec. 3.1), alongside a baseline initialized from the Qwen3-VL [3 pre-trained weight without any action pre-training. Out-of-the-box evaluation results are shown in Fig. 8. The overall success rate increases monotonically with the scale of pre-training data, rising from 26% without action pre-training to 75% with 100% of pre-training data. The gains from scaling pre-training data are particular pronounced on tasks that demand contact-rich manipulation. For instance, the baseline without pre-training fails completely on shoe tidying, whereas the model pre-trained on 100% of the data reaches a 75% success rate. Notably, utilizing only 12.5% of the pre-training data more than doubles the baseline's overall success rate (26% vs. 53%). While the marginal gains gradually moderate as the pre-training corpus grows, the performance shows no sign of saturation: doubling the data from 50% to

![](images/4a341814dd810f378d636380d2be47030cf803791e51e7837735e6208c50d120.jpg)

## Shoe Storage

![](images/d54ecf790d46a5dfaa9e28bb4c29e8a73be32f3e1263caa8517469d84123e6e0.jpg)

![](images/85c420845d5b2e275887d41041260f0b67a8a36b2e3581e35af9a19902248111.jpg)  
“Pick up the pair of the yellow sneakers and place it on the shoe rack."

![](images/a0048f7b212761c714a4cf6b53272f441f660487ce768ac47a140ae698d6a00e.jpg)

![](images/603bd2fdc8691a7139366114de029b80062e78ae19fb87617c9bc397db6ea3da.jpg)

## Bag Packing

![](images/f08f8555f1ff6ed1225b7380e80066cabbc0a0b5c96d77aca6039d4a7a90b359.jpg)  
"Unzip the backpack."

![](images/9bcb403e31c3408f7e6a6ca57e775e868107c4bfc8cf3f52c50abf769c09ff1b.jpg)  
"Open the backpack and put the car model into the backpack."

![](images/872a7257352af45db82fc651a113949c0230dc67ec20417fe3323b1f4fea5252.jpg)  
“Open the backpack and put the red can into the backpack."

![](images/5043ba7e1598a21b46e8acb490ef75b74944a7135421a5f5fdf7e299184cab91.jpg)  
“Open the backpack and put the blue pouch into the backpack."

![](images/1b6ea0a13c39d97238e15a6b57d7c645faa1ea05b242782971d1aa040dd2cde1.jpg)  
“Zip up the backpack."

## Table Organization

![](images/67fcd22e11fd3474cb95b5a1d9e3f41fa863231cea971ba915924ae9ecd2a055.jpg)

![](images/c53c97f81da8735573e105461dac7f6832eb0d54206d2aadefb1f38ce420eab5.jpg)  
“Pick up the sunglasses, place it into the grey glass case, and put the glass case into the storage box.

![](images/f85da674303edbf28f57ea57434b0bc78f7577f02468ea73ca05d0f3e8f22c23.jpg)  
"Pick up the toothpaste and place it into the storage box.'

![](images/f839252bfa3ff85dd8fca388f98bb2ab8da01f7cab29082aff8ee290819de6ff.jpg)  
“Pick up yellow glue stick and place it into the storage box.'

![](images/e90b7fa04274444e89714bc00c5205cfd76844b1cf22874d3b9674a9d7706efa.jpg)  
“Pick up the apple from the table and place it on the wooden plate."

![](images/3a599890207c5321d8f5e1956ccd5cf72d056ec1b6cffdc7646415959eae4154.jpg)  
"Pick up the red fluffy toy and place it into the white bin.'

![](images/153dca5471b1900a798693e6a6b9edb69042994d57c22f5d8770c86f73c1c4a9.jpg)  
“Pick up the mini tank car and place it into the white bin."

## Sofa Tidying

![](images/29d09079c3e409bc712781f771a34a6f940781ea7e19b85173084aa534cb83d6.jpg)  
"Pick up the white clothes and place it into the laundry basket."

![](images/3564db5ff7be39111ca8120fc51183cf738b470acd23dbc690ef36add5cf21a3.jpg)  
“Pick up the black pants and place into the laundry basket."

![](images/aa87bd7a2173d242ab234aa9788a0f18cbc070657a5f202ff8617480f6e533ad.jpg)  
“Pick up the cushion and place it upright."

Figure 7 Post-training Evaluation. We evaluate the post-trained model out-of-the-box across four tasks in novel environments. Crucially, both the environments and object instances are unseen during training.

100% yields an additional 6 percentage point improvement. These findings suggest that further scaling robot pre-training data remains a highly promising avenue for achieving stronger out-of-the-box performance in unseen environments.

## 3.2.2 Effectiveness of Scaling Model Size

We further investigate the impact of model scale during post-training. Specifically, We post-train the three size variants of Xiaomi-Robotics-1 (2B, 5B, and 10B) specified in Tab. 1. These variants are initialized from checkpoints pre-trained on 20k hours of UMI pre-training data. As shown in Fig. 8, the overall success rate increases monotonically with model size, rising from 61% for the 2B variant to 75% and 79% for the 5B and 10B variants, respectively. Similar to data scaling, the gains from model scaling are most pronounced on shoe tidying, where the success rate climbs from 58% (2B) to 75% (5B) and further to 92% (10B). Performance improves consistently with model size across three out of four tasks, with the 5B and 10B variants performing comparably (80% and 77%) on sofa tidying. Combined with the results in Sec. 3.2.1, these findings suggest that pre-training data scale and model size constitute two complementary axes for improving out-of-the-box performance in out-of-distribution settings. And a stronger pre-trained model is able to translate to better out-of-the-box real-robot performance after post-training.

![](images/8133d6e88cc1520f21714a4e47c65e327d0dcb9b4af1e076aa6ff97791729c83.jpg)

![](images/7d92d92de66cba064689f52f238e69f8824d4bb5161cd70066e31bda397c7c12.jpg)  
Figure 8 Quantitative Results of Post-training. We showcase the success rates of post-trained models across different pre-training data scales and model sizes.

## 3.3 Downstream Fine-tuning: Efficient Adaptation to New Tasks

A key desideratum of robot foundation models is their ability to efficiently adapt to novel tasks with minimal data. To investigate how Xiaomi-Robotics-1 performs in this setting, we fine-tune our post-trained model on a suite of four novel challenging tasks: phone packing, laundry loading, printer refilling, and box packing (Fig. 9). Crucially, these tasks are entirely held out from the in-house robot dataset. Each task introduces different complexities: phone packing requires bimanual coordination; laundry loading is a long-horizon mobile manipulation task involving multi-step instruction following; printer refilling demands handling of highly deformable sheets of paper; and box packing evaluates language grounding across multiple objects.

To evaluate data efficiency, we fine-tune the model under two settings. For the high-data setting, we leverage a total of 144 hours of data across all tasks. For the low-data setting, we sample 25% of the data for each task from the high-data setting, resulting in a subset of 36 hours in total. The average data per task for the low-data setting is less than 10 hours, with the maximum-data task, printer refilling, containing only 10.3 hours. This poses a significant challenge for policy learning. We fine-tune Xiaomi-Robotics-1 on the data of all four tasks using the asynchronous training recipe proposed in [8]. We compare our method against two baselines: $\pi _ { 0 . 5 }$ [5| and Xiaomi-Robotics-0 8]. For $\pi _ { 0 . 5 }$ , we follow the official OpenPi¹ fine-tuning protocol and fine-tune the base model on these tasks. For Xiaomi-Robotics-0 [8], we fine-tune its pre-trained model using the asynchronous setting. Each model is evaluated for 10 trials per task. We report both the average success rate and progress, where the progress measures partial task completion based on the task-specific milestones (Tab. 6). Results are shown in Fig. 10

Xiaomi-Robotics-1 outperforms the two baseline methods in terms of average success rates and progress in both low-data and high-data settings. In the low-data setting with less than 10 hours per task on average, it achieves an average success rate of 75% and an average progress of 90% across all four tasks, significantly outperforming $\pi _ { 0 . 5 }$ with a 40% success rate and a progress of 66%. The advantage of our method is most substantial on tasks requiring dexterous manipulation and mobile manipulation. In phone packing, Xiaomi-Robotics-1 outperforms both baseline methods substantially in the two data settings. In printer refilling, Xiaomi-Robotics-1 improves the success rate of the best baseline from 20% to 70% in the low-data setting, showcasing powerful capabilities in manipulating deformable objects. In laundry loading, our method shows strong robustness over the full task horizon where failure may occur at any stage, achieving an 80% success rate and a progress of 96% in the low-data setting. In this task, $\pi _ { 0 . 5 }$ struggles with opening the washing machine while Xiaomi-Robotics-0 fails to complete the task in the low-data setting. In box packing, all methods perform relatively well compared to other tasks, reaching 100% success rates when more task-specific data are available. Overall, all methods benefit from increasing the amount of fine-tuning

![](images/a513699af5dcd48095219a286ec54325075585ee758433a8cc39af89836f35a1.jpg)

![](images/4c92cfc51d23908815d22345cbbe013827c6c4165e05c4c0ccbaca8e6c8df6d3.jpg)

## Phone Packing

![](images/2707fd0e1e3040d5599ab60a44b2013fbc046e1837466cb3d55d9131ccc7faf3.jpg)  
“Pack phone”

![](images/fbd27082268d5f0db9a1f2fcd3877846e5c5a1bd9641b4e8da34c90aa020011f.jpg)

![](images/e70bbfe25f2365616efa9c82e319261f2c4fe831423b30845af84de03f5cc9c9.jpg)

## Laundry Loading

![](images/328ad49525524120f83454ab23f3f66abdaf71b7b40aaa62aa45d74a87cb7468.jpg)  
“Open the washing machine door”

![](images/ecd9a713dd778c1483a2b0eeb4b4eb2f43528a747d49b176a3040c85011f1cb8.jpg)  
“Put the laundry basket in front of the washing machine door"

![](images/e15dd3b02a8d6724666ad19460eaf5c1c3f390b3dfd6c2042fdc5a8a401c3541.jpg)  
“Put the clothes from the laundry basket into the washing machine\*

![](images/d6ea9d4e25a158016923eace20b7ca99f227536a4f336fc33412665ae55f6947.jpg)  
“Take away the laundry basket”

![](images/efc61150fac1cc8b8da71d87e747c94e6bcf0d23ed5d24a944861313c2012ab4.jpg)  
"Close the washing machine door

![](images/f1d4145434c5a9afa72e4eba490f62408fba0f12563e78a3d8c7ebe279756962.jpg)

![](images/3c7247ef53d254e7ad59b51bae938fcf771764407e29ad37b0db14b494089f6c.jpg)

## Printer Refilling

![](images/fe09b8d1a0f935c11852577c2d2c9f791556b865718628ef62aec1c62392aef3.jpg)  
“Refill printer paper”

![](images/db898fe06a8d68126ed7d9c5bd0dc6abbd7687684d2f7b2af52e5b89c0e6de64.jpg)

![](images/3f00680f7315515703653348e8c862479244bf104c2b00a91935e7bfdeaec0f4.jpg)

![](images/247b855335f99c7f365f7a7e525fd46bd2d28387efe3260dfd50e8cdfcb709bc.jpg)  
"Put the power strip into the box”

![](images/1992adc131320efa797d053df9d4cddeb864b527dc654d971cb71e9c5f1b64aa.jpg)  
"Put the rattle drum into the box”

## Box Packing

![](images/cee68325d013b11412588dedf16048a3e663b5704caef662fd247e00ed588372.jpg)  
"Put the lip glaze into the box”

![](images/58c139fabbc0c542525ed73804a0d5d13f2a41c028b9703cc8269ac953f05cf3.jpg)  
"Put the teddy bear into the box”

![](images/54e223f6f4d1d8c448a404f97a05b7f826b270f68f6f50d2e1c19d9122b2c4d0.jpg)  
"Put the facial cleanser into the box”

Figure 9 Downstream Fine-tuning Evaluation. We fine-tune the post-trained model on four new challenging tasks with a minimal amount of data.

data, but Xiaomi-Robotics-1 is substantially more data-efficient. We attribute this advantage to large-scale pre-training which exposes the model to diverse environments and tasks, and careful post-training alignment that aligns the strong manipulation capabilities acquired during pre-training to robot embodiments and instruction prompts. These results demonstrate that Xiaomi-Robotics-1 can serve as a strong foundation robot policy for efficient adaptation to novel tasks.

## 3.4 Simulation Benchmarks

In this section, we evaluate Xiaomi-Robotics-1 on four simulation benchmarks.

• RoboCasa [52]: RoboCasa features single-arm manipulation in realistic kitchen environments. The benchmark contains 24 everyday kitchen manipulation tasks spanning pick-and-place, articulated-object interaction, appliance operation, and coffee-making. In order to test generalization capabilities, it evaluates policies on unseen object instances and includes two scenes of which the styles were unseen in the training data among the five evaluation scenes. For training, we use the official set of 300 synthetic demonstrations provided by the benchmark. Following the standard evaluation protocol, we report the average success rate over 100 evaluation episodes per task across the five evaluation scenes. Results are shown in Tab. 2.

• RoboCasa365 [53]: RoboCasa365 extends RoboCasa into a large-scale simulation benchmark for evaluating general-purpose robot manipulation, with a particular emphasis on generalization beyond the training task distribution. It expands the original 24-task benchmark to 365 tasks across over 2,500 procedurally generated kitchen scenes and 3,200 object instances, covering both short-horizon manipulation and long-horizon mobile manipulation. This diversity introduces substantial variations in kitchen layouts, object appearances, and spatial relationships, testing whether policies remain robust across unseen scene and object configurations rather than overfitting. More importantly, RoboCasa365 explicitly evaluates generalization on task composition via a split featuring task templates that are unseen during training, We train our policy using the officially released dataset, which provides 100 demonstrations for each task. We follow the standard evaluation protocol and evaluate across 50 benchmark tasks, comprising 18 seen atomic tasks, 16 seen composite tasks, and 16 unseen composite tasks. The unseen composite tasks provide a zero-shot evaluation of whether the policy can recombine previously learned atomic skills and semantic knowledge to solve novel long-horizon tasks. Results are shown in Tab. 3

![](images/341ab2e2a89748fed635a6db0cd20f1c89c724268fe5394a334cade9b39e0cb1.jpg)  
Figure 10 Quantitative Results of Downstream Fine-tuning. We report the success rates and progresses of different models across the four different tasks.

• VLABench [87]: VLABench is a large-scale benchmark designed for comprehensive evaluation of languageconditioned manipulation. It comprises 100 task categories with over 2,000 object instances, emphasizing challenging scenarios involving semantic and distribution shifts. Specifically, VLABench introduces instructions with implicit human intentions, long-horizon tasks requiring multi-step reasoning, and evaluation settings that demand commonsense reasoning, category-level generalization, and robustness to unseen object appearances. These challenges are distributed across five evaluation tracks—In-distribution, Cross-Category, Commonsense, Instruction, and Texture—to thoroughly assess policy generalization. Beyond success rate (SR), VLABench introduces progress score (PS) and intention score (IS) to measure task completion quality and instruction understanding. Following the standard evaluation protocol, we train our policy using only the official training set from the In-distribution track, which contains 10 tasks with 500 demonstrations per task. We leverage chain-of-thought (CoT) labeling as in ERVLA [61] and train our model with a 50% probability on the next-token-prediction loss of CoT alongside the action loss. During evaluation, each task is tested across all five tracks with 50 evaluation episodes per track, resulting in 250 rollouts for each task and 2,500 rollouts in total. Results are shown in Tab. 4.

Table 2 Results on the RoboCasa Benchmark. We report the average success rate (%). The best and second-best results are highlighted in bold and underline, respectively.
<table><tr><td>Method</td><td>Avg. Success (%)</td></tr><tr><td>UVA [41]</td><td>50.0</td></tr><tr><td>UWM [96]</td><td>60.8</td></tr><tr><td>π0.5 [5]</td><td>62.1</td></tr><tr><td>π0-FAST [58]</td><td>63.6</td></tr><tr><td>GR00T N1.6 [54]</td><td>66.2</td></tr><tr><td>Cosmos Policy [32]</td><td>67.1</td></tr><tr><td>RLDX-1 [29]</td><td>70.6</td></tr><tr><td>World2Act [73]</td><td>72.6</td></tr><tr><td>Xiaomi-Robotics-1 (Ours)</td><td>74.5</td></tr></table>

Table 3 Results on the RoboCasa365 benchmark. We report task success rates (%). The best and second-best results are highlighted in bold and underline, respectively.
<table><tr><td>Method</td><td>Average</td><td>Atomic</td><td>Comp.-Seen</td><td>Comp.-Unseen</td></tr><tr><td>Diffusion Policy [16]</td><td>6.1</td><td>15.7</td><td>0.2</td><td>1.3</td></tr><tr><td>π0.5 [5]</td><td>16.9</td><td>39.6</td><td>7.1</td><td>1.2</td></tr><tr><td>GigaWorld-Policy 0.1 [79]</td><td>20.7</td><td>44.4</td><td>11.8</td><td>2.9</td></tr><tr><td>GR00T-N1.6 [54]</td><td>21.9</td><td>51.1</td><td>9.4</td><td>1.7</td></tr><tr><td>WorldDreamer [75]</td><td>35.3</td><td>66.3</td><td>26.7</td><td>9.0</td></tr><tr><td>Qwen-RobotManip [71]</td><td>35.9</td><td>68.6</td><td>20.1</td><td>14.9</td></tr><tr><td>RLDX-1 [29]</td><td>36.0</td><td>67.6</td><td>27.9</td><td>8.5</td></tr><tr><td>ABot-M0.5 [11]</td><td>40.4</td><td>75.9</td><td>38.3</td><td>2.7</td></tr><tr><td>ABot-M0.6 [11]</td><td>46.6</td><td>79.4</td><td>48.3</td><td>7.9</td></tr><tr><td>Xiaomi-Robotics-1 (Ours)</td><td>57.4</td><td>80.2</td><td>57.1</td><td>32.1</td></tr></table>

• RoboDojo [12]: RoboDojo is a unified simulation and real-world benchmark for comprehensively evaluating generalist robot manipulation policies. Unlike existing benchmarks that focus primarily on individual skills, RoboDojo provides a diverse and challenging suite comprising over 42 simulation tasks with varied object configurations, scene layouts, and language objectives. These tasks are organized into five core capability dimensions: Generalization (robustness to object/layout changes), Precision (finegrained control), Long-Horizon (multi-step execution), Memory (state-dependent manipulation), and Open (open-ended instruction following). Following the official protocol, we evaluate Xiaomi-Robotics-1 on the RoboDojo simulation benchmark and report the average score and success rate of each capability dimensions. Results are shown in Tab. 5.

Xiaomi-Robotics-1 achieves state-of-the-art results across all the four challenging benchmarks. It achieves an average success rate of 74.5% in RoboCasa. In RoboCasa365, it significantly outperforms the previous best methods by 10.8 percentage points, obtaining an average success rate of 57.4%. Specifically, it delivers the largest improvement on the most challenging split of Composite-Unseen, showcasing powerful generalization capabilities to task compositions that were unseen during training. In VLABench, Xiaomi-Robotics-1 achieves the highest average success rate and progress score, while maintaining a competitive intention score. This highlights its strong robustness to semantic and distribution shifts as well as superior long-horizon reasoning and language-conditioned manipulation capabilities. Notably, under cross-category and texture shifts, Xiaomi-Robotics-1 surpasses the strongest baseline by 6.0 and 15.2 percentage points in terms of success rate, respectively, demonstrating effective generalization to unseen objects and high resilience to visual appearance and background perturbations. In RoboDojo, Xiaomi-Robotics-1 significantly outperforms existing baseline methods, achieving absolute improvements of 7% and 5.13% in terms of average score and average success rate, respectively, over the second-best method. It ranks first across four out of five dimensions, indicating the model's outstanding overall performance across diverse task types. Notably, we do not incorporate history observation in the evaluation of the RoboDojo benchmark. Consequently, it scores lower on the memory dimension than Hy-Embodied-0.5-VLA [85] which explicitly models memory, yet it still significantly outperforms all other models.

Table 4 Results on the VLABench Benchmark. SR, PS, and IS denote success rate, progress score, and intention score, respectively. The best and second-best results are highlighted in bold and underline, respectively.
<table><tr><td rowspan="2">Method</td><td colspan="3">In-dist.</td><td colspan="3">Cross Category</td><td colspan="3">Commonsense</td></tr><tr><td>SR ↑1</td><td>PS ↑1</td><td>IS ↑</td><td>SR↑</td><td>PS↑</td><td>IS ↑</td><td>SR↑ PS ↑1</td><td></td><td>IS ↑</td></tr><tr><td>π0-FAST [58]</td><td>56.2</td><td>66.8</td><td>72.4</td><td>31.0</td><td>38.2</td><td>47.8</td><td>38.0</td><td>48.6</td><td>56.8</td></tr><tr><td>X-VLA [92]</td><td></td><td>67.8</td><td></td><td></td><td>25.1</td><td></td><td></td><td>48.2</td><td></td></tr><tr><td>ACoT-VLA [94]</td><td></td><td>66.1</td><td>79.8</td><td></td><td>38.9</td><td>54.1</td><td></td><td>37.8</td><td>52.3</td></tr><tr><td>π0.5 [5]</td><td>65.4</td><td>77.8</td><td>80.4</td><td>38.2</td><td>49.7</td><td>52.0</td><td>43.9</td><td>57.3</td><td>60.0</td></tr><tr><td>ERVLA [61]</td><td>69.7</td><td>81.1</td><td>84.2</td><td>47.0</td><td>61.0</td><td>66.4</td><td>44.0</td><td>55.0</td><td>57.2</td></tr><tr><td>Xiaomi-Robotics-1 (Ours)</td><td>75.6</td><td>85.0</td><td>79.8</td><td>53.0</td><td>66.6</td><td>66.4</td><td>48.4</td><td>58.3</td><td>58.2</td></tr><tr><td colspan="10"></td></tr><tr><td rowspan="2">Method</td><td>Instruction</td><td></td><td></td><td></td><td>Texture</td><td></td><td></td><td>Avg.</td><td></td></tr><tr><td>SR ↑ PS ↑ 1</td><td></td><td></td><td>IS ↑</td><td>SR↑</td><td>PS↑ IS ↑</td><td>SR↑</td><td></td><td>PS ↑ IS ↑</td></tr><tr><td>π0-FAST [58]</td><td>35.0</td><td>45.0</td><td>59.4</td><td>39.0</td><td>49.0</td><td>56.8</td><td>39.8</td><td>49.5</td><td>58.6</td></tr><tr><td>X-VLA [92]</td><td></td><td>63.1</td><td></td><td></td><td></td><td></td><td></td><td>51.1</td><td></td></tr><tr><td>ACoT-VLA [94]</td><td></td><td>39.6</td><td>56.8</td><td></td><td>54.6</td><td>74.6</td><td></td><td>47.4</td><td>63.5</td></tr><tr><td>π0.5 [5]</td><td>48.2</td><td>64.2</td><td>67.0</td><td>44.9</td><td>62.3</td><td>65.0</td><td>48.1</td><td>62.3</td><td>64.9</td></tr><tr><td>ERVLA [61]</td><td>58.0</td><td>70.2</td><td>73.8</td><td>47.4</td><td>62.3</td><td>70.6</td><td>53.2</td><td>65.9</td><td>70.4</td></tr><tr><td>Xiaomi-Robotics-1 (Ours)</td><td>55.8</td><td>66.8</td><td></td><td>70.2</td><td>62.6 74.9</td><td>74.8</td><td>59.1</td><td>70.3</td><td>69.9</td></tr></table>

## 4 Related Work

Scaling for Robot Learning. Research on the scaling laws of large language models (LLMs) demonstrates that performance improves predictably when data, compute, and model capacity are scaled in tandem [22, 27]. Recent advances in large language models |7, 72 and multi-modal foundation models 1-3, 63| further demonstrate substantial capability gains driven by scaling of data and model size. Motivated by these advancements, robot learning has increasingly embraced this scaling paradigm, enlarging both data and model sizes in pursuit of foundational generalist policies |4–6, 8, 31, 42, 54, 58, 65–67, 97]. Scaling robot learning, however, differs fundamentally from scaling models trained on web-scale data. Collecting real-robot trajectories requires costly and labor-intensive teleoperation, and the resulting data are often confined to a narrow slice of environments and tasks, severely constraining the data diversity. To alleviate this data bottleneck, recent work leverages portable UMI devices [17], enabling the collection of real-world manipulation trajectories without requiring physical robot embodiments |17, 46, 65, 66, 77, 90]. This lowers the barrier to data collection by facilitating in-the-wild collection across a highly diverse spectrum of unconstrained. real-world environments. Complementarily, another line of research leverages egocentric human manipulation trajectories to capture diverse behaviors across various tasks, objects, and environments, and subsequently converts them to robot data via representation alignment or motion retargeting |15, 40, 50]. In this work, we leverage over 100k hours of real-world manipulation trajectories collected from UMI devices. By varying data scale and model size, we investigate the scaling behavior of our model during pre-training and whether scaling translates to real-robot performance after post-training,

Robot Foundation Models. Recently, robot foundation models have emerged as a new paradigm for generalizable robot learning. By leveraging large-scale datasets, these models enable robust generalization across diverse environments and facilitate efficient adaptation to novel downstream tasks. World-action models (WAMs) [21,

Table 5Results on the RoboDojo Simulation Benchmark. Each entry reports score / success rate (%). The best and second-best results are highlighted in bold and underline, respectively.
<table><tr><td>Method</td><td>Generalization</td><td>Precision</td><td>Long-Horizon</td></tr><tr><td>GalaxeaVLA (G0) [26]</td><td>4.53 /2.83%</td><td>8.10 3.83%</td><td>12.60 5.58%</td></tr><tr><td>GigaWorld-Policy [79]</td><td>5.34 / 2.89%</td><td>6.15 1.83%</td><td>15.51 / 8.92%</td></tr><tr><td>StarVLA-α [80]</td><td>3.93 / 2.33%</td><td>9.90 / 4.33%</td><td>14.15 6.50%</td></tr><tr><td>Xiaomi-Robotics-0 [8]</td><td>7.43 5.56%</td><td>8.42 4.58%</td><td>13.51 6.92%</td></tr><tr><td>X-WAM [21]</td><td>7.39 3.33%</td><td>6.72 1.83%</td><td>17.47 9.08%</td></tr><tr><td>X-VLA [92]</td><td>10.48 6.78%</td><td>18.32 12.00%</td><td>16.53 9.75%</td></tr><tr><td>π0.5 [5]</td><td>13.37 8.17%</td><td>12.40 5.50%</td><td>23.54 14.67%</td></tr><tr><td>Spatial Forcing [35]</td><td>14.12 9.33%</td><td>17.33 10.58%</td><td>23.26 14.58%</td></tr><tr><td>Hy-Embodied-0.5-VLA [85]</td><td>11.77 8.39%</td><td>13.81 8.00%</td><td>25.74 14.92%</td></tr><tr><td>Xiaomi-Robotics-1 (Ours) 23.55</td><td>17.00%</td><td>26.69 18.83%</td><td>38.39 23.67%</td></tr><tr><td colspan="4"></td></tr><tr><td>Method</td><td>Memory</td><td>Open</td><td>Average</td></tr><tr><td>GalaxeaVLA (G0) [26]</td><td>3.17 1.89% 0.70</td><td>0.67%</td><td>5.82 2.96%</td></tr><tr><td>GigaWorld-Policy [79]</td><td>3.46 2.22%</td><td>0.54 / 0.50%</td><td>6.20 3.27%</td></tr><tr><td>StarVLA-α [80]</td><td>3.34 2.44%</td><td>0.68 /0.58%</td><td>6.40 3.24%</td></tr><tr><td>Xiaomi-Robotics-0 [8]</td><td>5.07 3.67%</td><td>0.22 0.17%</td><td>6.93 4.18%</td></tr><tr><td>X-WAM [21]</td><td>6.32 4.67%</td><td>0.57 7/ 0.25%</td><td>7.69 3.83%</td></tr><tr><td>X-VLA [92]</td><td>4.76 3.56%</td><td>0.55 0.50%</td><td>10.13 6.52%</td></tr><tr><td>π0.5 [5]</td><td>5.78 4.56%</td><td>1.98 1.67%</td><td>11.41 6.91%</td></tr><tr><td>Spatial Forcing [35]</td><td>5.43 4.11%</td><td>1.78 1.58%</td><td>12.38 8.04%</td></tr><tr><td>Hy-Embodied-0.5-VLA [85]</td><td>13.37 12.11%</td><td>0.65 0.58%</td><td>13.07 8.80%</td></tr><tr><td>Xiaomi-Robotics-1 (Ours)</td><td>7.81 6.56%</td><td>3.94 3.58%</td><td>20.07 13.93%</td></tr></table>

32, 36, 39, 51, 56, 68, 78, 81, 83, 86] and vision-language-action (VLA) models [4, 5, 8–10, 30, 38, 42, 60, 82, 92] represent two popular paradigms of robot foundation models. WAMs are generally built upon pre-trained video models. They explicitly model future observations or environment dynamics and use these predictions to facilitate action generation. Early approaches typically generate visual plans from language-specified goals and subsequently translate them into executable robot actions [18, 33, 37, 43, 95]. More recent work extends this paradigm to several directions, including jointly modeling future prediction and action generation [23, 32, 36, 41, 96], incorporating explicit 3D geometric structure [21, 39, 88, 91], and training on large-scale heterogeneous video-action data to improve zero-shot generalization and cross-embodiment transfer [81, 86]. In contrast, VLA models leverage pre-trained vision-language models as backbone, aiming to harness their rich, general semantic knowledge to facilitate robust action prediction [4, 5, 8–10, 24, 25, 30, 47, 54, 97]. Recent advances improve VLA models along three main axes. First, reasoning-oriented methods introduce intermediate representations (e.g., embodied reasoning tokens [13, 19, 34, 64, 84] and visual chains of thought [89, 93]) to support semantic, spatial, and temporal reasoning. Second, advances in action representation address the limitations of naive action discretization through learned tokenizers and flow matching |4, 20, 48, 58, 82]. Third, large-scale pre-training across heterogeneous datasets with different robot embodiments enables strong cross-embodiment transfer and robust downstream performance [8, 30, 54, 55, 69, 82]. Our work follows the VLA paradigm but focuses on a complementary question: the scaling behavior of robot foundation models. To support this investigation, we develop a scalable infrastructure to curate massive trajectory datasets with precise language annotations. Leveraging this data for large-scale training, we conduct extensive experiments to systematically explore the scaling properties of foundational VLA models.

## 5 Conclusions

In this work, we present Xiaomi-Robotics-1, a foundational vision-language-action (VLA) model that is able to follow instructions to perform a wide range of mobile manipulation tasks out-of-the-box in unseen environments, and efficiently adapt to novel challenging tasks with a minimal amount of data. During pre-training, we leverage over 100,000 hours of real-world manipulation trajectories, endowing the model with broad and generalizable manipulation capabilities. To scale training effectively, we propose an auto-labeling pipeline that annotates the large-scale dataset with detailed descriptions of scene state transitions as language prompts. In the post-training phase, we align these strong capabilities acquired during pre-training with robot embodiments and imperative instruction prompts using a cross-embodiment dataset. Extensive experiments demonstrate that the performance of Xiaomi-Robotics-1 consistently improves with increasing data scale and model size during pre-training. More importantly, the scaling property directly translates to out-of-the-box performance in unseen environments after post-training. Xiaomi-Robotics-1 can also serve as a powerful robot foundation model that is able to adapt to novel challenging real-robot tasks with minimal data. In addition, it achieves strong state-of-the-art results on four challenging simulation benchmarks. We hope this work can serve as a foundation for future exploration of scalable robot policies that can be deployed out-of-the-box in the real world.

## Contributions

Authors are listed in alphabetical order

## Core Contributors

• Jun Guo

• Piaopiao Jin

• Jason Li

• Peiyan Li

• Yingyan Li

• Futeng Liu

• Wanli Peng

• Optimus Qin

• Yifei Su

• Nan Sun

• Qiao Sun

• Runze Suo

• Heyun Wang

• Yunhong Wang

• Rujie Wu

• Caoyu Xia

• Lina Zhang

• Jack Zhao

## Contributors

• Guoliang Chen

• Wenlong Chen

• Xinze He

• Bin Li

• Qing Li

• Zhuorong Li

• Heng Qu

• Wenxuan Song

• Diyun Xiang

• Yifan Xie

• Peiran Xu

## Acknowledgment

• Hangjun Ye

• Wen Ye

• Han Zhao

• Quanyun Zhou

We express our sincere appreciation to the broader team for their tremendous support, including those not listed above: Li Jiang, Xiaohan Yu, Meichen Mu, Xiaoke Xilinjueluo, Qingyi Li, Qi Liu, Yayun Liu, Jun Xia, Feng Qiu, Donghao Wang, Yan Hou, Dong Wang, Liangliang He, Jiaxin Liu, Kang Zhou, Rui Cai, Shuoxue Bi, Yingchao Zhou, Kun Ma, Yiwei Zhou and Dongsheng Li.

## References

[1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023

[2] Niket Agarwal, Arslan Ali, Jon Allen, Martin Antolini, Adeline Aubame, Alisson Azzolini, Junjie Bai, Maciej Bala, Yogesh Balaji, Josh Bapst, et al. Cosmos 3: Omnimodal world models for physical ai. arXiv preprint arXiv:2606.02800, 2026.

[3] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025.

[4] Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, et al. π0: A vision-language-action flow model for general robot control arXiv preprint arXiv:2410.24164, 2024.

[5] Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, et al. π0.5: a vision-language-action model with open-world generalization. arXiv preprint arXiv:2504.16054, 2025.

[6] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Joseph Dabis, Chelsea Finn, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Jasmine Hsu, et al. Rt-1: Robotics transformer for real-world control at scale. arXiv preprint arXiv:2212.06817, 2022

[7] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.

[8] Rui Cai, Jun Guo, Xinze He, Piaopiao Jin, Jie Li, Bingxuan Lin, Futeng Liu, Wei Liu, Fei Ma, Kun Ma, et al. Xiaomi-robotics-0: An open-sourced vision-language-action model with real-time execution. arXiv preprint arXiv:2602.12684, 2026.

[9] Chi-Lam Cheang, Guangzeng Chen, Ya Jing, Tao Kong, Hang Li, Yifeng Li, Yuxiao Liu, Hongtao Wu, Jiafeng Xu, Yichu Yang, et al. Gr-2: A generative video-language-action model with web-scale knowledge for robot manipulation. arXiv preprint arXiv:2410.06158, 2024.

[10] Chilam Cheang, Sijin Chen, Zhongren Cui, Yingdong Hu, Liqun Huang, Tao Kong, Hang Li, Yifeng Li, Yuxiao Liu, Xiao Ma, et al. Gr-3 technical report. arXiv preprint arXiv:2507.15493, 2025.

[11] Ronghan Chen, Yandan Yang, Zuojin Tang, Dongjie Huo, Tong Lin, Haoning Wu, Haoyun Liu, Yuzhi Chen, Lulu Zheng, Botai Yuan, Tianlun Li, Mingxin Wang, Dekang Qi, Bin Hu, Wei Mei, Yuze Xuan, Haolong Yang, Yanqing Zhu, Mu Xu, Zhiheng Ma, and Xinyuan Chang. Abot-m0.5: Unified mobility-and-manipulation world action model. arXiv preprint arXiv:2607.00678, 2026.

[12] Tianxing Chen, Yue Chen, Zixuan Li, Junyuan Tang, Kailun Su, Haoran Lu, Weijie Wan, Baijun Chen, Songling Liu, Haowen Yan, Honghao Su, Zhiyang Dou, Kaixuan Wang, Dandan Zhang, Yunze Liu, Yan Qin, Qiwei Liang, Qiwei Wu, Zijian Lin, Wenwei Lin, Yuran Wang, Minghua He, Tianshu Wu, Ruihai Wu, Jingquan Zhou, Kai-Chong Lei, Haibao Yu, Yuanfeng Ji, Weiyang Jin, Guanyu Lin, Xiaofan Li, Qi Xiong, Renjing Xu, Zhongyu Li, Wenhao Chai, Enze Xie, Ziwei Wang, Yao Mu, Hao Dong, Wojciech Matusik, Mingyu Ding, Wenbo Ding, Ping Luo, and Masayoshi Tomizuka. Robodojo: A unified sim-and-real benchmark for comprehensive evaluation of generalist robot manipulation policies, 2026. URL https://arxiv.org/abs/2607.04434.

[13] William Chen, Suneel Belkhale, Suvir Mirchandani, Oier Mees, Danny Driess, Karl Pertsch, and Sergey Levine. Training strategies for efficient embodied reasoning. arXiv preprint arXiv:2505.08243, 2025.

[14] Xi Chen, Xiao Wang, Soravit Changpinyo, Anthony J Piergiovanni, Piotr Padlewski, Daniel Salz, Sebastian Goodman, Adam Grycner, Basil Mustafa, Lucas Beyer, et al. Pali: A jointly-scaled multilingual language-image model. arXiv preprint arXiv:2209.06794, 2022.

[15] Yangtao Chen, Zixuan Chen, Peiyang Wang, Yong-Lu Li, Jing Huo, Jieqi Shi, and Yang Gao. Wh0: Generative world models as scalable sources of egocentric human hand manipulation data. arXiv preprint arXiv:2606.22136, 2026.

[16] Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, and Shuran Song. Diffusion policy: Visuomotor policy learning via action diffusion. The International Journal of Robotics Research, 2024.

[17] Cheng Chi, Zhenjia Xu, Chuer Pan, Eric Cousineau, Benjamin Burchfiel, Siyuan Feng, Russ Tedrake, and Shuran Song. Universal manipulation interface: In-the-wild robot teaching without in-the-wild robots. arXiv preprint arXiv:2402.10329, 2024.

[18] Yilun Du, Sherry Yang, Bo Dai, Hanjun Dai, Ofir Nachum, Josh Tenenbaum, Dale Schuurmans, and Pieter Abbeel. Learning universal policies via text-guided video generation. Advances in neural information processing systems, 36:9156–9172, 2023.

[19] Haoquan Fang, Jiafei Duan, Donovan Clay, Sam Wang, Shuo Liu, Weikai Huang, Xiang Fan, Wei-Chuan Tsai, Shirui Chen, Yi Ru Wang, et al. Molmoact2: Action reasoning models for real-world deployment. arXiv preprint arXiv:2605.02881, 2026.

[20] Galaxea Team. Galaxea g0.5 technical report. 2026. URL https://opengalaxea.github.io/G05/.

[21] Jun Guo, Qiwei Li, Peiyan Li, Zilong Chen, Nan Sun, Yifei Su, Heyun Wang, Yuan Zhang, Xinghang Li, and Huaping Liu. Unified 4d world action modeling from video priors with asynchronous denoising. arXiv preprint arXiv:2604.26694, 2026.

[22] Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, DDL Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. Training compute-optimal large language models. arXiv preprint arXiv:2203.15556, 10, 2022

[23] Yucheng Hu, Yanjiang Guo, Pengchao Wang, Xiaoyu Chen, Yen-Jen Wang, Jianke Zhang, Koushil Sreenath Chaochao Lu, and Jianyu Chen. Video prediction policy: A generalist robot policy with predictive visual representations. arXiv preprint arXiv:2412.14803, 2024.

[24] Physical Intelligence, Ali Amin, Raichelle Aniceto, Ashwin Balakrishna, Kevin Black, Ken Conley, Grace Connors, James Darpinian, Karan Dhabalia, Jared DiCarlo, et al. π0.6: a vla that learns from experience. arXiv preprint arXiv:2511.14759, 2025.

[25] Physical Intelligence, Bo Ai, Ali Amin, Raichelle Aniceto, Ashwin Balakrishna, Greg Balke, Kevin Black, George Bokinsky, Shihao Cao, Thomas Charbonnier, et al. πo.7: a steerable generalist robotic foundation model with emergent capabilities. arXiv preprint arXiv:2604.15483, 2026.

[26] Tao Jiang, Tianyuan Yuan, Yicheng Liu, Chenhao Lu, Jianning Cui, Xiao Liu, Shuiqi Cheng, Jiyang Gao, Huazhe Xu, and Hang Zhao. Galaxea open-world dataset and g0 dual-system vla model. arXiv preprint arXiv:2509.00576, 2025.

[27] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020.

[28] Alexander Khazatsky, Karl Pertsch, Suraj Nair, Ashwin Balakrishna, Sudeep Dasari, Siddharth Karamcheti, Soroush Nasiriany, Mohan Kumar Srirama, Lawrence Yunliang Chen, Kirsty Ellis, et al. Droid: A large-scale in-the-wild robot manipulation dataset. arXiv preprint arXiv:2403.12945, 2024.

[29] Dongyoung Kim, Huiwon Jang, Myungkyu Koo, Suhyeok Jang, Taeyoung Kim, Beomjun Kim, Byungjun Yoon, Changsung Jang, Daewon Choi, Dongsu Han, et al. Rldx-1 technical report. arXiv preprint arXiv:2605.03269, 2026.

[30] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, et al. Openvla: An open-source vision-language-action model. arXiv preprint arXiv:2406.09246, 2024

[31] Moo Jin Kim, Chelsea Finn, and Percy Liang. Fine-tuning vision-language-action models: Optimizing speed and success. arXiv preprint arXiv:2502.19645, 2025.

[32] Moo Jin Kim, Yihuai Gao, Tsung-Yi Lin, Yen-Chen Lin, Yunhao Ge, Grace Lam, Percy Liang, Shuran Song, Ming-Yu Liu, Chelsea Finn, et al. Cosmos policy: Fine-tuning video models for visuomotor control and planning. arXiv preprint arXiv:2601.16163, 2026.

[33] Po-Chen Ko, Jiayuan Mao, Yilun Du, Shao-Hua Sun, and Joshua B Tenenbaum. Learning to act from actionless videos through dense correspondences. In International Conference on Learning Representations, volume 2024, pages 40938–40958, 2024.

[34] Jason Lee, Jiafei Duan, Haoquan Fang, Yuquan Deng, Shuo Liu, Boyang Li, Bohan Fang, Jieyu Zhang, Yi Ru Wang, Sangho Lee, et al. Molmoact: Action reasoning models that can reason in space. arXiv preprint arXiv:2508.07917, 2025.

[35] Fuhao Li, Wenxuan Song, Han Zhao, Jingbo Wang, Pengxiang Ding, Donglin Wang, Long Zeng, and Haoang Li. Spatial forcing: Implicit spatial representation alignment for vision-language-action model. arXiv preprint arXiv:2510.12276, 2025.

[36] Lin Li, Qihang Zhang, Yiming Luo, Shuai Yang, Ruilin Wang, Fei Han, Mingrui Yu, Zelin Gao, Nan Xue, Xing Zhu, et al. Causal world modeling for robot control. arXiv preprint arXiv:2601.21998, 2026.

[37] Peiyan Li, Hongtao Wu, Yan Huang, Chilam Cheang, Liang Wang, and Tao Kong. Gr-mg: Leveraging partiallyannotated data via multi-modal goal-conditioned policy. IEEE Robotics and Automation Letters, 10(2):1912-1919, 2025.

[38] Peiyan Li, Yixiang Chen, Hongtao Wu, Xiao Ma, Xiangnan Wu, Yan Huang, Liang Wang, Tao Kong, and Tieniu Tan. Bridgevla: Input-output alignment for efficient 3d manipulation learning with vision-language models. Advances in Neural Information Processing Systems, 38:63635–63673, 2026.

[39] Peiyan Li, Yixiang Chen, Yuan Xu, Jiabing Yang, Xiangnan Wu, Jun Guo, Nan Sun, Long Qian, Xinghang Li, Xin Xiao, et al. Multi-view video diffusion policy: A 3d spatio-temporal-aware video action model. arXiv preprint arXiv:2604.03181, 2026.

[40] Qixiu Li, Yu Deng, Yaobo Liang, Lin Luo, Lei Zhou, Chengtang Yao, Lingqi Zeng, Zhiyuan Feng, Huizhi Liang, Sicheng Xu, et al. Scalable vision-language-action model pretraining for robotic manipulation with real-life human activity videos. arXiv preprint arXiv:2510.21571, 2025.

[41] Shuang Li, Yihuai Gao, Dorsa Sadigh, and Shuran Song. Unified video action model. arXiv preprint arXiv:2503.00200, 2025.

[42] Xinghang Li, Peiyan Li, Minghuan Liu, Dong Wang, Jirong Liu, Bingyi Kang, Xiao Ma, Tao Kong, Hanbo Zhang, and Huaping Liu. Towards generalist robot policies: What matters in building vision-language-action models arXiv preprint arXiv:2412.14058, 2024.

[43] Junbang Liang, Ruoshi Liu, Ege Ozguroglu, Sruthi Sudhakar, Achal Dave, Pavel Tokmakov, Shuran Song, and Carl Vondrick. Dreamitate: Real-world visuomotor policy learning via video generation. arXiv preprint arXiv:2406.16862, 2024.

[44] Weixin Liang, Lili Yu, Liang Luo, Srinivasan Iyer, Ning Dong, Chunting Zhou, Gargi Ghosh, Mike Lewis, Wen-tau Yih, Luke Zettlemoyer, et al. Mixture-of-transformers: A sparse and scalable architecture for multi-modal foundation models. arXiv preprint arXiv:2411.04996, 2024.

[45] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024.

[46] Fangchen Liu, Chuanyu Li, Yihua Qin, Jing Xu, Pieter Abbeel, and Rui Chen. Vitamin: Learning contact-rich tasks through robot-free visuo-tactile manipulation interface. arXiv preprint arXiv:2504.06156, 2025.

[47] Songming Liu, Lingxuan Wu, Bangguo Li, Hengkai Tan, Huayu Chen, Zhengyi Wang, Ke Xu, Hang Su, and Jun Zhu. Rdt-1b: a diffusion foundation model for bimanual manipulation. In International Conference on Learning Representations, volume 2025, pages 29982–30009, 2025.

[48] Songming Liu, Bangguo Li, Kai Ma, Lingxuan Wu, Hengkai Tan, Xiao Ouyang, Hang Su, and Jun Zhu. Rdt2: Exploring the scaling limit of umi data towards zero-shot cross-embodiment generalization. arXiv preprint arXiv:2602.03310, 2026.

[49] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022.

[50] Hao Luo, Yicheng Feng, Wanpeng Zhang, Sipeng Zheng, Ye Wang, Haoqi Yuan, Jiazheng Liu, Chaoyi Xu, Qin Jin, and Zongqing Lu. Being-h0: vision-language-action pretraining from large-scale human videos. arXiv preprint arXiv:2507.15597, 2025.

[51] Teli Ma, Jia Zheng, Zifan Wang, Chunli Jiang, Andy Cui, Junwei Liang, and Shuo Yang. Dit4dit: Jointly modeling video dynamics and actions for generalizable robot control. arXiv preprint arXiv:2603.10448, 2026.

[52] Soroush Nasiriany, Abhiram Maddukuri, Lance Zhang, Adeet Parikh, Aaron Lo, Abhishek Joshi, Ajay Mandlekar, and Yuke Zhu. Robocasa: Large-scale simulation of everyday tasks for generalist robots. arXiv preprint arXiv:2406.02523, 2024.

[53] Soroush Nasiriany, Sepehr Nasiriany, Abhiram Maddukuri, and Yuke Zhu. Robocasa365: A large-scale simulation framework for training and benchmarking generalist robots. arXiv preprint arXiv:2603.04356, 2026.

[54] NVIDIA, Johan Bjorck, Nikita Cherniadev Fernando Castañeda, Xingye Da, Runyu Ding, Linxi "Jim" Fan, Yu Fang, Dieter Fox, Fengyuan Hu, Spencer Huang, Joel Jang, Zhenyu Jiang, Jan Kautz, Kaushil Kundalia, Lawrence Lao, Zhiqi Li, Zongyu Lin, Kevin Lin, Guilin Liu, Edith Llontop, Loic Magne, Ajay Mandlekar, Avnish Narayan, Soroush Nasiriany, Scott Reed, You Liang Tan, Guanzhi Wang, Zu Wang, Jing Wang, Qi Wang, Jiannan Xiang, Yuqi Xie, Yinzhen Xu, Zhenjia Xu, Seonghyeon Ye, Zhiding Yu, Ao Zhang, Hao Zhang, Yizhou Zhao, Ruijie Zheng, and Yuke Zhu. GR00T N1: An open foundation model for generalist humanoid robots. In ArXiv Preprint, March 2025.

[55] Abby O'Neill, Abdul Rehman, Abhiram Maddukuri, Abhishek Gupta, Abhishek Padalkar, Abraham Lee, Acorn Pooley, Agrim Gupta, Ajay Mandlekar, Ajinkya Jain, et al. Open x-embodiment: Robotic learning datasets and rt-x models: Open x-embodiment collaboration 0. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 6892–6903. IEEE, 2024.

[56] Jonas Pai, Liam Achenbach, Victoriano Montesinos, Benedek Forrai, Oier Mees, and Elvis Nava. mimic-video: Video-action models for generalizable robot control beyond vlas. arXiv preprint arXiv:2512.15692, 2025.

[57] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205, 2023.

[58] Karl Pertsch, Kyle Stachowicz, Brian Ichter, Danny Driess, Suraj Nair, Quan Vuong, Oier Mees, Chelsea Finn, and Sergey Levine. Fast: Efficient action tokenization for vision-language-action models. arXiv preprint arXiv:2501.09747, 2025.

[59] Haozhi Qi, Yen-Jen Wang, Toru Lin, Brent Yi, Yi Ma, Koushil Sreenath, and Jitendra Malik. Coordinated humanoid manipulation with choice policies. arXiv preprint arXiv:2512.25072, 2025.

[60] Delin Qu, Haoming Song, Qizhi Chen, Yuanqi Yao, Xinyi Ye, Yan Ding, Zhigang Wang, JiaYuan Gu, Bin Zhao, Dong Wang, et al. Spatialvla: Exploring spatial representations for visual-language-action model. arXiv preprint arXiv:2501.15830, 2025.

[61] Nan Sun, Yuan Zhang, Yongkun Yang, Wentao Zhao, Peiyan Li, Jun Guo, Wenxuan Song, Pengxiang Ding, Runze Suo, Yifei Su, et al. Revisiting embodied chain-of-thought for generalizable robot manipulation. arXiv preprint arXiv:2606.03784, 2026.

[62] Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.

[63] Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer, Damien Vincent, Zhufeng Pan, Shibo Wang, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024.

[64] Gemini Robotics Team, Saminda Abeyruwan, Joshua Ainslie, Jean-Baptiste Alayrac, Montserrat Gonzalez Arenas, Travis Armstrong, Ashwin Balakrishna, Robert Baruch, Maria Bauza, Michiel Blokzijl, et al. Gemini robotics: Bringing ai into the physical world. arXiv preprint arXiv:2503.20020, 2025.

[65] Generalist Team. Gen-0: Embodied foundation models that scale with physical interaction. Generalist AI Blog, 2025. https://generalistai.com/blog/gen-0.

[66] Generalist Team. Gen-1: Scaling embodied foundation models to mastery. Generalist AI Blog, 2026. https://generalistai.com/blog/gen-1.

[67] Genesis AI Team. Gene-26.5: Advancing robotic manipulation to human level. Genesis AI Blog, May 2026. URL https://genesis.ai/blog/gene-26-5-advancing-robotic-manipulation-to-human-level.

[68] MotuBrain Team, Chendong Xiang, Fan Bao, Haitian Liu, Hengkai Tan, Hongzhe Bi, James Li, Jiabao Liu, Jingrui Pang, Kiro Jing, et al. Motubrain: An advanced world action model for robot control. arXiv preprint arXiv:2604.27792, 2026.

[69] Octo Model Team, Dibya Ghosh, Homer Walke, Karl Pertsch, Kevin Black, Oier Mees, Sudeep Dasari, Joey Hejna, Tobias Kreiman, Charles Xu, et al. Octo: An open-source generalist robot policy. arXiv preprint arXiv:2405.12213, 2024.

[70] Qwen Team. Qwen3.5: Accelerating productivity with native multimodal agents, February 2026. URL https: //qwen.ai/blog?id=qwen3.5.

[71] Qwen Team. Qwen-robotmanip technical report: Alignment unlocks scale for robotic manipulation foundation models. 2026.

[72] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.

[73] An Dinh Vuong, Tuan Van Vo, Abdullah Sohail, Haoran Ding, Liang Ma, Xiaodan Liang, Anqing Duan, Ivan Laptev, and Ian Reid. World2act: Latent action post-training from world model dynamics. arXiv preprint arXiv:2603.10422, 2026.

[74] Homer Rich Walke, Kevin Black, Tony Z Zhao, Quan Vuong, Chongyi Zheng, Philippe Hansen-Estruch, Andre Wang He, Vivek Myers, Moo Jin Kim, Max Du, et al. Bridgedata v2: A dataset for robot learning at scale In Conference on Robot Learning, pages 1723–1736. PMLR, 2023

[75] Xiaofeng Wang, Zheng Zhu, Guan Huang, Boyuan Wang, Xinze Chen, and Jiwen Lu. Worlddreamer: Towards general world models for video generation via predicting masked tokens. arXiv preprint arXiv:2401.09985, 2024.

[76] Wei Wu, Fan Lu, Yunnan Wang, Shuai Yang, Shi Liu, Fangjing Wang, Qian Zhu, He Sun, Yong Wang, Shuailei Ma, et al. A pragmatic vla foundation model. arXiv preprint arXiv:2601.18692, 2026.

[77] Mengda Xu, Han Zhang, Yifan Hou, Zhenjia Xu, Linxi Fan, Manuela Veloso, and Shuran Song. Dexumi: Using human hand as the universal manipulation interface for dexterous manipulation. arXiv preprint arXiv:2505.21864, 2025.

[78] Sizhe Yang, Juncheng Mu, Tianming Wei, Chenhao Lu, Xiaofan Li, Linning Xu, Zhengrong Xue, Zhecheng Yuan, Dahua Lin, Jiangmiao Pang, et al. Memorywam: Efficient world action modeling with persistent memory. arXiv preprint arXiv:2606.20562, 2026.

[79] Angen Ye, Boyuan Wang, Chaojun Ni, Guan Huang, Guosheng Zhao, Hao Li, Hengtao Li, Jie Li, Jindi Lv, Jingyu Liu, Min Cao, Peng Li, Qiuping Deng, Wenjun Mei, Xiaofeng Wang, Xinze Chen, Xinyu Zhou, Yang Wang, Yifan Chang, Yifan Li, Yukun Zhou, Yun Ye, Zhichao Liu, and Zheng Zhu. Gigaworld-policy: An efficient action-centered world-action model. arXiv preprint arXiv:2603.17240, 2026.

[80] Jinhui Ye, Ning Gao, Senqiao Yang, Jinliang Zheng, Zixuan Wang, Yuxin Chen, Pengguang Chen, Yilun Chen, Shu Liu, and Jiaya Jia. Starvla-α: Reducing complexity in vision-language-action systems. In European Conference on Computer Vision (ECCV), 2026.

[81] Seonghyeon Ye, Yunhao Ge, Kaiyuan Zheng, Shenyuan Gao, Sihyun Yu, George Kurian, Suneel Indupuru, You Liang Tan, Chuning Zhu, Jiannan Xiang, et al. World action models are zero-shot policies. arXiv preprint arXiv:2602.15922, 2026.

[82] Ryan Yu, Pushi Zhang, Starrick Liu, Brae Liu, Miracle Kang, Shalfun Li, Lights Shi, Ellie Ma, Ping Yang, Chris Pan, et al. Wall-oss-0.5 technical report. arXiv preprint arXiv:2605.30877, 2026.

[83] Tianyuan Yuan, Zibin Dong, Yicheng Liu, and Hang Zhao. Fast-wam: Do world action models need test-time future imagination? arXiv preprint arXiv:2603.16666, 2026.

[84] Michał Zawalski, William Chen, Karl Pertsch, Oier Mees, Chelsea Finn, and Sergey Levine. Robotic control via embodied chain-of-thought reasoning. arXiv preprint arXiv:2407.08693, 2024.

[85] He Zhang, Lingzhu Xiang, Haitao Lin, Zeyu Huang, Minghui Wang, Dingyan Zhong, Yubo Dong, Yihao Wu, Yongming Rao, Dongsheng Zhang, et al. Hy-embodied-0.5-vla: From vision-language-action models to a real-world robot learning stack. arXiv preprint arXiv:2606.14409, 2026.

[86] Qihang Zhang, Lin Li, Luyao Zhang, Shuai Yang, Yiming Luo, Shuaiting Li, Ruilin Wang, Junke Wang, Jiahao Shao, Gangwei Xu, et al. Native video-action pretraining for generalizable robot control. arXiv preprint arXiv:2607.08639, 2026.

[87] Shiduo Zhang, Zhe Xu, Peiju Liu, Xiaopeng Yu, Yuan Li, Qinghui Gao, Zhaoye Fei, Zhangyue Yin, Zuxuan Wu, Yu-Gang Jiang, et al. Vlabench: A large-scale benchmark for language-conditioned robotics manipulation with long-horizon reasoning tasks. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 11142–11152, 2025.

[88] Haoyu Zhao, Xingyue Zhao, Siteng Huang, Xin Li, Deli Zhao, and Zhongyu Li. Rynnworld-4d: 4d embodied world models for robotic manipulation. arXiv preprint arXiv:2607.06559, 2026.

[89] Qingqing Zhao, Yao Lu, Moo Jin Kim, Zipeng Fu, Zhuoyang Zhang, Yecheng Wu, Zhaoshuo Li, Qianli Ma, Song Han, Chelsea Finn, et al. Cot-vla: Visual chain-of-thought reasoning for vision-language-action models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 1702–1713, 2025.

[90] Zhaxizhuom Zhaxizhuoma, Kehui Liu, Chuyue Guan, Zhongjie Jia, Ziniu Wu, Xin Liu, Tianyu Wang, Shuai Liang, Pengan Chen, Pingrui Zhang, et al. Fastumi: A scalable and hardware-independent universal manipulation interface with dataset. In Conference on Robot Learning, pages 3069–3093. PMLR, 2025.

[91] Haoyu Zhen, Qiao Sun, Hongxin Zhang, Junyan Li, Siyuan Zhou, Yilun Du, and Chuang Gan. Tesseract: learning 4d embodied world models. arXiv preprint arXiv:2504.20995, 2025.

[92] Jinliang Zheng, Jianxiong Li, Zhihao Wang, Dongxiu Liu, Xirui Kang, Yuchun Feng, Yinan Zheng, Jiayin Zou, Yilun Chen, Jia Zeng, et al. X-vla: Soft-prompted transformer as scalable cross-embodiment vision-language-action model. arXiv preprint arXiv:2510.10274, 2025.

[93] Ruijie Zheng, Yongyuan Liang, Shuaiyi Huang, Jianfeng Gao, Hal Daumé III, Andrey Kolobov, Furong Huang, and Jianwei Yang. Tracevla: Visual trace prompting enhances spatial-temporal awareness for generalist robotic policies. In International Conference on Learning Representations, volume 2025, pages 54277–54296, 2025.

[94] Linqing Zhong, Yi Liu, Yifei Wei, Ziyu Xiong, Maoqing Yao, Si Liu, and Guanghui Ren. Acot-vla: Action chain-of-thought for vision-language-action models. arXiv preprint arXiv:2601.11404, 2026.

[95] Siyuan Zhou, Yilun Du, Jiaben Chen, Yandong Li, Dit-Yan Yeung, and Chuang Gan. Robodreamer: Learning compositional world models for robot imagination. arXiv preprint arXiv:2404.12377, 2024.

[96] Chuning Zhu, Raymond Yu, Siyuan Feng, Benjamin Burchfiel, Paarth Shah, and Abhishek Gupta. Unified world models: Coupling video and action diffusion for pretraining on large robotic datasets. arXiv preprint arXiv:2504.02792, 2025.

[97] Brianna Zitkovich, Tianhe Yu, Sichun Xu, Peng Xu, Ted Xiao, Fei Xia, Jialin Wu, Paul Wohlhart, Stefan Welker, Ayzaan Wahid, et al. Rt-2: Vision-language-action models transfer web knowledge to robotic control. In Conference on Robot Learning, pages 2165–2183. PMLR, 2023.

Table 6Progress Definition for Evaluation on Efficient Adaptation to New Tasks. Each rollout is assigned a progress score from 0 to 100% according to completed task milestones.
<table><tr><td>Task</td><td>Progress milestones</td><td>Progress (%)</td></tr><tr><td>Phone Packing</td><td>Grasp the phone; place the phone into the box; grasp the instruction 10, 10, 30, 10, 10, 30 manual; place the manual into the box; grasp the lid; successfully close the lid.</td><td></td></tr><tr><td>Printer Refilling</td><td>Grasp the paper; complete the handover; successfully insert one end of the paper into the printer tray; fully insert the paper stack into the printer tray; return both robot arms to the resting pose.</td><td>20, 20, 20, 30, 10</td></tr><tr><td>Laundry Loading</td><td>Open the washing machine door; move the laundry basket to the door; transfer the clothes into the washing machine; remove the laundry basket; close the washing machine door.</td><td>20 each</td></tr><tr><td>Box Packing</td><td>Grasp and place each specified target object into the box according to the language instruction. Five target objects are evaluated in each rollout.</td><td>20 each</td></tr></table>

![](images/5ffd79846894e1011871c84b5501da2bdfc2e2058c3e00968c691443955bb098.jpg)  
Figure 11 Examples of UMI data in the Pre-training Dataset.

![](images/15bdf4416e716e3ed2a0e3b02620260f1ecf34dca4b93aef564ec9d3764a1299.jpg)

![](images/c5606614214f2683b2b42af08ac1d0ac85250033b7aed95f3afd978c6b5d4c1c.jpg)

![](images/c84315a22bcdd883636dbeb192fbf7900060d4ffa05a73f60501f2160031b865.jpg)

![](images/32c8c53ee9ab6728e039cd09a6674eb0721e3a0d63bd17ceed114a8bea891b14.jpg)

![](images/2069c8c41b70f5f962929a1df5f3ee96dad8c9c02c26b60e99a9480155a718eb.jpg)  
Pick up the brown bottle on the table and put it on the shelf in the refrigerator with the left hand; Pick up the plastic bag containing food on the table and put it on the refrigerator shelf with the right hand.

![](images/747a6701532680a29c6ceb448579d96fbd59bb69d12fd01ccf7ea5578156c1f9.jpg)

![](images/95a0cf4c1a9de38ce8079d31b17b0430673d2285609cea0cf192d2777fa5c37a.jpg)

![](images/1e2b4decd1e4e93c900a203961862e615624d3be500e88bff0e34d467a6faca1.jpg)

![](images/ecc9e173a2a70255facb1d4eeba7733fdb02ab505762138c8a4322ae7499ef93.jpg)

![](images/ef1bb2522a5e9c384fb50b5b56a1f46c1d6c0c51e0a57bf67ae17e0f621b2fd3.jpg)  
Pick up the pink pillow from the left side of the sofa and place it on the left side of the first pillow.

![](images/ca1dfa719a44b5fa7fee323cfc50af8cb7cba7c678a01c177a781b3c153f49d8.jpg)

![](images/9cb9976f4090df75b91b880fca09bb6117422d0b463b00ea3f823bc7f347972f.jpg)

![](images/9cc4aea68bfd705bd470fd4f22f2cd5080b84cbde9440850c8175537225e7ef5.jpg)

![](images/9c31eb3ae410363e2116640a7f8c97db7ba074233fba792f230ba6018b082115.jpg)

![](images/08f3d122b40f5c984d0013470fa6ded28766779f146107cb9109476a825a413f.jpg)  
Take the silver-packaged snacks from the white basket in front and place them in the nearest transparent box on the fourth layer of the front shelf.

![](images/2f8151391f0c9404e01a1a8f7ce43c5f913b91d47ef96ee37b1be2806441cd92.jpg)

![](images/e78c8e79679ab1a4f87c51efd927d565594e7d0207cd1f6846d9be312f3e5e7e.jpg)

![](images/a6564ec4f6754d89a4c81c25b7df80c36746fa860d3ece72002bea38ddb4b95c.jpg)

![](images/2c9b1912d51cfd4f91831bbcb44d370a826decced097f7176a74c6dbc80e99f8.jpg)

![](images/e493940a3c2a41f7615ce7d1d82fa01ef8919d9de1c1f429532aa54bc2bad53f.jpg)  
Take the blue and white box at the lower right of the sink and place it to the right of the green box on the right side of the faucet with the right hand.

![](images/dc7eb051132a66099116ea58755d90097c50a616a50d950f10695787b1322b70.jpg)

![](images/e1e9e37860931997f17d3471b1073cb7e3629aaaf8fd0d396ccf27c5a278dbe9.jpg)

![](images/48ae14520dc023be8e7478ac87a9d6db3b098c26694c6f900bc76da7ea101ca6.jpg)

![](images/0f293bbf7b4b95b019fd8fdac4adbb475706f9daafe5faad5044577e10c146db.jpg)

![](images/40232d0a07edfe1b204c9dd47e69b91e7398e77ca88e6802d5c9b1c307161e8d.jpg)  
Pick up a white sports shoe on the left foot from the ground with the left hand; Wipe the white sports shoes on the left foot clean with the right hand.

Figure 12 Examples of UMI data in the Post-training Dataset.