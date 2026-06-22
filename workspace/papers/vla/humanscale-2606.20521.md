# HumanScale: Egocentric Human Video Can Outperform Real-Robot Data for Embodied Pretraining

Juncheng Ma<sup>∗</sup> Jianxin Bi<sup>∗</sup> Yufan Deng Xuanran Zhai Kewei Zhang Ye Huang Bo Liang Shukai Gong Jiankai Tu Xiaotian Tang Jiaxin Li Kaiqi Chen   
Duomin Wang Yuqi Wang Bingyi Kang Eric Huang Zhiyang Dou Zhen Dong Enze Xie Wojciech Matusik Tat-Seng Chua Daquan Zhou<sup>†</sup>

PKU, NUS, MIT, UCSB, NVIDIA

<sup>∗</sup>Equal Contribution, <sup>†</sup>Corresponding Author

## Abstract

Embodied foundation models are expected to benefit from data scaling like large language models, but face a much tighter data bottleneck. Teleoperated real-robot trajectories remain the dominant pretraining source due to their precise action supervision and embodiment alignment, yet their scalability is limited by high collection cost, acquisition dificulty, and low behavioral and environmental diversity. These limitations have sparked interest in egocentric human video as a scalable, substantially lower-cost, and more diverse alternative for embodied model pretraining. However, its efectiveness compared to teleoperated real-robot data remains underexplored. To address this question, we conduct a systematic study comparing egocentric human video and teleoperated real-robot trajectories as pretraining data sources for embodied foundation models, under fixed post-training and validation protocols. Surprisingly, we find that egocentric data, when processed through a carefully designed filtering and labeling pipeline, is not merely a viable substitute for model pretraining but can lead to superior performance. With the same amount of pretraining data, models pretrained on egocentric data achieve a 24% lower validation loss on real-robot action prediction, as well as 52.5% and 90% higher success rates on in-distribution and out-of-distribution real-robot task execution, respectively. This finding verifies a scalable paradigm for embodied foundation models: pretrain on egocentric human video to learn diverse world representations, then adapt with a small amount of labeled real-robot data for action-space alignment. We hope this study encourages broader exploration of egocentric data and ofers guidance for data quality assessment before costly robot data collection. Code will be released at https://github.com/DAGroup-PKU/HumanNet/.

## 1 Introduction

Foundation models in language and vision have advanced through the joint scaling of data, model size, and compute: heterogeneous, internet-scale corpora provide broad supervision, while empirical scaling laws make the returns from further scaling increasingly predictable [1, 3, 7, 22, 32]. Embodied foundation models, including vision-language-action models (VLAs) and emerging world-action models (WAMs), seek to inherit this scaling recipe, but their data resources follow a diferent logic [5, 18, 29, 40]. The dominant source of embodied data is teleoperated real-robot trajectories, which provide valuable action supervision and direct embodiment alignment. Yet, unlike web corpora that can be passively harvested from naturally occurring human behavior, teleoperation data must be actively produced through physical robots, human operators, designed tasks, and controlled environments. This makes it dificult to scale and limits it to a narrow slice of real-world interaction diversity. As a result, the embodied pretraining stage that learns broadly transferable representations before embodiment-specific post-training faces a central tension: the data most aligned with

Egocentric video pretraining leads to better generalization than robot data pretraining.

![](images/effe1ca86a370a23d2a3f5dfb1c0b998ddc76d7d22979612bcc8f72b93fbf202.jpg)  
Figure 1 Egocentric human video leads to stronger generalization than robot data for embodied pretraining Left: Egocentric human video ofers massive accessible scale $( \sim 1 0 ^ { 6 }$ hours), low collection cost, and open-world diversity, but provides only pseudo action labels from human hand retargeting. Teleoperated robot data provides precise action labels yet is limited in scale $( \sim 1 0 ^ { 4 }$ public hours), costly to collect, and limited scene diversity. Middle: We compare egocentric video pretraining with real robot data pretraining using the same world-action model (WAM). Both pretrained models are post-trained on the same real-robot dataset and evaluated on both seen and unseen AgiBot tasks. Right: On out-of-distribution evaluations, egocentric pretraining exhibits clear scaling behavior, while real-robot pretraining saturates earlier. Real-world rollouts further show that ego-pretrained policies maintain high success rates under unseen-object shifts, whereas the no-pretraining baseline collapses.

downstream robot policies is poorly to provide the open-world coverage needed for broad generalization.

Egocentric human video ofers a natural way to address this coverage gap. Captured from a first-person perspective during everyday activity, it exposes models to contact-rich hand-object interactions, tool use, object state changes, and long-horizon behaviors at a scale that teleoperated robotics cannot easily approach. Recent work has begun to exploit egocentric video for embodied learning, including reusable visual representation learning, human-to-robot motion retargeting, cross-embodiment prior distillation, and downstream manipulation policy learning [10, 17, 24, 27, 30, 34, 39]. These results suggest that human video can provide useful pertraining signals despite lacking robot actions and exact embodiment alignment, but they do not measure whether this coverage advantage is competitive with teleoperated robot data under controlled, matched-scale pretraining. This leaves a basic question open: how does egocentric human video compare with real-robot data as a pretraining source?

We answer this question through a controlled, matched-scale comparison and find a clear result: egocentric human video pretraining leads to better generalization than real-robot pretraining. To isolate the efect of the pretraining source, we curate a 5,000-hour subset from the egocentric portion of HumanNet [9], selected for diversity in objects, skills, and environments, and compare it with an equal-scale multi-embodiment real-robot corpus. Both data sources are used to pretrain the same autoregressive world-action model, built on a Mixture-of-Transformers (MoT) architecture that predicts future video observations and subsequent actions. After pretraining, we post-train each model on the same set of real-robot tasks and evaluate it on two splits: held-out trajectories from the post-training tasks (Seen) and held-out tasks not used during post-training (Unseen). This design keeps model architecture, data scale, post-training data, and evaluation protocol fixed, making the pretraining source the primary variable.

Our key findings can be summarized as:

• Egocentric pretraining scales consistently. As the amount of egocentric pretraining data increases from hunreds of to thousands of hours, downstream validation loss decreases monotonically, indicating that additional egocentric video continues to provide useful embodied pretraining signal.

• Egocentric pretraining improves downstream generalization over real-robot pretraining. At matched scale and under the same pretraining–post-training protocol, egocentric pretraining achieves stronger performance than real-robot pretraining, with the largest gains on unseen tasks.

• The generalization advantage transfers to real-world execution. On real-robot rollouts, egocentric pretraining stays robust under distribution shift to unseen objects, while the no-pretraining baseline collapses, showing that the open-world prior holds up beyond validation loss.

## 2 Egocentric vs. Real-Robot Data: Coverage, Cost, and Alignment

A practical recipe for embodied models follows the pretraining–post-training paradigm that has been widely validated in language and vision-language modeling: pretraining learns general visual and physical representations from large heterogeneous corpora, while post-training adapts the model to a specific embodiment, camera configuration, and task distribution. Pretraining and post-training therefore emphasize diferent data properties. Pretraining benefits from coverage: broad exposure to scenes, objects, interactions, and behaviors that can support generalizable representations. Post-training, in contrast, places greater emphasis on alignment: embodiment-matched observations and actions that adapt the pretrained model to a target embodiment, camera setup, and task distribution. The choice of pretraining data source is thus not about which modality is intrinsically better, but about which supplies coverage at scale. We compare egocentric human video and teleoperated robot data along four axes: accessible scale, collection cost, acquisition dificulty, and diversity (Figure 2 and Table 1).

## 2.1 Accessible Scale

The two data supplies difer by orders of magnitude (Figure 1, Table 1). Despite years of community efort, individual robot releases remain in the hundreds to low thousands of hours, and even the most generous aggregations of the entire public supply total only ${ \sim } 2 \times 1 0 ^ { 4 }$ hours, comparable to what a single lab holds privately [4, 10, 25]. Egocentric video sits an order of magnitude higher: a single release such as Egocentric-100K already exceeds the entire aggregated robot supply several-fold [6], and frontier systems increasingly pair their robot data with far larger egocentric pools [25]. HumanNet [9], the dataset we build on, curates one million hours of human activity (over 800,000 hours of egocentric), from which we draw our 5,000-hour pretraining subset. This subset itself is beyond the reach of any open-sourced teleoperation dataset.

## 2.2 Collection Cost and Acquisition Difficulty

The scale gap follows directly from the cost structure. Robot teleoperation requires a robot, a teleoperation rig, a trained operator for the full duration, a staged scene, and ongoing maintenance; even the low-cost ALOHA platform costs ∼\$20k per station [38]. As cost is dominated by recurring labor and hardware, total volume scales only linearly with fleet size and budget [2, 18]. Egocentric data collection inverts this on every axis: footage is captured passively by an of-the-shelf camera or headset, or harvested from the web at no capture cost at all [14, 21, 26], which drives the marginal cost of an additional hour far lower. The remaining dificulty shifts from collection to curation (filtering, deduplication, privacy review, and pseudo-action labeling via hand-pose retargeting), a one-time, automatable compute cost rather than recurring labor. Human data is also more eficient per unit: minutes of egocentric demonstration can match or exceed far longer teleoperation sessions, with smoother motion and less idle time [34].

## 2.3 Diversity as a Function of Scale

Data amount alone is not suficient for efective embodied pretraining: what matters is whether each additional marginal hour exposes the model to new states, motions, interactions, and visual contexts. To compare egocentric human video and real-robot data under a controlled setting, we randomly sample approximately 2-hour subsets from each 5,000-hour data pool and compute the statistics shown in Figure 2. Consistent with previous findings in HumanEgo [34], the sampled egocentric human video also exhibits higher motion quality than real-robot data: human trajectories are smoother, as reflected by lower normalized jerk (Figure 2a), and contain substantially less action idle time (Figure 2b), indicating fewer stationary or uninformative segments.

![](images/64988143f30e04b082cdb110bc1bbad989d9b0f14be41bc86acd53d153142390.jpg)

![](images/775c99b30799b1f100a8ad1c2fc66573faca2a4f9be6e414aa5263a64fec1880.jpg)

![](images/8597ac8176e6104dd127dbcb1d98f98b4cf56169b69a754695d77463657f9746.jpg)

![](images/c183be2903ad4fb59142099fa994138cfd27236cb9fac1d8fcbde5ade4053a83.jpg)

![](images/f836bccd19bae4c28d4c86e5e4d86d1b2829f7410ab589f99c6b31af97c50726.jpg)

![](images/a5b7a5946e79ac9dc2c7516ecd0bb1a53255547a487a7dddf1d19c57bb50ce5f.jpg)  
Figure 2 Data diversity comparison between our egocentric human video and real-robot data. Following the visualization of HumanEgo [34], (a–b) motion-quality comparisons show that human trajectories exhibit smoother motion, as reflected by lower normalized jerk, and much less action idle time. (c–d) Spatial and trajectory-diversity comparisons show that human trajectories occupy a broader XZ workspace distribution and maintain larger inter-session positional spread over normalized time. (e–f) Semantic diversity comparisons show that egocentric human video contains a substantially richer interaction vocabulary and broader visual scene coverage than real-robot data. Together, these analyses indicate that egocentric human video provides a cleaner and more diverse pretraining substrate than real-robot teleoperation data at matched sampled duration.

Beyond motion quality, the more important question for scaling is whether the data continues to introduce non-redundant experience. At the matched subset scale, egocentric data is substantially more diverse than the collected real-robot data:

• Workspace coverage: real-robot motion is concentrated in the compact region reachable from a fixed workstation, whereas egocentric human motion spans a broader XZ workspace induced by unconstrained daily activities (Figure 2c).

• Inter-session variance: egocentric trajectories maintain larger positional variation across demonstrations, indicating that diferent demonstrations occupy more varied spatial configurations rather than repeatedly traversing a narrow motion manifold. Following HumanEgo [34], we quantify this efect by measuring positional spread across demonstrations over normalized time (Figure 2d).

• Interaction vocabulary: we count the set of unique verb-object pairs extracted from task descriptions. Robot data is constrained by scripted collection tasks and therefore repeats limited interactions, while egocentric video exhibits a long-tailed, open-vocabulary distribution of human-object interactions (Figure 2e).

• Visual scene coverage: we measure the diversity of scene semantics from caption-derived scene terms. Real-robot data saturates quickly because it is collected in bounded environments, while egocentric video covers a wider range of homes, workshops, kitchens, outdoor settings, objects, and surfaces (Figure 2f).

Together, these axes show that real-robot data saturation is structural, reflecting bounded environments, fixed workspace coverage, and scripted tasks, rather than an artifact of a particular collection efort. In contrast, egocentric human video is both individually informative and collectively less redundant: at the same sampled duration, it exposes the model to broader motion, spatial, interaction, and visual variation.

A division of labor. The analysis yields a clear division of labor in embodied training. Egocentric video dominates the axes that pretraining rewards, including scale, marginal cost, motion diversity, interaction diversity, and scene diversity, while its main weakness, the embodiment gap, is precisely what post-training can correct with a smaller amount of kinematically aligned robot data. Teleoperation data is strongest on the axis post-training needs, namely embodiment alignment, and weakest on the axes pretraining needs. The open question is empirical: at matched scale, does the diversity advantage of egocentric pretraining outweigh the kinematic alignment advantage of robot pretraining in the downstream policy? The rest of this paper answers in the afirmative through a controlled comparison.

Table 1 Landmark publicly released datasets with reported durations, sorted by scale. † marks datasets that aggregate other releases; ‡ marks datasets containing simulated trajectories.
<table><tr><td>Corpus</td><td>Hours</td><td>Acquisition</td><td>Coverage</td></tr><tr><td colspan="4">Egocentric human video</td></tr><tr><td>EgoDex [14]</td><td>829</td><td>Consumer headset, native hand pose</td><td>194 tabletop manipulation tasks</td></tr><tr><td>Ego4D [12]</td><td>3,670</td><td>931 camera wearers, daily life</td><td>74 locations, 9 countries</td></tr><tr><td>Sekai [21]</td><td>5,000+</td><td>Web harvest (egocentric POV)</td><td>Walking/exploration, global</td></tr><tr><td>Xperience-10M [31]</td><td>10,000</td><td>Wearable capture, 10M interactions</td><td>Open-world daily experience</td></tr><tr><td>EgoScale [39]†</td><td>20,854</td><td>Aggregation + hand retargeting</td><td>Dexterous manipulation</td></tr><tr><td>Egocentric-100K [6]</td><td>100,405</td><td>14,228 workers, head-mounted glasses</td><td>Industrial / factory operations</td></tr><tr><td colspan="4">Teleoperated real-robot</td></tr><tr><td>DROID [18]</td><td>350</td><td>50 collectors, 12 months</td><td>564 scenes, 84 tasks</td></tr><tr><td>Galaxea Open-World [16]</td><td>500</td><td>Single embodiment, in-the-wild</td><td>Homes, kitchens, retail, offices</td></tr><tr><td>MolmoAct2 BimanualYAM [11]</td><td>720+</td><td>Bimanual YAM arms</td><td>Largest open bimanual release</td></tr><tr><td>RoboMIND 2.0 [15]</td><td>1,000+</td><td>Bimanual mobile teleop, 310K+ trajs</td><td>Bimanual coordination tasks</td></tr><tr><td>Open X-Embodiment [29]†</td><td>~2,000-3,000</td><td>Pooling of 60 datasets, 1M+ trajs</td><td>22 embodiments, 527 skills</td></tr><tr><td>AgiBot World [2]</td><td>2,976</td><td>100-robot fleet, 1M+ trajs</td><td>217 tasks, 106 scenes, 5 domains</td></tr><tr><td>RoVid-X [10]†‡</td><td>10,000+</td><td>Open-source aggregation, 4M robot videos</td><td>1,300+ fine-grained robot skills</td></tr><tr><td>Being H-0.5 [25]††</td><td>~35,000</td><td>OXE + AgiBot + RoboMIND + RoboCOIN + ..</td><td>30 embodiments, incl. sim</td></tr><tr><td colspan="4">Reference points π0 corpus [4] &gt;10,000</td></tr><tr><td></td><td>1,000,000</td><td></td><td>Proprietary, inaccessible</td></tr><tr><td>HumanNet [9]</td><td></td><td>Web curation, egocentric + third-person</td><td>Open-world, long-tail interaction</td></tr></table>

## 3 Embodied Pretraining with Egocentric Human Video

We study egocentric pretraining with an autoregressive world action model that unifies video dynamics prediction and action inference through a Mix-of-Transformers (MoT) architecture. Specifically, the video expert is initialized from Wan 2.2, while the action expert is initialized via interpolation. To isolate the efect of the pretraining substrate, we compare the post-training performance of models pretrained on egocentric human video versus real-robot data. Throughout these comparisons, we rigorously hold the post-training data, compute budget, and evaluation protocol fixed.

The splits are designed around one principle: introduce a controlled distributional shift at each stage so that downstream performance isolates the quality of the pretraining substrate. We pretrain separately on egocentric and real-robot data, post-train on a real-robot dataset disjoint from both, and evaluate on two held-out splits that respectively measure in-distribution robustness and out-of-distribution extrapolation.

Stage 1: Pretraining data. We construct two pretraining sets matched at 5,000 hours but difering in collection:

Egocentric. Curated from the egocentric portion of HumanNet [9]; per-clip end-efector poses and gripper states are estimated from retargeted hand-pose signals as pseudo-action labels, placing it in the same action space as the robot data. It ofers open-world coverage of backgrounds, objects, and interaction skills.

Real-robot. Multi-embodiment trajectories with precise end-efector poses and gripper states, aggregated from various real-robot datasets. Although its environmental diversity and task scenarios remain bounded compared to the egocentric counterpart, it provides kinematically aligned embodied priors.

Stage 2: Post-training Data. For post-training, we curate a real-robot dataset from AgiBot World [2], selecting 15 manipulation tasks with 100 expert demonstrations per task, resulting in 1,500 trajectories in total. Compared with the pretraining robot data, this post-training set contains more diverse backgrounds and object instances, providing richer downstream manipulation scenarios for adapting the pretrained policy.

Evaluation Protocol. We evaluate models using validation flow-matching action loss on held-out Stage-2 robot data. We report results on two splits. The Seen split holds out trajectories from the 15 post-training tasks, where the task semantics are observed during post-training but the evaluation trajectories involve unseen object instances and variations, measuring robustness within the post-training task distribution. The Unseen split comprises 25 tasks that are not included in Stage-2 post-training, and serves as our primary evaluation for out-of-distribution generalization. This split tests whether the open-world prior learned from egocentric pretraining can better transfer to novel manipulation scenarios than a kinematically aligned prior learned from real-robot pretraining with relatively limited scene and object coverage.

![](images/54751d0413046371652a288b205f5946efee4d08638cf6a7bed7f0040aaa4730.jpg)

![](images/b156b5d1fd10e6ee0bd3623b2603e0f7df7d82a2d17e9479159be7c97ff3a2ff.jpg)  
Figure 3 Egocentric pretraining scales log-linearly while robot pretraining stalls on unseen tasks. Best post-training action loss as a function of pretraining hours on Seen and Unseen tasks. The blue curve shows ego pretraining, and the light orange curve overlays robot pretraining at matched scales. Dashed gray and blue lines denote the Wan2.2 [33] (no pretraining) and LingBot-VA [20] references.

## 4 Experimental Results and Analysis

In this section, we conduct empirical studies around two questions that together characterize the value of egocentric pretraining for embodied foundation models.

• Q1 (Ego Pretrain Scaling): Does egocentric pretraining exhibit scaling behavior in robot post-training?

• Q2 (Ego vs. Robot Pretrain): How does egocentric pretraining perform compared to real-robot pretraining?

We follow the training and evaluation protocol described in Section 3. We evaluate checkpoints throughout post-training and report the minimum validation loss on both in-distribution and out-of-distribution tasks. We compare our method with two baselines. The first is Wan2.2 without embodied pretraining. The second is LingBot-VA, which fine-tunes Wan2.2 on 20k hours of real-robot data and serves as a strong embodiedpretrained baseline. The complete validation action loss curves are shown in Fig. 4, and the analysis below summarizes each curve by its lowest loss.

## 4.1 Egocentric Pretraining Scales with Data

As egocentric pretraining scales from 100 to 5,000 hours, the best post-training action loss decreases monotonically in both evaluation settings (Fig. 3). Specifically, the loss drops from 0.0080 to 0.0067 on seen tasks and from 0.0234 to 0.0204 on unseen tasks, reaching values 35% and 24% lower than the $\mathrm { W a n 2 . 2 }$ baseline without pretraining. Across this range, the trend is well captured by a log-linear scaling law, ${ \mathcal { L } } = a - b \ln ( D )$ with $R ^ { 2 } = 0 . 8 6$ for seen tasks and $R ^ { 2 } = 0 . 9 4$ for unseen tasks. The fitted slope remains clearly negative up to 5,000 hours, suggesting that egocentric pretraining has not yet saturated and that further gains may be possible as the data scale increases.

We attribute this scaling behavior to the diversity of the egocentric data. Because our egocentric subsets are curated for broad coverage, increasing the number of hours does not simply repeat redundant trajectories, but introduces a wider range of manipulated objects, manipulation skills, and physical environments. The growing diversity benefits the two evaluation settings in complementary ways. For unseen tasks, out-of-distribution generalization depends more on the breadth of behavioral space observed during pretraining, and broader coverage therefore leads to lower loss on tasks held out from post-training. For seen tasks, the same diversity improves the pretrained representation. When the pretraining data are narrow, the model fits a limited set of human activities and transfers a correspondingly specialized initialization to post-training. In contrast, a diverse corpus encourages a more general and less task-specific representation, which provides a stronger starting point for adaptation to the target robot tasks.

![](images/3998fa0063e3ad2c853b3f77004728ee43161916e6c86f865deb38066b5dc4b6.jpg)

![](images/2f0abb522acb09b050e79de173afab082d40e380c9f4670743ecbe34f802a573.jpg)  
Figure 4 Complete validation action loss curves during post-training, for egocentric and real-robot pretraining at 100, 1,000, and 5,000 hours together with the Wan2.2 and LingBot-VA baselines. The left panel reports seen tasks (in-distribution), and the right panel reports unseen tasks (out-of-distribution).

## 4.2 Egocentric Pretraining Generalizes Better than Robot Pretraining

We compare egocentric and real-robot pretraining at matched data scales (Fig. 3). The two data sources exhibit distinct scaling behaviors across the two evaluation settings. On seen tasks, real-robot pretraining improves steadily with data scale and remains close to egocentric pretraining, reaching a loss of 0.0071 at 5,000 hours, compared with 0.0067 for egocentric pretraining. On unseen tasks, however, scaling real-robot data produces no consistent improvement: its loss remains near 0.025 across all scales and reaches 0.0254 at 5,000 hours, substantially higher than that of egocentric pretraining. In contrast, egocentric pretraining continues to improve on unseen tasks as the data scale increases, reaching 0.0204 at 5,000 hours and achieving a roughly 20% lower loss than real-robot pretraining at the same scale. These results indicate that the two sources are comparably efective for in-distribution transfer, whereas only egocentric data scales toward stronger out-of-distribution generalization.

## The diference derives from two factors:

• Diversity and information density. As analyzed in Section 2, egocentric data is intrinsically more diverse and covers a far wider range of tasks, objects, and backgrounds than teleoperated robot data, which is collected in a bounded set of laboratory setups. Matching the two sources by hours further understates this advantage, since an hour of egocentric video contains far more and cleaner trajectories than an hour of robot teleoperation. In our 100-hour recipe, for instance, the egocentric data comprises roughly 45,000 trajectories, whereas the real-robot data contains only about 8,000, as teleoperation is slowed by long idle intervals and the comparatively slow motion of the robot arm. Each hour of teleoperated data therefore carries substantially less information and ofers little advantage as a pretraining source.

• Limited generalization to unseen skills. We keep the real-robot pretraining tasks disjoint from both the post-training and the out-of-distribution evaluation tasks, and preserve this isolation as the real-robot corpus is scaled up. This control rules out any shortcut in which scaled robot data overlaps with the evaluation tasks, so the measured behavior reflects genuine generalization rather than leakage. However, scaling real-robot pretraining does not yield the generalization, as its loss on Unseen tasks stays nearly flat while the data grows, indicating that manipulation tasks collected in a constrained laboratory setting transfer poorly to genuinely unseen tasks.

![](images/eb5ed0f9f72048cfd717f013d24ea76cb9a92f578efc61d5554134efe00a79c4.jpg)  
Figure 5 Real-world rollouts on three tasks (place cup, pick up fruits, stamp), each under in-distribution and ood settings. The egocentric-pretrained model succeeds under distribution shift, whereas the baseline fails.

## 4.3 Real-World Robot Experiments

We validate our findings with real-robot rollouts on an AgiBot bimanual platform across three manipulation tasks: placing a cup on a coaster, sorting fruits and vegetables, and stamping. For each task we evaluate an in-distribution setting, whose objects are seen during post-training, and an out-of-distribution setting with previously unseen object instances. We compare the egocentric-pretrained model against the Wan 2.2 baseline under identical post-training. The advantage of egocentric pretraining is already visible during posttraining. On the fruit-and-vegetable sorting task (Fig. 6), the egocentric-pretrained initialization starts with a substantially lower loss and converges to a value approximately 2.4× lower than that of the no-pretraining baseline. This result indicates that the pretrained prior makes the downstream task easier to fit and is consistent with the observed gap in the following real-world rollouts.

Table 2 Real-robot performance on AgiBot bimanual platform. Success rate is averaged over tasks. Egocentric pretraining transfers to out-of-distribution objects with little degradation, while the baseline collapses.
<table><tr><td>Pretraining</td><td>In-distribution</td><td>Out-of-distribution</td></tr><tr><td>Wan2.2 (baseline)</td><td>40.0%</td><td>0.0%</td></tr><tr><td>Egocentric (ours)</td><td>92.5%</td><td>90.0%</td></tr></table>

As summarized in Table 2, the egocentric-pretrained model attains a 92.5% in-distribution success rate and retains 90.0% under distribution shift with a drop of only 2.5 points. The baseline, by contrast, reaches 40.0% in-distribution and degrades to 0% on ood trials, a collapse of 40 points. The degradation is small for egocentric pretraining but catastrophic for the baseline, indicating that the open-world prior learned from human video transfers, whereas a policy lacking this prior overfits to the narrow visual distribution of its post-training data and fails to generalize.

## 5 Related Work

Scaling robot learning with real-robot data. Teleoperated data is the dominant substrate for embodied foundation models. Open X-Embodiment federated over one million trajectories across 22 embodiments, showing cross-embodiment mixing improves robustness and zero-shot transfer [29]. RT-1/RT-2 established that scaling robot-hours and fine-tuning a web-pretrained VLM backbone yield semantic generalization [5, 40]. DROID [18], π [4], GR00T [28], and AgiBot World [2] pushed scale, diversity, and new embodiments. This data is kinematically aligned with the deployment policy, but its collection cost bounds supply to current robots, operators, and labs, yielding homogeneous backgrounds, objects, and interactions, as well as brittle out-of-distribution behavior.

Egocentric data for robot pretraining. Egocentric corpora such as EPIC-KITCHENS [8], Ego4D [12], and Ego-Exo4D [13] have long supported recognition and representation learning, with embodied transfer as an indirect application. A recent line targets egocentric video directly as a pretraining substrate: EgoMimic co-trains human and robot data [17]; EgoScale scales to 20,000+ hours with retargeted hand and wrist motion and reports a log-linear scaling law on dexterous manipulation [39]; Being-H0 distills a hand-trajectory prior for humanoids [24]; and HumanEgo shows minutes of egocentric demonstration can replace far longer teleoperation [34]. Yet no prior work compares egocentric and real-robot pretraining head-to-head at matched scale on a controlled post-training benchmark. We fill this gap, showing egocentric pretraining not only substitutes for but surpasses real-robot pretraining, with the largest gains on OOD generalization, and we establish the first such scaling curves for an autoregressive world-action model.

![](images/35827b59f57c351d62e885bca43c06ad33eae8e6c5e04986c8a0199afd9064fc.jpg)

![](images/c0a4cccbef20a6444abe9c3e380c7d6ad86143ed53d9c11c81784a657f4f80ee.jpg)  
Figure 6 Post-training action and video loss on the fruit-and-vegetable sorting task. Egocentric pretraining stays below the no-pretraining baseline throughout, converging to about 2.4× lower action loss as well as lower video loss.

Architectures for embodied intelligence. Embodied policy learning is organized around two architectural families. Vision-language-action (VLA) models add an action head to a pretrained vision-language backbone, as in RT-2 [40], π [4], GR00T [28], OpenVLA [19], RDT [23], and LingBot-VLA [35]. World-Action Models (WAMs) instead model future states and actions jointly, using video generation as a dense representation of how the world evolves under control. DreamZero [36] jointly denoises future video and actions in a single difusion process, while LingBot-VA [20] predicts them sequentially, first generating future video and then decoding the corresponding actions via causal autoregression. Since the imagine-then-execute paradigm incurs heavy test-time latency from iterative video rollout, Fast-WAM [37] retains video co-training but skips future generation at inference, showing that the benefit of video modeling lies mainly in shaping representations during training rather than imagining futures at test time. The two families exploit visual supervision diferently. Our experiments focus on the world-action model, where video generation provides a dense learning signal alongside action prediction; we leave a parallel study of VLA models to future work.

## 6 Conclusion

We presented a controlled, matched-scale comparison of egocentric human video and real-robot data as pretraining data source for embodied foundation models. From the perspective of pretraining data, egocentric video leads on the axes that pretraining rewards, namely scale, cost, and diversity, while its embodiment gap is the part post-training is meant to close. Empirically, under an identical post-training and evaluation protocol, egocentric pretraining scales with data and surpasses real-robot pretraining, with the largest gains on out-of-distribution generalization. We view these results as encouraging, but still preliminary.

Future Work. We have a few ongoing explorations. Due to the limited availability of real-robot data, the current training scheme is limited to 5000 hours. As more real-robot data become available, we plan to scale the egocentric corpus and pretraining budget substantially further. Besides, current evaluations are mainly based on world action models with WAN2.2 serving as the backbone of the video generation model. Currently, we are also evaluating the scaling behavior of egocentric data based on vision-language-action models (VLAs) across a broader range of robot embodiments, testing whether the advantage we observe persists at foundation-model scale and beyond the world-action model studied here.

## References

[1] Abdelrahman Abouelenin, Atabak Ashfaq, Adam Atkinson, Hany Awadalla, Nguyen Bach, Jianmin Bao, Alon Benhaim, Martin Cai, Vishrav Chaudhary, Congcong Chen, et al. Phi-4-mini technical report: Compact yet powerful multimodal language models via mixture-of-loras. arXiv preprint arXiv:2503.01743, 2025.

[2] AgiBot World Contributors. Agibot world colosseo: A large-scale manipulation platform for scalable and intelligent embodied systems. arXiv preprint arXiv:2503.06669, 2025.

[3] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025.

[4] Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, et al. π : A vision-language-action flow model for general robot control. arXiv preprint arXiv:2410.24164, 2024.

[5] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Joseph Dabis, Chelsea Finn, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Jasmine Hsu, et al. Rt-1: Robotics transformer for real-world control at scale. arXiv preprint arXiv:2212.06817, 2022.

[6] Build AI. Egocentric-100k: 100,000 hours of real-world egocentric video from factory workers. https:// huggingface.co/datasets/builddotai/Egocentric-100K, 2026.

[7] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271, 2024.

[8] Dima Damen, Hazel Doughty, Giovanni Maria Farinella, Antonino Furnari, Evangelos Kazakos, Jian Ma, Davide Moltisanti, Jonathan Munro, Toby Perrett, Will Price, et al. Rescaling egocentric vision: Collection, pipeline and challenges for epic-kitchens-100. International Journal of Computer Vision, 130(1):33–55, 2022.

[9] Yufan Deng and Daquan Zhou. Humannet: Scaling human-centric video learning to one million hours. arXiv preprint arXiv:2605.06747, 2026.

[10] Yufan Deng, Zilin Pan, Hongyu Zhang, Xiaojie Li, Ruoqing Hu, Yufei Ding, Yiming Zou, Yan Zeng, and Daquan Zhou. Rethinking video generation model for the embodied world. arXiv preprint arXiv:2601.15282, 2026.

[11] Haoquan Fang, Jiafei Duan, Donovan Clay, Sam Wang, Shuo Liu, Weikai Huang, Xiang Fan, Wei-Chuan Tsai, et al. Molmoact2: Action reasoning models for real-world deployment. arXiv preprint arXiv:2605.02881, 2026.

[12] Kristen Grauman, Andrew Westbury, Eugene Byrne, Zachary Chavis, et al. Ego4D: Around the world in 3,000 hours of egocentric video. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022.

[13] Kristen Grauman, Andrew Westbury, Lorenzo Torresani, Kris Kitani, et al. Ego-Exo4D: Understanding skilled human activity from first- and third-person perspectives, 2024. URL https://arxiv.org/abs/2311.18259.

[14] Ryan Hoque, Peide Huang, David J. Yoon, Mouli Sivapurapu, and Jian Zhang. Egodex: Learning dexterous manipulation from large-scale egocentric video, 2026. URL https://arxiv.org/abs/2505.11709.

[15] Chengkai Hou, Kun Wu, Jiaming Liu, Zhengping Che, Di Wu, Fei Liao, Guangrun Li, Jingyang He, et al. Robomind 2.0: A multimodal, bimanual mobile manipulation dataset for generalizable embodied intelligence. arXiv preprint arXiv:2512.24653, 2025.

[16] Tao Jiang, Tianyuan Yuan, Yicheng Liu, Chenhao Lu, Jianning Cui, Xiao Liu, Shuiqi Cheng, Jiyang Gao, et al. Galaxea open-world dataset and g0 dual-system vla model. arXiv preprint arXiv:2509.00576, 2025.

[17] Simar Kareer, Dhruv Patel, Ryan Punamiya, Pranay Mathur, Shuo Cheng, Chen Wang, Judy Hofman, and Danfei Xu. Egomimic: Scaling imitation learning via egocentric video, 2024. URL https://arxiv.org/abs/2410.24221.

[18] Alexander Khazatsky et al. Droid: A large-scale in-the-wild robot manipulation dataset, 2025. URL https: //arxiv.org/abs/2403.12945.

[19] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, Quan Vuong, Thomas Kollar, Benjamin Burchfiel, Russ Tedrake, Dorsa Sadigh, Sergey Levine, Percy Liang, and Chelsea Finn. Openvla: An open-source vision-language-action model. arXiv preprint arXiv:2406.09246, 2024.

[20] Lin Li, Qihang Zhang, Yiming Luo, Shuai Yang, Ruilin Wang, Fei Han, Mingrui Yu, Zelin Gao, Nan Xue, Xing Zhu, et al. Causal world modeling for robot control. arXiv preprint arXiv:2601.21998, 2026.

[21] Zhen Li, Chuanhao Li, Xiaofeng Mao, Shaoheng Lin, Ming Li, Shitian Zhao, Zhaopan Xu, Xinyue Li, et al. Sekai: A video dataset towards world exploration. arXiv preprint arXiv:2506.15675, 2025.

[22] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024.

[23] Songming Liu, Lingxuan Wu, Bangguo Li, Hengkai Tan, Huayu Chen, Zhengyi Wang, Ke Xu, Hang Su, and Jun Zhu. Rdt-1b: a difusion foundation model for bimanual manipulation. arXiv preprint arXiv:2410.07864, 2024.

[24] Hao Luo, Yicheng Feng, Wanpeng Zhang, Sipeng Zheng, Ye Wang, Haoqi Yuan, Jiazheng Liu, Chaoyi Xu, Qin Jin, and Zongqing Lu. Being-h0: Vision-language-action pretraining from large-scale human videos, 2025. URL https://arxiv.org/abs/2507.15597.

[25] Hao Luo, Ye Wang, Wanpeng Zhang, Sipeng Zheng, Ziheng Xi, Chaoyi Xu, Haiweng Xu, Haoqi Yuan, Chi Zhang, Yiqing Wang, Yicheng Feng, and Zongqing Lu. Being-h0.5: Scaling human-centric robot learning for cross-embodiment generalization, 2026. URL https://arxiv.org/abs/2601.12993.

[26] Antoine Miech, Dimitri Zhukov, Jean-Baptiste Alayrac, Makarand Tapaswi, Ivan Laptev, and Josef Sivic. Howto100m: Learning a text-video embedding by watching hundred million narrated video clips, 2019. URL https://arxiv.org/abs/1906.03327.

[27] Suraj Nair, Aravind Rajeswaran, Vikash Kumar, Chelsea Finn, and Abhinav Gupta. R3m: A universal visual representation for robot manipulation, 2022. URL https://arxiv.org/abs/2203.12601.

[28] NVIDIA et al. GR00T N1: An open foundation model for generalist humanoid robots, 2025. URL https: //arxiv.org/abs/2503.14734.

[29] Open X-Embodiment Collaboration et al. Open X-Embodiment: Robotic learning datasets and RT-X models, 2025. URL https://arxiv.org/abs/2310.08864.

[30] Ryan Punamiya et al. Egoverse: An egocentric human dataset for robot learning from around the world, 2026. URL https://arxiv.org/abs/2604.07607.

[31] Ropedia. Xperience-10m: A large-scale egocentric multimodal dataset with structured 3d/4d annotations, 2026. Dataset.

[32] Kimi Team, Angang Du, Bohong Yin, Bowei Xing, Bowen Qu, Bowen Wang, Cheng Chen, Chenlin Zhang, Chenzhuang Du, Chu Wei, et al. Kimi-vl technical report. arXiv preprint arXiv:2504.07491, 2025.

[33] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, Jiayu Wang, Jingfeng Zhang, Jingren Zhou, Jinkai Wang, Jixuan Chen, Kai Zhu, Kang Zhao, Keyu Yan, Lianghua Huang, Mengyang Feng, Ningyi Zhang, Pandeng Li, Pingyu Wu, Ruihang Chu, Ruili Feng, Shiwei Zhang, Siyang Sun, Tao Fang, Tianxing Wang, Tianyi Gui, Tingyu Weng, Tong Shen, Wei Lin, Wei Wang, Wei Wang, Wenmeng Zhou, Wente Wang, Wenting Shen, Wenyuan Yu, Xianzhong Shi, Xiaoming Huang, Xin Xu, Yan Kou, Yangyu Lv, Yifei Li, Yijing Liu, Yiming Wang, Yingya Zhang, Yitong Huang, Yong Li, You Wu, Yu Liu, Yulin Pan, Yun Zheng, Yuntao Hong, Yupeng Shi, Yutong Feng, Zeyinzi Jiang, Zhen Han, Zhi-Fan Wu, and Ziyu Liu. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

[34] Zhi Wang, Botao He, Kelin Yu, Seungjae Lee, Ruohan Gao, Furong Huang, and Yiannis Aloimonos. Humanego: Zero-shot robot learning from minutes of human egocentric videos. arXiv preprint, 2025.

[35] Wei Wu, Fan Lu, Yunnan Wang, Shuai Yang, Shi Liu, Fangjing Wang, Qian Zhu, He Sun, Yong Wang, Shuailei Ma, et al. A pragmatic vla foundation model. arXiv preprint arXiv:2601.18692, 2026.

[36] Seonghyeon Ye et al. World action models are zero-shot policies, 2026. URL https://arxiv.org/abs/2602.15922.

[37] Tianyuan Yuan, Zibin Dong, Yicheng Liu, and Hang Zhao. Fast-wam: Do world action models need test-time future imagination?, 2026. URL https://arxiv.org/abs/2603.16666.

[38] Tony Z. Zhao, Vikash Kumar, Sergey Levine, and Chelsea Finn. Learning fine-grained bimanual manipulation with low-cost hardware. In Robotics: Science and Systems (RSS), 2023.

[39] Ruijie Zheng, Dantong Niu, Yuqi Xie, Jing Wang, Mengda Xu, Yunfan Jiang, Fernando Castañeda, Fengyuan Hu, You Liang Tan, Letian Fu, Trevor Darrell, Furong Huang, Yuke Zhu, Danfei Xu, and Linxi Fan. Egoscale: Scaling dexterous manipulation with diverse egocentric human data, 2026. URL https://arxiv.org/abs/2602.16710.

[40] Brianna Zitkovich, Tianhe Yu, Sichun Xu, Peng Xu, Ted Xiao, Fei Xia, Jialin Wu, Paul Wohlhart, Stefan Welker, Ayzaan Wahid, et al. Rt-2: Vision-language-action models transfer web knowledge to robotic control. In Conference on Robot Learning, pages 2165–2183. PMLR, 2023.