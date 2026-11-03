<!-- page 1 of 28 -->

arXiv:2610.31670v2 [cs.CV] 23 Oct 2026

# Sensorimotor Alignment: A Platonic Proxy for Embodied Policy Success

Yunfei Xie<sup>1</sup>, Howard H. Qian<sup>1</sup>, Shiyi Lan<sup>2</sup>, Kaiyu Hang<sup>1</sup>, Chen Wei<sup>1†</sup>

<strong><sup>1</sup></strong>**Rice University,** <strong><sup>2</sup></strong>**NVIDIA**

## Abstract

Robot foundation models are typically evaluated through costly closed-loop rollouts, motivating low-cost offline proxies for model capability. The Platonic Representation Hypothesis (PRH) [21] suggests that as representations improve, modalities such as vision and language converge toward a shared statistical model of reality. We ask whether motion trajectories form another such modality: as embodied models become more capable, should their sensory representations converge toward the structure of physical motion? To test this, we propose **Sensorimotor Alignment** $( S ^ { 2 } ) _ { , }$ , a training-free offline metric that measures how well a model’s sensory representation aligns with the geometry of physical motion. We find that $S ^ { 2 }$ correlates positively with downstream robot task success across model families, making it a low-cost offline proxy for prioritizing models before closed-loop evaluation. $S ^ { 2 }$ can also identify promising vision-language model (VLM) backbones before costly fine-tuning into vision-language-action models (VLAs). Beyond this correlation, $S ^ { 2 }$ offers representation-level explanations for phenomena observed in VLAs and world action models (WAMs). ${ \mathsf { W i t h } }   S ^ { 2 } ,$ we extend PRH to robotics and make sensorimotor alignment measurable, offering a low-cost, representation-level view of robot foundation models that closed-loop success rates alone cannot reveal.

**Website** [https://yunfeixie233.github.io/s2/](https://yunfeixie233.github.io/s2/)

**Code** [https://github.com/yunfeixie233/sensorimotor-alignment](https://github.com/yunfeixie233/sensorimotor-alignment)

<small><span class="docvortex-page-footnote" data-block-type="page_footnote" style="color:#6b7280"><sup>†</sup>Corresponding author.</span></small>

<!-- page 2 of 28 -->

![Image block](doc:0a14520/tier:basic/page:2/block:1)

Figure 1 | Left: We test the Platonic Representation Hypothesis (PRH) in robotics, using image–text features as the sensory representation and end-effector trajectories as the motor representation: a more capable embodied model should have sensory representations that align more closely with this motor representation. To validate this, we define Sensorimotor Alignment (S<sup>2</sup>). Right: We measure S<sup>2</sup> across 12 vision-language-action models (VLAs) and find that S<sup>2</sup>correlates positively with SimplerEnv task success rate (Pearson 𝑟 = +0.717), providing empirical support for this PRH-inspired view of embodiment.

## 1. Introduction

Robot foundation models increasingly build on large pretrained vision-language models [3, 4], video models [50], and world action models [5, 15, 28]. Yet selecting and evaluating these models remains expensive: the most reliable test is still closed-loop rollout in simulation [30, 33] or the real world [1, 9], where evaluation is slow, noisy, and benchmark-dependent. This expense motivates low-cost offline proxies for policy capability.

A natural approach is to use existing perception-side proxies, such as visual question answering scores for VLMs [31, 57] or pixel-level video quality for world models [18, 23, 52]. However, these metrics measure passive perception or generation quality, not whether a representation is organized for action [24, 35, 47, 59]. Two scenes may look visually different but require similar robot motions; conversely, two visually similar scenes may demand different trajectories. For robot task success, a model must produce different motions across tasks even in similar scenes. This requires aligning the sensory representation with the motor representation.

This perspective connects robotics to the Platonic Representation Hypothesis (PRH) [21], which argues that as models of different modalities scale and solve diverse tasks, they are converging to a shared statistical model of reality in each of their own representation spaces. For example, images and text are two projections of the same underlying world. As vision and language models scale, they increasingly agree on the relative distances between data points. We bring PRH to robotics and ask: can physical robot motion be viewed as another projection of that same world, such that more capable embodied models better align their vision and language representations with the structure of motion trajectories?

We test this prediction of PRH: the more capable an embodied model is, the more closely its sensory representations align with its motor representation. In other words, episodes nearby in sensory representation space should also be nearby in motor representation space.

We formalize this as **Sensorimotor Alignment** (S<sup>2</sup>), a training-free metric, illustrated in Figure 1. For each episode in a robot dataset, a model’s image and text features form its sensory representation, and the recorded end-effector trajectory forms its motor representation. We measure alignment by comparing each episode’s nearest neighbors in sensory and motor spaces. More overlap between the two neighbor sets means stronger sensorimotor alignment.

2

<!-- page 3 of 28 -->

In Section 3, we show that $S ^ { 2 }$ correlates positively with downstream robot task success across model families, in simulated and real-world settings. The correlation is strong for Vision-Language-Action models (VLAs) in closed-loop manipulation, indicating that models with higher sensorimotor alignment tend to perform better. $\dot { S } ^ { 2 }$ computed on raw Vision-Language Models (VLMs) also correlates with the task success of the resulting VLAs, more strongly than their visual question answering (VQA) scores do. For World Action Models (WAMs), $S ^ { 2 }$ likewise correlates more strongly with task success than video-quality metrics such as FID [18], SSIM [52], and PSNR [19], and for real-robot policies, $S ^ { 2 }$ computed offline correlates with physical-robot task success on a shared set of tasks. Across these settings, sensorimotor alignment correlates with task success more closely than VQA or video-quality metrics.

Prior work has observed several phenomena in VLAs and WAMs only through closedloop results, without explaining them at the representation level. We use $\mathrm { ` S ^ { 2 } }$ to supply that explanation, as shown in Section 4. For example, fine-tuned VLAs often ignore the language instruction and act mainly on what they see; we find that their task success correlates with the $S ^ { 2 }$ of their vision features, not their text features.

Our contributions are threefold:

• Inspired by the Platonic Representation Hypothesis, we propose a new view of embodiment: more capable embodied models should align their sensory representations more closely with the structure of motion trajectories. To test this, we introduce $\left[ \overline { { S } } ^ { 2 } \right]$ a training-free offline metric that measures alignment between sensory and motor representations.

• Across model families, in simulation and on real robots, $S ^ { 2 }$ correlates positively with task success, making it a low-cost offline proxy for choosing models before closed-loop evaluation.

• Furthermore, we use $S ^ { 2 }$ to explain, at the representation level, several phenomena in VLAs and WAMs that were previously observed only through closed-loop results, making it a diagnostic tool for understanding how design choices shape embodied models.

We make sensorimotor alignment measurable, offering a low-cost, representation-level view of robot foundation models that closed-loop success rates alone cannot reveal.

## 2. Measuring Sensorimotor Alignment

Motivated by the Platonic Representation Hypothesis (PRH) [21], we hypothesize that the more capable an embodied model is, the more closely its sensory representations align with its motor representation. Here, sensory representations are the model’s image and text features, and each episode’s end-effector trajectory is its motor representation.

To make this alignment measurable, we define **Sensorimotor Alignment** $( S ^ { 2 } )$ by constructing both representations and comparing their neighborhoods, as follows:

**Reference dataset.** To measure cross-modal alignment, we use the DROID [26] and Bridge DataV2 [49] datasets, which pair sensory inputs with physical motions. From a dataset, we sample 𝑁 episodes and compute both representations from each episode. For episode $i ,$ w e randomly sample one image frame and pair it with the task instruction; this image–text pair forms the sensory input $\mathbf { x } _ { i }$ . Sampling details are provided in Appendix A.1.

**Sensory representation.** For each episode 𝑖, we encode the image frame and text instruction with the vision and text encoders of a robot foundation model, obtaining a sensory embedding

$$
\mathbf { z } _ { i } = f ( \mathbf { x } _ { i } ) \in \mathbb { R } ^ { d } ,
$$

where $f$ applies the vision encoder to the image and the text encoder to the instruction, and we average-pool the output token features into a single 𝑑-dimensional vector.

3

<!-- page 4 of 28 -->

![Image block](doc:0a14520/tier:basic/page:4/block:1)

Figure 2 | Measuring Sensorimotor Alignment $( S ^ { 2 } )$ by nearest-neighbor overlap. For the query episode (left), sensory neighbors (bottom right) have similar vision-text features, and motor neighbors (top right) have similar trajectories. Green checks mark shared neighbors; red crosses mark neighbors found only in the sensory set. A larger overlap means that episodes with similar sensory representations also follow similar motor trajectories.

**Motor representation.** For the sampled frame in $\mathbf { x } _ { i } ,$ we extract end-effector positions, orientations, and gripper states over the next ℎ steps, forming episode $i ^ { \prime }   \mathbf { s }$ motor representation:

$$
\pmb { \tau } _ { i } = ( x _ { i } , y _ { i } , z _ { i } , \phi _ { i } , \theta _ { i } , \psi _ { i } , g _ { i } ) \in \mathbb { R } ^ { h \times 7 } ,
$$

where row 𝑢 is the state 𝑢 steps after the sampled frame: $x , y ,$ 𝑧 are position coordinates, 𝜙, 𝜃, 𝜓 are roll, pitch, and yaw angles, and $g$ is the recorded gripper coordinate. We encode positions as vectors in $\mathbb { R } ^ { 3 }$ and orientation angles as rotations in SO(3) so that distances reflect the geometry of translations and rotations. Details are in Appendix A.2.

**Sensorimotor alignment.** We measure alignment with a mutual nearest-neighbor metric comparing the 𝑘-nearest-neighbor sets in sensory and motor spaces; greater overlap indicates stronger alignment between an episode’s sensory and motor representations.

We use cosine distance in sensory space and Dynamic Time Warping (DTW) [37, 41] in motor space to compare trajectories at different speeds. For episodes 𝑖 and 𝑗:

$$
d _ { \mathrm { s e n s o r y } } ( i , j ) = 1 - \mathsf { c o s } ( \mathbf { z } _ { i } , \mathbf { z } _ { j } ) , \qquad d _ { \mathrm { m o t o r } } ( i , j ) = \mathsf { D T W } ( \boldsymbol { \tau } _ { i } , \boldsymbol { \tau } _ { j } ) .
$$

Smaller distances mean closer neighbors. Appendices A.2–A.3 detail DTW and normalization.

For each episode $i ,$ we use $d _ { \mathrm { s e n s o r y } }$ to form the sensory 𝑘-nearest-neighbor set $\mathcal { N } _ { \mathrm { s e n s o r y } } ( i )$ and $d _ { \mathrm { m o t o r } }$ to form the motor 𝑘-nearest-neighbor set $\mathcal { N } _ { \mathrm { m o t o r } } ( i )$ . We then take the intersection of the two sets, divide its size by $k ,$ and average over all 𝑁 episodes [21]:

$$
m _ { \mathrm { N N } } = \frac { 1 } { N k } \sum _ { i = 1 } ^ { N } \left| \mathcal { N } _ { \mathrm { s e n s o r y } } ( i ) \cap \mathcal { N } _ { \mathrm { m o t o r } } ( i ) \right| .\tag{1}
$$

Greater overlap indicates that episodes with similar sensory representations also have similar motor trajectories, as shown in Figure 2.

In practice, our reported $S ^ { 2 }$ uses a centered, normalized variant of CKNNA with $k = 1 0$ neighbors unless stated otherwise; details are given in Appendix A.3.

## 3. Sensorimotor Alignment Correlates with Robot Task Success Rate

The Platonic Representation Hypothesis (PRH) [21] argues that as models of different modalities become more capable, their representations converge toward a shared statistical model of reality.

4

<!-- page 5 of 28 -->

Real-robot policies(d)

While the evidence for PRH comes from observational data such as images and text, robotics adds the motion the robot executes, which an embodied model must produce rather than observe. We test the prediction that models whose image–text representations align more closely with end-effector trajectories should achieve higher downstream task success.

We quantify this alignment using Sensorimotor Alignment $( S ^ { 2 } )$ , which compares episode neighborhoods in sensory and motor space, as defined in Section 2. A positive correlation between $S ^ { 2 }$ and task success would provide empirical support for this PRH-inspired view of embodiment. It would also support using $S ^ { 2 }$ as a low-cost offline proxy for deciding which models to evaluate in closed loop first.

Section 3.1 establishes this positive correlation across 3 model families and real-robot policies. Section 3.2 shows that $S ^ { 2 }$ correlates with task success more closely than VQA and video-quality metrics. Section 3.3 shows that the correlation holds across computation settings and drops in a negative control with a mismatched embodiment.

## 3.1. $S ^ { 2 }$ Correlates with Task Success Across Model Families

In Figure 3, we test 3 model families and real-robot policies, comparing each model’s $S ^ { 2 }$ on a reference dataset (𝑥-axis) with its downstream manipulation success rate (𝑦-axis).

![Chart block](doc:0a14520/tier:basic/page:5/block:7)

![Chart block](doc:0a14520/tier:basic/page:5/block:8)

![Chart block](doc:0a14520/tier:basic/page:5/block:9)

![Chart block](doc:0a14520/tier:basic/page:5/block:10)

Figure 3 | Sensorimotor Alignment $( S ^ { 2 } )$ correlates with task success across model families and in simulated and real-world evaluations. Each panel plots offline $S ^ { 2 }$ (𝑥-axis) against downstream task success rate (𝑦-axis) for (a) fine-tuned vision-language-action models (VLAs), (b) raw vision-language models (VLMs), (c) world action models (WAMs), and (d) real-robot policies. For raw VLMs, task success is measured after VLA fine-tuning. These correlations (Pearson’s 𝑟; unadjusted one-tailed 𝑝-values) support $S ^ { 2 }$ as an offline proxy across model families.

**Vision-language-action models (VLAs).** We test 12 VLAs against their SimplerEnv [30] success rates. In Figure 3a, the Pearson correlation $r   =   + 0 . 7 1 7$ is strong, with a model-bootstrap 95% confidence interval of [+0.436, +0.912]. Thus, models with higher sensorimotor alignment tend to achieve higher task success rates. The correlation holds across reference datasets, model families, scales, and VLA training setups, as shown in Figure 6 and Appendix B.2.

**Vision-language models (VLMs).** We ask whether $S ^ { 2 }$ computed on a raw VLM, before any robot fine-tuning, correlates with the task success of the VLA that is later fine-tuned from that backbone. We compute $S ^ { 2 }$ for 8 raw VLM backbones from VLM4VLA [59] and compare it with the SimplerEnv [30] success rates of the resulting VLAs. We find a strong correlation, $r = + 0 . 6 8 1$ with one-tailed $p = 0 . 0 3 1$ , as shown in Figure 3b. $S ^ { 2 }$ can therefore identify promising VLM backbones before costly VLA fine-tuning. Appendix B.3 lists the backbones and $S ^ { 2 }$ settings.

**World action models (WAMs).** We also test WAMs, which build on video generators rather than VLMs. We compute $S ^ { 2 }$ for 6 WAMs on the bimanual ALOHA dataset [61] and compare it with their success rates on RoboTwin 2.0 [33]. For WAMs, the sensory representation is the video-generator latent used for action decoding, and all 6 models share the same ALOHA trajectories and $S ^ { 2 }$ settings. $S ^ { 2 }$ again correlates strongly, $r = + 0 . 7 1 8$ with one-tailed $p = 0 . 0 5 4 ,$ , as shown in Figure 3c. Appendix B.4 lists the 6 WAMs and shared evaluation protocol.

5

<!-- page 6 of 28 -->

**Real-robot policies.** We test whether the correlation observed in simulation extends to real robots. Because real-world success rates depend on many factors, such as the task set, robot, cameras, and protocol, we take them all from one shared evaluation: the physical-robot results that MolmoB0T [11] reports for $5$ policies on 40 tasks. We then compute $\mathrm { ^ { \prime } S ^ { 2 } }$ for their public checkpoints on the same DROID reference dataset. We find a strong correlation with real-robot success, $r = + 0 . 8 3 1$ with one-tailed $p = 0 . 0 4 1$ , as shown in Figure 3d. Appendix B.5 describes the shared evaluation conditions and checkpoint scoring.

We further validate that $S ^ { 2 }$ correlates with downstream task success for frozen vision backbones under low-data behavior cloning and visual-reward learning, as shown in Appendix B.6.

## 3.2. $S ^ { 2 }$ Correlates with Task Success More Closely Than Alternative Metrics

Prior work asks whether offline scores correlate with downstream task performance. VLM4VLA [59] examines the correlation between VQA benchmark scores and the performance of the resulting VLA policies, finding weak or even negative correlations. For video models, Tian et al. [47] find that the quality of predicted video agrees inconsistently with downstream performance across tasks. A common limitation is that these metrics measure language understanding or pixel-level video quality, neither of which depends on the motion the robot must execute. $\widetilde { S } ^ { 2 }$ instead compares the nearest-neighbor sets of sensory embeddings and motor trajectories. As we show below, $S ^ { 2 }$ correlates more strongly with task success rate than any VQA or video-quality metric we evaluate.

$S ^ { 2 }$ **correlates with VLA success rate where VQA scores do not.** We compare $S ^ { 2 }$ with aggregate VQA scores as offline proxies, using the same 8 raw VLM backbones and evaluation setting as in Section 3.1. We find that VQA scores show virtually no correlation with VLA success rate, at $r   =   - 0 . 1 5 4$ (right panel of Figure 4). $S ^ { 2 } ,$ , by contrast, correlates strongly, at $r = + 0 . 6 8 1$ with one-tailed $p = 0 . 0 3 1$ (left panel of Figure 4).

![Chart block](doc:0a14520/tier:basic/page:6/block:6)

![Chart block](doc:0a14520/tier:basic/page:6/block:7)

Figure 4 | **Sensorimotor Alignment (**S<sup>2</sup>**) correlates with VLA success rate where VQA scores do not.** Across 8 raw vision-language models (VLMs) [59], $\vec { S ^ { 2 } }$ (left) correlates strongly with SimplerEnv task success after fine-tuning into vision-language-action models (VLAs), whereas visual question answering (VQA) scores (right) show virtually no correlation.

$S ^ { 2 }$ **correlates with WAM success rate more closely than video-quality metrics.** We compare $S ^ { 2 }$ with standard video-quality metrics (FID [18], SSIM [52], and PSNR [19]) on 3 of the WAMs, Motus [5], LingBot-VA [28], and Vidar [15], evaluated on RoboTwin. We find that these metrics correlate weakly with task success: SSIM reaches only $r = + 0 . 5 5 8$ and PSNR only $r = + 0 . 4 6 8 ,$ while FID correlates in the wrong direction, at $r = + 0 . 6 8 0 ,$ associating worse video quality with higher success (Figure 5). By contrast, $S ^ { 2 }$ on ALOHA correlates more strongly, at $r = + 0 . 9 9 9$

6

<!-- page 7 of 28 -->

![Chart block](doc:0a14520/tier:basic/page:7/block:1)

Sensorimotor Alignment (S <sup>2</sup> ) →

![Chart block](doc:0a14520/tier:basic/page:7/block:3)

![Chart block](doc:0a14520/tier:basic/page:7/block:4)

![Chart block](doc:0a14520/tier:basic/page:7/block:5)

Figure 5 | Sensorimotor Alignment $( S ^ { 2 } )$ correlates with WAM success rate more closely than video-quality metrics. For 3 world action models (WAMs) evaluated on RoboTwin, video-quality metrics (SSIM, PSNR, FID) show no significant correlations with task success. In contrast, only $S ^ { 2 }$ correctly recovers the success-rate ordering. Only 3 of the 6 WAMs are compared because the others do not generate future video during closed-loop task evaluation.

## 3.3. $S ^ { 2 }$ Is Robust to Evaluation Choices

We next check that the correlation does not depend on a particular choice of settings. We test this in two ways. First, we vary how $S ^ { 2 }$ is computed and find that the positive correlation with task success holds. Second, as a negative control, we deliberately compute $S ^ { 2 }$ on reference data from a mismatched embodiment, where a sound metric should lose this correlation. The correlation drops as expected, validating the design of $S ^ { 2 }$

![Chart block](doc:0a14520/tier:basic/page:7/block:9)

![Chart block](doc:0a14520/tier:basic/page:7/block:10)

Sensorimotor Alignment (S<sup>2</sup>)

![Chart block](doc:0a14520/tier:basic/page:7/block:12)

Figure 6 | The correlation transfers across reference datasets. For 12 vision-language-action models, Sensorimotor Alignment $( S ^ { 2 } )$ on DROID (Franka Panda, left) and BridgeDataV2 (WidowX, right) correlates strongly with the same SimplerEnv-WidowX task success rates.

Figure 7 | Stable correlations across horizons. For 12 vision-language-action models, Sensorimotor Alignment (S<sup>2</sup>) and SimplerEnv-WidowX success rate have Pearson $r > 0 . 5$ at nearly all tested horizons ℎ (future steps) on DROID and BridgeDataV2.

$S ^ { 2 }$ **is stable across computation choices.** In the first check, we vary eight settings: the reference dataset, the trajectory horizon, five implementation choices, and DTW warping. The correlation stays positive under all eight. We discuss the first two below; Appendix C.1 gives per-model $S ^ { 2 }$ by horizon, and Appendices C.2 and C.3 cover the other six.

The correlation transfers across reference datasets. We compute $S ^ { 2 }$ on both DROID (Franka Panda) and BridgeDataV2 [49] (WidowX), robots with different arm kinematics, and both correlate near $r   =   + 0 . 7$ with task success for the 12 VLAs on SimplerEnv (Figure 6).

The correlation is also stable across trajectory horizons. It stays above $r   =   0 . 5$ at all DROID horizons through $h   =   2 2 0$ and at BridgeDataV2 horizons from $h   =   3$ through 40 (Figure 7).

$S ^ { 2 }$ **is sensitive to embodiment mismatch.** In the negative control, we compute $S ^ { 2 }$ for 3 bimanual WAMs on a single-arm reference dataset (DROID) rather than the matched bimanual one (ALOHA). This embodiment mismatch drops the $S ^ { 2 } -$ –success correlation from $r   =   + 0 . 9 9 9$ to $r = + 0 . 2 4 6 ,$ indicating that $S ^ { 2 }$ responds to the difference in arm configuration rather than reporting alignment indiscriminately. Appendix C.4 gives the full comparison.

7

<!-- page 8 of 28 -->

## 4. Representation-Level Explanations with $S ^ { 2 }$

Several phenomena reported for VLAs and WAMs rest only on closed-loop results and still lack a representation-level explanation. Because $S ^ { 2 }$ correlates with task success, as shown in Section 3, we use it to explain each of these phenomena, as summarized below:

• **Modality dominance**: prior work observes that VLAs often struggle with instruction grounding: they act on what the scene visually suggests rather than on what the language instruction asks [13, 14, 60]. In Section 4.1, we show that after VLA fine-tuning, task success correlates with vision-feature $S ^ { 2 }$ but not text-feature $S ^ { 2 } ,$ , indicating that fine-tuned VLAs draw on what they see, not on the instruction, to produce their motion.

**Frozen encoders**: prior work observes that freezing the vision encoder during VLA finetuning degrades downstream task performance [59]. In Section 4.2, we show that freezing also prevents the $S ^ { 2 }$ gain from end-to-end fine-tuning, indicating that VLAs gain performance by bringing their sensory representation closer to the motor representation.

**Iterative denoising**: prior work observes that WAMs can skip iterative video denoising at test time, decoding actions without generating future video, with little loss in task success [56]. In Section 4.3, we show that the $S ^ { 2 }$ of WAM features is already high at the earliest denoising step and changes little afterwards, indicating that the first denoising steps already yield features aligned closely enough with robot trajectories to act on.

• **Mid-depth blocks**: prior work observes that mid-depth blocks of video diffusion transformers matter most for semantic reasoning [51]. In Section 4.4, we show that these blocks also carry the strongest $S ^ { 2 } )$ , indicating that the layers that reason about video semantics are also the layers most aligned with robot motion.

![Chart block](doc:0a14520/tier:basic/page:8/block:7)

![Chart block](doc:0a14520/tier:basic/page:8/block:8)

![Chart block](doc:0a14520/tier:basic/page:8/block:9)

![Chart block](doc:0a14520/tier:basic/page:8/block:10)

![Chart block](doc:0a14520/tier:basic/page:8/block:11)

![Chart block](doc:0a14520/tier:basic/page:8/block:12)

Figure 8 | Fine-tuned VLAs rely mainly on visual rather than language representations. We correlate Sensorimotor Alignment $( S ^ { 2 } )$ from vision, text, and joint vision-text features (left to right) with SimplerEnv task success rate. Top: For fine-tuned vision-language-action models (VLAs), vision and joint features correlate similarly strongly, while text features correlate weakly. Bottom: For raw vision-language models (VLMs), joint-feature $S ^ { 2 }$ correlates with task success after VLA fine-tuning, while neither modality alone reaches significance.

8

<!-- page 9 of 28 -->

## 4.1. What Carries the Correlation: Vision, Text, or Both?

VLAs often struggle with instruction grounding [13, 14, 60]. We use $S ^ { 2 }$ to explain this at the representation level, asking which modality the correlation with task success depends on: vision, text, or both. In Figure 8, we compute $S ^ { 2 }$ separately for each modality. We take the VLM output tokens read by the action head and form vision features from the image tokens, text features from the instruction tokens, and joint vision-text features from all tokens.

**Fine-tuned VLAs rely mainly on visual rather than language representations.** In the 12 fine-tuned $\mathrm { V L A s } , S ^ { 2 }$ from vision features correlates with SimplerEnv success rate at $r   =   + 0 . 6 9 4$ nearly matching the joint vision-text features at $r   =   + 0 . 6 9 9 ,$ , whereas text features show no significant correlation at $r   =   + 0 . 2 3 7$ . The raw VLMs reverse this pattern: before fine-tuning, text features correlate at $r   =   + 0 . 5 0 1 ,$ , more strongly than vision features at $r   =   + 0 . 4 2 5 ,$ , and joint vision-text features correlate most strongly at $r   =   + 0 . 6 8 1$

We infer that, after fine-tuning, VLAs produce their motion mainly from visual rather than language representations, which explains their weak instruction grounding. Appendix D.1 reports per-model $S ^ { 2 }$ at all feature positions.

## 4.2. Freezing Vision Encoders Limits $S ^ { 2 }$ Gain

![Chart block](doc:0a14520/tier:basic/page:9/block:6)

Figure 9 | Freezing the vision encoder reduces both Sensorimotor Alignment (S<sup>2</sup>) and task success rate. For Qwen2.5-VL-3B vision-language-action models fine-tuned on BridgeDataV2, we compare end-to-end training with a frozen vision encoder. Freezing lowers $S ^ { 2 }$ from joint vision-text features (top) and SimplerEnv success rate (bottom), linking lower success to weaker alignment. Trajectory horizon $h = 7 .$

VLM4VLA [59] reports that VLAs perform substantially worse when their vision encoder stays frozen during fine-tuning. We use $\bar { \mathsf { S } } ^ { 2 }$ to explain this drop at the representation level, testing whether freezing also prevents the sensorimotor-alignment gain from end-to-end fine-tuning.

In Figure 9, we compare 2 VLAs that follow the same VLM4VLA [59] recipe and differ only in whether the vision encoder is trained or kept frozen. Freezing the encoder reduces $S ^ { 2 }$ by 41% and SimplerEnv success rate by 40%. These results suggest that end-to-end fine-tuning improves performance by aligning the sensory representation more closely with the motor representation. Appendix D.2 gives the full horizon sweep.

## 4.3. Features at Early Denoising Steps Already Carry Strong Sensorimotor Alignment

Most WAMs generate future video through iterative denoising and decode actions from the resulting latents. Fast-WAM [56] shows that WAMs can skip this video generation and decode actions directly, with little loss in task success. We examine the representation-level reason: whether features at the earliest denoising steps already align with robot trajectories.

We measure $S ^ { 2 }$ for 3 WAMs [5, 15, 28] on RoboTwin [33] at 5 positions in each model’s native denoising schedule, from 0% to 100%, not at matched noise levels (Figure 10a). $S ^ { 2 }$ is already high at the earliest step and remains nearly flat, varying by about 6% at most from its initial value across the 3 models. We infer that early-step features are already sufficiently aligned with robot trajectories to generate actions. Because $S ^ { 2 }$ correlates with WAM success rate (Section 3.1), this result explains why skipping iterative video denoising barely lowers task success.

9

<!-- page 10 of 28 -->

![Chart block](doc:0a14520/tier:basic/page:10/block:1)

(a) $S ^ { 2 }$ is nearly flat across the schedule.

![Chart block](doc:0a14520/tier:basic/page:10/block:3)

(b) Block 14 has the strongest $S ^ { 2 }$ in all 3 WAMs.

Figure 10 | Sensorimotor Alignment (S<sup>2</sup>) is already high at early denoising steps and peaks at mid-depth blocks. For 3 world action models (WAMs), (a) $S ^ { 2 }$ remains nearly flat across denoising, suggesting early-step features already align with robot trajectories; (b) schedule-averaged $S ^ { 2 }$ peaks at Block 14, in the middle of the 30-block backbone.

## 4.4. Mid-Depth Blocks Carry the Strongest Sensorimotor Alignment

Chain-of-Steps [51] shows that mid-depth layers in Video Diffusion Transformers are especially important for semantic reasoning over objects and interactions. We ask whether the layers with the highest $S ^ { 2 }$ also lie in this mid-depth range.

In Figure 10b, we compute $S ^ { 2 }$ at different depths of 3 WAMs [5, 15, 28] on RoboTwin. All 3 share a 30-block video backbone; we extract features from Blocks 0, 7, 14, 21, and 29. Block 14 yields the highest $S ^ { 2 }$ in every model. This peak falls in the mid-depth range that Chain-of-Steps [51] identifies as semantically important, suggesting that the layers handling video semantic reasoning align most closely with robot motion. Details are in Appendix D.3.

## 5. Related Work

**Representation learning in robotics.** Many robotics studies pretrain visual representations to improve task performance [32, 34, 40]. However, there remains limited understanding of which representational properties lead to downstream task success. A few studies ask what makes a pretrained vision encoder useful for robot learning. Hu et al. [20] evaluate 14 frozen vision encoders and find that the correlation between an encoder’s offline score and downstream task success rate is not stable across policy-learning methods. Burns et al. [8] find that emergent segmentation predicts manipulation success under visual distribution shifts, but only for Vision Transformers. Theia [43] hypothesizes that higher feature-norm entropy improves robot learning, but reports the supporting correlations only within separate groups of encoders. These studies examine only vision encoders rather than a broader range of embodied models, and none of them accounts for the trajectories the robot executes when measuring the representation. By contrast, $S ^ { 2 }$ compares 𝑘-nearest-neighbor sets in sensory and motor spaces. The same $S ^ { 2 }$ measure correlates with task success across VLAs, raw VLMs, WAMs, and the encoders that Hu et al. [20] evaluate.

**Platonic Representation Hypothesis.** Recent work has applied the Platonic Representation Hypothesis (PRH) [21] to several new domains. In image generation, REPA [54] aligns diffusiontransformer features with self-supervised vision encoders, improving training speed and quality. CREPA [22] extends this recipe to video diffusion. In multimodal understanding, VIRAL [53] re-aligns multimodal large language models (LLMs) with their vision encoders, recovering visual detail lost in multimodal training. For cross-modal alignment, Schnaus et al. [42] match independently trained vision and language encoders through their shared geometry, without paired data. PRH has not yet been brought into robotics; we extend it to robotics by treating

10

<!-- page 11 of 28 -->

robot trajectories as another projection of the world, and find that more capable embodied models align their sensory representations more closely with it.

## 6. Conclusion

We introduce Sensorimotor Alignment $( S ^ { 2 } )$ , a training-free offline metric of alignment between sensory representations and end-effector trajectories. Across VLAs, raw VLMs, and WAMs in simulation, and across real-robot policies, $S ^ { 2 }$ correlates positively with downstream task success, supporting a PRH-inspired view of embodiment. $S ^ { 2 }$ correlates more strongly with task success than VQA scores and video-quality metrics, and explains, at the representation level, VLA and WAM phenomena previously observed only through closed-loop results: modality dominance, the performance drop from freezing the vision encoder, the small task-success cost of skipping iterative denoising, and the importance of mid-depth blocks for semantic reasoning. We use $S ^ { \tilde { 2 } }$ only offline as a low-cost model-selection and diagnostic tool before closed-loop evaluation, but this correlation motivates optimizing embodied models directly for sensorimotor alignment.

**Limitations.** Our study has three main limitations. (1) The positive relationship between $S ^ { 2 }$ and task success is correlational across the tested models; we have not yet demonstrated that sensorimotor alignment can be used directly during training to improve task success rates. (2) Models with similar $S ^ { 2 }$ values can still achieve different success rates, so $S ^ { 2 }$ is best used as a low-cost offline proxy for identifying promising models and VLM backbones before costly fine-tuning or closed-loop testing, which remains the final step of model selection. (3) Our evaluation covers limited model families, benchmarks, and training recipes, few models per comparison, and a single real-robot setting; broader generalization remains to be tested.

## References

[1] Pranav Atreya, Karl Pertsch, Tony Lee, Moo Jin Kim, Arhan Jain, Artur Kuramshin, Clemens Eppner, Cyrus Neary, Edward Hu, Fabio Ramos, et al. RoboArena: Distributed real-world evaluation of generalist robot policies. In CoRL, 2025.

[2] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-VL technical report. arXiv:2511.21631, 2025.

[3] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-VL technical report. arXiv:2502.13923, 2025.

[4] Lucas Beyer, Andreas Steiner, André Susano Pinto, Alexander Kolesnikov, Xiao Wang, Daniel Salz, Maxim Neumann, Ibrahim Alabdulmohsin, Michael Tschannen, Emanuele Bugliarello, et al. PaliGemma: A versatile 3B VLM for transfer. arXiv:2407.07726, 2024.

[5] Hongzhe Bi, Hengkai Tan, Shenghao Xie, Zeyuan Wang, Shuhe Huang, Haitian Liu, Ruowen Zhao, Yao Feng, Chendong Xiang, Yinze Rong, et al. Motus: A unified latent action world model. In CVPR, 2026.

[6] Johan Bjorck, Fernando Castañeda, Nikita Cherniadev, Xingye Da, Runyu Ding, Linxi Fan, Yu Fang, Dieter Fox, Fengyuan Hu, Spencer Huang, et al. GR00T N1: An open foundation model for generalist humanoid robots. arXiv:2503.14734, 2025.

11

<!-- page 12 of 28 -->

[7] Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, et al. 𝜋<sub>0</sub>: A vision-languageaction flow model for general robot control. In RSS, 2025.

[8] Kaylee Burns, Zach Witzel, Jubayer Ibn Hamid, Tianhe Yu, Chelsea Finn, and Karol Hausman. What makes pre-trained visual representations successful for robust manipulation? In CoRL, 2024.

[9] Yiting Chen, Kenneth Kimble, Edward H Adelson, Tamim Asfour, Podshara Chanrungmaneekul, Sachin Chitta, Yash Chitambar, Ziyang Chen, Ken Goldberg, Danica Kragic, et al. ManipulationNet: An infrastructure for benchmarking real-world robot manipulation with physical skill challenges and embodied multimodal reasoning. arXiv:2603.04363, 2026.

[10] Yixiang Chen, Peiyan Li, Yuan Xu, Qisen Ma, et al. FlowWAM: Optical flow as a unified action representation for world action models. arXiv:2607.13017, 2026.

[11] Abhay Deshpande, Maya Guru, Rose Hendrix, Snehal Jauhri, et al. MolmoB0T: Large-scale simulation enables zero-shot manipulation. In CoRL, 2026.

[12] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, et al. An image is worth 16x16 words: Transformers for image recognition at scale. In ICLR, 2021.

[13] Yu Fang, Yuchun Feng, Dong Jing, Jiaqi Liu, Yue Yang, Zhenyu Wei, Daniel Szafir, and Mingyu Ding. When vision overrides language: Evaluating and mitigating counterfactual failures in VLAs. In CoRL, 2026.

[14] Senyu Fei, Siyin Wang, Junhao Shi, Zihao Dai, Jikun Cai, Pengfang Qian, Li Ji, Xinzhe He, Shiduo Zhang, Zhaoye Fei, et al. LIBERO-Plus: A progressive robustness benchmark for visual-language-action models. In CVPR, 2026.

[15] Yao Feng, Hengkai Tan, Xinyi Mao, Chendong Xiang, Guodong Liu, Shuhe Huang, Hang Su, and Jun Zhu. Vidar: Embodied video diffusion model for generalist manipulation. arXiv:2507.12898, 2025.

[16] Abhishek Gupta, Vikash Kumar, Corey Lynch, Sergey Levine, and Karol Hausman. Relay policy learning: Solving long-horizon tasks via imitation and reinforcement learning. In CoRL, 2019.

[17] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In CVPR, 2016.

[18] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. GANs trained by a two time-scale update rule converge to a local Nash equilibrium. In NeurIPS, 2017.

[19] Alain Horé and Djemel Ziou. Image quality metrics: PSNR vs. SSIM. In ICPR, 2010.

[20] Yingdong Hu, Renhao Wang, Li Erran Li, and Yang Gao. For pre-trained vision models in motor control, not all policy learning methods are created equal. In ICML, 2023.

[21] Minyoung Huh, Brian Cheung, Tongzhou Wang, and Phillip Isola. The Platonic representation hypothesis. In ICML, 2024.

[22] Sungwon Hwang, Hyojin Jang, Kinam Kim, Minho Park, and Jaegul Choo. Cross-frame representation alignment for fine-tuning video diffusion models. arXiv:2506.09229, 2025.

12

<!-- page 13 of 28 -->

[23] Sadeep Jayasumana, Srikumar Ramalingam, Andreas Veit, Daniel Glasner, Ayan Chakrabarti, and Sanjiv Kumar. Rethinking FID: Towards a better evaluation metric for image generation. In CVPR, 2024.

[24] Nikita Kachaev, Mikhail Kolosov, Daniil Zelezetsky, Alexey K Kovalev, and Aleksandr I Panov. Don’t blind your VLA: Aligning visual representations for OOD generalization. In AAMAS, 2026.

[25] Kairos Team, Fei Wang, Shan You, Qiming Zhang, et al. Kairos: A regret-aware native world-action model stack for physical AI. arXiv:2606.16533, 2026.

[26] Alexander Khazatsky, Karl Pertsch, Suraj Nair, Ashwin Balakrishna, Sudeep Dasari, Siddharth Karamcheti, Soroush Nasiriany, Mohan Kumar Srirama, Lawrence Yunliang Chen, Kirsty Ellis, et al. DROID: A large-scale in-the-wild robot manipulation dataset. In RSS, 2024.

[27] Simon Kornblith, Mohammad Norouzi, Honglak Lee, and Geoffrey Hinton. Similarity of neural network representations revisited. In ICML, 2019.

[28] Lin Li, Qihang Zhang, Yiming Luo, Shuai Yang, Ruilin Wang, Luyao Zhang, Mingrui Yu, Zelin Gao, Nan Xue, Boyu Zhou, et al. Causal world modeling for robot control. In RSS, 2026.

[29] Qixiu Li, Yaobo Liang, Zeyu Wang, Lin Luo, Xi Chen, Mozheng Liao, Fangyun Wei, Yu Deng, Sicheng Xu, Yizhong Zhang, et al. CogACT: A foundational visionlanguage-action model for synergizing cognition and action in robotic manipulation. arXiv:2411.19650, 2024.

[30] Xuanlin Li, Kyle Hsu, Jiayuan Gu, Karl Pertsch, Oier Mees, Homer Rich Walke, Chuyuan Fu, Ishikaa Lunawat, Isabel Sieh, Sean Kirmani, et al. Evaluating real-world robot manipulation policies in simulation. In CoRL, 2024.

[31] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. MMBench: Is your multi-modal model an all-around player? In ECCV, 2024.

[32] Arjun Majumdar, Karmesh Yadav, Sergio Arnaud, Jason Ma, Claire Chen, Sneha Silwal, Aryan Jain, Vincent-Pierre Berges, Tingfan Wu, Jay Vakil, et al. Where are we in the search for an artificial visual cortex for embodied intelligence? In NeurIPS, 2023.

[33] Yao Mu, Tianxing Chen, Zanxin Chen, Shijia Peng, Zhiqian Lan, Zeyu Gao, Zhixuan Liang, Qiaojun Yu, Yude Zou, Mingkun Xu, et al. RoboTwin: Dual-arm robot benchmark with generative digital twins. In CVPR, 2025.

[34] Suraj Nair, Aravind Rajeswaran, Vikash Kumar, Chelsea Finn, and Abhinav Gupta. R3M: A universal visual representation for robot manipulation. In CoRL, 2022.

[35] Jonas Pai, Liam Achenbach, Oliver Sanchez, Stefanos Charalambous, Victoriano Montesinos, Benedek Forrai, Oier Mees, and Elvis Nava. mimic-video: Video-action models for generalizable robot control beyond VLAs. In RSS, 2026.

[36] Zhiliang Peng, Wenhui Wang, Li Dong, Yaru Hao, Shaohan Huang, Shuming Ma, Qixiang Ye, and Furu Wei. KOSMOS-2: Grounding multimodal large language models to the world. In ICLR, 2024.

13

<!-- page 14 of 28 -->

[37] Howard Qian, Yiting Chen, Yunfei Xie, Kejia Ren, Podshara Chanrungmaneekul, Gaotian Wang, Bowen Wen, Chen Wei, and Kaiyu Hang. RoboTok: A scalable data engine for internet demonstration video retrieval and dexterous manipulation learning. arXiv:2609.03199, 2026.

[38] Delin Qu, Haoming Song, Qizhi Chen, Yuanqi Yao, Xinyi Ye, Yan Ding, Zhigang Wang, JiaYuan Gu, Bin Zhao, Dong Wang, et al. SpatialVLA: Exploring spatial representations for visual-language-action models. In RSS, 2025.

[39] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, et al. Learning transferable visual models from natural language supervision. In ICML, 2021.

[40] Ilija Radosavovic, Tete Xiao, Stephen James, Pieter Abbeel, Jitendra Malik, and Trevor Darrell. Real-world robot learning with masked visual pre-training. In CoRL, 2022.

[41] H. Sakoe and S. Chiba. Dynamic programming algorithm optimization for spoken word recognition. IEEE TASSP, 1978.

[42] Dominik Schnaus, Thomas Dagès, Daniel Cremers, Xi Wang, and Phillip Isola. Shared geometry as a Rosetta stone: Cross-modal alignment without paired data. arXiv:2610.09411, 2026.

[43] Jinghuan Shang, Karl Schmeckpeper, Brandon B. May, Maria Vittoria Minniti, Tarik Ke lestemur, David Watkins, and Laura Herlant. Theia: Distilling diverse vision foundation models for robot learning. In CoRL, 2024.

[44] Le Song, Alex Smola, Arthur Gretton, Justin Bedo, and Karsten Borgwardt. Feature selection via dependence maximization. Journal of Machine Learning Research, 13:1393–1434, 2012.

[45] StarVLA Community. StarVLA: A Lego-like codebase for vision-language-action model developing. arXiv:2604.05014, 2026.

[46] Andreas Steiner, André Susano Pinto, Michael Tschannen, Daniel Keysers, Xiao Wang, Yonatan Bitton, Alexey Gritsenko, Matthias Minderer, Anthony Sherbondy, Shangbang Long, et al. PaliGemma 2: A family of versatile VLMs for transfer. arXiv:2412.03555, 2024.

[47] Stephen Tian, Chelsea Finn, and Jiajun Wu. A control-centric benchmark for video prediction. In ICLR, 2023.

[48] Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, et al. SigLIP 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features. arXiv:2502.14786, 2025.

[49] Homer Rich Walke, Kevin Black, Abraham Lee, Moo Jin Kim, Max Du, Chongyi Zheng, Tony Z Zhao, Philippe Hansen-Estruch, Quan Vuong, Andre Wang He, et al. BridgeData V2: A dataset for robot learning at scale. In CoRL, 2023.

[50] Wan Team, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv:2503.20314, 2025.

[51] Ruisi Wang, Zhongang Cai, Fanyi Pu, Junxiang Xu, Wanqi Yin, Maijunxian Wang, Ran Ji, Chenyang Gu, Bo Li, Ziqi Huang, et al. Demystifying video reasoning. In ECCV, 2026.

[52] Zhou Wang, A.C. Bovik, H.R. Sheikh, and E.P. Simoncelli. Image quality assessment: From error visibility to structural similarity. IEEE TIP, 2004.

14

<!-- page 15 of 28 -->

[53] Heeji Yoon, Jaewoo Jung, Junwan Kim, Hyungyu Choi, Heeseong Shin, Sangbeom Lim, Honggyu An, Chaehyun Kim, Jisang Han, Donghyun Kim, et al. Visual representation alignment for multimodal large language models. arXiv:2509.07979, 2025.

[54] Sihyun Yu, Sangkyung Kwak, Huiwon Jang, Jongheon Jeong, Jonathan Huang, Jinwoo Shin, and Saining Xie. Representation alignment for generation: Training diffusion transformers is easier than you think. In ICLR, 2025.

[55] Tianhe Yu, Deirdre Quillen, Zhanpeng He, Ryan Julian, Karol Hausman, Chelsea Finn, and Sergey Levine. Meta-World: A benchmark and evaluation for multi-task and meta reinforcement learning. In CoRL, 2019.

[56] Tianyuan Yuan, Zibin Dong, Yicheng Liu, and Hang Zhao. Fast-WAM: Do world action models need test-time future imagination? In NeurIPS, 2026.

[57] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. MMMU: A massive multi-discipline multimodal understanding and reasoning benchmark for expert AGI. In CVPR, 2024.

[58] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In ICCV, 2023.

[59] Jianke Zhang, Xiaoyu Chen, Yanjiang Guo, Yucheng Hu, and Jianyu Chen. VLM4VLA: Revisiting vision-language-models in vision-language-action models. In ICLR, 2026.

[60] Ninghao Zhang, Bin Zhu, Shijie Zhou, and Jingjing Chen. Restoring linguistic grounding in VLA models via train-free attention recalibration. In ECCV, 2026.

[61] Tony Z Zhao, Vikash Kumar, Sergey Levine, and Chelsea Finn. Learning fine-grained bimanual manipulation with low-cost hardware. In RSS, 2023.

[62] Yuke Zhu, Josiah Wong, Ajay Mandlekar, Roberto Martín-Martín, et al. robosuite: A modular simulation framework and benchmark for robot learning. arXiv:2009.12293, 2020.

15

<!-- page 16 of 28 -->

## Appendix

## Contents

- A. Sensorimotor Alignment Metric . . 17
- A.1. Reference Datasets and Sampling . 17
- A.2. Trajectory Distance. .17
- A.3. Alignment Estimator. . 18
- B. Experiments on $S ^ { 2 }$ and Task Success . 19
- B.1. Vision-Language-Action Models . 20
- B.2. Confound Controls for the VLA Correlation . 20
- B.3. Raw Vision-Language Models . 20
- B.4. World Action Models . . 21
- B.5. Real-Robot Policies . . 21
- B.6. Frozen Vision Encoders Under Downstream Policy Learning . 21
- C. Robustness of $S ^ { 2 }$ . 22
- C.1. Trajectory Horizon . . 22
- C.2. Implementation Choices . 24
- C.3. DTW Alignment . . . 24
- C.4. Sensitivity to Embodiment Mismatch . 25
- D. Details of the Representation-Level Explanations 25
- D.1. S<sup>2</sup> Across Feature Positions . . 26
- D.2. Frozen Versus Unfrozen Vision Encoder . . 27
- D.3. WAM Features Along the Denoising Schedule. . 28

16

<!-- page 17 of 28 -->

## A. Sensorimotor Alignment Metric

This section gives the full computation of $S ^ { 2 }$ summarized in Section 2: the reference datasets and how we sample from them, the trajectory distance that defines motor neighbors, and the estimator that scores alignment between sensory and motor neighbor sets.

## A.1. Reference Datasets and Sampling

We use three reference datasets: the single-arm DROID and BridgeDataV2 datasets for ${ \mathrm { V L A s } } ,$ raw VLMs, frozen vision encoders, and real-robot policies, and the bimanual ALOHA dataset for WAMs. DROID and BridgeDataV2 contain teleoperated robot demonstrations with RGB images, proprioceptive states, and language task annotations. Each state is a 7D vector $\mathbf { s } =$ $( x , y , z , \phi , \theta , \psi , g )$ , where $( x , y , z )$ is the end-effector position in meters, $( \phi , \theta , \psi )$ are the roll, pitch, and yaw angles in radians, and 𝑔 is the recorded gripper position. This shared format supports the same DTW and $S ^ { 2 }$ computation on both single-arm datasets.

**BridgeDataV2.** BridgeDataV2 [49] contains 53,192 WidowX episodes recorded at 5 Hz. We use the recorded 7D state and retain 38,307 episodes with non-empty task labels.

**DROID.** DROID [26] contains 95,658 episodes on a Franka Panda arm recorded at 15 Hz, spanning 86 task categories across 564 scenes in offices, kitchens, and homes. We form the 7D state by concatenating the 6D Cartesian end-effector pose with the gripper position, which is 0 when open and 1 when closed. We retain episodes with a language annotation. As the observation image, we use the exterior\_image\_1\_left camera at 180×320, the fixed exterior view closest to the third-person camera of BridgeDataV2.

**ALOHA.** For WAMs evaluated on bimanual RoboTwin, we sample $N   =   8 3 9$ trajectory windows from the bimanual ALOHA dataset [61]. Each trajectory row is the recorded 14- dimensional joint-position action vector, with six joints and a gripper coordinate per arm.

**Anchor sampling across horizons.** We compute $S ^ { 2 }$ at several trajectory horizons ℎ. To keep these evaluations comparable, we draw one sample set at a maximum horizon $H _ { \mathrm { m a x } }$ and reuse it at every shorter horizon. The sampling rule draws an integer anchor index $t _ { i }$ for episode 𝑖 of length $L _ { i }$ uniformly from $\{ 0 , \cdots , L _ { i } - 1 - H _ { \operatorname* { m a x } } \}$ . At horizon $1 \leq h \leq H _ { \operatorname* { m a x } } ,$ row 𝑢 of $\tau _ { i }$ is the recorded state at frame $t _ { i } + u ,$ for $u = 1 , \cdots , h$ . The image at frame $t _ { i }$ and the task instruction form the fixed sensory input $\mathbf { x } _ { i } ,$ whose embedding $\mathbf { z } _ { i }$ is computed once. The horizon sweep therefore changes only the trajectory length, not the episodes, anchors, or sensory inputs. On BridgeDataV2, we set $H _ { \mathrm { m a x } }   =   4 0$ . From a pool of $^ { 7 , 7 6 2 }$ episodes with $L _ { i } > 4 0 ,$ we retain non-empty task labels and at most three episodes per distinct task instruction, leaving $N = 3 { , } 2 4 5$ . We sweep $h \in \{ 1 , 3 , 7 , 1 5 , 2 5 , 4 0 \}$ . On DROID, we set $H _ { \mathrm { m a x } } = 2 2 0$ , which covers about the same fraction of the dataset-wide mean episode length. The sampling rule also requires $L _ { i } \geq 2 H _ { \operatorname* { m a x } } { + } 1 { = } 4 4 1$ so that anchors are not concentrated at the start of short episodes. For the $N   =   6 { , } 0 0 0$ retained episodes, we sweep ℎ ∈ {1, 3, 7, 15, 25, 40, 75, 110, 150, 220}.

## A.2. Trajectory Distance

For BridgeDataV2 and DROID, we compare end-effector trajectories in each robot’s recorded base frame. We do not transform trajectories into a shared world or camera frame. Appendix C.2 compares this choice with positions expressed relative to the start of each trajectory.

**Local cost.** For the single-arm trajectories, let $\mathbf { p } _ { i , u } = ( x _ { i , u } , y _ { i , u } , z _ { i , u } ) ^ { \top }$ and let $\mathbf { q } _ { i , u } \in \mathbb { R } ^ { 4 }$ be the unit quaternion obtained from the recorded roll, pitch, and yaw using extrinsic rotations about

17

<!-- page 18 of 28 -->

the $x , y ,$ then 𝑧 axes. Quaternions q and −q represent the same rotation in $S \mathrm { O } ( 3 )$ , the group of three-dimensional rotations. For steps 𝑢 and 𝑣 of episodes 𝑖 and 𝑗, the local cost is

$$
\begin{aligned}d_{\mathrm{pos}} &= \| \mathbf{p}_{i,u} - \mathbf{p}_{j,\nu} \|_2, \qquad d_{\mathrm{grip}} = |g_{i,u} - g_{j,\nu}|, \\d_{\mathrm{rot}} &= 2 \arccos \left( \min \{ 1, |\mathbf{q}_{i,u}^\top \mathbf{q}_{j,\nu}| \} \right), \\c_{u\nu} &= w_{\mathrm{pos}} d_{\mathrm{pos}} + w_{\mathrm{rot}} d_{\mathrm{rot}} + w_{\mathrm{grip}} d_{\mathrm{grip}}.\end{aligned}
$$

The pair $( i , j )$ is suppressed in $c _ { u v }$ and the component distances. All weights default to one. Components use meters, radians, and recorded gripper values without further scaling.

**Dynamic time warping.** We use the symmetric2 DTW recurrence with no warping window. For a fixed trajectory pair, let $D _ { u v }$ be the accumulated cost through steps $u , v ,$ with

$$
D _ { 1 1 } = c _ { 1 1 } , \qquad D _ { u 1 } = \sum _ { a = 1 } ^ { u } c _ { a 1 } , \qquad D _ { 1 \nu } = \sum _ { b = 1 } ^ { v } c _ { 1 b } .
$$

For $u , \nu \geq 2 ,$

$$
D _ { u \nu } = \operatorname* { m i n } \{ D _ { u - 1 , \nu } + c _ { u \nu } , \; D _ { u , \nu - 1 } + c _ { u \nu } , \; D _ { u - 1 , \nu - 1 } + 2 c _ { u \nu } \} .
$$

For two ℎ-step trajectories, the distance is $D _ { h h }$ without path-length normalization; $h = 1$ gives the single-state cost $c _ { 1 1 }$ . ALOHA trajectories (Appendix A.1) use the same DTW recurrence with Euclidean distance between 14-dimensional joint-position vectors as the local cost.

## A.3. Alignment Estimator

We follow the progression from HSIC to CKA and centered kernel nearest-neighbor alignment (CKNNA) in Appendix A of Huh et al. [21], specializing their construction to a sensory kernel and a motor neighbor matrix. CKA scores alignment across all sample pairs; CKNNA restricts the comparison to mutual nearest neighbors, which measures alignment locally.

**Similarity matrices.** For the 𝑁 paired samples in Section 2, normalize sensory features as

$$
\widehat { \mathbf { z } } _ { i } = \frac { \mathbf { z } _ { i } } { \operatorname* { m a x } ( \| \mathbf { z } _ { i } \| _ { 2 } , \varepsilon _ { z } ) } , \qquad \varepsilon _ { z } = 1 0 ^ { - 1 2 } .
$$

Let $\mathbf { K } , \mathbf { L } \in \mathbb { R } ^ { N \times N }$ denote the sensory and motor matrices used in the score, with entries

$$
K _ { i j } = \widehat { \mathbf { z } } _ { i } ^ { \top } \widehat { \mathbf { z } } _ { j } , \qquad L _ { i j } = \mathbb { 1 } \{ j \in \mathcal { N } _ { \mathrm { m o t o r } } ( i ) \} .
$$

Here $K _ { i j } = \cos ( \mathbf { z } _ { i } , \mathbf { z } _ { j } )$ , with zero features mapped to zero, so K is the inner-product (linear) kernel on ℓ<sub>2</sub>-normalized features used by Huh et al. [21]. The motor set $\mathcal { N } _ { \mathrm { m o t o r } } ( i )$ contains the 𝑘 other samples with smallest DTW distance, as defined in Appendix A.2. Thus DTW selects motor neighbors, and L assigns them unit weight; L is a directed similarity matrix, not necessarily a positive-semidefinite kernel. For $1 \leq k < N ,$ we write $\mathcal { N } _ { \mathbf { K } } ( i )$ and $\mathcal { N } _ { \mathrm { L } } ( i )$ for the sets of 𝑘 other sample indices with the largest entries in row 𝑖 of K and $\mathbf { L } ;$ these are the sensory set $\mathcal { N } _ { \mathrm { s e n s o r y } } ( i )$ and motor set $\mathcal { N } _ { \mathrm { m o t o r } } ( i )$ from Section 2.

**HSIC and centered kernel alignment.** For two kernel matrices A, $, \mathbf { B } \; \in \; \mathbb { R } ^ { N \times N }$ , let $\mathbf { H } =$ $\begin{array} { r } { \mathbf { I } _ { N } - \frac { 1 } { N } \mathbf { 1 } \mathbf { 1 } ^ { \top } } \end{array}$ be the centering matrix, where $\mathbf { I } _ { N }$ is the identity and $\mathbf { 1 } \in \mathbb { R } ^ { N }$ is the all-ones vector. Define $\overline { { \mathbf { A } } } = \mathbf { H } \mathbf { A } \mathbf { H }$ and $\mathbf { \overline { { B } } } = \mathbf { H } \mathbf { B } \mathbf { H } .$ The biased empirical Hilbert–Schmidt independence criterion (HSIC) and its normalized counterpart, centered kernel alignment (CKA) [27], are

$$
\begin{aligned}\mathsf{HSIC_b}(\mathbf{A}, \mathbf{B}) &= \frac{\mathsf{tr}(\overline{\mathbf{A}}   \overline{\mathbf{B}})}{(N - 1)^2}, \\\mathsf{CKA}(\mathbf{A}, \mathbf{B}) &= \frac{\mathsf{HSIC_b}(\mathbf{A}, \mathbf{B})}{\sqrt{\mathsf{HSIC_b}(\mathbf{A}, \mathbf{A})   \mathsf{HSIC_b}(\mathbf{B}, \mathbf{B})}}.\end{aligned}
$$

18

<!-- page 19 of 28 -->

Here tr denotes the trace and ⊤ denotes transpose; the subscript b identifies the biased empirical estimator. Expanding the trace writes HSIC as a sum over all ordered sample pairs,

$$
\operatorname { t r } ( { \overline { { \mathbf { A } } } }   { \overline { { \mathbf { B } } } } ) = \sum _ { i } \sum _ { j } { \overline { { A } } } _ { i j }   { \overline { { B } } } _ { j i } ,
$$

so CKA measures alignment globally, across every pair of samples, and is defined when both self terms are positive [21, 27]. Restricting this sum to mutual nearest neighbors, as below, biases the measure toward local structure.

For nearest-neighbor alignment, we use the diagonal-corrected estimator of Song et al. [44], as in the released CKNNA implementation of Huh et al. $[ 2 1 ] . ^ { 1 }$ Let $\widetilde { \mathbf { A } } , \widetilde { \mathbf { B } }$ be copies of A, B with zero diagonals. For $N > 3 ,$ define

$$
\widehat { \mathsf { H S I C } } _ { \mathfrak { u } } ( \mathbf { A } , \mathbf { B } ) = \frac { 1 } { N ( N - 3 ) } \left[ \mathsf { t r } ( \widetilde { \mathbf { A } } \widetilde { \mathbf { B } } ) + \frac { ( \mathbf { 1 } ^ { \top } \widetilde { \mathbf { A } } \mathbf { 1 } ) ( \mathbf { 1 } ^ { \top } \widetilde { \mathbf { B } } \mathbf { 1 } ) } { ( N - 1 ) ( N - 2 ) } - \frac { 2   \mathbf { 1 } ^ { \top } \widetilde { \mathbf { A } } \widetilde { \mathbf { B } } \mathbf { 1 } } { N - 2 } \right] .
$$

The subscript u denotes this unbiased estimator, which corrects the finite-sample bias of HSIC under i.i.d. sampling. We apply the same algebra to the directed neighbor matrices: the trace term is $\begin{array} { r } { \sum _ { i , j } \widetilde { A } _ { i j } \widetilde { B } _ { j i } , } \end{array}$ and we do not symmetrize the matrices.

**Centered kernel nearest-neighbor alignment.** For A, $\mathbf { B } \in \{ \mathbf { K } , \mathbf { L } \}$ , using the neighbor sets $\mathcal { N } _ { \mathbf { K } }$ and $\mathrm { { \cal N } L }$ defined above, we define the shared-neighbor mask for each matrix pair by

$$
\alpha _ { i j } ( \mathbf { A } , \mathbf { B } ) = \mathbb { I } \{ j \in \mathcal { N } _ { \mathbf { A } } ( i ) \cap \mathcal { N } _ { \mathbf { B } } ( i ) \} ,
$$

where $\mathbb { 1 } \{ \cdot \}$ is one when its condition holds and zero otherwise. Self-neighbors are excluded, and motor-neighbor indices are reused across models. The mask is directed: $\alpha _ { i j }$ need not equal $\alpha _ { j i }$ . As in Huh et al. [21], “mutual” means membership in both neighbor sets.

Using ⊙ for entrywise multiplication, define

$$
\mathsf { A l i g n } _ { \mathsf { k n n } } ( \mathbf { A } , \mathbf { B } ) = \widehat { \mathsf { H S I C } } _ { \mathsf { u } } ( \pmb { \alpha } ( \mathbf { A } , \mathbf { B } ) \odot \mathbf { A } , \pmb { \alpha } ( \mathbf { A } , \mathbf { B } ) \odot \mathbf { B } )   .
$$

This follows the released CKNNA code’s masking-before-HSIC computation. Separate masks cover the cross term (K, L) and the self terms $( \mathbf { K } , \mathbf { K } )$ and $( \mathbf { L } , \mathbf { L } )$ . Because L is binary, $\pmb { \alpha } ( \mathbf { K } , \mathbf { L } ) \odot \mathbf { L } =$ $\pmb { \alpha } ( \mathbf { K } , \mathbf { L } )$ : shared neighbors retain sensory similarities and receive unit motor weights.

The reported score follows the CKNNA normalization:

$$
S ^ { 2 } = \frac { \mathrm { A l i g n } _ { \mathrm { k n n } } ( \mathbf { K } , \mathbf { L } ) } { \sqrt { \operatorname* { m a x } \{ \mathrm { A l i g n } _ { \mathrm { k n n } } ( \mathbf { K } , \mathbf { K } ) \mathrm { A l i g n } _ { \mathrm { k n n } } ( \mathbf { L } , \mathbf { L } ) , \varepsilon _ { A } \} } } , \qquad \varepsilon _ { A } = 1 0 ^ { - 1 0 } .\tag{2}
$$

Following the relaxation of Huh et al. [21], treating all shared neighbors as equidistant replaces the sensory weights by one, and the resulting count of shared neighbors recovers the overlap of Equation 1: $\begin{array} { r } { \widehat { \sum _ { i , j } \alpha _ { i j } ( { \bf K } , { \bf L } ) }   =   N k   m _ { \mathrm { N N } } , } \end{array}$ so 𝑚 $\begin{array} { r } { { \bf \Pi } _ { \mathrm { N N } }   =   ( N k ) ^ { - 1 } \sum _ { i , j } \alpha _ { i j } ( \dot { \bf K } , { \bf L } )   \in   [ 0 , 1 ] } \end{array}$ . The directed, centered score $S ^ { 2 }$ keeps the sensory weights and is not guaranteed to lie in this interval.

## B. Experiments on $S ^ { 2 }$ and Task Success

This section details the experiments in Section 3.1 by model family, plus VLA confound controls and a test of frozen vision encoders under downstream policy learning.

<small><span class="docvortex-page-footnote" data-block-type="page_footnote" style="color:#6b7280"><sup>1</sup>[https://github.com/minyoungg/platonic-rep/blob/main/metrics.py](https://github.com/minyoungg/platonic-rep/blob/main/metrics.py)</span></small>

19

<!-- page 20 of 28 -->

## B.1. Vision-Language-Action Models

The 12 evaluated VLAs include 5 StarVLA [45] variants (Qwen2.5-GR00T-Bridge, Qwen2.5- GR00T-RT-1-Bridge, Qwen2.5-OFT-RT-1-Bridge, Qwen3-GR00T-RT-1-Bridge, and Qwen3-OFT-RT-1-Bridge), CogACT [29] in Small, Base, and Large sizes, GR00T-N1.5-Bridge, GR00T-N1.6- Bridge [6], Pi0-Bridge [7], and SpatialVLA-Bridge [38]. They cover 4 action-head types (flow matching, diffusion, autoregressive, and deterministic MLP) and 3 VLM training regimes, from frozen to end-to-end. We use reported SimplerEnv success rates from the model papers and public evaluation repositories, including [https://github.com/starVLA/starVLA](https://github.com/starVLA/starVLA) and [https://github.com/DelinQu/SimplerEnv-OpenVLA](https://github.com/DelinQu/SimplerEnv-OpenVLA).

## B.2. Confound Controls for the VLA Correlation

The 12 VLAs in Appendix B.1 differ in fine-tuning recipe, model family and scale, and architecture group. Three checks show that none of these factors alone drives the correlation. Each check retains its main-paper experiment’s reference data and feature settings.

**Training recipe.** The raw-VLM experiment in Appendix B.3 holds the recipe fixed. All 8 VLMs become VLAs under the same VLM4VLA [59] recipe: the same action head, objective, BridgeDataV2 data, and hyperparameters. The correlation remains strong at $r   =   + 0 . 6 8 1$ with one-tailed $p   =   0 . 0 3 1$ , so the fine-tuning recipe does not drive the correlation.

**Model family and scale.** CogACT [29] Small, Base, and Large share the same VLM backbone and action-head design and differ only in action-head size. Within this family, the correlation is also positive, at $r   =   + 0 . 8 2 6$ . With only 3 models, this check has little statistical power (one-tailed $p   =   0 . 1 9 1 )$ , but its direction matches the full cohort.

**Architecture groups.** We remove each of the 5 architecture groups in turn and recompute the correlation on the remaining models, with results in Table 1. The correlation stays positive in all 5 cases, from $r   =   + 0 . 5 2 6$ to $r   =   + 0 . 7 8 5 ,$ , and remains significant at the 0.05 level in 4 of them. The exception is removing the 5 StarVLA variants, which leaves 7 models with one-tailed $p = 0 . 1 1 3 .$

Table 1 | Leave-one-out over architecture groups for 12 fine-tuned vision-language-action models. Each row removes one group and recomputes the correlation between Sensorimotor Alignment (S<sup>2</sup>) and SimplerEnv-WidowX task success rate. The remaining model counts 𝑛 are 7, 9, 10, 11, and 11 in row order.

| Removed group | Pearson's 𝑟 | One-tailed 𝑝 |
| --- | --- | --- |
| StarVLA | +0.526 | 0.113 |
| CogACT | +0.785 | 0.006 |
| GR00T | +0.745 | 0.007 |
| Pi0 | +0.783 | 0.002 |
| SpatialVLA | +0.687 | 0.010 |

## B.3. Raw Vision-Language Models

We evaluate VLM4VLA’s [59] 8 raw VLMs: KosMos-2 [36], PaliGemma [4], PaliGemma 2 [46], Qwen2.5-VL-3B and Qwen2.5-VL-7B [3], and Qwen3-VL-2B, Qwen3-VL-4B, and Qwen3-VL-8B [2]. Their success rates come from the VLAs that VLM4VLA fine-tunes from each backbone on BridgeDataV2 and evaluates on SimplerEnv. We compute $S ^ { 2 }$ on the raw backbones before any robot fine-tuning, using DROID [26] as the reference dataset, which differs from the fine-tuning data in robot and scenes. We use ℎ = 3 and pool image and instruction tokens.

20

<!-- page 21 of 28 -->

## B.4. World Action Models

We evaluate 6 WAMs: Vidar [15], LingBot-VA [28], Motus [5], Fast-WAM [56], FlowWAM [10], and Kairos [25]. For all 6 models, we compute $S ^ { 2 }$ on identical bimanual ALOHA trajectories, fixing feature position, trajectory horizon, number of neighbors, and DTW computation. The reported comparison uses video-backbone Block 7 features at $h = 1 5$ and $k = 1 0$

## B.5. Real-Robot Policies

We use success rates from MolmoB0T’s physical-robot evaluation [11], which tests all policies on the same tasks, robots, and cameras. Policies run on 3 Franka DROID platforms in 4 environments (kitchen, workroom, bedroom, and office) at 2 institutions. Each environment has 10 pick-and-place tasks with 3 trials per task, giving 120 trials per policy. A trial succeeds if the policy places the named object stably in or on the receptacle within 900 steps. The evaluation compares 5 policies, three MolmoB0T variants (MolmoB0T, MolmoB0T-Img, and MolmoB0T-Pi0) and two baselines $( \pi _ { 0 } \text {-} \mathrm { D R O I D }$ and $\pi_{0.5} -DROID)$ . We compute $S ^ { 2 }$ from public checkpoints joint LLM-output features on the 6,000-episode DROID subset at $h = 1 1 0$ . This is an exploratory horizon; the prespecified $h = 7$ gives $r = + 0 . 4 2 9$ with one-tailed $p = 0 . 2 3 6$

## B.6. Frozen Vision Encoders Under Downstream Policy Learning

We ask whether $S ^ { 2 }$ can rank frozen vision backbones without training a policy. Hu et al. [20] provide a benchmark: they evaluate 14 pretrained vision backbones (9 ResNet-50 [17] and 5 ViT-B/16 [12]) on 21 tasks in Meta-World [55], Robosuite [62], and Franka-Kitchen [16]. They freeze each backbone and train policies on top, so only the backbone changes. We compute $S ^ { 2 }$ for their released checkpoints on the same 6,000-episode DROID subset as in the main paper, with all other settings unchanged. We take the interquartile mean (IQM) success rates from their paper for two policy-learning methods. Behavior cloning (BC) learns from expert demonstrations, and we use the smallest budget of 1 demonstration per task (5 on Robosuite). The visual reward function (VRF) method uses reinforcement learning, with rewards computed from distances between the backbone’s embeddings of agent and expert observations.

O ResNet-50

![Chart block](doc:0a14520/tier:basic/page:21/block:8)

ViT-B/16

![Chart block](doc:0a14520/tier:basic/page:21/block:10)

Figure 11 | Sensorimotor Alignment $( S ^ { 2 } )$ of frozen vision encoders correlates with downstream policy success rate. Points represent the 14 frozen backbones of Hu et al. [20], scored by $S ^ { 2 }$ on DROID without policy training. The vertical axis shows the interquartile mean (IQM) of task success rates over 21 tasks, using low-data behavior cloning (left) or reinforcement learning with visual reward functions (right).

We find a strong correlation between the backbone’s $S ^ { 2 }$ and downstream task success under both policy-learning methods, as shown in Figure 11. For BC, $r   =   + 0 . 6 7 7$ with one-tailed

21

<!-- page 22 of 28 -->

$p \; = \; 0 . 0 0 4$ For VRF, $r   =   + 0 . 7 7 4$ with one-tailed $p \; < \; 0 . 0 0 1$ Both methods rely directly on the frozen representation, and the correlation weakens when that reliance decreases. As the BC demonstration budget grows, the correlation drops to $r   =   + 0 . 2 3 6$ at the largest budget, consistent with more demonstrations letting the policy compensate for a weaker representation. For reinforcement learning from task rewards, which Hu et al. [20] find too variable across random seeds to evaluate vision backbones reliably, we find no positive correlation on two Meta-World seed sets or on Robosuite, where 𝑟 lies between −0.193 and −0.002. Thus, $S ^ { 2 }$ correlates with task success when the learning method relies directly on the frozen representation.

## C. Robustness of $S ^ { 2 }$

This section extends Section 3.3, which checks that the correlation does not depend on a particular choice of settings. Appendices C.1–C.3 cover the first check, across the trajectory horizon, five implementation choices, and DTW warping. Appendix C.4 gives the negative control, a mismatch in arm configuration.

## C.1. Trajectory Horizon

Tables 2 to 4 report $S ^ { 2 }$ for the 12 finetuned VLAs at all 10 DROID horizons $( N   =   6 { , } 0 0 0 ,   k   =   1 0 )$ , for the joint, vision-token, and text-token features selected from the same LLM output; Figure 7 summarizes the joint-feature correlation on both reference datasets. Models are sorted by SimplerEnv WidowX success rate (SR). Bottom rows give Spearman’s $\rho$ and Pearson’s 𝑟 with success rate, each followed by its one-tailed 𝑝-value for a positive association. We find that $S ^ { 2 }$ from joint vision-text features and from vision features correlates significantly with the success rate at every horizon, with 𝑟 from +0.610 to +0.717 and from +0.554 to +0.703, respectively. Text-feature $\overline { { S ^ { 2 } } }$ is not significant under Pearson’s 𝑟 at any horizon, where 𝑟 ranges from +0.227 to +0.274, consistent with the modality analysis in Section 4.1.

22

<!-- page 23 of 28 -->

Table 2 | Sensorimotor Alignment $( S ^ { 2 } )$ from joint vision-text features at the language-model output versus SimplerEnv-WidowX task success rate (SR) across trajectory horizons ℎ. We score 12 fine-tuned vision-language-action models on DROID $( N = 6 { , } 0 0 0$ episodes, $k = 1 0$ neighbors). Bottom rows report Spearman’s 𝜌 and Pearson’s 𝑟 with their one-tailed 𝑝-values. Bold marks the horizon of peak Pearson 𝑟 $( h = 7 )$

| Model | SR (%) | ℎ=1 ℎ=3 ℎ=7 ℎ=15 ℎ=25 ℎ=40 ℎ=75 | ℎ=110 | ℎ=150 | ℎ=220 |
| --- | --- | --- | --- | --- | --- |
| Qwen2.5-GR00T-Bridge | 71.4 | 0.034 0.034 0.036 0.040 0.043 0.051 0.065 | 0.076 | 0.086 | 0.097 |
| Qwen3-GR00T-RT-1-Bridge | 65.3 | 0.028 0.027 0.029 0.032 0.036 0.044 0.057 | 0.071 | 0.083 | 0.096 |
| Qwen2.5-GR00T-RT-1-Bridge | 63.6 | 0.034 0.034 0.035 0.040 0.042 0.048 0.062 | 0.071 | 0.080 | 0.090 |
| CogACT-Large | 58.3 | 0.031 0.031 0.033 0.037 0.040 0.048 0.062 | 0.075 | 0.086 | 0.097 |
| GR00T-N1.6-Bridge | 57.1 | 0.023 0.024 0.026 0.029 0.031 0.038 0.051 | 0.064 | 0.076 | 0.091 |
| CogACT-Base | 51.3 | 0.031 0.031 0.032 0.037 0.040 0.049 0.062 | 0.074 | 0.086 | 0.098 |
| CogACT-Small | 51.0 | 0.031 0.031 0.032 0.036 0.040 0.047 0.060 | 0.071 | 0.083 | 0.092 |
| Pi0-Bridge | 47.9 | 0.017 0.017 0.017 0.020 0.022 0.025 0.033 | 0.044 | 0.053 | 0.061 |
| Qwen2.5-OFT-RT-1-Bridge | 44.2 | 0.024 0.023 0.024 0.028 0.030 0.036 0.049 | 0.061 | 0.070 | 0.082 |
| Qwen3-OFT-RT-1-Bridge | 42.7 | 0.024 0.024 0.025 0.029 0.031 0.038 0.051 | 0.064 | 0.075 | 0.086 |
| SpatialVLA-Bridge | 42.7 | 0.021 0.022 0.022 0.024 0.026 0.033 0.044 | 0.054 | 0.063 | 0.073 |
| GR00T-N1.5-Bridge | 36.5 | 0.023 0.023 0.024 0.027 0.030 0.036 0.048 | 0.058 | 0.069 | 0.080 |
| Spearman 𝜌 |  | 0.736 0.764 0.802 0.788 0.767 0.767 0.774 | 0.746 | 0.774 | 0.725 |
| 𝑝 |  | 0.003 0.002 0.001 0.001 0.002 0.002 0.002 | 0.003 | 0.002 | 0.004 |
| Pearson 𝑟 |  | 0.692 0.699 0.717 0.696 0.689 0.683 0.647 | 0.646 | 0.629 | 0.610 |
| 𝑝 |  | 0.006 0.006 0.004 0.006 0.007 0.007 0.012 | 0.012 | 0.014 | 0.018 |

Table 3 | Sensorimotor Alignment $( S ^ { 2 } )$ from vision features at the language-model output versus SimplerEnv-WidowX task success rate (SR) across trajectory horizons ℎ. We score 12 fine-tuned vision-language-action models on DROID $( N = 6 { , } 0 0 0$ episodes, 𝑘 = 10 neighbors). Bottom rows report Spearman’s 𝜌 and Pearson’s 𝑟 with their one-tailed 𝑝-values. Bold marks the horizon of peak Pearson 𝑟 $( h = 1 \bar { 5 } )$

| Model | SR (%) | ℎ=1 ℎ=3 ℎ=7 ℎ=15 ℎ=25 ℎ=40 ℎ=75 | ℎ=110 | ℎ=150 | ℎ=220 |
| --- | --- | --- | --- | --- | --- |
| Qwen2.5-GR00T-Bridge | 71.4 | 0.034 0.034 0.035 0.039 0.043 0.049 0.062 | 0.074 | 0.084 | 0.093 |
| Qwen3-GR00T-RT-1-Bridge | 65.3 | 0.026 0.026 0.027 0.031 0.034 0.041 0.054 | 0.067 | 0.080 | 0.092 |
| Qwen2.5-GR00T-RT-1-Bridge | 63.6 | 0.033 0.033 0.035 0.038 0.041 0.047 0.059 | 0.070 | 0.079 | 0.088 |
| CogACT-Large | 58.3 | 0.030 0.030 0.032 0.035 0.039 0.047 0.061 | 0.073 | 0.084 | 0.095 |
| GR00T-N1.6-Bridge | 57.1 | 0.024 0.024 0.026 0.029 0.032 0.039 0.051 | 0.064 | 0.077 | 0.091 |
| CogACT-Base | 51.3 | 0.031 0.030 0.032 0.036 0.040 0.048 0.060 | 0.073 | 0.085 | 0.098 |
| CogACT-Small | 51.0 | 0.030 0.030 0.031 0.035 0.039 0.046 0.059 | 0.070 | 0.082 | 0.092 |
| Pi0-Bridge | 47.9 | 0.017 0.017 0.017 0.020 0.022 0.025 0.033 | 0.044 | 0.053 | 0.061 |
| Qwen2.5-OFT-RT-1-Bridge | 44.2 | 0.022 0.022 0.023 0.026 0.028 0.035 0.046 | 0.058 | 0.068 | 0.079 |
| Qwen3-OFT-RT-1-Bridge | 42.7 | 0.024 0.024 0.025 0.028 0.030 0.037 0.050 | 0.063 | 0.074 | 0.087 |
| SpatialVLA-Bridge | 42.7 | 0.022 0.022 0.022 0.025 0.026 0.033 0.045 | 0.055 | 0.064 | 0.075 |
| GR00T-N1.5-Bridge | 36.5 | 0.023 0.023 0.024 0.027 0.029 0.035 0.048 | 0.059 | 0.069 | 0.079 |
| Spearman 𝜌 |  | 0.767 0.757 0.767 0.746 0.746 0.739 0.725 | 0.711 | 0.683 | 0.704 |
| 𝑝 |  | 0.002 0.002 0.002 0.003 0.003 0.003 0.004 | 0.005 | 0.007 | 0.005 |
| Pearson 𝑟 |  | 0.686 0.694 0.701 0.703 0.690 0.647 0.606 | 0.618 | 0.601 | 0.554 |
| 𝑝 |  | 0.007 0.006 0.006 0.005 0.007 0.011 0.018 | 0.016 | 0.019 | 0.031 |

Table 4 | Sensorimotor Alignment $( S ^ { 2 } )$ from text features at the language-model output versus SimplerEnv-WidowX task success rate (SR) across trajectory horizons ℎ. We score 12 fine-tuned vision-language-action models on DROID $( N = 6 { , } 0 0 0$ episodes, 𝑘 = 10 neighbors). Bottom rows report Spearman’s 𝜌 and Pearson’s 𝑟 with their one-tailed 𝑝-values. Bold marks the highest Spearman 𝜌 (ℎ = 150).

| Model | SR (%) | ℎ=1 ℎ=3 ℎ=7 ℎ=15 ℎ=25 ℎ=40 ℎ=75 | ℎ=110 | ℎ=150 | ℎ=220 |
| --- | --- | --- | --- | --- | --- |
| Qwen2.5-GR00T-Bridge | 71.4 | 0.018 0.019 0.019 0.022 0.025 0.030 0.039 | 0.046 | 0.053 | 0.059 |
| Qwen3-GR00T-RT-1-Bridge | 65.3 | 0.015 0.015 0.016 0.018 0.020 0.024 0.031 | 0.040 | 0.048 | 0.057 |
| Qwen2.5-GR00T-RT-1-Bridge | 63.6 | 0.022 0.021 0.022 0.026 0.028 0.032 0.040 | 0.048 | 0.053 | 0.058 |
| CogACT-Large | 58.3 | 0.037 0.037 0.038 0.042 0.045 0.054 0.068 | 0.081 | 0.091 | 0.104 |
| GR00T-N1.6-Bridge | 57.1 | 0.008 0.008 0.008 0.010 0.010 0.014 0.020 | 0.024 | 0.029 | 0.033 |
| CogACT-Base | 51.3 | 0.035 0.035 0.037 0.041 0.044 0.053 0.068 | 0.080 | 0.090 | 0.104 |
| CogACT-Small | 51.0 | 0.030 0.029 0.030 0.034 0.038 0.044 0.057 | 0.064 | 0.070 | 0.080 |
| Pi0-Bridge | 47.9 | 0.014 0.015 0.016 0.019 0.021 0.024 0.033 | 0.043 | 0.052 | 0.061 |
| Qwen2.5-OFT-RT-1-Bridge | 44.2 | 0.019 0.018 0.018 0.022 0.024 0.028 0.036 | 0.044 | 0.051 | 0.060 |
| Qwen3-OFT-RT-1-Bridge | 42.7 | 0.013 0.013 0.014 0.015 0.016 0.020 0.026 | 0.034 | 0.040 | 0.045 |
| SpatialVLA-Bridge | 42.7 | 0.014 0.014 0.014 0.016 0.017 0.022 0.030 | 0.036 | 0.043 | 0.049 |
| GR00T-N1.5-Bridge | 36.5 | 0.010 0.010 0.010 0.011 0.011 0.014 0.019 | 0.024 | 0.029 | 0.033 |
| Spearman 𝜌 |  | 0.438 0.452 0.452 0.452 0.452 0.452 0.501 | 0.452 | 0.508 | 0.291 |
| 𝑝 |  | 0.077 0.070 0.070 0.070 0.070 0.070 0.049 | 0.070 | 0.046 | 0.180 |
| Pearson 𝑟 |  | 0.227 0.238 0.245 0.265 0.274 0.259 0.254 | 0.265 | 0.255 | 0.233 |
| 𝑝 |  | 0.239 0.229 0.222 0.203 0.194 0.208 0.213 | 0.202 | 0.212 | 0.233 |

23

<!-- page 24 of 28 -->

## C.2. Implementation Choices

We vary one choice at a time, keeping the main-paper defaults otherwise: the 6,000-episode DROID subset, $h   =   7 ,   k   =   1 0 ,$ , the joint vision-text features at the language-model output, and the 12 finetuned VLAs. Table 5 reports the Pearson 𝑟 between $S ^ { 2 }$ and SimplerEnv success rate for each setting. All settings of $k ,$ normalization, component weights, rotation distance, and pooling keep the correlation positive and significant, with 𝑟 between +0.634 and +0.752.

Table 5 | Sensitivity of the Sensorimotor Alignment (S<sup>2</sup>)–success-rate correlation to implementation choices. Pearson’s 𝑟 and one-tailed 𝑝 compare $S ^ { 2 }$ with SimplerEnv-WidowX task success rate across 12 fine-tuned visionlanguage-action models. Each block varies one choice; ∗ marks the default.

| Setting | r | Setting | r | p |
| --- | --- | --- | --- | --- |
| Number of neighbors k | 0.0076 | $( 1 , { \bar { 1 } } , 1 ) ^ { * }$ | Weights (position, rotation, gripper) |  |
| $k { = } 2$ | +0.679 |  | +0.717 (0.5, 1, 1) | 0.0043 |
| $k = 5$ | +0.667 |  | +0.698 | 0.0058 |
| $k = 1 0 ^ { * }$ | +0.717 |  | +0.708 | 0.0050 |
| $k { = } 1 5$ | +0.665 |  | +0.694 | 0.0062 |
| $k = 2 0$ | +0.644 |  | +0.703 | 0.0054 |
| Trajectory normalization |  |  | (1,1,0.5) +0.702 | 0.0055 |
| Absolute pose* | +0.717 |  | +0.699 | 0.0057 |
| Anchor-relative | +0.752 |  | +0.745 | 0.0027 |
| Rotation distance |  |  | +0.658 | 0.0100 |
| SO(3) geodesic* | +0.717 0.0043 |  | (1,1,0) +0.634 | 0.0135 |
| Quaternion squared chordal | +0.686 |  |  |  |
| XYZ Euler $\ell _ { 2 }$ | +0.660 | 0.0069 0.0097 |  |  |
| Token pooling |  |  |  |  |
| Masked mean | +0.717 | 0.0043 |  |  |
| Masked max | +0.653 |  |  |  |
| Masked min | +0.730 | 0.0107 0.0035 |  |  |

**Number of neighbors** 𝑘. We use 𝑘 =10 in the main paper, following Huh et al. [21]. Across $k   \in   \{ 2 , 5 , 1 0 , 1 5 , 2 0 \}$ , 𝑟 ranges from +0.644 to +0.717, with a mild peak at the default 𝑘 =10.

**Trajectory normalization.** The anchor-relative setting replaces each position with its offset from the starting frame and keeps rotation and gripper state unchanged. It gives a slightly higher 𝑟 than absolute poses, possibly because removing the workspace offset emphasizes the shape of the motion. We keep absolute poses for simplicity.

**Component weights.** The DTW step cost in Appendix A.2 combines position, rotation, and gripper-state distances with weights $w _ { \mathrm { p o s } } , w _ { \mathrm { r o t } } ,$ , and $w _ { \mathtt { g r i p } } ,$ all equal to one by default. Halving, doubling, or removing one weight keeps 𝑟 between +0.634 and +0.745. Removing rotation or gripper lowers 𝑟 most, suggesting both carry information beyond position.

**Rotation distance and token pooling.** The SO(3) geodesic gives the highest 𝑟 of the three rotation distances. The quaternion alternative uses the scaled squared chordal quantity $1 - | \mathbf { q } ^ { \top } \mathbf { q } ^ { \prime } |$ the Euler alternative uses the $\ell _ { 2 }$ distance between recorded angle triples, without wrapping angular differences. Each alternative is rescaled to match the mean geodesic distance on 40,000 sampled state pairs. For pooling the LLM-output tokens, masked min gives a slightly higher 𝑟 than the default masked mean, and masked max gives the lowest.

## C.3. DTW Alignment

$S ^ { 2 }$ aligns trajectory pairs with DTW before measuring their distance, so the same motion at different speeds counts as similar. To test whether the correlation depends on this warping, we compare DTW with the lockstep distance $\begin{array} { r } { d _ { \mathrm { l o c k } } ( i , j )   =   \sum _ { u = 1 } ^ { h } c _ { u u } , } \end{array}$ which matches frames by

24

<!-- page 25 of 28 -->

time index using the local cost in Appendix A.2. The lockstep sum weights every matched state once, whereas symmetric2 DTW weights diagonal transitions twice. Table 6 reports the comparison. DTW raises $S ^ { 2 }$ for all 12 VLAs from $h   =   7$ onward, and the gain grows with the horizon, from 1.2% at $h   =   7$ to $3 4 . 1 \%$ at $h   =   2 2 0 ,$ , since longer trajectories contain more variation in speed. Because DTW raises the $S ^ { 2 }$ of all models by a similar amount, $( 3 4 \pm 4 ) \%$ (mean ± standard deviation across models) at $h   =   2 2 0 _ { , }$ , the correlation with task success barely changes. Across the 10 horizons, 𝑟 ranges from +0.617 to +0.713 without DTW and from +0.610 to +0.717 with DTW, a relative difference between −3.2% and +0.6%. The $S ^ { 2 } .$ –success-rate correlation is therefore not an artifact of warping.

Table 6 | Comparing Sensorimotor Alignment $( S ^ { 2 } )$ with and without Dynamic Time Warping (DTW). DTW raises absolute $S ^ { 2 }$ more at longer trajectory horizons ℎ, while its Pearson correlation 𝑟 with SimplerEnv-WidowX task success rate (SR) stays stable. Each relative change is $1 0 0 ( \nu _ { \mathrm { D T W } } / \nu _ { \mathrm { n o D T W } } - 1 ) \%$ for the quantity 𝑣 in that block.

<table><tr><td rowspan="2"></td><td colspan="3">(mean over 12 VLAs)  $S ^ { 2 }$ </td><td colspan="3">S2-SR Pearson r</td></tr><tr><td>w/DTW Horizon h</td><td>w/o DTW</td><td>rel. change</td><td>w/DTW</td><td>w/o DTW</td><td>rel. change</td></tr><tr><td>1</td><td>0.0268</td><td>0.0268</td><td>+0.0%</td><td>+0.692</td><td>+0.692</td><td>+0.0%</td></tr><tr><td>3</td><td>0.0268</td><td>0.0269</td><td>-0.6%</td><td>+0.699</td><td>+0.700</td><td>-0.1%</td></tr><tr><td>7</td><td>0.0280</td><td>0.0277</td><td>+1.2%</td><td>+0.717</td><td>+0.713</td><td>+0.6%</td></tr><tr><td>15</td><td>0.0314</td><td>0.0302</td><td>+4.1%</td><td>+0.696</td><td>+0.710</td><td>-2.0%</td></tr><tr><td>25</td><td>0.0342</td><td>0.0317</td><td>+7.7%</td><td>+0.689</td><td>+0.711</td><td>-3.2%</td></tr><tr><td>40</td><td>0.0410</td><td>0.0365</td><td>+12.4%</td><td>+0.683</td><td>+0.680</td><td>+0.4%</td></tr><tr><td>75</td><td>0.0537</td><td>0.0442</td><td>+21.7%</td><td>+0.647</td><td>+0.663</td><td>-2.5%</td></tr><tr><td>110</td><td>0.0653</td><td>0.0533</td><td>+22.4%</td><td>+0.646</td><td>+0.661</td><td>-2.2%</td></tr><tr><td>150</td><td>0.0758</td><td>0.0602</td><td>+25.9%</td><td>+0.629</td><td>+0.633</td><td>-0.7%</td></tr><tr><td>220</td><td>0.0870</td><td>0.0649</td><td>+34.1%</td><td>+0.610</td><td>+0.617</td><td>-1.1%</td></tr></table>

## C.4. Sensitivity to Embodiment Mismatch

For the negative control in Section 3.3, we score 3 bimanual WAMs evaluated on RoboTwin with two reference datasets that differ in arm configuration, as shown in Figure 12. With the matched bimanual ALOHA reference [61], $S ^ { 2 }$ recovers the success-rate order of all 3 models, at $r   =   + 0 . 9 9 9 .$ With the single-arm DROID reference, the correlation drops to $r   =   + 0 . 2 4 6$ with $p   =   0 . 4 2 1$ $S ^ { 2 }$ therefore responds to the mismatch in arm configuration rather than reporting alignment indiscriminately. This sensitivity complements the stability across single-arm references in Section 3.3, where DROID and BridgeDataV2 differ in robot, scenes, and tasks yet give nearly the same correlation for the 12 VLAs. A reference dataset therefore need not come from the exact target robot, but it should match the target’s arm configuration.

![Chart block](doc:0a14520/tier:basic/page:25/block:6)

![Chart block](doc:0a14520/tier:basic/page:25/block:7)

Figure 12 | Sensorimotor Alignment $( S ^ { 2 } )$ **is sensitive to a mismatch in arm configuration.** For 3 world action models (WAMs) evaluated on bimanual RoboTwin, the matched bimanual ALOHA reference (left) recovers the success-rate ordering, while the single-arm DROID reference (right) does not.

## D. Details of the Representation-Level Explanations

This section gives the full data behind the representation-level explanations in Section 4.

25

<!-- page 26 of 28 -->

## D.1. $S ^ { 2 }$ Across Feature Positions

Section 4.1 compares $S ^ { 2 }$ from vision, text, and joint features at the LLM output. Here we extend the comparison to every feature position, from the vision-encoder output (ViT-out) through the language-model input (LLM-in) to the language-model output (LLM-out).

Correlation with task success. Table 7 reports the correlation between $S ^ { 2 }$ and SimplerEnv success rate for the 12 fine-tuned VLAs at each feature position on DROID $( N = 6 , 0 0 0 , h = 7 ) ,$ Section 4.1 reports $h   =   3 ,$ , and Appendix C.1 gives every horizon. The correlation holds at every visual feature position and grows with depth, from $r   =   + 0 . 5 6 7$ at the vision-encoder output to $r   =   + 0 . 7 1 7$ at the LLM output. Text-only features show no significant correlation at either position. This agrees with reports that VLAs follow visual cues despite conflicting instructions [13, 60].

Table 7 | Correlation between Sensorimotor Alignment (S<sup>2</sup>) and task success by feature position. Pearson’s 𝑟 and one-tailed 𝑝 compare $S ^ { 2 }$ with SimplerEnv-WidowX task success rate across 12 fine-tuned vision-language-action models on DROID $( N   =   6 { , } 0 0 0$ episodes, trajectory horizon ℎ = 7). ViT-out, LLM-in, and LLM-out denote the vision-encoder output, language-model input, and language-model output; ∗ marks the default feature.

| Feature position | Pearson's 𝑟 | One-tailed 𝑝 |
| --- | --- | --- |
| ViT-out | +0.567 | 0.0274 |
| LLM-in vision-text | +0.580 | 0.0239 |
| LLM-out vision-only | +0.701 | 0.0056 |
| LLM-out vision-text<sup>∗</sup> | +0.717 | 0.0043 |
| LLM-in text-only | -0.325 | 0.8486 |
| LLM-out text-only | +0.245 | 0.2215 |

$S ^ { 2 }$ by model group. Table 8 reports the mean $S ^ { 2 }$ at each feature position for 3 model groups on DROID $( N   =   6 { , } 0 0 0 , h   =   7 )$ : 3 standalone vision encoders, the 8 raw VLMs, and the 12 fine-tuned VLAs. Table 9 gives the per-model values. For the vision, text, and joint vision-text features of Section 4.1, we average-pool the selected output tokens and apply the feature normalization in Appendix A.3. The VLA group has higher mean $S ^ { 2 }$ than the raw-VLM group at every compared position. The relative difference is larger at the LLM output, from +29.9% to +45.0%, than at the vision-encoder output and LLM input, from +12.7% to +18.1%.

Table 8 | Mean Sensorimotor Alignment $( S ^ { 2 } )$ by feature position on DROID $( N = 6 { , } 0 0 0$ episodes, trajectory horizon $h = 7 )$ . ViT-out, LLM-in, and LLM-out denote the vision-encoder output, language-model input, and language-model output; standalone vision encoders have only ViT-out features. The last column is $100(\overline{\mathrm{S}^{2}}_{\mathrm{VLA}} / \overline{\mathrm{S}^{2}}_{\mathrm{VLM}} - 1)\%$ , where each bar denotes the model-group mean.

| Feature | Vision encoder (mean, 𝑛=3) | Raw VLM (mean, 𝑛=8) | Finetuned VLA (mean, 𝑛=12) | Relative change |
| --- | --- | --- | --- | --- |
| ViT-out | 0.0246 | 0.0247 | 0.0278 | +12.7% |
| LLM-in vision-only | - | 0.0244 | 0.0275 | +12.7% |
| LLM-in vision-text | - | 0.0233 | 0.0275 | +18.1% |
| LLM-out vision-only | - | 0.0212 | 0.0275 | +29.9% |
| LLM-out text-only | - | 0.0140 | 0.0202 | +45.0% |
| LLM-out vision-text | - | 0.0204 | 0.0280 | +37.4% |

26

<!-- page 27 of 28 -->

Table 9 | Per-model Sensorimotor Alignment $( S ^ { 2 } )$ by feature position on DROID (𝑁 = 6,000 episodes, trajectory horizon ℎ = 7). ViT-out, LLM-in, and LLM-out denote the vision-encoder output, language-model input, and language-model output. Vision-only, text-only, and vision-text select image tokens, instruction tokens, or both from the same model input. Group means are in Table 8.

<table><tr><td>Model</td><td>ViT-out</td><td>LLM-in vision-only</td><td>LLM-in text-only</td><td>LLM-in vision-text</td><td>LLM-out vision-only</td><td>LLM-out text-only</td><td>LLM-out vision-text</td></tr><tr><td colspan="8">Pretrained vision encoders, no LLM attached (n=3)</td></tr><tr><td>CLIP-ViT-L/14 [39]</td><td>0.0240</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SigLIP-SO400M [58]</td><td>0.0247</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SigLIP-2 Large [48]</td><td>0.0250</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="8">Raw VLMs from VLM4VLA (n =8)</td></tr><tr><td>KosMos-2</td><td>0.0257</td><td>0.0238</td><td>0.0093</td><td>0.0239</td><td>0.0214</td><td>0.0151</td><td>0.0216</td></tr><tr><td>PaliGemma 1</td><td>0.0249</td><td>0.0251</td><td>0.0082</td><td>0.0193</td><td>0.0193</td><td>0.0160</td><td>0.0191</td></tr><tr><td>PaliGemma 2</td><td>0.0250</td><td>0.0251</td><td>0.0087</td><td>0.0222</td><td>0.0196</td><td>0.0142</td><td>0.0192</td></tr><tr><td>Qwen2.5-VL-3B</td><td>0.0241</td><td>0.0239</td><td>0.0079</td><td>0.0239</td><td>0.0185</td><td>0.0139</td><td>0.0188</td></tr><tr><td>Qwen2.5-VL-7B</td><td>0.0231</td><td>0.0234</td><td>0.0083</td><td>0.0234</td><td>0.0188</td><td>0.0127</td><td>0.0190</td></tr><tr><td>Qwen3-VL-2B</td><td>0.0242</td><td>0.0237</td><td>0.0089</td><td>0.0237</td><td>0.0236</td><td>0.0127</td><td>0.0217</td></tr><tr><td>Qwen3-VL-4B</td><td>0.0253</td><td>0.0254</td><td>0.0089</td><td>0.0254</td><td>0.0239</td><td>0.0135</td><td>0.0232</td></tr><tr><td>Qwen3-VL-8B</td><td>0.0250</td><td>0.0245</td><td>0.0088</td><td>0.0245</td><td>0.0240</td><td>0.0136</td><td>0.0207</td></tr><tr><td colspan="8">Finetuned VLAs (n=12)</td></tr><tr><td>Qwen2.5-GR00T-Bridge</td><td>0.0344</td><td>0.0335</td><td>0.0078</td><td>0.0335</td><td>0.0354</td><td>0.0192</td><td>0.0363</td></tr><tr><td>Qwen2.5-GR00T-RT-1-Bridge</td><td>0.0355</td><td>0.0345</td><td>0.0077</td><td>0.0345</td><td>0.0350</td><td>0.0224</td><td>0.0353</td></tr><tr><td>Qwen2.5-OFT-RT-1-Bridge</td><td>0.0265</td><td>0.0265</td><td>0.0082</td><td>0.0265</td><td>0.0234</td><td>0.0185</td><td>0.0242</td></tr><tr><td>Qwen3-GR00T-RT-1-Bridge</td><td>0.0264</td><td>0.0266</td><td>0.0085</td><td>0.0266</td><td>0.0274</td><td>0.0158</td><td>0.0292</td></tr><tr><td>Qwen3-OFT-RT-1-Bridge</td><td>0.0263</td><td>0.0257</td><td>0.0086</td><td>0.0257</td><td>0.0248</td><td>0.0136</td><td>0.0254</td></tr><tr><td>CogACT Small</td><td>0.0301</td><td>0.0303</td><td>0.0097</td><td>0.0303</td><td>0.0313</td><td>0.0300</td><td>0.0319</td></tr><tr><td>CogACT Base</td><td>0.0301</td><td>0.0298</td><td>0.0096</td><td>0.0298</td><td>0.0319</td><td>0.0368</td><td>0.0324</td></tr><tr><td>CogACT Large</td><td>0.0298</td><td>0.0298</td><td>0.0097</td><td>0.0298</td><td>0.0317</td><td>0.0379</td><td>0.0328</td></tr><tr><td>GR00T-N1.5-Bridge</td><td>0.0252</td><td>0.0244</td><td>0.0090</td><td>0.0243</td><td>0.0238</td><td>0.0099</td><td>0.0240</td></tr><tr><td>GR00T-N1.6-Bridge</td><td>0.0251</td><td>0.0245</td><td>0.0090</td><td>0.0245</td><td>0.0258</td><td>0.0082</td><td>0.0255</td></tr><tr><td>Pi0-Bridge</td><td>0.0176</td><td>0.0178</td><td>0.0090</td><td>0.0180</td><td>0.0174</td><td>0.0161</td><td>0.0174</td></tr><tr><td>SpatialVLA-Bridge</td><td>0.0266</td><td>0.0263</td><td>0.0087</td><td>0.0265</td><td>0.0219</td><td>0.0145</td><td>0.0217</td></tr></table>

## D.2. Frozen Versus Unfrozen Vision Encoder

Following VLM4VLA [59], both VLAs in Section 4.2 attach an FCDecoder action head to Qwen2.5-VL-3B [3] and are fine-tuned on BridgeDataV2. Training differs only in whether the vision encoder is frozen. We compare each configuration’s checkpoint with the best observed success rate: step 10,000 for the unfrozen model and step 30,000 for the frozen model. Table 10 compares all 10 horizons using mean-pooled LLM-output vision-text features, as in the main experiments, rather than the action-token state read by the action head. The frozen model has lower $S ^ { 2 }$ at every horizon, with relative reductions from 38.1% at ℎ=1 to 49.4% at ℎ=220.

Table 10 | Sensorimotor Alignment $( S ^ { 2 } )$ with a frozen and an unfrozen vision encoder across trajectory horizons ℎ. Qwen2.5-VL-3B vision-language-action models trained with the same recipe, evaluated using joint vision-text features at the language-model output on DROID (𝑁 = 6,000 episodes). Relative change is $1 0 0 ( \mathrm { S } _ { \mathrm { f r o z e n } } ^ { 2 } / \mathrm { S } _ { \mathrm { u n f r o z e n } } ^ { 2 } - 1 ) \%$

| Horizon ℎ | Unfrozen S<sup>2</sup> | Frozen S<sup>2</sup> | Relative change |
| --- | --- | --- | --- |
| 1 | 0.0231 | 0.0143 | -38.1% |
| 3 | 0.0232 | 0.0141 | -39.0% |
| 7 | 0.0245 | 0.0144 | -41.1% |
| 15 | 0.0268 | 0.0158 | -41.2% |
| 25 | 0.0295 | 0.0166 | -43.8% |
| 40 | 0.0352 | 0.0191 | -45.7% |
| 75 | 0.0481 | 0.0258 | -46.4% |
| 110 | 0.0614 | 0.0326 | -46.9% |
| 150 | 0.0733 | 0.0390 | -46.8% |
| 220 | 0.0861 | 0.0435 | -49.4% |

27

<!-- page 28 of 28 -->

## D.3. WAM Features Along the Denoising Schedule

Table 11 reports $S ^ { 2 }$ for 3 WAMs at 7 feature positions and 5 points of each model’s denoising schedule. Figure 10a plots the final-layer-norm row, and Figure 10b plots each row’s schedule average. Block 14 has the highest $S ^ { 2 }$ at 13 of the 15 model and schedule points. Columns give Motus steps 1, 3, 5, 8, and 10 of 10; LingBot-VA steps 0, 6, 13, 19, and 25 of 25; and Vidar steps 0, 5, 10, 15, and 19 of 20; the last Motus and Vidar features precede the final update.

Table 11 | Layer-wise Sensorimotor Alignment $( S ^ { 2 } )$ along the denoising schedule for 3 world action models (WAMs). Columns label 5 selected positions in each model’s native schedule, not matched noise levels. Motus and LingBot-VA use 500 episodes over 50 RoboTwin tasks; Vidar uses 480 episodes over 48 tasks. Rows give 6 video-backbone positions and the text embedding. Block 14 (bold) has the highest $S ^ { 2 }$ at 13 of the 15 model and schedule positions. The random baseline is the mean ± standard deviation (divisor 10) over 10 draws of independent standard-normal features with the same shape.

| Feature | 0% earliest | 25% | 50% | 75% | 100 % |
| --- | --- | --- | --- | --- | --- |
|  |  |  |  |  | latest |
| Motus Block 0 | random baseline = 0.012 ± 0.002 | 0.203 | 0.307 | 0.321 | 0.324 |
| Block 7 | 0.044 0.342 | 0.388 | 0.399 | 0.390 | 0.378 |
| Block 14 | 0.479 | 0.492 | 0.482 | 0.479 | 0.471 |
| Block 21 | 0.454 | 0.448 | 0.447 | 0.436 | 0.424 |
| Block 29 | 0.420 | 0.424 | 0.419 | 0.421 | 0.405 |
| Final layer norm | 0.425 | 0.421 | 0.421 | 0.418 | 0.402 |
| Text embedding | 0.325 | 0.325 | 0.325 | 0.325 | 0.325 |
|  |  |  |  |  |  |
| LingBot-VA | random baseline = 0.012 ± 0.001 | 0.160 | 0.238 |  |  |
| Block 0 Block 7 | 0.142 | 0.399 | 0.431 | 0.326 | 0.384 |
| Block 14 | 0.404 0.498 | 0.504 | 0.514 | 0.449 0.514 | 0.455 0.513 |
| Block 21 | 0.422 | 0.435 | 0.428 | 0.445 | 0.449 |
| Block 29 | 0.423 | 0.440 | 0.441 | 0.447 | 0.448 |
| Final layer norm | 0.425 | 0.429 | 0.431 | 0.439 | 0.451 |
| Text embedding |  | 0.355 | 0.355 | 0.355 |  |
|  | 0.355 |  |  |  | 0.355 |
| Vidar | random baseline : | 二 | 0.013 ± | 0.004 |  |
| Block 0 | 0.483 | 0.477 | 0.459 | 0.429 | 0.422 |
| Block 7 | 0.467 | 0.484 | 0.485 | 0.479 | 0.469 |
| Block 14 | 0.504 | 0.555 | 0.531 | 0.518 | 0.473 |
| Block 21 | 0.508 | 0.510 | 0.500 | 0.488 | 0.475 |
| Block 29 | 0.487 | 0.490 | 0.489 | 0.481 | 0.474 |
| Final layer norm | 0.487 | 0.491 | 0.491 | 0.481 | 0.476 |
| Text embedding | 0.467 | 0.467 | 0.467 | 0.467 | 0.467 |

**Behavioral validation.** To check that $S ^ { 2 }$ along the denoising schedule correlates with task success, we vary the number of video denoising steps in LingBot-VA [28]. We measure $S ^ { 2 }$ and closed-loop success rate on the RoboTwin turn\_switch task over 100 episodes. Reducing the steps from the default 25 to 5 lowers the success rate from 56% to 48% and $S ^ { 2 }$ from 0.288 to 0.241, so $S ^ { 2 }$ and task success change together.

28