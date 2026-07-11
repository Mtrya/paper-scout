Title: GigaWorld-1: A Roadmap to Build World Models for Robot Policy Evaluation

URL Source: https://arxiv.org/html/2607.02642

Markdown Content:
GigaAI

Tsinghua University

Project Page: [https://open-gigaai.github.io/giga-world-1/](https://open-gigaai.github.io/giga-world-1/)

Alphabetical Order: Angyuan Ma Bohan Li  Chaojun Ni  Guo Li  Guan Huang  Guosheng Zhao  Hao Li  Hengtao Li  Jingyu Liu  Jiwen Lu  Qiuping Deng  Tingdong Yu  Xuancheng Xu  Xinyu Zhou  Xiuwei Xu  Xinze Chen  Xiaofeng Wang 

Xiaoyu Tian  Yang Wang  Yifan Chang  Yukun Zhou  Yun Ye  Zhenyu Wu  Zhanqian Wu  Zheng Zhu

###### Abstract

Evaluating embodied robot foundation models remains a critical bottleneck; unlike large language models efficiently assessed via digital benchmarks, robotic policies require slow, costly real-world rollouts limited by hardware and human supervision, which has driven interest in world models as surrogate policy evaluators, yet the key properties that make a world model reliable for policy assessment remain poorly understood. This work presents a systematic study of world models for robotic policy evaluation and introduces WMBench, a benchmark constructed from real-robot teleoperation data and matched policy rollouts covering diverse manipulation tasks to enable controlled comparisons across model families, action encodings, rollout horizons, and evaluation metrics. Using WMBench, we analyze 7 video world models, 4 action representation schemes, and over 324,000 simulated policy rollouts paired with real robot executions, further enriching our analysis with large-scale community submissions from the CVPR 2026 GigaBrain Challenge, curated synthetic trajectories, and a training videos spanning more than 12,000 hours. Our experiments deliver three core insights: evaluator quality is dominated by long-horizon, action-faithful rollout consistency rather than short-term visual realism; pretraining gains stem not only from data scale but from balancing general world knowledge with robot-specific controllability; and architectural choices including action encoding, memory design, and evaluator-focused post-training strongly determine alignment with real-world robot behavior. Drawing on these results, we derive a practical design roadmap and realize it in GigaWorld-1, a world model specially optimized for policy evaluation, and we fully release our code, models, datasets, and toolkits to advance scalable evaluation research for embodied foundation models.

![Image 1: Refer to caption](https://arxiv.org/html/2607.02642v1/x1.png)

Figure 1: This paper analyzes 324,000 world-model-simulated rollouts, 7 video world models, 4 action representation paradigms. Complemented by more than 12,000 hours of training data and guided by the roadmap we establish for building world models tailored to robot policy evaluation, we introduce GigaWorld-1.

\abscontent

## 1 Introduction

Efficient evaluation is critical to the iterative improvement and performance tuning of large foundation models. In language modeling, evaluation incurs relatively low overhead: newly saved model checkpoints can be rapidly assessed against standardized benchmarks [[37](https://arxiv.org/html/2607.02642#bib.bib37), [121](https://arxiv.org/html/2607.02642#bib.bib121), [154](https://arxiv.org/html/2607.02642#bib.bib154), [61](https://arxiv.org/html/2607.02642#bib.bib61), [24](https://arxiv.org/html/2607.02642#bib.bib24), [94](https://arxiv.org/html/2607.02642#bib.bib94), [47](https://arxiv.org/html/2607.02642#bib.bib47), [40](https://arxiv.org/html/2607.02642#bib.bib40), [140](https://arxiv.org/html/2607.02642#bib.bib140), [141](https://arxiv.org/html/2607.02642#bib.bib141), [28](https://arxiv.org/html/2607.02642#bib.bib28)], meaning evaluation rarely becomes a bottleneck. Robotics, however, presents a fundamentally distinct scenario. Validating a robot policy typically demands repeated real-world rollouts on physical robot hardware, which necessitates continuous human monitoring and occupies robot hardware for lengthy evaluation cycles. As a result, evaluation emerges as the primary bottleneck holding back progress in robot policy models.

This bottleneck is especially severe for recent robot foundation models such as vision-language-action models and world-action models [[66](https://arxiv.org/html/2607.02642#bib.bib66), [14](https://arxiv.org/html/2607.02642#bib.bib14), [43](https://arxiv.org/html/2607.02642#bib.bib43), [13](https://arxiv.org/html/2607.02642#bib.bib13), [20](https://arxiv.org/html/2607.02642#bib.bib20), [12](https://arxiv.org/html/2607.02642#bib.bib12), [51](https://arxiv.org/html/2607.02642#bib.bib51), [98](https://arxiv.org/html/2607.02642#bib.bib98), [84](https://arxiv.org/html/2607.02642#bib.bib84), [143](https://arxiv.org/html/2607.02642#bib.bib143), [45](https://arxiv.org/html/2607.02642#bib.bib45), [102](https://arxiv.org/html/2607.02642#bib.bib102), [134](https://arxiv.org/html/2607.02642#bib.bib134)]. Although these models are becoming increasingly capable, reliable evaluation still depends heavily on real-robot execution. As reported in OpenVLA [[50](https://arxiv.org/html/2607.02642#bib.bib50)], it requires 100 human hours for 2,500 rollouts evaluation, and physical robots cannot yield fully consistent reset states across trials. Classical simulation can partially reduce cost, but its utility is limited by the sim-to-real gap, alongside prohibitive overhead associated with scene-wise digital twin construction [[22](https://arxiv.org/html/2607.02642#bib.bib22), [82](https://arxiv.org/html/2607.02642#bib.bib82), [62](https://arxiv.org/html/2607.02642#bib.bib62), [57](https://arxiv.org/html/2607.02642#bib.bib57), [145](https://arxiv.org/html/2607.02642#bib.bib145), [80](https://arxiv.org/html/2607.02642#bib.bib80), [44](https://arxiv.org/html/2607.02642#bib.bib44), [161](https://arxiv.org/html/2607.02642#bib.bib161), [86](https://arxiv.org/html/2607.02642#bib.bib86), [85](https://arxiv.org/html/2607.02642#bib.bib85), [116](https://arxiv.org/html/2607.02642#bib.bib116)]. World models offer a compelling middle ground. Recent progress in video generation and action-conditioned world modeling [[15](https://arxiv.org/html/2607.02642#bib.bib15), [132](https://arxiv.org/html/2607.02642#bib.bib132), [33](https://arxiv.org/html/2607.02642#bib.bib33), [109](https://arxiv.org/html/2607.02642#bib.bib109), [46](https://arxiv.org/html/2607.02642#bib.bib46), [159](https://arxiv.org/html/2607.02642#bib.bib159), [117](https://arxiv.org/html/2607.02642#bib.bib117), [150](https://arxiv.org/html/2607.02642#bib.bib150), [111](https://arxiv.org/html/2607.02642#bib.bib111), [103](https://arxiv.org/html/2607.02642#bib.bib103), [66](https://arxiv.org/html/2607.02642#bib.bib66), [53](https://arxiv.org/html/2607.02642#bib.bib53)] reveals that learned world models can capture rich visual dynamics and, to some extent, controllable physical evolution [[23](https://arxiv.org/html/2607.02642#bib.bib23), [97](https://arxiv.org/html/2607.02642#bib.bib97), [114](https://arxiv.org/html/2607.02642#bib.bib114), [68](https://arxiv.org/html/2607.02642#bib.bib68), [147](https://arxiv.org/html/2607.02642#bib.bib147), [1](https://arxiv.org/html/2607.02642#bib.bib1), [48](https://arxiv.org/html/2607.02642#bib.bib48)]. If such models can interact with robot policies and accurately preserve the relative success or failure of their rollouts, they could serve as efficient policy evaluators, reducing dependence on repeated real-world testing. However, current literature [[34](https://arxiv.org/html/2607.02642#bib.bib34), [89](https://arxiv.org/html/2607.02642#bib.bib89), [91](https://arxiv.org/html/2607.02642#bib.bib91), [107](https://arxiv.org/html/2607.02642#bib.bib107), [58](https://arxiv.org/html/2607.02642#bib.bib58), [32](https://arxiv.org/html/2607.02642#bib.bib32), [10](https://arxiv.org/html/2607.02642#bib.bib10)] mostly demonstrates that world models _can_ be used for evaluation, while leaving open the more fundamental question of which designs are reliable for building world models as policy evaluator.

This paper addresses that gap. Rather than presenting a single new evaluator and reporting one headline number, we ask a broader scientific question: _what matters in building world models for evaluating robot policies?_ Our aim is to move the field from proof-of-concept demonstrations toward principled design rules. We focus on three concrete questions. First, how should one systematically evaluate whether a world model is a good policy evaluator, beyond generic video quality metrics? Second, how do pretraining and training data affect evaluator quality? Third, which architectural and algorithmic design choices most strongly influence evaluator reliability?

To answer these questions, we construct WMBench, a benchmark centered on paired real-world and world-model rollouts. The benchmark covers eight task families, including rigid and deformable manipulation, and contains both teleoperated expert data and policy rollout data collected from multiple policy checkpoints. This design enables us to measure not only whether a world model generates plausible videos, but whether it preserves the relative outcomes of policies seen in the real world. On top of WMBench, we perform a large-scale controlled study over 7 world models, 4 action representation schemes, 324,000+ world model rollouts, and a diverse set of evaluator metrics. Notably, the study is augmented with community submissions from a public challenge associated with CVPR 2026 1 1 1[https://gigaai-research.github.io/GigaBrain-Challenge-2026/](https://gigaai-research.github.io/GigaBrain-Challenge-2026/), which broadens the model design space beyond our in-house variants.

Our analysis leads to a clear picture. The best world-model evaluators are not simply the models with the most photorealistic frames. They are the models that remain action-faithful over long horizons, preserve pretrained world knowledge under robot-domain adaptation, and expose architectural pathways for stable iterative rollout. Drawing on these key insights, we propose GigaWorld-1, which formalize a roadmap spanning data curation, world model training, and downstream policy evaluation. Empirically, GigaWorld-1 boosts evaluator-alignment metrics by 14.9% compared to competitive state-of-the-art baselines. To facilitate follow-up investigations into scalable evaluation pipelines for embodied foundation models, we fully open-source our code, pre-trained model checkpoints, curated datasets, and auxiliary toolkits.

The contributions of this paper are four-folds:

*   •
We formulate _world model as policy evaluator_ as a first-class research problem and identify the central factors that govern whether a world model can predict policy quality in a way that matches real-world execution.

*   •
We introduce WMBench, a benchmark with human teleoperation trajectories and robot policy rollout data. Our exhaustive experiments on this benchmark cover 7 distinct world models, 4 alternative action representations, 8 robotic manipulation tasks, and over 324,000 evaluation rollouts, from which we distill a set of critical empirical conclusions.

*   •
We provide a systematic empirical study showing how evaluator reliability depends on metric design, pretraining and data composition, and architectural choices such as action representation, memory, and reinforcement-learning-based post-training.

*   •
We summarize these findings into a practical design map and instantiate them in GigaWorld-1, which is trained using over 12000 hours data. Our model outperforms strong baselines by 14.9% on the core evaluator-alignment metric. To foster subsequent research on scalable evaluation for embodied foundation models, we fully open-source our code, pre-trained model weights, curated datasets, and auxiliary toolkits.

## 2 Related Work

### 2.1 Video Diffusion Models as World Models

With the rapid advancement of video generation technologies, a number of powerful foundation models [[110](https://arxiv.org/html/2607.02642#bib.bib110), [39](https://arxiv.org/html/2607.02642#bib.bib39), [133](https://arxiv.org/html/2607.02642#bib.bib133), [99](https://arxiv.org/html/2607.02642#bib.bib99), [30](https://arxiv.org/html/2607.02642#bib.bib30), [95](https://arxiv.org/html/2607.02642#bib.bib95), [104](https://arxiv.org/html/2607.02642#bib.bib104), [25](https://arxiv.org/html/2607.02642#bib.bib25), [65](https://arxiv.org/html/2607.02642#bib.bib65), [64](https://arxiv.org/html/2607.02642#bib.bib64)] have made it possible to generate videos with high visual fidelity, strong perceptual quality, and extended temporal duration. Building upon these capabilities, controllable video generation [[73](https://arxiv.org/html/2607.02642#bib.bib73), [79](https://arxiv.org/html/2607.02642#bib.bib79), [74](https://arxiv.org/html/2607.02642#bib.bib74), [78](https://arxiv.org/html/2607.02642#bib.bib78), [75](https://arxiv.org/html/2607.02642#bib.bib75), [72](https://arxiv.org/html/2607.02642#bib.bib72), [76](https://arxiv.org/html/2607.02642#bib.bib76), [71](https://arxiv.org/html/2607.02642#bib.bib71), [77](https://arxiv.org/html/2607.02642#bib.bib77), [125](https://arxiv.org/html/2607.02642#bib.bib125), [127](https://arxiv.org/html/2607.02642#bib.bib127), [126](https://arxiv.org/html/2607.02642#bib.bib126), [146](https://arxiv.org/html/2607.02642#bib.bib146), [118](https://arxiv.org/html/2607.02642#bib.bib118), [113](https://arxiv.org/html/2607.02642#bib.bib113), [112](https://arxiv.org/html/2607.02642#bib.bib112)] enables generated videos to follow specific control conditions, thereby opening up new possibilities for downstream applications such as autonomous driving [[163](https://arxiv.org/html/2607.02642#bib.bib163), [31](https://arxiv.org/html/2607.02642#bib.bib31), [142](https://arxiv.org/html/2607.02642#bib.bib142), [155](https://arxiv.org/html/2607.02642#bib.bib155), [156](https://arxiv.org/html/2607.02642#bib.bib156)] and embodied intelligence [[36](https://arxiv.org/html/2607.02642#bib.bib36), [103](https://arxiv.org/html/2607.02642#bib.bib103)]. A world model is expected to predict future states by jointly considering the current state and control signals. Video generation models are therefore naturally well suited to serve as foundation models for world modeling [[100](https://arxiv.org/html/2607.02642#bib.bib100), [138](https://arxiv.org/html/2607.02642#bib.bib138), [135](https://arxiv.org/html/2607.02642#bib.bib135), [35](https://arxiv.org/html/2607.02642#bib.bib35), [124](https://arxiv.org/html/2607.02642#bib.bib124), [63](https://arxiv.org/html/2607.02642#bib.bib63)]. Existing studies [[137](https://arxiv.org/html/2607.02642#bib.bib137), [42](https://arxiv.org/html/2607.02642#bib.bib42), [160](https://arxiv.org/html/2607.02642#bib.bib160), [152](https://arxiv.org/html/2607.02642#bib.bib152)] transform control conditions into manually triggered directional inputs, which stimulate the generative model’s ability to predict future states. By further integrating Forcing-style techniques, these approaches enable real-time, indefinitely long video generation with extremely low latency.

### 2.2 World Models for Robotic Learning

World models have emerged as a cornerstone in embodied AI and robotic learning, primarily advancing the field through four distinct paradigms [[162](https://arxiv.org/html/2607.02642#bib.bib162)]. First, acting as _data engines_, world models generate large-scale, diverse synthetic data to scale up policy training [[111](https://arxiv.org/html/2607.02642#bib.bib111), [103](https://arxiv.org/html/2607.02642#bib.bib103), [55](https://arxiv.org/html/2607.02642#bib.bib55), [26](https://arxiv.org/html/2607.02642#bib.bib26), [67](https://arxiv.org/html/2607.02642#bib.bib67), [128](https://arxiv.org/html/2607.02642#bib.bib128), [2](https://arxiv.org/html/2607.02642#bib.bib2), [117](https://arxiv.org/html/2607.02642#bib.bib117), [150](https://arxiv.org/html/2607.02642#bib.bib150), [119](https://arxiv.org/html/2607.02642#bib.bib119), [151](https://arxiv.org/html/2607.02642#bib.bib151)]. Second, functioning as _policies_, they integrate predictive dynamics directly into the action-generation loop, yielding robust end-to-end controllers [[134](https://arxiv.org/html/2607.02642#bib.bib134), [12](https://arxiv.org/html/2607.02642#bib.bib12), [1](https://arxiv.org/html/2607.02642#bib.bib1), [136](https://arxiv.org/html/2607.02642#bib.bib136), [139](https://arxiv.org/html/2607.02642#bib.bib139), [105](https://arxiv.org/html/2607.02642#bib.bib105), [158](https://arxiv.org/html/2607.02642#bib.bib158), [131](https://arxiv.org/html/2607.02642#bib.bib131), [56](https://arxiv.org/html/2607.02642#bib.bib56)]. Third, serving as _interaction environments_, world models provide learned, visually rich testbeds for closed-loop planning or reinforcement learning [[32](https://arxiv.org/html/2607.02642#bib.bib32), [1](https://arxiv.org/html/2607.02642#bib.bib1), [148](https://arxiv.org/html/2607.02642#bib.bib148), [88](https://arxiv.org/html/2607.02642#bib.bib88), [149](https://arxiv.org/html/2607.02642#bib.bib149), [87](https://arxiv.org/html/2607.02642#bib.bib87), [36](https://arxiv.org/html/2607.02642#bib.bib36), [123](https://arxiv.org/html/2607.02642#bib.bib123)]. Fourth, operating as _value critics_, world models assess observations to provide long-horizon return estimates, guiding action selection and bootstrapping reinforcement learning [[51](https://arxiv.org/html/2607.02642#bib.bib51), [102](https://arxiv.org/html/2607.02642#bib.bib102), [70](https://arxiv.org/html/2607.02642#bib.bib70), [157](https://arxiv.org/html/2607.02642#bib.bib157), [27](https://arxiv.org/html/2607.02642#bib.bib27)].

However, these paradigms do not directly answer whether a world model can serve as a reliable _policy evaluator_: an external mechanism for judging whether a policy would successfully complete a task under realistic visual and physical dynamics. This distinction motivates a closer look at policy evaluation itself.

### 2.3 Robot Policy Evaluation

The evaluation of robotic policies has traditionally relied on a spectrum from real-world testing to simulation-based benchmarking. At one end, _real-world evaluation_ provides the most trustworthy assessment by directly measuring execution in target environments [[129](https://arxiv.org/html/2607.02642#bib.bib129), [4](https://arxiv.org/html/2607.02642#bib.bib4)]. However, real-robot testing is notoriously expensive, slow, and difficult to scale comprehensively under broad distribution shifts. To address this scalability bottleneck, _traditional simulators_ offer cheap, repeatable, and safe testing environments, underpinning numerous standard benchmarks [[22](https://arxiv.org/html/2607.02642#bib.bib22), [54](https://arxiv.org/html/2607.02642#bib.bib54), [82](https://arxiv.org/html/2607.02642#bib.bib82), [83](https://arxiv.org/html/2607.02642#bib.bib83), [62](https://arxiv.org/html/2607.02642#bib.bib62)]. Yet, classical simulation often struggles with the sim-to-real gap, particularly in visually complex, contact-rich, or deformable scenarios.

Recently, utilizing _world models as policy evaluators_ has emerged as a compelling alternative to bridge this gap [[32](https://arxiv.org/html/2607.02642#bib.bib32), [1](https://arxiv.org/html/2607.02642#bib.bib1)]. By operating directly in visually open environments and leveraging large-scale spatiotemporal priors, world-model-based evaluators offer the scalability of simulation alongside a higher degree of visual and physical realism. They provide the distinct advantage of scalable, dynamic, and realistic evaluation without the manual modeling effort required by traditional simulators. Rather than asking whether learned evaluation is possible in principle, our study systematically asks what properties make such world-model-based evaluation trustworthy, scalable, and aligned with real-robot outcomes.

## 3 Preliminaries

![Image 2: Refer to caption](https://arxiv.org/html/2607.02642v1/x2.png)

Figure 2: World model as policy evaluator framework. A world model serves as a policy evaluator by iteratively receiving policy actions and predicting future observations. Reliable evaluation requires not only visual quality, but also action-faithful rollout and agreement with real-world policy outcomes.

We consider a robot policy \pi that receives an observation o_{t}, optional robot state s_{t}, and a task instruction l, and outputs an action a_{t}=\pi(o_{t},s_{t},l). In real-world evaluation, the policy interacts with the physical environment and yields a trajectory:

\tau^{\mathrm{real}}=\{(o_{t},s_{t},a_{t})\}_{t=1}^{T},(1)

from which we can estimate task success, failure modes, and other performance signals.

When a world model M_{\theta} is used as an evaluator, the policy instead interacts with a learned environment. Given the initial observation, task instruction, and optionally the robot state, the model predicts future observations conditioned on the policy actions:

\hat{o}_{t+1:t+H}\sim M_{\theta}(\cdot\mid o_{\leq t},s_{\leq t},a_{\leq t},l),(2)

where H denotes the rollout horizon. Iterating this process yields a world-model trajectory:

\tau^{\mathrm{wm}}=\{(\hat{o}_{t},s_{t},a_{t})\}_{t=1}^{H}.(3)

The role of the evaluator is not merely to generate visually plausible observations; it is to preserve the decision-relevant properties of the real trajectory. In particular, for a policy set \{\pi_{i}\}_{i=1}^{N}, we care about whether world-model-based scores preserve the ranking, success prediction, and risk profile observed in the real world.

We therefore define the primary evaluator target as agreement between world-model and real-world policy outcomes. Let S^{\mathrm{real}}(\pi) be the empirical success rate of policy \pi in real rollouts, and let S^{\mathrm{wm}}(\pi) be the predicted success rate inferred from world-model rollouts. A key alignment measure is the ranking correlation:

\rho=\mathrm{Corr}\!\left(S^{\mathrm{real}}(\pi),S^{\mathrm{wm}}(\pi)\right),(4)

evaluated across policies, checkpoints, tasks, or rollout conditions. Throughout the paper, this evaluator-world agreement serves as the central quantity of interest.

## 4 WMBench: A Benchmark for World Models as Policy Evaluators

To rigorously assess whether a world model can faithfully replace real-world physical execution for policy evaluation, we introduce WMBench. This section details its construction logic: we first describe the dataset composition and processing pipelines (Sec. [4.1](https://arxiv.org/html/2607.02642#S4.SS1 "4.1 Data Source ‣ 4 WMBench: A Benchmark for World Models as Policy Evaluators ‣ GigaWorld-1: A Roadmap to Build World Models for Robot Policy Evaluation")), then outline the four-step closed-loop evaluation protocol (Sec. [4.2](https://arxiv.org/html/2607.02642#S4.SS2 "4.2 Evaluation Protocol ‣ 4 WMBench: A Benchmark for World Models as Policy Evaluators ‣ GigaWorld-1: A Roadmap to Build World Models for Robot Policy Evaluation")), and finally define the hierarchical metric system used to measure both generation quality and evaluator reliability (Sec. [4.3](https://arxiv.org/html/2607.02642#S4.SS3 "4.3 Metric System ‣ 4 WMBench: A Benchmark for World Models as Policy Evaluators ‣ GigaWorld-1: A Roadmap to Build World Models for Robot Policy Evaluation")).

### 4.1 Data Source

WMBench is constructed to bridge the gap between visual generation and physical policy evaluation. It also serves as the official benchmark for the GigaBrain Challenge @ CVPR 2026 World Model Track, with its dataset open-sourced on HF Mirror 2 2 2[https://hf-mirror.com/datasets/open-gigaai/CVPR-2026-WorldModel-Track-Dataset](https://hf-mirror.com/datasets/open-gigaai/CVPR-2026-WorldModel-Track-Dataset), accumulating over 50,000 downloads.

Data source. The benchmark corpus comprises 2,989 paired trajectories across eight tasks, drawn from two complementary sources: a teleoperated real-world dataset covering varied manipulations and camera views, and a policy-rollout dataset generated by GigaBrain [[101](https://arxiv.org/html/2607.02642#bib.bib101)] checkpoints containing both successes and failures, the ratio of teleoperated and rollout data is near 1:1. To ensure evaluation integrity, the train-test split strictly enforces _episode-disjointness_ (no test trajectory overlaps with training data), _diversity preservation_ (stressing generalization over memorization), and _outcome balance_ (distinguishing visually similar success/failure cases). After filtering, this yields a final training set of 82,470 seconds and a test set of 7,200 seconds.

Data cleaning. We perform conservative data cleaning before benchmark release. Specifically, we systematically remove corrupted or truncated videos, clips with camera desynchronization, trajectories with missing robot states, and episodes where control timestamps cannot be aligned to observations. Furthermore, rollouts whose outcome labels are ambiguous after human verification are excluded, and near-duplicate teleoperation trajectories are collapsed to reduce redundancy and improve effective data diversity.

Large-scale rollout dataset. To rigorously analyze evaluator reliability (as discussed later in Sec. [5](https://arxiv.org/html/2607.02642#S5 "5 What Matters in Building World Models For Evaluating Robot Policies? ‣ GigaWorld-1: A Roadmap to Build World Models for Robot Policy Evaluation")), we further curated a massive annotated rollout dataset. From the submissions of over 100 participating teams in the CVPR 2026 challenge, we sampled 324,000 world model rollout segments. These segments were sequentially chained together in a closed-loop manner to form complete, long-horizon policy-world model interaction episodes (with each complete episode comprising approximately 20 to 30 rollout segments). Ultimately, these full long-horizon interaction episodes were subjected to meticulous human annotation based on a four-level ordinal scale, which is defined as World Model as Evaluator Score (WMES):

*   •
Score 3 (accurate outcome & high fidelity): The generated rollout successfully predicts the same task outcome (success/failure) as the real-world reference. Additionally, the visual execution is of high quality: the robot action and object states perfectly align with the reference, exhibiting no obvious object distortion and maintaining realistic physics and collision dynamics.

*   •
Score 2 (accurate outcome & low fidelity): The generated rollout predicts the correct final outcome, but the intermediate visual generation is flawed. The process may suffer from noticeable object distortion, unrealistic physics/collisions, or slight misalignment in the robot’s action trajectory.

*   •
Score 1 (incorrect outcome & high fidelity): The generated rollout fails to predict the correct task outcome (e.g., the reference succeeds but the model predicts failure, or vice versa). However, the video itself remains visually and temporally stable, with the robot arm generally following the conditioned action even if the object interaction is incorrect.

*   •
Score 0 (incorrect outcome & low fidelity): The generated rollout completely fails to match the reference outcome and suffers from severe generation collapse. Both the robot action and object state are visually unstable, highly distorted, or physically nonsensical.

To ensure fairness and consistency, each rollout was independently scored by three annotators, with a fourth senior annotator conducting random spot checks. A comprehensive evaluation rubric is available online 3 3 3[https://gigaai-research.github.io/GigaBrain-Challenge-2026/guide/evaluation-rubric.html](https://gigaai-research.github.io/GigaBrain-Challenge-2026/guide/evaluation-rubric.html). These annotated rollouts provide the critical foundation for the experimental analysis in the subsequent sections.

### 4.2 Evaluation Protocol

As illustrated in Figure [3](https://arxiv.org/html/2607.02642#S4.F3 "Fig. 3 ‣ 4.2 Evaluation Protocol ‣ 4 WMBench: A Benchmark for World Models as Policy Evaluators ‣ GigaWorld-1: A Roadmap to Build World Models for Robot Policy Evaluation"), the WMBench evaluation protocol mimics the real-world deployment of a world model as a policy evaluator through four standardized steps:

Step 1: real-world policy data collection. We first train policy models on real-robot data and collect physical closed-loop rollouts from multiple policy checkpoints. For each episode, we record the initial observation, task instruction, multi-view rollout video, and human-annotated success labels.

Step 2: world model training and holdout. World models are trained on the designated training split. The test episodes, including their specific object layouts and initial states, are strictly held out to ensure the world model is evaluated on its generalization ability rather than memorization.

Step 3: closed-loop rollout in world models. Starting from the first frame of a held-out test episode, the target policy predicts an action. The world model then takes this action and the current state to predict future observations across synchronized views. This predicted observation is fed back to the policy for the next step, forming a closed-loop execution until task termination.

Step 4: metric calculation and outcome assessment. The generated trajectories are then comprehensively evaluated using the metric system defined in Sec. [4.3](https://arxiv.org/html/2607.02642#S4.SS3 "4.3 Metric System ‣ 4 WMBench: A Benchmark for World Models as Policy Evaluators ‣ GigaWorld-1: A Roadmap to Build World Models for Robot Policy Evaluation"). This involves two parallel streams: (1) calculating automatic metrics that capture visual fidelity and physical motion dynamics, and (2) assessing the final rollout score (WMES) via a hybrid evaluator (human annotation or VLM). This dual assessment verifies not only the generation quality but also whether the simulated outcome faithfully matches the real-world ground truth.

![Image 3: Refer to caption](https://arxiv.org/html/2607.02642v1/x3.png)

Figure 3: WMBench evaluation pipeline. The four-step protocol includes (1) collecting real-world policy rollouts, (2) training world models on a strict split, (3) executing closed-loop policy rollouts inside the learned world model, and (4) assessing metrics and outcomes to measure alignment with real-world conclusions.

### 4.3 Metric System

Our metric system separates _outcome evaluation_ from _rollout diagnostics_. The outcome-based WMES introduced above measures whether a generated rollout supports task-level decision making. For diagnostic analysis, we select a subset of automatic metrics from WorldArena [[96](https://arxiv.org/html/2607.02642#bib.bib96)], keeping their original definitions and evaluation protocols whenever the metric is shared. These diagnostics describe frame quality, representation fidelity, geometry, semantics, interaction, motion, and long-horizon rollout behavior. We organize them into three families: _frame and representation fidelity_; _geometry, semantics, and interaction_; and _motion and long-horizon rollout_. Among them, Aesthetic Quality, Image Quality, JEPA Similarity, Semantic Alignment, Subject Consistency, and Trajectory Accuracy are the six diagnostics reported in the summary tables of Sec. [6.5](https://arxiv.org/html/2607.02642#S6.SS5 "6.5 Final Evaluation of GigaWorld-1 ‣ 6 Final Design Map and GigaWorld-1 ‣ GigaWorld-1: A Roadmap to Build World Models for Robot Policy Evaluation").

Frame and representation fidelity. These metrics characterize perceptual quality, content stability, and feature-level similarity:

*   •
_Image Quality and Aesthetic Quality._ Image Quality measures frame clarity and technical distortions such as blur, overexposure, noise, and compression artifacts using a MUSIQ-style no-reference image-quality predictor [[49](https://arxiv.org/html/2607.02642#bib.bib49)]. Aesthetic Quality measures visual appeal, lighting, and color composition using the LAION/CLIP aesthetic-predictor protocol [[52](https://arxiv.org/html/2607.02642#bib.bib52)]. We average both scores over frames.

*   •
_JEPA Similarity._ Generated and reference videos are encoded with a frozen V-JEPA video encoder [[11](https://arxiv.org/html/2607.02642#bib.bib11)], and their feature distributions are compared with polynomial-kernel MMD. Higher scores indicate closer high-level spatiotemporal alignment with reference rollouts [[69](https://arxiv.org/html/2607.02642#bib.bib69)].

*   •
_Subject Consistency and Background Consistency._ Subject Consistency measures object stability across frames using DINO-style features, while Background Consistency measures scene stability using CLIP-style features [[19](https://arxiv.org/html/2607.02642#bib.bib19), [92](https://arxiv.org/html/2607.02642#bib.bib92)]. Both compare the current frame with the first and previous frames and apply a dynamic-degree penalty so that near-static rollouts cannot obtain artificially high consistency scores.

*   •
_Photometric Consistency._ Photometric Consistency measures pixel-level texture stability with optical-flow-based average endpoint error. We report the normalized inverse score, with the same dynamic-degree adjustment used to avoid rewarding static videos.

Geometry, semantics, and interaction. These metrics quantify whether the generated rollout preserves spatial structure, task semantics, and action-conditioned effects:

*   •
_Geometry Accuracy and Perspectivity._ Geometry Accuracy compares generated and reference depth maps from a monocular depth estimator after median-based scale alignment [[130](https://arxiv.org/html/2607.02642#bib.bib130)]. Perspectivity is a Qwen3-VL-judged [[6](https://arxiv.org/html/2607.02642#bib.bib6)] 3D-plausibility score over scale variation with depth, lighting consistency, and occlusion relationships, normalized from a Likert-scale output to [0,1].

*   •
_Semantic Alignment and Instruction Following._ Instruction Following uses a Qwen3-VL [[6](https://arxiv.org/html/2607.02642#bib.bib6)] judge to assess whether the rollout matches the requested action type, target object, and task state. Semantic Alignment first uses Qwen2.5-VL [[9](https://arxiv.org/html/2607.02642#bib.bib9)] to produce structured descriptions of the generated and reference videos, then computes normalized CLIP-text similarity between the two descriptions.

*   •
_Interaction Quality._ Interaction Quality uses a prompted Qwen3-VL judge to evaluate robot-object contact, force transfer, friction, inertia, and boundary integrity. The 1–5 Likert score is normalized to [0,1].

*   •
_Trajectory Accuracy._ Trajectory Accuracy evaluates whether the robot arm follows the reference execution path. Arm bounding boxes are extracted with a SAM-style segmentation/detection model [[18](https://arxiv.org/html/2607.02642#bib.bib18)], converted into center trajectories after filtering and interpolation, and compared with the reference trajectory using normalized dynamic time warping (NDTW) [[81](https://arxiv.org/html/2607.02642#bib.bib81)]. Higher scores indicate better spatial-temporal alignment and task-stage ordering.

Motion and long-horizon rollout. These metrics capture short-term motion behavior and long-horizon autoregressive degradation:

*   •
_Dynamic Degree and Flow Score._ Dynamic Degree and Flow Score are RAFT-based optical-flow diagnostics [[106](https://arxiv.org/html/2607.02642#bib.bib106)]. Dynamic Degree focuses on the top 5% highest-motion pixels to capture salient robot/object motion, while Flow Score averages optical-flow magnitude over all pixels and frames to capture global motion intensity.

*   •
_Motion Smoothness._ Motion Smoothness uses a frame-interpolation model to assess temporal coherence: intermediate frames predicted from neighboring frames are compared with real intermediate frames, with motion-aware weighting to avoid over-rewarding static backgrounds [[144](https://arxiv.org/html/2607.02642#bib.bib144)]. Higher scores indicate smoother and more physically coherent motion.

*   •
_PSNR, FID, and FVD._ For long-horizon rollouts, we additionally report standard reconstruction and distributional video-generation metrics: PSNR for pixel-level fidelity, FID [[38](https://arxiv.org/html/2607.02642#bib.bib38)] for frame-level image distribution distance, and FVD [[108](https://arxiv.org/html/2607.02642#bib.bib108)] for video-level spatiotemporal distribution distance.

## 5 What Matters in Building World Models For Evaluating Robot Policies?

### 5.1 Question I: How Should Evaluator Quality Be Assessed?

If world models are to become practical evaluators, we must determine not only which metrics reliably align with WMES, but also what evaluation protocol best reflects real deployment. In this section, we answer three practical questions. First, leveraging the 324,000 manually annotated rollout trajectories introduced in Sec. [4.1](https://arxiv.org/html/2607.02642#S4.SS1 "4.1 Data Source ‣ 4 WMBench: A Benchmark for World Models as Policy Evaluators ‣ GigaWorld-1: A Roadmap to Build World Models for Robot Policy Evaluation"), we systematically analyze the correlation between various visual/motion metrics and WMES. Second, we show that evaluator quality should be judged under long-horizon and OOD rollout settings rather than by short-horizon video generation quality alone. Third, to overcome the bottleneck of human annotation, we introduce a scalable VLM-assisted outcome annotation pipeline, whose predictions achieve near-perfect agreement with human expert WMES rankings.

Formally, for a submitted automatic metric m and the ground-truth WMES score c, we compute the Pearson correlation over the set of valid submissions \Omega_{m} using pairwise deletion:

\rho(m,c)=\frac{\sum_{i\in\Omega_{m}}(m_{i}-\bar{m})(c_{i}-\bar{c})}{\sqrt{\sum_{i\in\Omega_{m}}(m_{i}-\bar{m})^{2}}\sqrt{\sum_{i\in\Omega_{m}}(c_{i}-\bar{c})^{2}}}.(5)

where m_{i} and c_{i} are the metric value and WMES score for the i-th submission, respectively, and \bar{m} and \bar{c} are their sample means. To estimate uncertainty, we compute 95% confidence intervals using non-parametric bootstrap resampling with 10{,}000 iterations. A metric with an upper confidence bound below zero is considered a negative predictor, meaning higher metric scores indicate lower WMES performance.

For a metric group \mathcal{G}, the group-level score is the mean of its metric-level correlations:

\rho(\mathcal{G},c)=\frac{1}{|\mathcal{G}|}\sum_{m\in\mathcal{G}}\rho(m,c).(6)

This protocol makes the evaluation target explicit: a good automatic metric should rank world models similarly to their WMES score.

![Image 4: Refer to caption](https://arxiv.org/html/2607.02642v1/x4.png)

Figure 4: Metric-group correlation with WMES. Metrics submitted to the WMBench are grouped into visual fidelity, geometry, semantics, dynamics, interaction, and appearance stability categories. Visual fidelity and geometry are the strongest group-level predictors, while appearance stability is negatively correlated with WMES.

![Image 5: Refer to caption](https://arxiv.org/html/2607.02642v1/x5.png)

Figure 5: Pearson correlation matrix over all submitted metrics. The full metric-level heatmap shows that Subject Consistency, Perspectivity, JEPA Similarity, Instruction Following, Image Quality, and Aesthetic Quality correlate strongly with WMES, whereas appearance-stability metrics and Interaction Quality are much less reliable.

As shown in Figures [4](https://arxiv.org/html/2607.02642#S5.F4 "Fig. 4 ‣ 5.1 Question I: How Should Evaluator Quality Be Assessed? ‣ 5 What Matters in Building World Models For Evaluating Robot Policies? ‣ GigaWorld-1: A Roadmap to Build World Models for Robot Policy Evaluation") and [5](https://arxiv.org/html/2607.02642#S5.F5 "Fig. 5 ‣ 5.1 Question I: How Should Evaluator Quality Be Assessed? ‣ 5 What Matters in Building World Models For Evaluating Robot Policies? ‣ GigaWorld-1: A Roadmap to Build World Models for Robot Policy Evaluation"), the empirical study yields two critical findings:

Finding 1: visual and geometric fidelity dominate WMES prediction. At the group level, Visual Fidelity has the highest correlation (\rho=0.78), followed by Geometry (\rho=0.71) and Semantics (\rho=0.59). At the individual-metric level, Subject Consistency (\rho=0.88) and Perspectivity (\rho=0.86) are the strongest predictors, followed closely by Instruction Following (\rho=0.84). This suggests that a world model’s ability to act as a reliable evaluator depends primarily on preserving recognizable subjects, viewpoint geometry, and semantic task information, rather than merely generating smooth motion. By contrast, Semantic Alignment alone has a weak correlation (\rho=0.11), showing that high-level semantic labels are insufficient if they do not capture whether the rollout preserves the policy-relevant geometric state.

Finding 2: degenerate metrics mislead evaluator ranking. Surprisingly, we identify several metrics that correlate negatively with WMES, including Background Consistency (\rho=-0.45), Photometric Consistency (\rho=-0.42), and Interaction Quality (\rho=-0.11). The first two fail because a trivial baseline that generates a completely static video can achieve high appearance stability while ignoring all actions and therefore failing entirely at policy evaluation. Interaction Quality is also unreliable because it is obtained by querying a VLM about physical consistency, and current VLMs still cannot judge physical realism robustly enough for fine-grained evaluator ranking. Metrics that do not explicitly penalize action-ignorance or cannot faithfully assess physical consistency can therefore favor degenerate world models over those that correctly model physical interaction.

These correlation results identify which automatic metrics are predictive, but evaluator quality still cannot be reduced to snapshot fidelity alone. Two additional findings emerge when we examine iterative rollout quality and robustness under distribution shift.

Finding 3: evaluator quality must be assessed under long-horizon rollout, not single-step video generation. In practical use, world models are rolled out autoregressively, so small state errors compound over time and can eventually change the policy-level conclusion. We therefore assess long-horizon quality chunk-by-chunk over 40 seconds using PSNR for multi-view reconstruction and FID/FVD for perceptual and temporal quality. As shown in Table [4](https://arxiv.org/html/2607.02642#S5.T4 "Tab. 4 ‣ 5.3 Question III: How Do Model Design Choices Matter? ‣ 5 What Matters in Building World Models For Evaluating Robot Policies? ‣ GigaWorld-1: A Roadmap to Build World Models for Robot Policy Evaluation"), strong short-horizon video generators do not necessarily remain strong evaluators over long rollout horizons. Generic backbones such as Wan, Cosmos, LTX, and SVD often begin with plausible short segments but then suffer from viewpoint drift, object-identity collapse, and texture accumulation, with SVD showing especially severe late-stage degradation. This shows that evaluator quality should be judged by whether a model can preserve actionable state information under repeated autoregressive feedback, rather than by one-step video-generation quality alone.

To scale outcome annotation beyond exhaustive human inspection, we train a VLM evaluator on generated rollout videos. As illustrated in Figure [6](https://arxiv.org/html/2607.02642#S5.F6 "Fig. 6 ‣ 5.1 Question I: How Should Evaluator Quality Be Assessed? ‣ 5 What Matters in Building World Models For Evaluating Robot Policies? ‣ GigaWorld-1: A Roadmap to Build World Models for Robot Policy Evaluation"), each rollout is presented as a synchronized three-view video together with the task-specific prompt, and the evaluator predicts both an ordinal outcome score and structured aspect-level assessments. The core challenge is to ensure that the VLM reflects human judgement on policy outcome rather than overfitting to superficial visual artifacts.

![Image 6: Refer to caption](https://arxiv.org/html/2607.02642v1/)

Figure 6: VLM-assisted Rollout Evaluator. Given a three-view rollout video and a task-specific evaluation prompt, the LoRA-tuned Qwen3-VL evaluator predicts a WMES score and produces evidence-grounded rationales with structured aspect-level assessments of _overall video quality_, _instruction following_, and _physical adherence_.

Finding 4: outcome-centric supervision is essential for scalable VLM evaluation. We fine-tune Qwen3-VL-8B-Instruct using LoRA with structured supervision that couples the overall WMES score with evidence-grounded rationales, including a concise summary rationale and aspect-level assessments of _overall video quality_, _instruction following_, and _physical adherence_. LoRA adapters with rank 16, scaling factor 32, and dropout 0.05 are applied to the attention projection layers. Each rollout video is sampled at 2 fps, with 15–32 frames used as the visual input.

Since rationale tokens greatly outnumber the overall score token, a standard language-modeling objective would under-emphasize the final outcome prediction. Therefore, we adopt a score-focused training objective with token-type-aware loss weighting. Specifically, the overall score token receives the highest weight of 8.0, format and structure-related tokens receive a weight of 1.0 to preserve complete and parseable outputs, and free-form evidence and rationale tokens receive lower length-adaptive weights with a minimum of 0.05 to prevent verbose explanations from dominating optimization.

Finding 5: VLM evaluators achieve near-perfect agreement with human ratings. We evaluate the reliability of VLM-based evaluation by comparing VLM-predicted scores with human ground-truth annotations, as summarized in Table [1](https://arxiv.org/html/2607.02642#S5.T1 "Tab. 1 ‣ 5.1 Question I: How Should Evaluator Quality Be Assessed? ‣ 5 What Matters in Building World Models For Evaluating Robot Policies? ‣ GigaWorld-1: A Roadmap to Build World Models for Robot Policy Evaluation"). The VLM assigns each video a WMES on the same 0–3 ordinal scale used by human annotators, and these video-level predictions are further aggregated to compute method-level WMES. Across 5,000+ videos, the VLM achieves 87.80% exact agreement and 99.16% adjacent agreement, with only 0.84% of videos differing by two score levels. It also obtains low error values, with MAE 0.1304 and RMSE 0.3836, and a quadratic weighted kappa of 0.7349, indicating strong ordinal agreement with human ratings. The predictions further correlate well with human scores, achieving Spearman correlation 0.7574 and Kendall \tau_{b} 0.7507. These results show that VLM-based evaluation closely follows human judgment trends and provides a reliable proxy for method-level WMES comparison.

Acc. \uparrow Adj. Acc. \uparrow Large Err. \downarrow MAE \downarrow RMSE \downarrow QWK \uparrow Spearman \uparrow Kendall \tau_{b}\uparrow W-F1 \uparrow|\mathrm{Bias}|\downarrow
0.8780 0.9916 0.0084 0.1304 0.3836 0.7349 0.7574 0.7507 0.8744 0.0455

Table 1: Overall agreement metrics on 5,000+ videos. We compare VLM-predicted scores against human ground-truth scores. Adjacent accuracy counts predictions within one score level, while large error denotes predictions that differ by two score levels. W-F1 denotes weighted F1.

### 5.2 Question II: How Do Pretraining and Training Data Matter?

The second question concerns the origin of evaluator capability. A natural hypothesis is that larger generative models or models trained on more robot data should always be better evaluators. Our results reveal a more nuanced answer: evaluator quality depends not only on scale, but also on whether the pretrained model contains transferable physical priors and whether post-training preserves them under robot-conditioned rollout.

Finding 6: transferable physical priors matter more than raw scale in pretraining. A controlled pretraining-data ablation across large video foundation models is not realistic in practice, since the original datasets and training pipelines are typically unavailable. We therefore analyze a representative set of open-source video backbones with different scales and pretraining sources. Following the evidence from Question I, we summarize each model using the core evaluator-relevant metrics that best align with WMES, while avoiding appearance-stability metrics that can be misleading for evaluator ranking. The key question is whether evaluator quality comes primarily from parameter count, or from the compatibility between pretrained priors and robot-conditioned rollout. As summarized in Figure [11](https://arxiv.org/html/2607.02642#S6.F11 "Fig. 11 ‣ 6.5.1 Comparison with Open-Source Baselines ‣ 6.5 Final Evaluation of GigaWorld-1 ‣ 6 Final Design Map and GigaWorld-1 ‣ GigaWorld-1: A Roadmap to Build World Models for Robot Policy Evaluation") and Table [9](https://arxiv.org/html/2607.02642#S6.T9 "Tab. 9 ‣ 6.5.1 Comparison with Open-Source Baselines ‣ 6.5 Final Evaluation of GigaWorld-1 ‣ 6 Final Design Map and GigaWorld-1 ‣ GigaWorld-1: A Roadmap to Build World Models for Robot Policy Evaluation"), the evidence favors the latter. Cosmos-Predict2.5 4 4 4 We also plan to evaluate Cosmos-3 [[1](https://arxiv.org/html/2607.02642#bib.bib1)]. However, its multiview version has not yet been publicly released. We are currently in discussion with the Cosmos team, and will update the comparison once access to the latest multiview model becomes available.[[3](https://arxiv.org/html/2607.02642#bib.bib3)] is the strongest baseline (AVG 0.6123), suggesting that robotics and autonomous-driving pretraining provides useful physical priors. Among general-purpose backbones, Wan 2.2 5B [[109](https://arxiv.org/html/2607.02642#bib.bib109)] is the strongest (AVG 0.5948) and still surpasses the much larger LTX 2.3 (22B, AVG 0.5775) [[33](https://arxiv.org/html/2607.02642#bib.bib33)], while CogVideoX [[132](https://arxiv.org/html/2607.02642#bib.bib132)] remains intermediate (AVG 0.5620). By contrast, SVD [[15](https://arxiv.org/html/2607.02642#bib.bib15)] performs worst overall (AVG 0.5569) and especially struggles on Trajectory Accuracy (0.0926), indicating weak temporal and action-related dynamics despite similar scale. These comparisons show that a larger model is not automatically a better evaluator; what matters is whether the pretrained model contains transferable physical priors that can be effectively adapted to robot-conditioned prediction.

Table 2: Per-metric breakdown of training-data composition ablation. Values show absolute scores. Delta (\Delta) indicates the change from the GigaData-only baseline.

Data Recipe Aesthetic Image Quality JEPA Sim.Photo Cons.Semantic Subject Trajectory Average
GigaData 0.34355 0.6904 0.8141 0.3187 0.8896 0.7212 0.2566 0.5654
GigaData + AgiBot 0.3802 0.7042 0.5715 0.6218 0.8710 0.8613 0.1482 0.5940
\Delta+0.03665+0.0138-0.2426+0.3031-0.0186+0.1401-0.1084+0.0286
GigaData + PhysData 0.3388 0.6974 0.7678 0.6261 0.8798 0.7409 0.2497 0.6144
\Delta-0.00475+0.0070-0.0463+0.3074-0.0098+0.0197-0.0069+0.0490

Finding 7: broad physical videos provide the best overall trade-off for evaluator training. To study post-training data composition, we adopt Wan2.1 1.3B [[109](https://arxiv.org/html/2607.02642#bib.bib109)] as a common backbone and continue training it with different data mixtures. Here, GigaData denotes our curated Giga-collected robot demonstrations with calibrated robot trajectories and multi-view observations, while PhysData denotes the internet and physics video corpus that provides broad visual-physical priors (see Sec. [6.1](https://arxiv.org/html/2607.02642#S6.SS1 "6.1 Data Sources and Data Curation ‣ 6 Final Design Map and GigaWorld-1 ‣ GigaWorld-1: A Roadmap to Build World Models for Robot Policy Evaluation") for the full data taxonomy). Following Question I, we interpret this ablation mainly through the metrics that are most predictive of evaluator quality—especially subject/geometry-related fidelity and the overall retained-metric average—while treating appearance-stability signals only as auxiliary diagnostics. A GigaData-only recipe is naturally a strong baseline because its distribution is closest to the benchmark test domain. However, relying on GigaData alone can also overfit the model to a narrow robot setting and gradually erase the broader world knowledge inherited from pretraining. This is exactly where PhysData helps: robotic manipulation involves diverse contact patterns, object dynamics, and scene changes, and general physical videos reinforce the underlying world knowledge needed to model such interactions. As shown in Table [2](https://arxiv.org/html/2607.02642#S5.T2 "Tab. 2 ‣ 5.2 Question II: How Do Pretraining and Training Data Matter? ‣ 5 What Matters in Building World Models For Evaluating Robot Policies? ‣ GigaWorld-1: A Roadmap to Build World Models for Robot Policy Evaluation"), GigaData + PhysData yields the strongest overall result, improving the average score from 0.5654 to 0.6144 (+0.0490). The largest gain comes from Photometric Consistency (+0.3074), together with smaller improvements in Image Quality and Subject Consistency, indicating that broad physical videos improve realism and robustness without moving the model too far away from the target robot distribution. Although JEPA Similarity, Semantic Alignment, and Trajectory Accuracy decrease slightly, PhysData still provides the best overall trade-off because it restores general world knowledge while preserving the task-relevant bias already present in GigaData.

Finding 8: robot-specific data mainly improves embodiment fidelity, but introduces a sharper trade-off. Adding AgiBot data also improves the overall score, from 0.5654 to 0.5940 (+0.0286), but the gain is more selective. In light of Question I, the key evidence here is not the rise of appearance-oriented metrics by itself, but the trade-off between embodiment-sensitive gains such as Subject Consistency and structurally important metrics such as JEPA Similarity and Trajectory Accuracy. AgiBot substantially improves Aesthetic Quality (+0.0367), Photometric Consistency (+0.3031), and especially Subject Consistency (+0.1401), showing that robot-specific data is valuable for embodiment and camera-view fidelity. However, its diversity is still narrower than PhysData, and introducing demonstrations from other robot embodiments can create additional mismatch when the evaluation target is already well covered by GigaData. This is reflected in the much larger drop in JEPA Similarity (-0.2426) and Trajectory Accuracy (-0.1084), indicating that narrow robot-domain data can over-specialize the generator and weaken broader structural or motion-related generalization. In other words, AgiBot can be helpful when evaluating on closely related robot platforms, but under the current GigaData-centered setting, adding broad physical videos is the better choice. The broader lesson is that evaluator-oriented training is inherently a balancing problem: robot data is useful for embodiment refinement, but broad physical data provides a better overall trade-off for reliable policy evaluation.

Table 3: Control-condition ablation. All metrics are higher-is-better (\uparrow). Cell colors indicate per-metric ranking: green = best, yellow = second best, and red = worst.

Method Control Type Traj. Acc. \uparrow Dynamic \uparrow Smooth \uparrow Flow \uparrow Subject \uparrow Photo. \uparrow
Wan 2.1 1.3B I2V None 0.1576 0.2429 0.4997 0.0971 0.5568 0.2185
Wan 2.1 1.3B Control Cross-attention 0.1620 0.1049 0.4525 0.0624 0.3573 0.1853
Wan 2.1 1.3B Control ControlNet 0.2566 0.3083 0.5197 0.1412 0.7212 0.3187
Wan 2.1 1.3B Control Channel concat 0.3528 0.3566 0.5747 0.2179 0.8600 0.3206

### 5.3 Question III: How Do Model Design Choices Matter?

The third question examines how architectural choices affect evaluator reliability once the data pipeline is fixed. We focus on two design axes that directly determine whether a world model can support faithful policy assessment: the action-control interface and long-horizon memory.

Finding 9: Action control must be injected through a spatially aligned interface. The interface between policy actions and the world model is central. A visually plausible rollout can still be a poor evaluator if the robot or object trajectory does not follow the intended action. We therefore compare four control interfaces: no explicit control, cross-attention control, ControlNet-style spatial control, and channel-concatenated control maps.

As shown in Table [3](https://arxiv.org/html/2607.02642#S5.T3 "Tab. 3 ‣ 5.2 Question II: How Do Pretraining and Training Data Matter? ‣ 5 What Matters in Building World Models For Evaluating Robot Policies? ‣ GigaWorld-1: A Roadmap to Build World Models for Robot Policy Evaluation"), and consistent with Question I, the primary judge of action control is Trajectory Accuracy, while Dynamic Degree, Motion Smoothness, Flow Score, and Subject Consistency serve as auxiliary diagnostics for whether better control also preserves coherent interaction. Cross-attention provides only a marginal improvement in Trajectory Accuracy over the image-to-video baseline (0.1620 vs. 0.1576) and degrades related motion metrics, suggesting that attention-side action tokens are easily overwhelmed by appearance and semantic tokens. ControlNet-style conditioning is stronger, improving Trajectory Accuracy to 0.2566 because the control signal enters as a spatial feature.

The strongest result comes from channel-concatenated control maps, which achieve the best score across all paired metrics, including Trajectory Accuracy (0.3528), Dynamic Degree (0.3566), Motion Smoothness (0.5747), Flow Score (0.2179), Subject Consistency (0.8600), and Photometric Consistency (0.3206). This indicates that the most reliable action representation is not merely explicit, but spatially aligned with the noisy latent from the beginning of denoising. Concretely, as detailed in Sec. [6.2](https://arxiv.org/html/2607.02642#S6.SS2 "6.2 Architecture and Control Interface ‣ 6 Final Design Map and GigaWorld-1 ‣ GigaWorld-1: A Roadmap to Build World Models for Robot Policy Evaluation"), the control is implemented as a unified pixel-aligned representation derived from calibrated robot and camera geometry: the head view uses a rendered end-effector pose map to encode manipulation intent, while the wrist views use ray maps to encode view-dependent camera motion. These view-specific controls are then encoded into a shared control latent and concatenated with the noisy video latent throughout autoregressive generation, which is particularly important in multi-view settings where camera motion and object motion are easily confounded.

Table 4: Long-horizon rollout quality averaged every 8 seconds. Each 8-second interval averages eight 10-frame chunks. Cell colors indicate per-interval ranking across models: green = best, yellow = second best, and red = worst.

Model Metric 0–8s 8–16s 16–24s 24–32s 32–40s
SVD PSNR \uparrow 14.05 8.71 7.24 6.82 6.88
FID \downarrow 142.84 381.25 423.45 422.32 419.21
FVD \downarrow 173.63 350.12 433.91 444.74 443.76
Cosmos2.5 PSNR \uparrow 13.65 13.75 13.40 13.13 12.83
FID \downarrow 235.74 253.58 253.26 243.52 260.84
FVD \downarrow 203.03 201.03 242.69 271.12 289.74
LTX-Video PSNR \uparrow 13.38 13.67 12.98 12.82 12.74
FID \downarrow 257.83 320.51 293.35 288.67 295.33
FVD \downarrow 217.17 218.13 232.83 236.50 260.78
Wan 2.2 PSNR \uparrow 14.35 11.57 10.97 10.53 10.09
FID \downarrow 216.02 253.02 245.95 245.64 266.63
FVD \downarrow 169.22 183.10 231.50 259.89 300.10
Wan 2.1 PSNR \uparrow 14.46 13.63 13.63 13.49 13.37
FID \downarrow 219.67 344.39 333.66 326.30 316.77
FVD \downarrow 197.46 264.74 284.11 302.71 320.52
Wan 2.1+Mem.PSNR \uparrow 19.82 17.01 17.52 17.51 17.41
FID \downarrow 40.58 95.22 111.78 112.66 121.61
FVD \downarrow 35.30 68.87 76.15 84.56 98.34

Finding 10: reliable evaluators require persistent memory for long-horizon rollout. Iterative rollout creates temporal accumulation error: each generated window becomes part of the next conditioning context, so small visual or geometric mistakes can be amplified into large state errors. This is especially harmful for policy evaluation, because a rollout may look plausible in the first few frames but produce the wrong policy conclusion once object identity, camera geometry, or contact state drifts. We therefore evaluate long-horizon rollout quality chunk by chunk over 40 seconds, using PSNR for multi-view reconstruction and FID/FVD for perceptual and temporal quality. This metric choice directly follows Finding 3: once the evaluation target becomes repeated autoregressive rollout rather than single-step generation, temporal and long-horizon fidelity become the relevant evidence. As shown in Table [4](https://arxiv.org/html/2607.02642#S5.T4 "Tab. 4 ‣ 5.3 Question III: How Do Model Design Choices Matter? ‣ 5 What Matters in Building World Models For Evaluating Robot Policies? ‣ GigaWorld-1: A Roadmap to Build World Models for Robot Policy Evaluation"), adding memory on top of the Wan 2.1 1.3B backbone substantially improves long-horizon rollout quality across all intervals. Concretely, the memory is implemented as a hierarchical history buffer with a persistent first-frame anchor and short-, mid-, and long-range temporal memories, so the model retains both the original scene identity and recent motion context; the full implementation is described in Sec. [6.2](https://arxiv.org/html/2607.02642#S6.SS2 "6.2 Architecture and Control Interface ‣ 6 Final Design Map and GigaWorld-1 ‣ GigaWorld-1: A Roadmap to Build World Models for Robot Policy Evaluation"). The long-horizon comparison also reveals several failure modes in memory-free or generic video models, including viewpoint drift, object-identity collapse, exposure artifacts, and eventual degeneration into visually meaningless frames. Qualitatively, memory reduces abrupt background jumps and preserves long-term scene identity; without it, the rollout may begin plausibly but gradually lose the state information needed for correct policy evaluation. These results show that short-horizon video quality can be misleading, and that persistent memory is necessary for reliable long-horizon evaluation.

## 6 Final Design Map and GigaWorld-1

The empirical study above can be summarized as a data-to-model-to-evaluation design map for world models as policy evaluators. At the data level, evaluator quality depends on balancing generic world knowledge with robot-specific controllability. At the model level, the best designs expose explicit low-level action representation, preserve spatial alignment, and include memory to stabilize long-horizon rollout. At the evaluation level, the decisive target is not visual realism in isolation, but agreement with real-world policy success under both in-distribution and OOD conditions.

We instantiate these principles in GigaWorld-1, our final evaluator-oriented world model. As shown in Table [5](https://arxiv.org/html/2607.02642#S6.T5 "Tab. 5 ‣ 6 Final Design Map and GigaWorld-1 ‣ GigaWorld-1: A Roadmap to Build World Models for Robot Policy Evaluation"), GigaWorld-1 is built from Wan [[109](https://arxiv.org/html/2607.02642#bib.bib109)] backbones of two scales, [1.3B] and [5B], and uses a training corpus composed of real robot trajectories, policy rollout data, egocentric videos, simulation-derived data, and challenge-submitted generated rollouts after quality filtering. The comparisons in Sec. [5](https://arxiv.org/html/2607.02642#S5 "5 What Matters in Building World Models For Evaluating Robot Policies? ‣ GigaWorld-1: A Roadmap to Build World Models for Robot Policy Evaluation") suggest that both Cosmos and Wan provide strong pretrained priors for evaluator-oriented world modeling. We ultimately adopt Wan as the backbone because it is the strongest general-purpose open backbone in our controlled comparison, while also offering a more mature ecosystem for redesign and engineering. Data cleaning removes corrupted and duplicated clips, while data labeling includes success annotation, action synchronization, and quality control for multi-view consistency. On the modeling side, GigaWorld-1 combines the recommended explicit control interface, hierarchical memory, relative temporal encoding, and a progressive multi-stage training pipeline to improve evaluator stability under long-horizon rollout.

Table 5: Design map of GigaWorld-1.

Component Design choice Notes
Backbone Wan-[1.3B / 5B]open baseline with competitive performance and a mature ecosystem
Training data physical videos + open-source robot + egocentric + Giga-collected data broad physical priors with embodiment diversity
Data curation quality + motion + distribution filtering removes noisy, static, and misaligned samples
Structured supervision semantic masks + depth + fast-slow captions improves geometry and task grounding
Action interface explicit pixel-aligned representation EE pose maps + ray maps with released calibration toolkit
Long-horizon module memory-augmented rollout first-frame anchor + hierarchical history
Temporal encoding Relative RoPE reduces position drift in long autoregressive rollout
Training recipe progressive multi-stage training foundation pretraining + AR learning + optional scene LoRA + distillation

### 6.1 Data Sources and Data Curation

#### 6.1.1 Data Composition

The success of large-scale embodied foundation models relies heavily on the quality, diversity, and scale of training data. The ablation results in Sec. [5](https://arxiv.org/html/2607.02642#S5 "5 What Matters in Building World Models For Evaluating Robot Policies? ‣ GigaWorld-1: A Roadmap to Build World Models for Robot Policy Evaluation") show that broad physical-video priors are especially helpful for evaluator training: adding PhysData provides the best overall trade-off by improving realism and robustness while preserving the task-relevant bias of robot data. Guided by this finding, and further motivated by the need for broader embodiment and scene diversity, we construct a heterogeneous corpus of approximately 12,980 hours from four complementary sources: internet and physics videos, open-source robot datasets, human-centric egocentric data, and Giga-collected robot demonstrations, as illustrated in Figure [7](https://arxiv.org/html/2607.02642#S6.F7 "Fig. 7 ‣ 6.1.1 Data Composition ‣ 6.1 Data Sources and Data Curation ‣ 6 Final Design Map and GigaWorld-1 ‣ GigaWorld-1: A Roadmap to Build World Models for Robot Policy Evaluation") and summarized in Table [6](https://arxiv.org/html/2607.02642#S6.T6 "Tab. 6 ‣ 6.1.1 Data Composition ‣ 6.1 Data Sources and Data Curation ‣ 6 Final Design Map and GigaWorld-1 ‣ GigaWorld-1: A Roadmap to Build World Models for Robot Policy Evaluation").

Specifically, the corpus includes about 1,298 hours of internet and physics videos for generic physical dynamics, 5,377 hours of open-source robot demonstrations, 2,411 hours of egocentric human-hand data, and 3,894 hours of Giga-collected humanoid and dual-arm demonstrations. This composition covers both broad visual-physical priors and embodiment-specific manipulation behaviors across humanoid robots, dual-arm manipulators, single-arm manipulators, and dexterous hands.

![Image 7: Refer to caption](https://arxiv.org/html/2607.02642v1/x7.png)

Figure 7: Data construction pipeline. Multi-source data is filtered, balanced, and automatically annotated before being incorporated into the world-model training corpus.

Table 6: Overview of the training corpus.

Category Representative Sources Views Robot Type Hours Modality
Physical Data Internet Videos, Physics Videos Single view N/A\sim 1298 RGB Video
Open-source Robot Data Open X, AgiBot Single & Multi Views Single-arm, Dual-arm and Humanoid\sim 5377 Robot Demonstration
Human-centric Data EgoDex, SynData Ego-centric Human Hands\sim 2411 RGB + Hand Pose
Giga-collected Data Giga Humanoid, Giga Dual-arm Single & Multi Views Humanoid, Dual-arm\sim 3894 Robot Demonstration
Total Mixed Mixed Multi Types\sim 12,980 Multi-modal

#### 6.1.2 Data Curation

Raw aggregation alone is insufficient for training physically consistent world models, as the four data sources above differ substantially in visual style and supervision. Internet and physics videos provide broad physical priors but contain noisy web artifacts; open-source robot datasets such as AgiBot [[17](https://arxiv.org/html/2607.02642#bib.bib17)], RoboMind [[122](https://arxiv.org/html/2607.02642#bib.bib122)], and Galaxea [[45](https://arxiv.org/html/2607.02642#bib.bib45)] cover diverse embodiments but use inconsistent camera and action formats; egocentric human videos emphasize hand-object interaction but lack robot trajectories; and Giga-collected demonstrations provide calibrated robot trajectories. We therefore apply a video-level quality gate followed by semantic filtering to remove corrupted, low-quality, uninformative, or misaligned samples before balancing the final training mixture.

Video quality filter. We formulate video quality filtering as a clip-level acceptance test over decoded frames. For a video clip v=\{x_{t}\}_{t=1}^{T}, we first verify metadata consistency, frame decodability, timestamp monotonicity, and resolution validity; clips with codec failures, unreadable frames, duplicate-frame collapse, or abnormal aspect ratios are removed. From K uniformly sampled frames, we compute a low-level image-quality vector:

q(x_{i})=\big[s(x_{i}),e(x_{i}),n(x_{i}),c(x_{i}),b(x_{i})\big],(7)

where s,e,n,c,b denote sharpness, exposure validity, noise level, contrast, and compression/block-artifact scores, respectively. The aggregate image-quality score is:

Q_{\mathrm{img}}(v)=\frac{1}{K}\sum_{i=1}^{K}w^{\top}q(x_{i}),(8)

and clips are rejected when Q_{\mathrm{img}}(v)<\tau_{\mathrm{img}}, which removes videos with dirty or blurred frames, severe under/over-exposure, sensor noise, compression artifacts, or other degradations that make physical state estimation unreliable. In parallel, an aesthetic-semantic scorer A(v) filters web videos with irrelevant overlays, extreme occlusion, poor framing, or non-manipulation content.

We further evaluate temporal integrity by comparing adjacent sampled frames in both histogram and embedding spaces:

D_{t}=\lambda_{h}\bigl(1-\operatorname{sim}(h_{t},h_{t+1})\bigr)+\lambda_{\phi}\bigl(1-\cos(\phi_{t},\phi_{t+1})\bigr),(9)

where h_{t} is a color histogram and \phi_{t} is a visual embedding. Large spikes in D_{t} indicate scene jumps, stitching errors, black screens, or dropped frames, while persistently small changes indicate frozen or static clips. The final video-level gate is:

\mathbbm{1}_{\mathrm{keep}}(v)=\mathbbm{1}\!\left[Q_{\mathrm{img}}(v)\geq\tau_{\mathrm{img}},A(v)\geq\tau_{\mathrm{aes}},\max_{t}D_{t}\leq\tau_{\mathrm{jump}},\operatorname{Var}_{t}(D_{t})\geq\tau_{\mathrm{static}}\right].(10)

Videos that pass this gate are then constrained to training-compliant temporal windows; overly short clips are removed, while long demonstrations are segmented by task index or temporal boundaries so that each resulting clip preserves a coherent manipulation attempt.

Motion and trajectory filter. After visual quality filtering, we further evaluate whether a clip contains physically meaningful manipulation dynamics. Given consecutive frames (x_{t},x_{t+1}), we estimate a dense optical-flow field F_{t}(u)\in\mathbb{R}^{2} over pixels u\in\Omega and define the frame-level motion magnitude as:

M_{t}=\frac{1}{|\Omega|}\sum_{u\in\Omega}\|F_{t}(u)\|_{2}.(11)

The clip-level kinematic score is then computed as M(v)=\frac{1}{T-1}\sum_{t=1}^{T-1}M_{t}. Clips with M(v)<\tau_{\mathrm{motion}} are removed as static or uninformative, except for task segments explicitly labeled as waiting, holding, or contact stabilization. To reject physically implausible trajectories, we additionally penalize high-frequency motion artifacts using the temporal acceleration of the flow signal:

J(v)=\frac{1}{T-2}\sum_{t=2}^{T-1}|M_{t+1}-2M_{t}+M_{t-1}|,(12)

and discard clips with J(v)>\tau_{\mathrm{jerk}}, which captures abrupt oscillations, discontinuous camera/robot motion, and unstable hand-object contacts. For robot demonstrations, we further project calibrated action maps, including joint commands, end-effector poses, and gripper states, onto video frames. A vision-language verifier then checks whether the observed object and robot motion are consistent with the action stream, filtering samples with synchronization errors, calibration drift, or action-observation mismatches.

Distribution filter. After filtering, the remaining clips are hierarchically balanced across source type, embodiment, camera view, task family, and motion intensity. The curated data is then annotated with task descriptions, success labels, action synchronization metadata, and quality tags before being used for world-model training.

#### 6.1.3 Giga DataCrafter

After curation, Giga DataCrafter converts raw videos into structured supervision for world-model training. For each retained clip v, the system produces three synchronized annotation streams: semantic masks, monocular depth, and language descriptions. These annotations expose object-level geometry and task semantics without requiring expensive manual labeling.

Semantic map annotation. We use Segment Anything Model 2 (SAM2) [[93](https://arxiv.org/html/2607.02642#bib.bib93)] to obtain frame-level semantic masks for manipulable objects, robot arms, hands, and task-relevant background regions. Given a sampled frame x_{t}, SAM2 produces a set of masks:

\mathcal{S}_{t}=\{(m_{t}^{k},c_{t}^{k})\}_{k=1}^{K_{t}},(13)

where m_{t}^{k}\in\{0,1\}^{H\times W} is the binary mask and c_{t}^{k} is the corresponding semantic category or region tag. Masks are propagated and checked across neighboring frames to improve temporal consistency, yielding object-centric supervision for contact, occlusion, and spatial grounding.

Depth annotation. We apply Depth Anything 3 (DA3) [[60](https://arxiv.org/html/2607.02642#bib.bib60)] to estimate a dense depth map d_{t}\in\mathbb{R}^{H\times W} for each annotated frame. The resulting depth stream provides geometric cues for hand-object distance, object support relations, and camera-view consistency. For multi-view robot data, depth predictions are further normalized per camera and checked against calibration metadata when available, producing a unified geometric signal across embodiments and viewpoints.

Caption annotation. A Vision-Language Model (VLM) [[7](https://arxiv.org/html/2607.02642#bib.bib7), [8](https://arxiv.org/html/2607.02642#bib.bib8), [115](https://arxiv.org/html/2607.02642#bib.bib115), [115](https://arxiv.org/html/2607.02642#bib.bib115)] understands video content [[21](https://arxiv.org/html/2607.02642#bib.bib21)] and generates language annotations through a fast–slow captioning system. The fast stream produces high-frequency short-term captions for local subtasks, such as reaching, grasping, lifting, placing, tool use, gripper-object contact, and object displacement. The slow stream produces low-frequency long-term captions that describe the environment in detail, including scene layout, object attributes, workspace constraints, and persistent task context. Formally, each clip is annotated as:

\mathcal{C}(v)=\{\mathcal{C}_{\mathrm{short}}(v),\mathcal{C}_{\mathrm{long}}(v)\},(14)

where \mathcal{C}_{\mathrm{short}} is updated on short local windows and \mathcal{C}_{\mathrm{long}} is updated sparsely and shared across longer temporal spans. This fast-slow design avoids repeatedly invoking large VLMs during world-model training: captions are computed offline once, cached with the video, and then reused as lightweight conditioning signals, reducing GPU cost while preserving both high-frequency subtask semantics and low-frequency environmental context.

### 6.2 Architecture and Control Interface

GigaWorld-1 is designed as an evaluator-oriented world model rather than a generic video generator. Its architecture follows three principles derived from the design map in Figure [8](https://arxiv.org/html/2607.02642#S6.F8 "Fig. 8 ‣ 6.2 Architecture and Control Interface ‣ 6 Final Design Map and GigaWorld-1 ‣ GigaWorld-1: A Roadmap to Build World Models for Robot Policy Evaluation"): preserve the spatiotemporal priors of a large pretrained video backbone, inject robot actions through an explicit and geometrically aligned interface, and maintain state across iterative rollouts so that long-horizon policy outcomes remain stable. To retain pretrained generative priors while adapting efficiently to robot domains, the VAE, text encoder, and frozen backbone components are reused, while trainable LoRA adapters [[41](https://arxiv.org/html/2607.02642#bib.bib41)] and lightweight control pathways specialize the autoregressive diffusion transformer.

![Image 8: Refer to caption](https://arxiv.org/html/2607.02642v1/x8.png)

Figure 8: Overall architecture of GigaWorld-1. The model is built as an autoregressive diffusion-transformer world generator with parameter-efficient LoRA adaptation. Historical frames are encoded through memory patchification, future noisy latents are encoded through patchification, and structured controls such as actions, depth, semantic maps, and captions are injected as temporally aligned conditions. Frozen VAE and text-encoder components preserve pretrained visual and semantic priors, while trainable LoRA adapters and control modules adapt the DiT backbone to embodied rollout generation. Generated windows are decoded by the VAE and appended back into the rollout history for subsequent prediction.

Figure [8](https://arxiv.org/html/2607.02642#S6.F8 "Fig. 8 ‣ 6.2 Architecture and Control Interface ‣ 6 Final Design Map and GigaWorld-1 ‣ GigaWorld-1: A Roadmap to Build World Models for Robot Policy Evaluation") summarizes the overall architecture of GigaWorld-1. The model extends a pretrained video diffusion backbone into an autoregressive world generator by combining memory patchification, temporally aligned control injection, hierarchical history guidance, relative temporal position encoding, and LoRA-based parameter-efficient adaptation. This design keeps the pretrained VAE and language encoder fixed, while concentrating robot-domain learning in the AR DiT adapters and control branches.

#### 6.2.1 Autoregressive World Generation

Most video diffusion backbones are trained as bidirectional denoising models over fixed-length clips, which limits their ability to synthesize long-horizon interaction trajectories. Inspired by [[138](https://arxiv.org/html/2607.02642#bib.bib138)], we reformulate world generation as an autoregressive video-continuation problem. Given a historical latent sequence:

X_{\mathrm{hist}}=\{x_{1},x_{2},\ldots,x_{t}\},(15)

and a future latent window:

X_{\mathrm{future}}=\{x_{t+1},x_{t+2},\ldots,x_{t+T_{f}}\},(16)

the model learns the conditional distribution:

p(X_{\mathrm{future}}\mid X_{\mathrm{hist}},C),(17)

where C denotes the control condition, including action maps, semantic maps, depth, captions, or task context. During training, the future window is corrupted by the diffusion process:

X_{\tau}=\alpha_{\tau}X_{\mathrm{future}}+\sigma_{\tau}\epsilon,\qquad\epsilon\sim\mathcal{N}(0,I),(18)

and the network predicts the denoising target:

\epsilon_{\theta}=f_{\theta}(X_{\tau},H_{t},C,\tau),(19)

where H_{t} denotes the historical memory state. During inference, each generated future segment is appended to the history buffer and used for the next prediction step, yielding the factorization:

p(X_{1:T}\mid C)=\prod_{k=1}^{K}p(X_{k}\mid H_{k-1},C).(20)

This converts a fixed-window diffusion model into an autoregressive world generator that can produce long-horizon rollouts while preserving the original denoising-based sampling procedure.

#### 6.2.2 Unified Control Injection

To support controllable robot world generation, we introduce a unified control representation that explicitly incorporates both robot manipulation intent and camera geometry [[90](https://arxiv.org/html/2607.02642#bib.bib90), [59](https://arxiv.org/html/2607.02642#bib.bib59)]. Since different camera views exhibit distinct motion characteristics, we employ view-specific control signals while projecting them into a shared latent space.

EE pose control. The head camera is approximately static with respect to the robot base. Consequently, future scene evolution is primarily determined by robot actions rather than camera motion. For this view, we adopt an end-effector (EE) pose map as the control signal. Specifically, future end-effector trajectories are projected into the image plane, producing a spatially aligned representation that encodes arm position, orientation, and gripper state [[120](https://arxiv.org/html/2607.02642#bib.bib120)]. This representation provides an explicit description of future manipulation intent and enables the model to associate robot actions with subsequent scene changes [[29](https://arxiv.org/html/2607.02642#bib.bib29)].

Ray map control. Unlike the head camera, wrist cameras are rigidly attached to the robot arms and move together with the end effectors. In this setting, most visual variations are caused by viewpoint changes rather than object motion. Therefore, we use a ray map as the control signal for wrist views [[153](https://arxiv.org/html/2607.02642#bib.bib153)]. For each pixel, the ray map stores the ray origin and normalized ray direction in the world coordinate system, providing an explicit representation of camera geometry [[5](https://arxiv.org/html/2607.02642#bib.bib5)]. Since the ray map changes synchronously with robot motion, it helps the model distinguish appearance changes induced by camera motion from those caused by scene dynamics.

Let C_{t}^{\mathrm{ee}} be the rendered EE pose map for the head view, and C_{t}^{\mathrm{ray}} be the ray map for the wrist view. We concatenate them along the image-width dimension to obtain a unified control map:

C_{t}=\operatorname{Concat}_{W}(C_{t}^{\mathrm{ee}},C_{t}^{\mathrm{ray}}),(21)

where \operatorname{Concat}_{W}(\cdot) denotes width-wise concatenation. The control sequence C=\{C_{t}\}_{t=1}^{T} is then encoded into the latent control representation:

Z_{\mathrm{ctrl}}=E(C).(22)

During autoregressive generation, the k-th generation window uses the temporally aligned control segment:

Z_{\mathrm{ctrl}}^{(k)}=Z_{\mathrm{ctrl}}[t_{k}:t_{k}+T_{f}],(23)

where T_{f} is the temporal length of the current generation chunk. The denoising network predicts the noise conditioned on the noisy latent, the aligned control latent, the hierarchical history memory, and the diffusion timestep:

\epsilon_{\theta}=f_{\theta}(X_{\tau},Z_{\mathrm{ctrl}}^{(k)},H_{t},\tau).(24)

Unlike one-shot conditioning, this formulation continuously provides temporally aligned control signals throughout autoregressive generation. By combining EE pose control for static head views and ray map control for dynamic wrist views in a unified latent representation, the model jointly captures robot kinematics, camera geometry, and scene dynamics, enabling accurate long-horizon prediction of robot-object interactions.

#### 6.2.3 Hierarchical History Injection

Autoregressive generation must preserve both short-term motion continuity and long-term scene consistency under a bounded context budget. Instead of conditioning on all previous frames, we represent history with a multi-scale memory:

H_{t}=\{H_{t}^{(L)},H_{t}^{(M)},H_{t}^{(S)}\},(25)

where H_{t}^{(S)} captures local motion continuity and recent scene changes, H_{t}^{(M)} stores intermediate action evolution and object interactions, and H_{t}^{(L)} preserves global scene layout, object identity, and environmental state.

To mitigate identity, color, and scene drift, we additionally keep the first-frame latent x_{\mathrm{anchor}} as a persistent anchor. The final history state is:

\widetilde{H}_{t}=\{x_{\mathrm{anchor}},H_{t}^{(L)},H_{t}^{(M)},H_{t}^{(S)}\}.(26)

The denoising network is then conditioned on the control latent and the anchored hierarchical history:

\epsilon_{\theta}=f_{\theta}(X_{\tau},Z_{\mathrm{ctrl}}^{(k)}\widetilde{H}_{t},\tau).(27)

Since the anchor is never evicted during memory updates, each generation step retains access to the original appearance statistics, while the hierarchical memory provides multi-scale temporal context. This combination improves long-horizon consistency while controlling memory consumption.

#### 6.2.4 Hierarchical History Guidance Attention

As shown in Figure [8](https://arxiv.org/html/2607.02642#S6.F8 "Fig. 8 ‣ 6.2 Architecture and Control Interface ‣ 6 Final Design Map and GigaWorld-1 ‣ GigaWorld-1: A Roadmap to Build World Models for Robot Policy Evaluation"), the historical and noisy contexts exhibit fundamentally different statistics and should therefore be treated differently. The historical context contains previously generated observations that have already been aligned with the task objective and environmental dynamics. In contrast, the noisy context corresponds to the future latent window that requires denoising and prediction. Therefore, the role of historical context is not to be regenerated, but rather to guide the generation of future observations.

To achieve this, we maintain a hierarchical history memory consisting of a first-frame anchor and multi-scale temporal memories:

X_{\mathrm{Hist}}=\left\{X_{\mathrm{Anchor}},X_{\mathrm{Long}},X_{\mathrm{Mid}},X_{\mathrm{Short}}\right\}.(28)

The anchor memory preserves the initial scene configuration and object identity, while long-, mid-, and short-term memories capture scene-level, task-level, and motion-level information, respectively.

In the self-attention layer, we compute the query, key, and value tensors for the noisy and historical contexts, denoted Q_{\mathrm{Noisy}},K_{\mathrm{Noisy}},V_{\mathrm{Noisy}} and Q_{\mathrm{Hist}},K_{\mathrm{Hist}},V_{\mathrm{Hist}} respectively. The self-attention is computed as:

X_{\mathrm{Self}}=\operatorname{Attention}\left([Q_{\mathrm{Noisy}},Q_{\mathrm{Hist}}],[K_{\mathrm{Noisy}},K_{\mathrm{Hist}}],[V_{\mathrm{Noisy}},V_{\mathrm{Hist}}]\right).(29)

Unlike conventional self-attention, the hierarchical history memory serves as a persistent guidance source that provides long-term scene context, task progress, and motion continuity for future prediction.

In cross-attention, semantic information is injected through the task description. Since the historical memory has already accumulated semantic information from previous generation windows, repeatedly conditioning historical tokens on the same task description is unnecessary. Therefore, we apply cross-attention only to the current noisy window:

X_{\mathrm{Cross}}=\operatorname{Attention}\left(Q_{\mathrm{Noisy}},K_{\mathrm{Task}},V_{\mathrm{Task}}\right).(30)

The final representation is obtained by combining self-attention and cross-attention outputs:

X=X_{\mathrm{Self}}+X_{\mathrm{Cross}}.(31)

#### 6.2.5 Relative RoPE

Long-horizon autoregressive generation suffers from temporal position drift when a model trained with fixed absolute temporal indices is evaluated beyond its training horizon. To avoid exposing the model to unseen positional distributions, we use Relative Rotary Position Embedding, referred to as Relative RoPE.

At each autoregressive step, we build a local temporal coordinate system for the concatenation of the historical context and the current generation window. Given a history length T_{h} and a future window length T_{f}, the concatenated sequence has length T_{h}+T_{f}. We assign local temporal positions as:

\mathcal{P}_{\mathrm{hist}}=\{0,1,\ldots,T_{h}-1\},\qquad\mathcal{P}_{\mathrm{future}}=\{T_{h},T_{h}+1,\ldots,T_{h}+T_{f}-1\}.(32)

Equivalently, all tokens in the current autoregressive step use local temporal positions from:

\mathcal{P}=\{0,1,\ldots,T_{h}+T_{f}-1\}.(33)

These local positions are reinitialized at every generation step and therefore do not depend on the absolute timestamp of the generated video. For a token located at local temporal position p\in\mathcal{P}, rotary embeddings are applied as:

Q_{p}^{\prime}=R(p)Q_{p},\qquad K_{p}^{\prime}=R(p)K_{p},(34)

where R(p) denotes the rotary embedding operator evaluated at the local temporal position p. In this way, the model always observes the same range of temporal positions during both training and inference, which helps reduce repetitive motion and temporal instability in long rollouts.

#### 6.2.6 Prompt Transition via Spherical Linear Interpolation

Long-video generation often requires smooth semantic transitions across different temporal stages. Directly switching between prompts may introduce abrupt changes in scene appearance, motion patterns, and object behaviors. As illustrated in Figure [9](https://arxiv.org/html/2607.02642#S6.F9 "Fig. 9 ‣ 6.2.6 Prompt Transition via Spherical Linear Interpolation ‣ 6.2 Architecture and Control Interface ‣ 6 Final Design Map and GigaWorld-1 ‣ GigaWorld-1: A Roadmap to Build World Models for Robot Policy Evaluation"), we achieve gradual transitions between semantic conditions by interpolating text embeddings using Spherical Linear Interpolation (SLERP).

![Image 9: Refer to caption](https://arxiv.org/html/2607.02642v1/x9.png)

Figure 9: Prompt transition via spherical linear interpolation. Instead of abruptly switching between prompt embeddings, GigaWorld-1 samples intermediate text conditions along the spherical path between two semantic endpoints. These interpolated embeddings are injected into successive autoregressive windows, producing smoother changes in scene appearance, motion pattern, and task phase.

Given two text embeddings \mathbf{e}_{1},\mathbf{e}_{2}\in\mathbb{R}^{d} corresponding to two prompts, we first compute the angle between them:

\theta=\arccos\left(\frac{\mathbf{e}_{1}^{\top}\mathbf{e}_{2}}{\|\mathbf{e}_{1}\|\|\mathbf{e}_{2}\|}\right).(35)

For an interpolation coefficient t\in[0,1], the interpolated embedding is defined as

\operatorname{SLERP}(\mathbf{e}_{1},\mathbf{e}_{2},t)=\frac{\sin((1-t)\theta)}{\sin\theta}\mathbf{e}_{1}+\frac{\sin(t\theta)}{\sin\theta}\mathbf{e}_{2}.(36)

To guide autoregressive long-video generation, we uniformly sample a sequence of interpolation coefficients:

t_{i}=\frac{i}{N-1},\qquad i=0,\ldots,N-1,(37)

and obtain a sequence of text conditions

\mathbf{e}^{(i)}=\operatorname{SLERP}(\mathbf{e}_{1},\mathbf{e}_{2},t_{i}).(38)

These interpolated embeddings are progressively injected into successive generation windows, allowing the world model to smoothly evolve from one semantic state to another. Compared with conventional linear interpolation, SLERP preserves the angular structure of the embedding space and better maintains semantic consistency throughout long-horizon generation.

### 6.3 Progressive Training Pipeline

We train GigaWorld-1 with a progressive curriculum that separates robot-domain pretraining, autoregressive rollout learning, scene-level adaptation, and few-step distillation. As summarized in Figure [10](https://arxiv.org/html/2607.02642#S6.F10 "Fig. 10 ‣ 6.3 Progressive Training Pipeline ‣ 6 Final Design Map and GigaWorld-1 ‣ GigaWorld-1: A Roadmap to Build World Models for Robot Policy Evaluation"), the first two stages are required for obtaining a controllable long-horizon world model. Scene adaptation is optional and is used only when additional in-domain demonstrations are available. In contrast, DMD2 distillation is part of the default recipe, since evaluator-scale policy rollout requires fast generation. ODE distillation is used only as an optional warm start for DMD2: it provides a smoother initialization for the student sampler, but the final few-step model is trained by DMD2.

![Image 10: Refer to caption](https://arxiv.org/html/2607.02642v1/images/Training_Pipeline.png)

Figure 10: Training pipeline of GigaWorld-1. The model is first adapted into a robot world foundation model, converted into an autoregressive world generator, and finally compressed through optional ODE warm start and required DMD2 distillation for few-step rollout. Dashed branches denote optional modules that can be skipped.

Stage 1: robot world foundation model. We first initialize from a pretrained video backbone and continue training on the curated multi-source robot corpus. This stage learns a bidirectional robot video prior that captures embodiment-dependent kinematics, object motion, contact events, and camera-specific dynamics. Let x_{0} denote a clean latent video, x_{t} its noisy counterpart at continuous time t, and u_{t} the target velocity. The model is optimized with the flow-matching objective:

\mathcal{L}_{\mathrm{FM}}=\mathbb{E}\left[\|u_{t}-u_{\theta}(x_{t},t)\|_{2}^{2}\right].(39)

where u_{\theta} denotes the predicted velocity field. This stage does not impose autoregressive causality; instead, it transfers general spatiotemporal knowledge into the embodied domain and provides the initialization for controlled rollout learning.

Stage 2: autoregressive world modeling. The bidirectional foundation model is then converted into an autoregressive world model using Relative RoPE, Hierarchical History Injection, the First-Frame Anchor, and Unified Control Injection. Given historical memory H_{t}, temporally aligned controls C, and a noisy future latent window X_{\tau}, the model learns to denoise future observations conditioned on past context and robot actions:

\mathcal{L}_{\mathrm{AR}}=\mathbb{E}\left[\|\epsilon-\epsilon_{\theta}(X_{\tau},H_{t},C,\tau)\|_{2}^{2}\right].(40)

During training, historical frames are sampled from real trajectories; during inference, generated windows are appended to the memory buffer and reused as context for subsequent prediction. This training-inference alignment is critical for reducing compounding error in long-horizon policy evaluation.

Stage 3: scene adaptation LoRA (optional). For deployment in a specific workspace or robot cell, we optionally perform Low-Rank Adaptation (LoRA). The backbone parameters remain frozen and only low-rank matrices are optimized:

W=W_{0}+BA.(41)

This stage adapts the model to scene-specific appearance, lighting, camera calibration, object inventory, and task distribution while preserving the general dynamics learned from the full corpus. If no scene-specific data is available, the Stage-2 model is directly passed to the distillation stage.

Stage 4: few-step distillation. To accelerate inference, we distill the autoregressive teacher into a few-step student. ODE distillation is optional and serves as a warm-start procedure. It constructs teacher-student trajectory pairs by integrating the teacher dynamics and minimizing the discrepancy between the teacher ODE solution and the student prediction:

\mathcal{L}_{\mathrm{ODE}}=\mathbb{E}_{x_{t},t}\left[\left\|\hat{x}^{\mathrm{teacher}}_{0}(x_{t},t)-\hat{x}^{\mathrm{student}}_{0}(x_{t},t)\right\|_{2}^{2}\right].(42)

DMD2 is then applied as the mandatory few-step distillation objective. It combines teacher distribution matching, score consistency, and adversarial supervision so that the student remains faithful to the multi-step teacher while producing realistic rollouts with substantially fewer denoising steps:

\mathcal{L}_{\mathrm{DMD2}}=\lambda_{\mathrm{dm}}\mathcal{L}_{\mathrm{distill}}+\lambda_{\mathrm{score}}\mathcal{L}_{\mathrm{score}}+\lambda_{\mathrm{GAN}}\mathcal{L}_{\mathrm{GAN}}.(43)

The detailed optimization settings are reported in Tables [7](https://arxiv.org/html/2607.02642#S6.T7 "Tab. 7 ‣ 6.3 Progressive Training Pipeline ‣ 6 Final Design Map and GigaWorld-1 ‣ GigaWorld-1: A Roadmap to Build World Models for Robot Policy Evaluation") and [8](https://arxiv.org/html/2607.02642#S6.T8 "Tab. 8 ‣ 6.3 Progressive Training Pipeline ‣ 6 Final Design Map and GigaWorld-1 ‣ GigaWorld-1: A Roadmap to Build World Models for Robot Policy Evaluation"). Table [7](https://arxiv.org/html/2607.02642#S6.T7 "Tab. 7 ‣ 6.3 Progressive Training Pipeline ‣ 6 Final Design Map and GigaWorld-1 ‣ GigaWorld-1: A Roadmap to Build World Models for Robot Policy Evaluation") covers the non-distillation stages, while Table [8](https://arxiv.org/html/2607.02642#S6.T8 "Tab. 8 ‣ 6.3 Progressive Training Pipeline ‣ 6 Final Design Map and GigaWorld-1 ‣ GigaWorld-1: A Roadmap to Build World Models for Robot Policy Evaluation") isolates the optional ODE warm start and the required DMD2 distillation stage.

Table 7: Detailed training hyperparameters for the non-distillation stages. Stage is used as the horizontal axis. AdamW uses \beta_{1}=0.9, \beta_{2}=0.999, \epsilon=10^{-8}. This table excludes ODE and DMD2 distillation.

Configuration Stage 1 Stage 2
Global Batch Size 32 32
Optimizer AdamW AdamW
Learning Rate 5{\times}10^{-5}1{\times}10^{-4}
Learning Rate Schedule Cosine Cosine
Training Steps 13k 36k
Gradient Clipping 1.0 1.0
LoRA Rank 128 256
LoRA Alpha 128 256
Numerical Precision BFloat16 BFloat16
GPU Usage 32 NVIDIA H20 32 NVIDIA H20

Table 8: Detailed training hyperparameters for the distillation stages. ODE is optional and serves as a warm start, whereas DMD2 is required for the final few-step model. AdamW uses \beta_{1}=0.0, \beta_{2}=0.999, \epsilon=10^{-8}, and weight decay 10^{-3}.

Configuration ODE DMD2
Global Batch Size 32 32
Real-score CFG Weight–3.0
Optimizer AdamW AdamW
Learning Rate (G_{\theta})2.0{\times}10^{-6}2.0{\times}10^{-6}
Learning Rate (p_{\mathrm{fake}})–4.0{\times}10^{-7}
Learning Rate Schedule (G_{\theta} and p_{\mathrm{fake}})Cosine Cosine
Learning Rate Warmup Step––
Gradient Clipping (G_{\theta} and p_{\mathrm{fake}})10.0 10.0
LoRA Rank (G_{\theta})256 256
LoRA Rank (p_{\mathrm{fake}})–256
LoRA Alpha (G_{\theta})256 256
LoRA Alpha (p_{\mathrm{fake}})–256
TTUR–5
GAN Head Layers–5, 15, 25, 35, 39
GAN Head Dim–768
GAN Start Step–1000
EMA Decay 0.99 0.99
EMA Start Step 250 750
Training Steps 3759 2250
Numerical Precision BFloat16 BFloat16
GPU Usage 32 NVIDIA H20 32 NVIDIA H20

### 6.4 System Efficiency Optimization

To further accelerate both training and inference, we replace several memory-bound PyTorch operators with custom kernels and distributed execution strategies. These optimizations target the dominant cost centers of the AR diffusion transformer, including normalization, rotary position embedding, attention, VAE decoding, and long-sequence parallelism. Unless otherwise specified, the optimized kernels support both forward and backward propagation, allowing the same implementation to be used during large-scale training.

SageAttention. Attention dominates the runtime of long-window video diffusion transformers. We integrate SageAttention as a drop-in backend for compatible training and inference settings. By improving memory access patterns and using optimized low-precision execution, SageAttention reduces attention overhead and enables larger history memories and denser control tokens under the same GPU budget.

TinyVAE decoding. Full VAE decoding is costly during rollout inspection. We therefore use a lightweight TinyVAE decoder based on TAESD [[16](https://arxiv.org/html/2607.02642#bib.bib16)] for preview decoding, while retaining the full VAE decoder for final evaluation and paper-quality visualization. This allows fast qualitative inspection during development without changing the final evaluation protocol.

Ulysses sequence parallelism. To support long-horizon rollouts, we adopt Ulysses sequence parallelism and shard the token dimension across multiple GPUs. This reduces per-device activation storage approximately in proportion to the number of GPUs, allowing larger temporal windows and richer control conditions.

Inference acceleration benchmark. We evaluate inference efficiency on an H20 96G GPU using 1920{\times}480 videos with 99 frames. Attention-kernel optimization alone provides 1.25{\times}–1.31{\times} acceleration, while combining SageAttention with six-step DMD2 and Ulysses sequence parallelism reaches up to 35.93{\times} speedup.

Flash normalization. We fuse LayerNorm and RMSNorm in Triton by combining statistic computation, normalization, and affine transformation into a single kernel. Instead of materializing full normalized activations for backward propagation, the kernel caches only row-wise statistics, reducing intermediate memory usage. Internal reductions are computed in FP32, while inputs and outputs remain in the original training precision.

Flash rotary position embedding. We implement a fused Triton kernel for Rotary Position Embedding (RoPE), avoiding repeated reshape, chunk, and indexing operations. The backward pass reuses the same kernel with the inverse rotation. By retaining only the pre-computed sine and cosine tables, the fused RoPE kernel reduces activation memory footprint and improves throughput for long-sequence rollout generation.

### 6.5 Final Evaluation of GigaWorld-1

We evaluate the final GigaWorld-1 from multiple perspectives to validate our design choices. Overall, the experiments confirm the effectiveness of the core components of GigaWorld-1: the robot-oriented architecture improves benchmark performance, mixed training data enhances generalization, direct control leads to more accurate trajectories, and the final model produces more stable long-horizon rollouts than general-purpose video generation baselines.

#### 6.5.1 Comparison with Open-Source Baselines

We first compare GigaWorld-1 with representative general video generation and world-model baselines under a controlled setting: all backbones are post-trained on the same curated corpus described in Sec. [6.1](https://arxiv.org/html/2607.02642#S6.SS1 "6.1 Data Sources and Data Curation ‣ 6 Final Design Map and GigaWorld-1 ‣ GigaWorld-1: A Roadmap to Build World Models for Robot Policy Evaluation") and evaluated under the same rollout protocol. Following the metric taxonomy in Sec. [4.3](https://arxiv.org/html/2607.02642#S4.SS3 "4.3 Metric System ‣ 4 WMBench: A Benchmark for World Models as Policy Evaluators ‣ GigaWorld-1: A Roadmap to Build World Models for Robot Policy Evaluation") and the empirical findings in Sec. [5](https://arxiv.org/html/2607.02642#S5 "5 What Matters in Building World Models For Evaluating Robot Policies? ‣ GigaWorld-1: A Roadmap to Build World Models for Robot Policy Evaluation"), we retain six core evaluator-relevant metric(Aesthetic Quality, Image Quality, JEPA Similarity, Semantic Alignment, Subject Consistency, and Trajectory Accuracy) and report their normalized average as the main summary score. We exclude appearance-stability metrics such as Background and Photometric Consistency, which can mislead evaluator ranking, and analyze long-horizon generation metrics separately in the rollout section. As shown in Figure [11](https://arxiv.org/html/2607.02642#S6.F11 "Fig. 11 ‣ 6.5.1 Comparison with Open-Source Baselines ‣ 6.5 Final Evaluation of GigaWorld-1 ‣ 6 Final Design Map and GigaWorld-1 ‣ GigaWorld-1: A Roadmap to Build World Models for Robot Policy Evaluation") and Table [9](https://arxiv.org/html/2607.02642#S6.T9 "Tab. 9 ‣ 6.5.1 Comparison with Open-Source Baselines ‣ 6.5 Final Evaluation of GigaWorld-1 ‣ 6 Final Design Map and GigaWorld-1 ‣ GigaWorld-1: A Roadmap to Build World Models for Robot Policy Evaluation"), GigaWorld-1-Plus achieves the best overall result with an average score of 0.6834, followed by GigaWorld-1-Nano at 0.6717. Among the open-source baselines, Cosmos-Predict2.5 is the strongest (0.6123), while Wan 2.2 5B is the strongest general-purpose model (0.5948). Compared with these two strongest baselines, GigaWorld-1-Plus improves the average score by 11.6% over Cosmos-Predict2.5 and by 14.9% over Wan 2.2 5B. More importantly, the gains are concentrated on the evaluator-critical metrics identified in Question I: GigaWorld-1-Plus achieves the best JEPA Similarity (0.9337), Semantic Alignment (0.8926), and Trajectory Accuracy (0.3561), while matching the best Subject Consistency score (0.8883).

![Image 11: Refer to caption](https://arxiv.org/html/2607.02642v1/x10.png)

Figure 11: Model architecture comparison. Left: mean score across six evaluation metrics. Right: radar plot of individual metric scores. GigaWorld-1-Nano achieves the second-best overall score and performs particularly well in JEPA Similarity, Semantic Alignment, Subject Consistency, and Trajectory Accuracy.

Table 9: Model architecture comparison. Robot-oriented models generally outperform generic video backbones on embodied rollout evaluation. Cell background colors indicate ranking: green = 1st, yellow = 2nd, red = last. All metrics are higher-is-better (\uparrow).

Model Size Type Aesthetic\uparrow Image\uparrow JEPA\uparrow Semantic\uparrow Subject\uparrow Trajectory\uparrow AVG\uparrow
SVD 1.5B General 0.2861 0.6497 0.6454 0.8411 0.8267 0.0926 0.5569
Wan 2.1 1.3B I2V 1.3B General 0.3422 0.6856 0.6002 0.8705 0.5568 0.1576 0.5355
LTX 2.3 22B General 0.3900 0.6967 0.5380 0.8678 0.8248 0.1479 0.5775
CogVideoX 5B General 0.3303 0.6775 0.6437 0.8633 0.6963 0.1609 0.5620
Wan 2.2 5B TI2V 5B General 0.3538 0.6980 0.5853 0.8789 0.8883 0.1643 0.5948
Cosmos-Predict2.5 2B Robot/Auto 0.3491 0.7184 0.6781 0.8764 0.8747 0.1770 0.6123
GigaWorld-1-Nano 1.3B Robot/Auto 0.3538 0.6802 0.8911 0.8920 0.8600 0.3528 0.6717
GigaWorld-1-Plus 5B Robot/Auto 0.3534 0.6765 0.9337 0.8926 0.8883 0.3561 0.6834

#### 6.5.2 Long-Horizon Rollout Quality

We evaluate long-horizon video rollout quality chunk by chunk, using 10-frame chunks corresponding to approximately 1 second. PSNR is computed across multi-view settings, while FID and FVD are computed on the head view. As shown in Figure [12](https://arxiv.org/html/2607.02642#S6.F12 "Fig. 12 ‣ 6.5.2 Long-Horizon Rollout Quality ‣ 6.5 Final Evaluation of GigaWorld-1 ‣ 6 Final Design Map and GigaWorld-1 ‣ GigaWorld-1: A Roadmap to Build World Models for Robot Policy Evaluation") and Table [4](https://arxiv.org/html/2607.02642#S5.T4 "Tab. 4 ‣ 5.3 Question III: How Do Model Design Choices Matter? ‣ 5 What Matters in Building World Models For Evaluating Robot Policies? ‣ GigaWorld-1: A Roadmap to Build World Models for Robot Policy Evaluation"), GigaWorld-1 consistently achieves the best PSNR, FID, and FVD over 40 seconds of autoregressive generation. Qualitatively, generic baselines suffer from viewpoint drift, object-identity collapse, and accumulated texture artifacts, as shown in Figure [13](https://arxiv.org/html/2607.02642#S6.F13 "Fig. 13 ‣ 6.5.2 Long-Horizon Rollout Quality ‣ 6.5 Final Evaluation of GigaWorld-1 ‣ 6 Final Design Map and GigaWorld-1 ‣ GigaWorld-1: A Roadmap to Build World Models for Robot Policy Evaluation") (more videos are shown in the project page). In contrast, GigaWorld-1 avoids these failure modes by using hierarchical history memory and unified control injection (Sec. [6.2](https://arxiv.org/html/2607.02642#S6.SS2 "6.2 Architecture and Control Interface ‣ 6 Final Design Map and GigaWorld-1 ‣ GigaWorld-1: A Roadmap to Build World Models for Robot Policy Evaluation")), which explicitly constrain robot motion and preserve long-term scene identity.

![Image 12: Refer to caption](https://arxiv.org/html/2607.02642v1/images/rollout.png)

Figure 12: Long-horizon rollout dynamics. PSNR (\uparrow) and FID (\downarrow) are measured over successive 10-frame rollout chunks, showing how reconstruction fidelity and perceptual quality evolve with rollout length.

![Image 13: Refer to caption](https://arxiv.org/html/2607.02642v1/x11.png)

Figure 13: Long-horizon model comparison.GigaWorld-1 is compared with general video generation and world-model baselines under the same rollout evaluation protocol.

#### 6.5.3 Impact of Memory and Prompt Interpolation

We conduct a qualitative ablation to isolate the effect of historical memory and prompt interpolation in a multi-stage dual-arm manipulation task (Figure [14](https://arxiv.org/html/2607.02642#S6.F14 "Fig. 14 ‣ 6.5.3 Impact of Memory and Prompt Interpolation ‣ 6.5 Final Evaluation of GigaWorld-1 ‣ 6 Final Design Map and GigaWorld-1 ‣ GigaWorld-1: A Roadmap to Build World Models for Robot Policy Evaluation")). Without memory, the rollout suffers from background jumps and inconsistent scene layouts. While adding memory stabilizes the workspace, abrupt prompt changes can cause the model to over-condition on previous subtasks. Combining memory with Spherical Linear Interpolation (SLERP) resolves this trade-off, enabling smoother semantic transitions from one task phase to the next.

![Image 14: Refer to caption](https://arxiv.org/html/2607.02642v1/images/mem.png)

Figure 14: Effect of memory and prompt interpolation (left arm moves to a predefined observation pose, while the right arm places a towel into the blue box). Without memory, the rollout suffers from background jumps and inconsistent scene layout. Adding history memory stabilizes the workspace, but a fixed or abruptly switched prompt can make the model over-conditioned on the previous subtask during stage transitions. Combining memory with prompt interpolation preserves background consistency while enabling smoother semantic transition from the left-arm observation pose to the right-arm towel-placement action.

#### 6.5.4 OOD Generalization

We evaluate out-of-distribution (OOD) generalization by testing whether the world model preserves physical behavior under shifts in object appearance, object category, scene background, and policy outcome. As shown in Figure [15](https://arxiv.org/html/2607.02642#S6.F15 "Fig. 15 ‣ 6.5.4 OOD Generalization ‣ 6.5 Final Evaluation of GigaWorld-1 ‣ 6 Final Design Map and GigaWorld-1 ‣ GigaWorld-1: A Roadmap to Build World Models for Robot Policy Evaluation"), GigaWorld-1 successfully handles changes in container color, object contents (e.g., different food types), and table textures without geometry collapse. Crucially, the fourth row demonstrates action-outcome generalization: the model accurately simulates both successful placements and failures (e.g., spilling), which is an essential property for a reliable policy evaluator.

![Image 15: Refer to caption](https://arxiv.org/html/2607.02642v1/images/ood.png)

Figure 15: OOD generalization cases. From top to bottom, the rows evaluate generalization to object color and container appearance, object content changes, background and table-surface changes, and action-outcome variation covering both successful and failed executions.

#### 6.5.5 Closed-Loop Policy Consistency

Replay fidelity alone does not imply a world model can evaluate policies. In closed-loop use, small errors in visual state or contact can compound and alter the task outcome. We evaluate whether GigaWorld-1 preserves the same success/failure conclusion as physical-robot execution using the hybrid VLM-evaluator on the WMBench closed-loop tasks (Table [10](https://arxiv.org/html/2607.02642#S6.T10 "Tab. 10 ‣ 6.5.5 Closed-Loop Policy Consistency ‣ 6.5 Final Evaluation of GigaWorld-1 ‣ 6 Final Design Map and GigaWorld-1 ‣ GigaWorld-1: A Roadmap to Build World Models for Robot Policy Evaluation")).

As shown in Figure [16](https://arxiv.org/html/2607.02642#S6.F16 "Fig. 16 ‣ 6.5.5 Closed-Loop Policy Consistency ‣ 6.5 Final Evaluation of GigaWorld-1 ‣ 6 Final Design Map and GigaWorld-1 ‣ GigaWorld-1: A Roadmap to Build World Models for Robot Policy Evaluation"), task-level success-rate alignment provides a direct measure of evaluator calibration. The challenge baselines often overestimate policy success, reflecting a common bias where generated rollouts look visually plausible but fail to penalize poor actions. In contrast, GigaWorld-1 follows the real-world diagonal much more closely, indicating better preservation of relative task difficulty.

Table 10: Closed-loop policy consistency tasks. Each task is decomposed into subtask-level outcome checks so that agreement can be measured not only at the task level, but also at contact- and manipulation-sensitive intermediate stages.

Task Task Type Subtasks
task1 Put banana into basket Grasp banana; Place banana into basket
task2 Put green bowl into pink plate Grasp green bowl; Place green bowl into pink plate
task3 Fold paper boxes Grasp carton flaps; Gripper in position; Press down carton lids; Reset
task4 Pour the fries into the box Move the takeout box over the plate; Open the takeout box; Pour out the fries; Press down on the takeout box

![Image 16: Refer to caption](https://arxiv.org/html/2607.02642v1/images/multi_task_success_rate_fit.png)

Figure 16: Task-level success-rate alignment. Real-robot success rates are compared with generated success rates under closed-loop policy rollout. The gray dashed line indicates perfect agreement. GigaWorld-1 has a fitted line closer to the diagonal than the challenge baselines, indicating better calibration of task difficulty across the evaluated tasks and subtasks.

![Image 17: Refer to caption](https://arxiv.org/html/2607.02642v1/x12.png)

Figure 17: Success-rate bias across world models and subtasks. Bars show \mathrm{Gen}-\mathrm{Real} success-rate differences: green indicates overestimation of real-world success, red indicates underestimation, and values near zero indicate closer agreement.

Success-rate bias analysis (Figure [17](https://arxiv.org/html/2607.02642#S6.F17 "Fig. 17 ‣ 6.5.5 Closed-Loop Policy Consistency ‣ 6.5 Final Evaluation of GigaWorld-1 ‣ 6 Final Design Map and GigaWorld-1 ‣ GigaWorld-1: A Roadmap to Build World Models for Robot Policy Evaluation")) further compares generated and real robot outcomes across manipulation subtasks. GigaWorld-1 shows smaller overall \mathrm{Gen}-\mathrm{Real} deviations than cvpr_world_challenge_model_1, suggesting closer calibration to real-world success rates. In contrast, cvpr_world_challenge_model_2 and cvpr_world_challenge_model_3 tend to over-confidently predict success, whereas GigaWorld-1 provides a more balanced estimate across subtasks.

Therefore, closed-loop policy consistency is stricter than replay quality. A useful world model must be calibrated not only to visual realism, but also to the success and failure distribution induced by a policy interacting with the generated state. The results show that GigaWorld-1 better preserves closed-loop policy conclusions than generic video-generation baselines, while also highlighting an important direction for future work: reducing the optimistic bias of video generation models on contact-sensitive failure.

## 7 Discussion and Conclusion

Our study suggests that the central difficulty of learned robot evaluation is not video generation alone, but preserving _policy-relevant causality_ under iterative rollout. This insight explains why seemingly strong video models can still be poor evaluators: they may generate plausible local motion while drifting away from the action-conditioned evolution required for reliable policy comparison. More broadly, GigaWorld-1 explores how to build a world model that can serve as a reliable policy evaluator under a cost-controllable recipe. In this setting, evaluator design cannot be reduced to a single axis such as model size or robot-data volume alone. What matters is the ability to preserve broad world knowledge, remain controllable under action input, and sustain long-horizon consistency while keeping the overall evaluation pipeline practical enough for repeated use.

The work also has practical implications for benchmark design and deployment. If future robot foundation models are to iterate as quickly as digital foundation models, the community needs reliable, reusable evaluator infrastructure. WMBench is intended as a step in this direction: not merely another benchmark with a fixed leaderboard, but a tool for studying why evaluator conclusions succeed or fail. We do not encourage optimizing narrowly for leaderboard gains on WMBench. Instead, the broader goal is to use large-scale experimentation to identify which modeling choices and which evaluation signals are truly reliable for world models as policy evaluators. Beyond benchmark analysis, we have already integrated GigaWorld-1 into our internal policy-evaluation pipeline and conducted extensive testing at scale. This experience suggests that world-model-based evaluation can already provide meaningful support for policy iteration in practice, and that continued accumulation of rollout data should further strengthen GigaWorld-1 as both a simulator and an evaluator. The public challenge component further demonstrates that community-driven model exploration can enrich this research question by broadening the diversity of evaluator candidates.

At the same time, our conclusions should be interpreted with several limitations in mind. First, although WMBench covers eight task families, it does not yet exhaust the full space of mobile manipulation, dexterous in-hand manipulation, or safety-critical autonomy. Second, our study focuses primarily on video-centric world models; other structured state-space or hybrid 3D approaches may exhibit different trade-offs. Third, while VLM-assisted success labeling substantially reduces annotation cost, final evaluator assessment still benefits from human verification in uncertain cases. Looking ahead, several directions appear especially promising for world models as simulators and policy evaluators. We also plan to continue expanding the benchmark with more data, especially more rollout data, so that evaluator analysis can be grounded in broader and harder closed-loop settings. In parallel, more accurate metrics for measuring outcome faithfulness, action-conditioned consistency, and evaluator reliability remain important for making world-model-based policy evaluation more trustworthy. Recent community efforts on omni world models, such as Cosmos-3 [[1](https://arxiv.org/html/2607.02642#bib.bib1)], suggest that absorbing richer multimodal training data may improve both controllability and physical grounding. In parallel, scaling model parameters remains an important direction, not as a substitute for evaluator-oriented design, but as a complementary path for improving capacity, world knowledge retention, and long-horizon robustness. Altogether, we hope these efforts will accelerate the adoption of world models as policy evaluators in both research and industrial practice, while extending these results toward richer reward modeling, counterfactual evaluation, and certified uncertainty.

## References

*   Agarwal et al. [2026] Niket Agarwal, Arslan Ali, Jon Allen, Martin Antolini, Adeline Aubame, Alisson Azzolini, Junjie Bai, Maciej Bala, Yogesh Balaji, Josh Bapst, et al. Cosmos 3: Omnimodal world models for physical ai. _arXiv preprint arXiv:2606.02800_, 2026. 
*   Alhaija et al. [2025] Hassan Abu Alhaija, Jose Alvarez, Maciej Bala, Tiffany Cai, Tianshi Cao, Liz Cha, Joshua Chen, Mike Chen, Francesco Ferroni, Sanja Fidler, et al. Cosmos-transfer1: Conditional world generation with adaptive multimodal control. _arXiv preprint arXiv:2503.14492_, 2025. 
*   Ali et al. [2025] Arslan Ali, Junjie Bai, Maciej Bala, Yogesh Balaji, Aaron Blakeman, Tiffany Cai, Jiaxin Cao, Tianshi Cao, Elizabeth Cha, Yu-Wei Chao, et al. World simulation with video foundation models for physical ai. _arXiv preprint arXiv:2511.00062_, 2025. 
*   Atreya et al. [2025] Pranav Atreya, Karl Pertsch, Tony Lee, Moo Jin Kim, Arhan Jain, Artur Kuramshin, Clemens Eppner, Cyrus Neary, Edward Hu, Fabio Ramos, et al. Roboarena: Distributed real-world evaluation of generalist robot policies. _arXiv preprint arXiv:2506.18123_, 2025. 
*   Attal et al. [2023] Benjamin Attal, Jia-Bin Huang, Christian Richardt, Michael Zollhoefer, Johannes Kopf, Matthew O’Toole, and Changil Kim. Hyperreel: High-fidelity 6-dof video with ray-conditioned sampling. In _CVPR_, pages 16610–16620, 2023. 
*   Bai et al. [2025a] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, Mei Li, Kaixin Li, Zicheng Lin, Junyang Lin, Xuejing Liu, Jiawei Liu, Chenglong Liu, Yang Liu, Dayiheng Liu, Shixuan Liu, Dunjie Lu, Ruilin Luo, Chenxu Lv, Rui Men, Lingchen Meng, Xuancheng Ren, Xingzhang Ren, Sibo Song, Yuchong Sun, Jun Tang, Jianhong Tu, Jianqiang Wan, Peng Wang, Pengfei Wang, Qiuyue Wang, Yuxuan Wang, Tianbao Xie, Yiheng Xu, Haiyang Xu, Jin Xu, Zhibo Yang, Mingkun Yang, Jianxin Yang, An Yang, Bowen Yu, Fei Zhang, Hang Zhang, Xi Zhang, Bo Zheng, Humen Zhong, Jingren Zhou, Fan Zhou, Jing Zhou, Yuanzhi Zhu, and Ke Zhu. Qwen3-vl technical report. _arXiv preprint arXiv:2511.21631_, 2025a. 
*   Bai et al. [2025b] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, Mei Li, Kaixin Li, Zicheng Lin, Junyang Lin, Xuejing Liu, Jiawei Liu, Chenglong Liu, Yang Liu, Dayiheng Liu, Shixuan Liu, Dunjie Lu, Ruilin Luo, Chenxu Lv, Rui Men, Lingchen Meng, Xuancheng Ren, Xingzhang Ren, Sibo Song, Yuchong Sun, Jun Tang, Jianhong Tu, Jianqiang Wan, Peng Wang, Pengfei Wang, Qiuyue Wang, Yuxuan Wang, Tianbao Xie, Yiheng Xu, Haiyang Xu, Jin Xu, Zhibo Yang, Mingkun Yang, Jianxin Yang, An Yang, Bowen Yu, Fei Zhang, Hang Zhang, Xi Zhang, Bo Zheng, Humen Zhong, Jingren Zhou, Fan Zhou, Jing Zhou, Yuanzhi Zhu, and Ke Zhu. Qwen3-vl technical report. _arXiv preprint arXiv:2511.21631_, 2025b. 
*   Bai et al. [2025c] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. _arXiv preprint arXiv:2502.13923_, 2025c. 
*   Bai et al. [2025d] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. _arXiv preprint arXiv:2502.13923_, 2025d. 
*   Bar et al. [2025] Amir Bar, Gaoyue Zhou, Danny Tran, Trevor Darrell, and Yann LeCun. Navigation world models. In _CVPR_, pages 15791–15801, 2025. 
*   Bardes et al. [2024] Adrien Bardes, Quentin Garrido, Jean Ponce, Xinlei Chen, Michael Rabbat, Yann LeCun, Mido Assran, and Nicolas Ballas. V-JEPA: Latent video prediction for visual representation learning, 2024. URL [https://openreview.net/forum?id=WFYbBOEOtv](https://openreview.net/forum?id=WFYbBOEOtv). 
*   Bi et al. [2025] Hongzhe Bi, Hengkai Tan, Shenghao Xie, Zeyuan Wang, Shuhe Huang, Haitian Liu, Ruowen Zhao, Yao Feng, Chendong Xiang, Yinze Rong, et al. Motus: A unified latent action world model. _arXiv preprint arXiv:2512.13030_, 2025. 
*   Bjorck et al. [2025] Johan Bjorck, Fernando Castañeda, Nikita Cherniadev, Xingye Da, Runyu Ding, Linxi Fan, Yu Fang, Dieter Fox, Fengyuan Hu, Spencer Huang, et al. Gr00t n1: An open foundation model for generalist humanoid robots. _arXiv preprint arXiv:2503.14734_, 2025. 
*   Black et al. [2024] Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, et al. pi0: A vision-language-action flow model for general robot control. _arXiv preprint arXiv:2410.24164_, 2024. 
*   Blattmann et al. [2023] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. _arXiv preprint arXiv:2311.15127_, 2023. 
*   Bohan [2023] Ollin Boer Bohan. Tiny autoencoder for stable diffusion. [https://github.com/madebyollin/taesd](https://github.com/madebyollin/taesd), 2023. GitHub repository. 
*   Bu et al. [2025] Qingwen Bu, Jisong Cai, Li Chen, Xiuqi Cui, Yan Ding, Siyuan Feng, Xindong He, Xu Huang, et al. Agibot world colosseo: A large-scale manipulation platform for scalable and intelligent embodied systems. In _IROS_, 2025. 
*   Carion et al. [2025] Nicolas Carion, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris, Chaitanya Ryali, Kalyan Vasudev Alwala, Haitham Khedr, Andrew Huang, Jie Lei, Tengyu Ma, Baishan Guo, Arpit Kalla, Markus Marks, Joseph Greer, Meng Wang, Peize Sun, Roman Rädle, Triantafyllos Afouras, Effrosyni Mavroudi, Katherine Xu, Tsung-Han Wu, Yu Zhou, Liliane Momeni, Rishi Hazra, Shuangrui Ding, Sagar Vaze, Francois Porcher, Feng Li, Siyuan Li, Aishwarya Kamath, Ho Kei Cheng, Piotr Dollár, Nikhila Ravi, Kate Saenko, Pengchuan Zhang, and Christoph Feichtenhofer. Sam 3: Segment anything with concepts, 2025. URL [https://arxiv.org/abs/2511.16719](https://arxiv.org/abs/2511.16719). 
*   Caron et al. [2021] Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers, 2021. URL [https://arxiv.org/abs/2104.14294](https://arxiv.org/abs/2104.14294). 
*   Cen et al. [2025] Jun Cen, Chaohui Yu, Hangjie Yuan, Yuming Jiang, Siteng Huang, Jiayan Guo, Xin Li, Yibing Song, Hao Luo, Fan Wang, et al. Worldvla: Towards autoregressive action world model. _arXiv preprint arXiv:2506.21539_, 2025. 
*   Chen et al. [2026a] Shuimu Chen, Yuteng Chen, Yuanshen Guan, Zebang Cheng, Zeyu Zhang, Shengqian Qin, Bin Xia, Jiaran Li, Wenming Yang, and Fei Ma. Reflect-r1: Evidence-driven reflection for self-correction in long video understanding, 2026a. URL [https://arxiv.org/abs/2606.27922](https://arxiv.org/abs/2606.27922). 
*   Chen et al. [2025] Tianxing Chen, Zanxin Chen, Baijun Chen, Zijian Cai, Yibin Liu, Qiwei Liang, Zixuan Li, Xianliang Lin, Yiheng Ge, Zhenyu Gu, et al. Robotwin 2.0: A scalable data generator and benchmark with strong domain randomization for robust bimanual robotic manipulation. _arXiv preprint arXiv:2506.18088_, 2025. 
*   Chen et al. [2026b] Yuzhi Chen, Ronghan Chen, Dongjie Huo, Yandan Yang, Dekang Qi, Haoyun Liu, Tong Lin, Shuang Zeng, Junjin Xiao, Xinyuan Chang, et al. Abot-physworld: Interactive world foundation model for robotic manipulation with physics alignment. _arXiv preprint arXiv:2603.23376_, 2026b. 
*   Cobbe et al. [2021] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. _arXiv preprint arXiv:2110.14168_, 2021. 
*   DeepMind [2025] Google DeepMind. Introducing veo 3, our video generation model with expanded creative controls – including native audio and extended videos. [Online], 2025. URL [https://deepmind.google/models/veo/](https://deepmind.google/models/veo/). 
*   Dong et al. [2025] Zhehao Dong, Xiaofeng Wang, Zheng Zhu, Yirui Wang, Yang Wang, Yukun Zhou, Boyuan Wang, Chaojun Ni, Runqi Ouyang, Wenkang Qin, et al. Emma: Generalizing real-world robot manipulation via generative visual transfer. _arXiv preprint arXiv:2509.22407_, 2025. 
*   Fan et al. [2026] Liaoyuan Fan, Zetian Xu, Chen Cao, Wenyao Zhang, Mingqi Yuan, and Jiayu Chen. Aim: Intent-aware unified world action modeling with spatial value maps. _arXiv preprint arXiv:2604.11135_, 2026. 
*   Fu et al. [2025a] Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. In _CVPR_, pages 24108–24118, 2025a. 
*   Fu et al. [2025b] Xiao Fu, Xintao Wang, Xian Liu, Jianhong Bai, Runsen Xu, Pengfei Wan, Di Zhang, and Dahua Lin. Learning video generation for robotic manipulation with collaborative trajectory control. _arXiv preprint arXiv:2506.01943_, 2025b. 
*   Gao et al. [2025] Yu Gao, Haoyuan Guo, Tuyen Hoang, Weilin Huang, Lu Jiang, Fangyuan Kong, Huixia Li, Jiashi Li, Liang Li, Xiaojie Li, et al. Seedance 1.0: Exploring the boundaries of video generation models. _arXiv preprint arXiv:2506.09113_, 2025. 
*   Guo et al. [2026] Xiangyu Guo, Zhanqian Wu, Kaixin Xiong, Ziyang Xu, Lijun Zhou, Gangwei Xu, Shaoqing Xu, Haiyang Sun, Bing Wang, Guang Chen, et al. Genesis: Multimodal driving scene generation with spatio-temporal and cross-modal consistency. _NeurIPS_, 38:60431–60455, 2026. 
*   Guo et al. [2025] Yanjiang Guo, Lucy Xiaoyang Shi, Jianyu Chen, and Chelsea Finn. Ctrl-world: A controllable generative world model for robot manipulation. _arXiv preprint arXiv:2510.10125_, 2025. 
*   HaCohen et al. [2024] Yoav HaCohen, Nisan Chiprut, Benny Brazowski, Daniel Shalem, Dudu Moshe, Eitan Richardson, Eran Levin, Guy Shiran, Nir Zabari, Ori Gordon, et al. Ltx-video: Realtime video latent diffusion. _arXiv preprint arXiv:2501.00103_, 2024. 
*   He et al. [2026a] Haoran He, Yang Zhang, Liang Lin, Zhongwen Xu, and Ling Pan. Pre-trained video generative models as world simulators. In _AAAI_, volume 40, pages 4645–4653, 2026a. 
*   He et al. [2025] Xianglong He, Chunli Peng, Zexiang Liu, Boyang Wang, Yifan Zhang, Qi Cui, Fei Kang, Biao Jiang, Mengyin An, Yangyang Ren, et al. Matrix-game 2.0: An open-source real-time and streaming interactive world model. _arXiv preprint arXiv:2508.13009_, 2025. 
*   He et al. [2026b] Ziheng He, Yixiang Chen, Ning Yang, Zhanqian Wu, Qisen Ma, Yuan Xu, Jiabing Yang, Peiyan Li, Xiangnan Wu, Xiaofeng Wang, et al. Skip: Sparse keyframe interpolation paradigm for efficient embodied world models. _arXiv preprint arXiv:2606.00664_, 2026b. 
*   Hendrycks et al. [2020] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. In _ICLR_, 2020. 
*   Heusel et al. [2017] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. _NeurIPS_, 2017. 
*   Hong et al. [2023] Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. Cogvideo: Large-scale pretraining for text-to-video generation via transformers. In _ICLR_, 2023. 
*   Hsieh et al. [2024] Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, Yang Zhang, and Boris Ginsburg. Ruler: What’s the real context size of your long-context language models? _arXiv preprint arXiv:2404.06654_, 2024. 
*   Hu et al. [2021] Edward J Hu, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. In _ICLR_, 2021. 
*   Huang et al. [2026] Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self forcing: Bridging the train-test gap in autoregressive video diffusion. _NeurIPS_, 38:167283–167308, 2026. 
*   Intelligence et al. [2025] Physical Intelligence, Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, et al. \pi_{0.5}: a vision-language-action model with open-world generalization. _arXiv preprint arXiv:2504.16054_, 2025. 
*   James et al. [2020] Stephen James, Zicong Ma, David Rovick Arrojo, and Andrew J Davison. Rlbench: The robot learning benchmark & learning environment. _IEEE Robotics and Automation Letters_, 5(2):3019–3026, 2020. 
*   Jiang et al. [2025a] Tao Jiang, Tianyuan Yuan, Yicheng Liu, Chenhao Lu, Jianning Cui, Xiao Liu, Shuiqi Cheng, Jiyang Gao, Huazhe Xu, and Hang Zhao. Galaxea open-world dataset and g0 dual-system vla model. _arXiv preprint arXiv:2509.00576_, 2025a. 
*   Jiang et al. [2025b] Yuxin Jiang, Shengcong Chen, Siyuan Huang, Liliang Chen, Pengfei Zhou, Yue Liao, Xindong He, Chiming Liu, Hongsheng Li, Maoqing Yao, et al. Enerverse-ac: Envisioning embodied environments with action condition. _arXiv preprint arXiv:2505.09723_, 2025b. 
*   Jimenez et al. [2024] Carlos E Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik Narasimhan. Swe-bench: Can language models resolve real-world github issues? In _ICLR_, volume 2024, pages 54107–54157, 2024. 
*   Kang et al. [2024] Bingyi Kang, Yang Yue, Rui Lu, Zhijie Lin, Yang Zhao, Kaixin Wang, Gao Huang, and Jiashi Feng. How far is video generation from world model: A physical law perspective. _arXiv preprint arXiv:2411.02385_, 2024. 
*   Ke et al. [2021] Junjie Ke, Qifei Wang, Yilin Wang, Peyman Milanfar, and Feng Yang. Musiq: Multi-scale image quality transformer. In _ICCV_, pages 5148–5157, October 2021. 
*   Kim et al. [2024] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, et al. Openvla: An open-source vision-language-action model. _arXiv preprint arXiv:2406.09246_, 2024. 
*   Kim et al. [2026] Moo Jin Kim, Yihuai Gao, Tsung-Yi Lin, Yen-Chen Lin, Yunhao Ge, Grace Lam, Percy Liang, Shuran Song, Ming-Yu Liu, Chelsea Finn, et al. Cosmos policy: Fine-tuning video models for visuomotor control and planning. _arXiv preprint arXiv:2601.16163_, 2026. 
*   LAION-AI [2022] LAION-AI. Aesthetic predictor, 2022. URL [https://github.com/LAION-AI/aesthetic-predictor](https://github.com/LAION-AI/aesthetic-predictor). Accessed: 2024. 
*   Lang et al. [2026] Xiaolei Lang, Yang Wang, Yukun Zhou, Chaojun Ni, Kerui Li, Jiagang Zhu, Tianze Liu, Jiajun Lv, Xingxing Zuo, Yun Ye, et al. Vag: Dual-stream video-action generation for embodied data synthesis. _arXiv preprint arXiv:2604.09330_, 2026. 
*   Lei et al. [2026] Huashuo Lei, Wenxuan Song, Huarui Zhang, Jieyuan Pei, Jiayi Chen, Haodong Yan, Han Zhao, Pengxiang Ding, Zhipeng Zhang, Lida Huang, et al. Robomemarena: A comprehensive and challenging robotic memory benchmark. _arXiv preprint arXiv:2605.10921_, 2026. 
*   Li et al. [2025a] Haoyun Li, Ivan Zhang, Runqi Ouyang, Xiaofeng Wang, Zheng Zhu, Zhiqin Yang, Zhentao Zhang, Boyuan Wang, Chaojun Ni, Wenkang Qin, et al. Mimicdreamer: Aligning human and robot demonstrations for scalable vla training. _arXiv preprint arXiv:2509.22199_, 2025a. 
*   Li et al. [2026] Kerui Li, Zhe Jing, Xiaofeng Wang, Zheng Zhu, Yukun Zhou, Guan Huang, Dongze Li, Qingkai Yang, and Huaibo Huang. Stableidm: Stabilizing inverse dynamics model against manipulator truncation via spatio-temporal refinement. _arXiv preprint arXiv:2604.17887_, 2026. 
*   Li et al. [2024] Xuanlin Li, Kyle Hsu, Jiayuan Gu, Karl Pertsch, Oier Mees, Homer Rich Walke, Chuyuan Fu, Ishikaa Lunawat, Isabel Sieh, Sean Kirmani, et al. Evaluating real-world robot manipulation policies in simulation. _arXiv preprint arXiv:2405.05941_, 2024. 
*   Li et al. [2025b] Yaxuan Li, Yichen Zhu, Junjie Wen, Chaomin Shen, and Yi Xu. Worldeval: World model as real-world robot policies evaluator. _arXiv preprint arXiv:2505.19017_, 2025b. 
*   Liao et al. [2025] Yue Liao, Pengfei Zhou, Siyuan Huang, Donglin Yang, Shengcong Chen, Yuxin Jiang, Yue Hu, Jingbin Cai, Si Liu, Jianlan Luo, et al. Genie envisioner: A unified world foundation platform for robotic manipulation. _arXiv preprint arXiv:2508.05635_, 2025. 
*   Lin et al. [2025] Haotong Lin, Sili Chen, Junhao Liew, Donny Y Chen, Zhenyu Li, Guang Shi, Jiashi Feng, and Bingyi Kang. Depth anything 3: Recovering the visual space from any views. _arXiv preprint arXiv:2511.10647_, 2025. 
*   Lin et al. [2022] Stephanie Lin, Jacob Hilton, and Owain Evans. Truthfulqa: Measuring how models mimic human falsehoods. In _Proceedings of the 60th annual meeting of the association for computational linguistics (volume 1: long papers)_, pages 3214–3252, 2022. 
*   Liu et al. [2023] Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, and Peter Stone. Libero: Benchmarking knowledge transfer for lifelong robot learning. _NeurIPS_, 36:44776–44791, 2023. 
*   Liu et al. [2025a] Jiuming Liu, Jinru Han, Lihao Liu, Angelica I Aviles-Rivero, Chaokang Jiang, Zhe Liu, and Hesheng Wang. Mamba4d: Efficient 4d point cloud video understanding with disentangled spatial-temporal state space models. In _CVPR_, pages 17626–17636, 2025a. 
*   Liu et al. [2025b] Jiuming Liu, Weicai Ye, Guangming Wang, Chaokang Jiang, Lei Pan, Jinru Han, Zhe Liu, Guofeng Zhang, and Hesheng Wang. Difflow3d: Hierarchical diffusion models for uncertainty-aware 3d scene flow estimation. _TPAMI_, 2025b. 
*   Liu et al. [2026a] Jiuming Liu, Mengmeng Liu, Siting Zhu, Yunpeng Zhang, Jiangtao Li, Michael Ying Yang, Francesco Nex, Hao Cheng, and Hesheng Wang. Arflow: Auto-regressive optical flow estimation for arbitrary-length videos via progressive next-frame forecasting. In _ICLR_, 2026a. 
*   Liu et al. [2026b] Jiuming Liu, Chaojun Ni, Mengmeng Liu, Chensheng Peng, Fangjinhua Wang, Sitian Shen, Marc Pollefeys, Masayoshi Tomizuka, Ayush Tewari, and Per Ola Kristensson. Towards interactive video world modeling: Frontiers, challenges, benchmarks, and future trends. _arXiv preprint arXiv:2606.01164_, 2026b. 
*   Liu et al. [2026c] Liu Liu, Xiaofeng Wang, Guosheng Zhao, Keyu Li, Wenkang Qin, Jiagang Zhu, Jiaxiong Qiu, Guan Huang, and Zhizhong Su. Robotransfer: Controllable geometry-consistent video diffusion for manipulation policy transfer. In _CVPR_, pages 1410–1420, 2026c. 
*   Liu et al. [2024] Shaowei Liu, Zhongzheng Ren, Saurabh Gupta, and Shenlong Wang. Physgen: Rigid-body physics-grounded image-to-video generation. In _ECCV_, pages 360–378. Springer, 2024. 
*   Luo et al. [2024] Ge Ya Luo, Gian Mario Favero, Zhi Hao Luo, Alexia Jolicoeur-Martineau, and Christopher Pal. Beyond fvd: Enhanced evaluation metrics for video generation quality, 2024. URL [https://arxiv.org/abs/2410.05203](https://arxiv.org/abs/2410.05203). 
*   Lv et al. [2026] Jindi Lv, Hao Li, Jie Li, Yifei Nie, Fankun Kong, Yang Wang, Xiaofeng Wang, Zheng Zhu, Chaojun Ni, Qiuping Deng, Hengtao Li, Jiancheng Lv, and Guan Huang. Viva: A video-generative value model for robot reinforcement learning. _arXiv preprint arXiv:2604.08168_, 2026. 
*   Ma et al. [2024a] Yue Ma, Yingqing He, Xiaodong Cun, Xintao Wang, Siran Chen, Xiu Li, and Qifeng Chen. Follow your pose: Pose-guided text-to-video generation using pose-free videos. In _AAAI_, volume 38, pages 4117–4125, 2024a. 
*   Ma et al. [2024b] Yue Ma, Hongyu Liu, Hongfa Wang, Heng Pan, Yingqing He, Junkun Yuan, Ailing Zeng, Chengfei Cai, Heung-Yeung Shum, Wei Liu, et al. Follow-your-emoji: Fine-controllable and expressive freestyle portrait animation. In _SIGGRAPH Asia_, pages 1–12, 2024b. 
*   Ma et al. [2025a] Yue Ma, Kunyu Feng, Zhongyuan Hu, Xinyu Wang, Yucheng Wang, Mingzhe Zheng, Xuanhua He, Chenyang Zhu, Hongyu Liu, Yingqing He, et al. Controllable video generation: A survey. _arXiv preprint arXiv:2507.16869_, 2025a. 
*   Ma et al. [2025b] Yue Ma, Yulong Liu, Qiyuan Zhu, Ayden Yang, Kunyu Feng, Xinhua Zhang, Zhifeng Li, Sirui Han, Chenyang Qi, and Qifeng Chen. Follow-your-motion: Video motion transfer via efficient spatial-temporal decoupled finetuning. _arXiv preprint arXiv:2506.05207_, 2025b. 
*   Ma et al. [2025c] Yue Ma, Zexuan Yan, Hongyu Liu, Hongfa Wang, Heng Pan, Yingqing He, Junkun Yuan, Ailing Zeng, Chengfei Cai, Heung-Yeung Shum, et al. Follow-your-emoji-faster: Towards efficient, fine-controllable, and expressive freestyle portrait animation. _arXiv preprint arXiv:2509.16630_, 2025c. 
*   Ma et al. [2025d] Yue Ma, Yingqing He, Hongfa Wang, Andong Wang, Leqi Shen, Chenyang Qi, Jixuan Ying, Chengfei Cai, Zhifeng Li, Heung-Yeung Shum, et al. Follow-your-click: Open-domain regional image animation via motion prompts. In _AAAI_, volume 39, pages 6018–6026, 2025d. 
*   Ma et al. [2025e] Yue Ma, Kunyu Feng, Xinhua Zhang, Hongyu Liu, David Junhao Zhang, Jinbo Xing, Yinhan Zhang, Ayden Yang, Zeyu Wang, and Qifeng Chen. Follow-your-creation: Empowering 4d creation through video inpainting. _arXiv preprint arXiv:2506.04590_, 2025e. 
*   Ma et al. [2026a] Yue Ma, Xinyu Wang, Qianli Ma, Qinghe Wang, Mingzhe Zheng, Xiangpeng Yang, Hao Li, Chongbo Zhao, Jixuan Ying, Harry Yang, et al. Group editing: Edit multiple images in one go. _arXiv preprint arXiv:2603.22883_, 2026a. 
*   Ma et al. [2026b] Yue Ma, Zhikai Wang, Tianhao Ren, Mingzhe Zheng, Hongyu Liu, Jiayi Guo, Mark Fong, Yuxuan Xue, Zixiang Zhao, Konrad Schindler, et al. Fastvmt: Eliminating redundancy in video motion transfer. _arXiv preprint arXiv:2602.05551_, 2026b. 
*   McLean et al. [2025] Reginald McLean, Evangelos Chatzaroulas, Luc McCutcheon, Frank Röder, Tianhe Yu, Zhanpeng He, K.R. Zentner, Ryan Julian, J K Terry, Isaac Woungang, Nariman Farsad, and Pablo Samuel Castro. Meta-world+: An improved, standardized, RL benchmark. In _The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track_, 2025. 
*   Müller [2007] Meinard Müller. _Information retrieval for music and motion_. Springer, 2007. 
*   Nasiriany et al. [2024] Soroush Nasiriany, Abhiram Maddukuri, Lance Zhang, Adeet Parikh, Aaron Lo, Abhishek Joshi, Ajay Mandlekar, and Yuke Zhu. Robocasa: Large-scale simulation of everyday tasks for generalist robots. _arXiv preprint arXiv:2406.02523_, 2024. 
*   Nasiriany et al. [2026] Soroush Nasiriany, Sepehr Nasiriany, Abhiram Maddukuri, and Yuke Zhu. Robocasa365: A large-scale simulation framework for training and benchmarking generalist robots. _arXiv preprint arXiv:2603.04356_, 2026. 
*   Ni et al. [2025a] Chaojun Ni, Cheng Chen, Xiaofeng Wang, Zheng Zhu, Wenzhao Zheng, Boyuan Wang, Tianrun Chen, Guosheng Zhao, Haoyun Li, Zhehao Dong, et al. Swiftvla: Unlocking spatiotemporal dynamics for lightweight vla models at minimal overhead. _arXiv preprint arXiv:2512.00903_, 2025a. 
*   Ni et al. [2025b] Chaojun Ni, Jie Li, Haoyun Li, Hengyu Liu, Xiaofeng Wang, Zheng Zhu, Guosheng Zhao, Boyuan Wang, Chenxin Li, Guan Huang, et al. Wonderfree: Enhancing novel view quality and cross-view consistency for 3d scene exploration. _arXiv preprint arXiv:2506.20590_, 2025b. 
*   Ni et al. [2025c] Chaojun Ni, Xiaofeng Wang, Zheng Zhu, Weijie Wang, Haoyun Li, Guosheng Zhao, Jie Li, Wenkang Qin, Guan Huang, and Wenjun Mei. Wonderturbo: Generating interactive 3d world in 0.72 seconds. _arXiv preprint arXiv:2504.02261_, 2025c. 
*   Ni et al. [2025d] Chaojun Ni, Guosheng Zhao, Xiaofeng Wang, Zheng Zhu, Wenkang Qin, Xinze Chen, Guanghong Jia, Guan Huang, and Wenjun Mei. Recondreamer-rl: Enhancing reinforcement learning via diffusion-based scene reconstruction. _arXiv preprint arXiv:2508.08170_, 2025d. 
*   Ni et al. [2025e] Chaojun Ni, Guosheng Zhao, Xiaofeng Wang, Zheng Zhu, Wenkang Qin, Guan Huang, Chen Liu, Yuyin Chen, Yida Wang, Xueyang Zhang, et al. Recondreamer: Crafting world models for driving scene reconstruction via online restoration. In _CVPR_, pages 1559–1569, 2025e. 
*   Qin et al. [2024] Yiran Qin, Zhelun Shi, Jiwen Yu, Xijun Wang, Enshen Zhou, Lijun Li, Zhenfei Yin, Xihui Liu, Lu Sheng, Jing Shao, et al. Worldsimbench: Towards video generation models as world simulators. _arXiv preprint arXiv:2410.18072_, 2024. 
*   Qiu et al. [2026] Boxiang Qiu, Liliang Chen, Yue Liao, Nan Wang, Lintao Wang, Jiayi Luo, Wenzhi Zhao, Shengcong Chen, Di Chen, Ye Li, et al. Ge-sim 2.0: A roadmap towards comprehensive closed-loop video world simulators for robotic manipulation. _arXiv preprint arXiv:2605.27491_, 2026. 
*   Quevedo et al. [2025] Julian Quevedo, Ansh Kumar Sharma, Yixiang Sun, Varad Suryavanshi, Percy Liang, and Sherry Yang. Worldgym: World model as an environment for policy evaluation. _arXiv preprint arXiv:2506.00613_, 2025. 
*   Radford et al. [2021] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision, 2021. URL [https://arxiv.org/abs/2103.00020](https://arxiv.org/abs/2103.00020). 
*   Ravi et al. [2025] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, et al. Sam 2: Segment anything in images and videos. In _ICLR_, volume 2025, pages 28085–28128, 2025. 
*   Rein et al. [2023] David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R Bowman. Gpqa: A graduate-level google-proof q&a benchmark. _arXiv preprint arXiv:2311.12022_, 2023. 
*   Seedance et al. [2025] Team Seedance, Heyi Chen, Siyan Chen, Xin Chen, Yanfei Chen, Ying Chen, Zhuo Chen, Feng Cheng, Tianheng Cheng, Xinqi Cheng, et al. Seedance 1.5 pro: A native audio-visual joint generation foundation model. _arXiv preprint arXiv:2512.13507_, 2025. 
*   Shang et al. [2026a] Yu Shang, Zhuohang Li, Yiding Ma, Weikang Su, Xin Jin, Ziyou Wang, Lei Jin, Xin Zhang, Yinzhou Tang, Haisheng Su, et al. Worldarena: A unified benchmark for evaluating perception and functional utility of embodied world models. _arXiv preprint arXiv:2602.08971_, 2026a. 
*   Shang et al. [2026b] Yu Shang, Xin Zhang, Yinzhou Tang, Lei Jin, Chen Gao, Wei Wu, and Yong Li. Roboscape: Physics-informed embodied world model. _NeurIPS_, 38:63674–63698, 2026b. 
*   Shen et al. [2025] Yichao Shen, Fangyun Wei, Zhiying Du, Yaobo Liang, Yan Lu, Jiaolong Yang, Nanning Zheng, and Baining Guo. Videovla: Video generators can be generalizable robot manipulators. _arXiv preprint arXiv:2512.06963_, 2025. 
*   Sterling [2023] Spencer Sterling. Zeroscope. [Online], 2023. 
*   Sun et al. [2025] Wenqiang Sun, Haiyu Zhang, Haoyuan Wang, Junta Wu, Zehan Wang, Zhenwei Wang, Yunhong Wang, Jun Zhang, Tengfei Wang, and Chunchao Guo. Worldplay: Towards long-term geometric consistency for real-time interactive world modeling. _arXiv preprint arXiv:2512.14614_, 2025. 
*   Team et al. [2025a] GigaBrain Team, Angen Ye, Boyuan Wang, Chaojun Ni, Guan Huang, Guosheng Zhao, Haoyun Li, Jie Li, Jiagang Zhu, Lv Feng, et al. Gigabrain-0: A world model-powered vision-language-action model. _arXiv preprint arXiv:2510.19430_, 2025a. 
*   Team et al. [2026a] GigaBrain Team, Boyuan Wang, Chaojun Ni, Guan Huang, Guosheng Zhao, Hao Li, Jie Li, Jindi Lv, Jingyu Liu, Lv Feng, et al. Gigabrain-0.5 m*: a vla that learns from world model-based reinforcement learning. _arXiv preprint arXiv:2602.12099_, 2026a. 
*   Team et al. [2025b] GigaWorld Team, Angen Ye, Boyuan Wang, Chaojun Ni, Guan Huang, Guosheng Zhao, Haoyun Li, Jiagang Zhu, Kerui Li, Mengyuan Xu, et al. Gigaworld-0: World models as data engine to empower embodied ai. _arXiv preprint arXiv:2511.19861_, 2025b. 
*   Team et al. [2025c] Kling Team, Jialu Chen, Yuanzheng Ci, Xiangyu Du, Zipeng Feng, Kun Gai, Sainan Guo, Feng Han, Jingbin He, Kang He, et al. Kling-omni technical report. _arXiv preprint arXiv:2512.16776_, 2025c. 
*   Team et al. [2026b] MotuBrain Team, Chendong Xiang, Fan Bao, Haitian Liu, Hengkai Tan, Hongzhe Bi, James Li, Jiabao Liu, Jingrui Pang, Kiro Jing, et al. Motubrain: An advanced world action model for robot control. _arXiv preprint arXiv:2604.27792_, 2026b. 
*   Teed and Deng [2020] Zachary Teed and Jia Deng. Raft: Recurrent all-pairs field transforms for optical flow. In _ECCV_, pages 402–419. Springer, 2020. 
*   Tseng et al. [2025] Wei-Cheng Tseng, Jinwei Gu, Qinsheng Zhang, Hanzi Mao, Ming-Yu Liu, Florian Shkurti, and Lin Yen-Chen. Scalable policy evaluation with video world models. _arXiv preprint arXiv:2511.11520_, 2025. 
*   Unterthiner et al. [2018] Thomas Unterthiner, Sjoerd Van Steenkiste, Karol Kurach, et al. Towards accurate generative models of video: A new metric & challenges. _arXiv preprint arXiv:1812.01717_, 2018. 
*   Wan et al. [2025a] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, Jiayu Wang, Jingfeng Zhang, Jingren Zhou, Jinkai Wang, Jixuan Chen, Kai Zhu, Kang Zhao, Keyu Yan, Lianghua Huang, Mengyang Feng, Ningyi Zhang, Pandeng Li, Pingyu Wu, Ruihang Chu, Ruili Feng, Shiwei Zhang, Siyang Sun, Tao Fang, Tianxing Wang, Tianyi Gui, Tingyu Weng, Tong Shen, Wei Lin, Wei Wang, Wei Wang, Wenmeng Zhou, Wente Wang, Wenting Shen, Wenyuan Yu, Xianzhong Shi, Xiaoming Huang, Xin Xu, Yan Kou, Yangyu Lv, Yifei Li, Yijing Liu, Yiming Wang, Yingya Zhang, Yitong Huang, Yong Li, You Wu, Yu Liu, Yulin Pan, Yun Zheng, Yuntao Hong, Yupeng Shi, Yutong Feng, Zeyinzi Jiang, Zhen Han, Zhi-Fan Wu, and Ziyu Liu. Wan: Open and advanced large-scale video generative models. _arXiv preprint arXiv:2503.20314_, 2025a. 
*   Wan et al. [2025b] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. _arXiv preprint arXiv:2503.20314_, 2025b. 
*   Wang et al. [2025a] Boyuan Wang, Xinpan Meng, Xiaofeng Wang, Zheng Zhu, Angen Ye, Yang Wang, Zhiqin Yang, Chaojun Ni, Guan Huang, and Xingang Wang. Embodiedreamer: Advancing real2sim2real transfer for policy training via embodied world modeling. _arXiv preprint arXiv:2507.05198_, 2025a. 
*   Wang et al. [2025b] Boyuan Wang, Runqi Ouyang, Xiaofeng Wang, Zheng Zhu, Guosheng Zhao, Chaojun Ni, Guan Huang, Lihong Liu, and Xingang Wang. Humandreamer-x: Photorealistic single-image human avatars reconstruction via gaussian restoration. _arXiv preprint arXiv:2504.03536_, 2025b. 
*   Wang et al. [2025c] Boyuan Wang, Xiaofeng Wang, Chaojun Ni, Guosheng Zhao, Zhiqin Yang, Zheng Zhu, Muyang Zhang, Yukun Zhou, Xinze Chen, Guan Huang, Lihong Liu, and Xingang Wang. Humandreamer: Generating controllable human-motion videos via decoupled generation. _arXiv preprint arXiv:2503.24026_, 2025c. 
*   Wang et al. [2026a] Chen Wang, Chuhao Chen, Yiming Huang, Zhiyang Dou, Yuan Liu, Jiatao Gu, and Lingjie Liu. Physctrl: Generative physics for controllable and physics-grounded video generation. _NeurIPS_, 38:167907–167932, 2026a. 
*   Wang et al. [2024a] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. _arXiv preprint arXiv:2409.12191_, 2024a. 
*   Wang et al. [2025d] Weijie Wang, Jiagang Zhu, Zeyu Zhang, Xiaofeng Wang, Zheng Zhu, Guosheng Zhao, Chaojun Ni, Haoxiao Wang, Guan Huang, Xinze Chen, et al. Drivegen3d: Boosting feed-forward driving scene generation with efficient video diffusion. _arXiv preprint arXiv:2510.15264_, 2025d. 
*   Wang et al. [2024b] Xiaofeng Wang, Zheng Zhu, Guan Huang, Xinze Chen, Jiagang Zhu, and Jiwen Lu. Drivedreamer: Towards real-world-drive world models for autonomous driving. In _ECCV_, 2024b. 
*   Wang et al. [2024c] Xiaofeng Wang, Zheng Zhu, Guan Huang, Boyuan Wang, Xinze Chen, and Jiwen Lu. Worlddreamer: Towards general world models for video generation via predicting masked tokens. _arXiv preprint arXiv:2401.09985_, 2024c. 
*   Wang et al. [2026b] Xiaofeng Wang, Kang Zhao, Feng Liu, Jiayu Wang, Guosheng Zhao, Xiaoyi Bao, Zheng Zhu, and Yingya Zhang. Egovid-5m: A large-scale video-action dataset for egocentric videos generation. _NeurIPS_, 38, 2026b. 
*   Wang et al. [2025e] Yuang Wang, Chao Wen, Haoyu Guo, Sida Peng, Minghan Qin, Hujun Bao, Xiaowei Zhou, and Ruizhen Hu. Precise action-to-video generation through visual action prompts. In _ICCV_, pages 12713–12724, 2025e. 
*   Wang et al. [2024d] Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, et al. Mmlu-pro: A more robust and challenging multi-task language understanding benchmark. _NeurIPS_, 37:95266–95290, 2024d. 
*   Wu et al. [2024] Kun Wu, Chengkai Hou, Jiaming Liu, Zhengping Che, Xiaozhu Ju, Zhuqin Yang, Meng Li, Yinuo Zhao, Zhiyuan Xu, Guang Yang, et al. Robomind: Benchmark on multi-embodiment intelligence normative data for robot manipulation. _arXiv preprint arXiv:2412.13877_, 2024. 
*   Wu et al. [2026] Zhenyu Wu, Xiuwei Xu, Yukun Zhou, Yifan Li, Qiuping Deng, Xiaofeng Wang, Zheng Zhu, Bingyao Yu, Ziwei Wang, Jiwen Lu, et al. imac: Translating actions into motion and contact images for embodied world models. _arXiv preprint arXiv:2606.09813_, 2026. 
*   Xiang et al. [2025] Jiannan Xiang, Yi Gu, Zihan Liu, Zeyu Feng, Qiyue Gao, Yiyan Hu, Benhao Huang, Guangyi Liu, Yichi Yang, Kun Zhou, et al. Pan: A world model for general, interactable, and long-horizon world simulation. _arXiv preprint arXiv:2511.09057_, 2025. 
*   Xu et al. [2025a] Xuancheng Xu, Ming Tao, and Bing-Kun Bao. Clgc: Continuous layout guidance for consistent text-to-video editing. In _ICME_, pages 1–6. IEEE, 2025a. 
*   Xu et al. [2026a] Xuancheng Xu, Gengyun Jia, and Bing-Kun Bao. Disco-lora: Disentangled composition of content, style, and motion for multi-concept video customization. _arXiv preprint arXiv:2606.26668_, 2026a. 
*   Xu et al. [2026b] Xuancheng Xu, Yaning Li, Sisi You, and Bing-Kun Bao. Smrabooth: Subject and motion representation alignment for customized video generation. In _CVPR_, pages 16130–16141, 2026b. 
*   Xu et al. [2025b] Yuan Xu, Jiabing Yang, Xiaofeng Wang, Yixiang Chen, Zheng Zhu, Bowen Fang, Guan Huang, Xinze Chen, Yun Ye, Qiang Zhang, et al. Egodemogen: Novel egocentric demonstration generation enables viewpoint-robust manipulation. _arXiv preprint arXiv:2509.22578_, 2025b. 
*   Yakefu et al. [2025] Adina Yakefu, Bin Xie, Chongyang Xu, Enwen Zhang, Erjin Zhou, Fan Jia, Haitao Yang, Haoqiang Fan, Haowei Zhang, Hongyang Peng, et al. Robochallenge: Large-scale real-robot evaluation of embodied policies. _arXiv preprint arXiv:2510.17950_, 2025. 
*   Yang et al. [2024a] Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything v2. _arXiv:2406.09414_, 2024a. 
*   Yang et al. [2026] Ning Yang, Yan Huang, Kaiwen Peng, Ziheng He, Kai Wang, Cui Miao, Kailin Lyu, Guo Li, Xiaofeng Wang, Zheng Zhu, et al. Wam-nav: Asymmetric latent world-action modeling for unified visual navigation. _arXiv preprint arXiv:2606.04907_, 2026. 
*   Yang et al. [2024b] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. _arXiv preprint arXiv:2408.06072_, 2024b. 
*   Yang et al. [2024c] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. _arXiv preprint arXiv:2408.06072_, 2024c. 
*   Ye et al. [2026a] Angen Ye, Boyuan Wang, Chaojun Ni, Guan Huang, Guosheng Zhao, Hao Li, Hengtao Li, Jie Li, Jindi Lv, Jingyu Liu, et al. Gigaworld-policy: An efficient action-centered world–action model. _arXiv preprint arXiv:2603.17240_, 2026a. 
*   Ye et al. [2025] Deheng Ye, Fangyun Zhou, Jiacheng Lv, Jianqi Ma, Jun Zhang, Junyan Lv, Junyou Li, Minwen Deng, Mingyu Yang, Qiang Fu, et al. Yan: Foundational interactive video generation. _arXiv preprint arXiv:2508.08601_, 2025. 
*   Ye et al. [2026b] Seonghyeon Ye, Yunhao Ge, Kaiyuan Zheng, Shenyuan Gao, Sihyun Yu, George Kurian, Suneel Indupuru, You Liang Tan, Chuning Zhu, Jiannan Xiang, et al. World action models are zero-shot policies. _arXiv preprint arXiv:2602.15922_, 2026b. 
*   Yin et al. [2025] Tianwei Yin, Qiang Zhang, Richard Zhang, William T Freeman, Fredo Durand, Eli Shechtman, and Xun Huang. From slow bidirectional to fast autoregressive video diffusion models. In _CVPR_, pages 22963–22974, 2025. 
*   Yuan et al. [2026a] Shenghai Yuan, Yuanyang Yin, Zongjian Li, Xinwei Huang, Xiao Yang, and Li Yuan. Helios: Real real-time long video generation model. _arXiv preprint arXiv:2603.04379_, 2026a. 
*   Yuan et al. [2026b] Tianyuan Yuan, Zibin Dong, Yicheng Liu, and Hang Zhao. Fast-wam: Do world action models need test-time future imagination? _arXiv preprint arXiv:2603.16666_, 2026b. 
*   Yue et al. [2024] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In _CVPR_, pages 9556–9567, 2024. 
*   Yue et al. [2025] Xiang Yue, Tianyu Zheng, Yuansheng Ni, Yubo Wang, Kai Zhang, Shengbang Tong, Yuxuan Sun, Botao Yu, Ge Zhang, Huan Sun, et al. Mmmu-pro: A more robust multi-discipline multimodal understanding benchmark. In _Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_, pages 15134–15186, 2025. 
*   Zeng et al. [2025] Kai Zeng, Zhanqian Wu, Kaixin Xiong, Xiaobao Wei, Xiangyu Guo, Zhenxin Zhu, Kalok Ho, Lijun Zhou, Bohan Zeng, Ming Lu, et al. Rethinking driving world model as synthetic data generator for perception tasks. _arXiv preprint arXiv:2510.19195_, 2025. 
*   Zhai et al. [2025] Andy Zhai, Brae Liu, Bruno Fang, Chalse Cai, Ellie Ma, Ethan Yin, Hao Wang, Hugo Zhou, James Wang, Lights Shi, et al. Igniting vlms toward the embodied space. _arXiv preprint arXiv:2509.11766_, 2025. 
*   Zhang et al. [2024a] Guozhen Zhang, Chunxu Liu, Yutao Cui, Xiaotong Zhao, Kai Ma, and Limin Wang. Vfimamba: Video frame interpolation with state space models. _NeurIPS_, 37:107225–107248, 2024a. 
*   Zhang et al. [2025a] Shiduo Zhang, Zhe Xu, Peiju Liu, Xiaopeng Yu, Yuan Li, Qinghui Gao, Zhaoye Fei, Zhangyue Yin, Zuxuan Wu, Yu-Gang Jiang, et al. Vlabench: A large-scale benchmark for language-conditioned robotics manipulation with long-horizon reasoning tasks. In _ICCV_, pages 11142–11152, 2025a. 
*   Zhang et al. [2025b] Shiyi Zhang, Junhao Zhuang, Zhaoyang Zhang, Ying Shan, and Yansong Tang. Flexiact: Towards flexible action control in heterogeneous scenarios. In _Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers_, pages 1–11, 2025b. 
*   Zhang et al. [2024b] Tianyuan Zhang, Hong-Xing Yu, Rundi Wu, Brandon Y Feng, Changxi Zheng, Noah Snavely, Jiajun Wu, and William T Freeman. Physdreamer: Physics-based interaction with 3d objects via video generation. In _ECCV_, pages 388–406. Springer, 2024b. 
*   Zhao et al. [2025a] Guosheng Zhao, Chaojun Ni, Xiaofeng Wang, Zheng Zhu, Xueyang Zhang, Yida Wang, Guan Huang, Xinze Chen, Boyuan Wang, Youyi Zhang, et al. Drivedreamer4d: World models are effective data machines for 4d driving scene representation. In _CVPR_, pages 12015–12026, 2025a. 
*   Zhao et al. [2025b] Guosheng Zhao, Xiaofeng Wang, Chaojun Ni, Zheng Zhu, Wenkang Qin, Guan Huang, and Xingang Wang. Recondreamer++: Harmonizing generative and reconstructive models for driving scene representation. _arXiv preprint arXiv:2503.18438_, 2025b. 
*   Zhao et al. [2025c] Guosheng Zhao, Xiaofeng Wang, Zheng Zhu, Xinze Chen, Guan Huang, Xiaoyi Bao, and Xingang Wang. Drivedreamer-2: Llm-enhanced world models for diverse driving video generation. In _AAAI_, volume 39, pages 10412–10420, 2025c. 
*   Zhao et al. [2026a] Guosheng Zhao, Yaozeng Wang, Xiaofeng Wang, Zheng Zhu, Tingdong Yu, Guan Huang, Yongchen Zai, Ji Jiao, Changliang Xue, Xiaole Wang, et al. Unidrivedreamer: A single-stage multimodal world model for autonomous driving. _arXiv preprint arXiv:2602.02002_, 2026a. 
*   Zhao et al. [2026b] Min Zhao, Hongzhou Zhu, Kaiwen Zheng, Zihan Zhou, Bokai Yan, Xinyuan Li, Xiao Yang, Chongxuan Li, and Jun Zhu. Causal forcing++: Scalable few-step autoregressive diffusion distillation for real-time interactive video generation. _arXiv preprint arXiv:2605.15141_, 2026b. 
*   Zheng et al. [2024] Guangcong Zheng, Teng Li, Rui Jiang, Yehao Lu, Tao Wu, and Xi Li. Cami2v: Camera-controlled image-to-video diffusion model. _arXiv preprint arXiv:2410.15957_, 2024. 
*   Zhong et al. [2024] Wanjun Zhong, Ruixiang Cui, Yiduo Guo, Yaobo Liang, Shuai Lu, Yanlin Wang, Amin Saied, Weizhu Chen, and Nan Duan. Agieval: A human-centric benchmark for evaluating foundation models. In _Findings of the association for computational linguistics: NAACL 2024_, pages 2299–2314, 2024. 
*   Zhou et al. [2026a] Jiawei Zhou, Zhenxin Zhu, Lingyi Du, Linye Lyu, Lijun Zhou, Zhanqian Wu, Hongcheng Luo, Zhuotao Tian, Bing Wang, Guang Chen, et al. Toward physically consistent driving video world models under challenging trajectories. _arXiv preprint arXiv:2603.24506_, 2026a. 
*   Zhou et al. [2026b] Lijun Zhou, Hongcheng Luo, Zhenxin Zhu, Cheng Chi, Mingfei Tu, Kaixin Xiong, Lei Gong, Zhanqian Wu, Zehan Zhang, Fangzhen Li, et al. Xiaomi ev world model: A joint world model integrating reconstruction and generation for autonomous driving. _arXiv preprint arXiv:2605.18137_, 2026b. 
*   Zhou et al. [2026c] Pengfei Zhou, Shengcong Chen, Di Chen, Jiaxu Wang, Rongjun Jin, Bingwen Zhu, Yike Pan, Songen Gu, Kuanning Wang, Shufeng Nan, Xingyu Qiu, Chenhao Qiu, Pu Yang, Yunuo Cai, Jianxiong Gao, Yifan Li, Yanwei Fu, Xiangyu Yue, Zhi Chen, and Jianlan Luo. \tau_{0}-WM: A unified video-action world model for robotic manipulation. _arXiv preprint arXiv:2606.01027_, 2026c. 
*   Zhou et al. [2026d] Yang Zhou, Xiaofeng Wang, Hao Shao, Letian Wang, Guosheng Zhao, Jiangnan Shao, Jiagang Zhu, Tingdong Yu, Zheng Zhu, Guan Huang, et al. Drivedreamer-policy: A geometry-grounded world-action model for unified generation and planning. _arXiv preprint arXiv:2604.01765_, 2026d. 
*   Zhu et al. [2025a] Fangqi Zhu, Hongtao Wu, Song Guo, Yuxiao Liu, Chilam Cheang, and Tao Kong. Irasim: A fine-grained world model for robot manipulation. In _ICCV_, pages 9834–9844, 2025a. 
*   Zhu et al. [2026] Hongzhou Zhu, Min Zhao, Guande He, Hang Su, Chongxuan Li, and Jun Zhu. Causal forcing: Autoregressive diffusion distillation done right for high-quality real-time interactive video generation. _arXiv preprint arXiv:2602.02214_, 2026. 
*   Zhu et al. [2020] Yuke Zhu, Josiah Wong, Ajay Mandlekar, Roberto Martín-Martín, Abhishek Joshi, Kevin Lin, Abhiram Maddukuri, Soroush Nasiriany, and Yifeng Zhu. robosuite: A modular simulation framework and benchmark for robot learning. _arXiv preprint arXiv:2009.12293_, 2020. 
*   Zhu et al. [2024] Zheng Zhu, Xiaofeng Wang, Wangbo Zhao, Chen Min, Bohan Li, Nianchen Deng, Min Dou, Yuqi Wang, Botian Shi, Kai Wang, et al. Is sora a world simulator? a comprehensive survey on general world models and beyond. _arXiv preprint arXiv:2405.03520_, 2024. 
*   Zhu et al. [2025b] Ziyue Zhu, Zhanqian Wu, Zhenxin Zhu, Lijun Zhou, Haiyang Sun, Bing Wan, Kun Ma, Guang Chen, Hangjun Ye, Jin Xie, et al. Worldsplat: Gaussian-centric feed-forward 4d scene generation for autonomous driving. _arXiv preprint arXiv:2509.23402_, 2025b.

