Title: Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models

URL Source: https://arxiv.org/html/2607.03751

Markdown Content:
Xinyi Xie 1 Zican Hu 1 Zhanyu Liu 1 Yicheng Dong 1 Wenhao Wu 1

Zhenhong Sun 2 Haoran Li 3 Chunlin Chen 1 Zhi Wang 1 Pichao Wang 4

1 Nanjing University 

2 Australian National University 

3 Institute of Automation, Chinese Academy of Sciences 

4 Nvidia 

{xinyixie,zicanhu,zhanyuliu,yichengdong,wenhaowu}@smail.nju.edu.cn 

zhenhong.sun@anu.edu.au lihaoran2015@ia.ac.cn 

{clchen,zhiwang}@nju.edu.cn pichaowang@gmail.com

###### Abstract

Vision-Language-Action (VLA) models acquire broad embodied capabilities through large-scale pretraining, yet their generalization remains far more fragile than that of LLMs and VLMs. The prevailing remedy, post-training via supervised fine-tuning or reinforcement learning, improves task-specific performance but narrows the generalist capability that makes pretraining valuable. We identify a key bottleneck: VLA failures stem not only from action generation but also from action evaluation. A diagnostic pass@k study confirms that frozen VLAs already contain competent behaviors in their output distribution, with overall success rates rising from 33% at pass@1 to 92% at pass@32. Inspired by this, we propose SVA (S earch, V alue, and A ct), a simple framework that equips frozen VLA policies with long-term consequence awareness. SVA first uses Monte-Carlo tree search in simulation to fully explore the VLA’s output distribution and collect diverse trajectories annotated with empirical returns; this knowledge is then distilled into a lightweight Q-value model that predicts the expected consequence of candidate actions; at deployment, the frozen VLA proposes multiple candidates and the evaluator selects the one with the highest uncertainty-regularized Q-value, requiring no simulator access. By decoupling action proposal from consequence evaluation, SVA preserves the generalization capacity of the VLA backbone while substantially improving task success rates. Experiments across embodied benchmarks show that SVA consistently improves generalization on unseen tasks and exhibits strong test-time scaling behavior. Strikingly, SVA enables a 9B VLA to outperform a 27B VLA by 7 points at 27% lower inference latency, suggesting that scaling test-time evaluation is more cost-effective than scaling model size.

4 4 footnotetext: This work does not relate to the author’s position at Nvidia.

> Keywords: VLAs, Test-Time scaling, Monte-Carlo Tree Search, Action Evaluation

### 1 Introduction

Vision-Language-Action (VLA) models have emerged as a promising paradigm for building generalist robotic agents, leveraging large-scale pretraining on diverse vision-language and robotic data to acquire broad embodied capabilities[[5](https://arxiv.org/html/2607.03751#bib.bib38 "RT-1: robotics transformer for real-world control at scale"), [55](https://arxiv.org/html/2607.03751#bib.bib21 "RT-2: vision-language-action models transfer web knowledge to robotic control")]. Despite encouraging progress from models such as OpenVLA[[18](https://arxiv.org/html/2607.03751#bib.bib22 "OpenVLA: an open-source vision-language-action model")], \pi_{0.5}[[15](https://arxiv.org/html/2607.03751#bib.bib47 "π0.5: A vision-language-action model with open-world generalization")], and GR00T[[3](https://arxiv.org/html/2607.03751#bib.bib49 "GR00T N1: an open foundation model for generalist humanoid robots")], VLAs remain far more fragile than their LLM/VLM counterparts[[1](https://arxiv.org/html/2607.03751#bib.bib40 "GPT-4 technical report")], frequently failing on tasks that lie only modestly outside their training distribution[[22](https://arxiv.org/html/2607.03751#bib.bib33 "Evaluating real-world robot manipulation policies in simulation")]. The prevailing remedy is post-training, via supervised fine-tuning (SFT) on curated demonstrations[[39](https://arxiv.org/html/2607.03751#bib.bib42 "Octo: an open-source generalist robot policy")] or reinforcement learning (RL) with environment rewards[[8](https://arxiv.org/html/2607.03751#bib.bib46 "ConRFT: a reinforced fine-tuning method for VLA models via consistency policy"), [50](https://arxiv.org/html/2607.03751#bib.bib45 "RLinf-VLA: a unified and efficient framework for reinforcement learning of vision-language-action models")], both of which require updating the VLA backbone. Yet updating a multi-billion-parameter backbone is computationally expensive[[42](https://arxiv.org/html/2607.03751#bib.bib23 "TinyVLA: towards fast, data-efficient vision-language-action models for robotic manipulation"), [31](https://arxiv.org/html/2607.03751#bib.bib50 "FAST: efficient action tokenization for vision-language-action models")], and the resulting specialist policies tend to narrow the generalist capacity acquired during pretraining[[17](https://arxiv.org/html/2607.03751#bib.bib51 "Fine-tuning vision-language-action models: optimizing speed and success")]. This tension between task performance and generalization preservation is particularly acute for VLAs, whose generalization is already far more limited than that of LLMs/VLMs and far more costly to lose[[26](https://arxiv.org/html/2607.03751#bib.bib11 "LIBERO: benchmarking knowledge transfer for lifelong robot learning")]. How to strengthen VLA performance without sacrificing its hard-won generalist capabilities is therefore a central and pressing challenge.

![Image 1: Refer to caption](https://arxiv.org/html/2607.03751v1/x1.png)

Figure 1: Post-training vs. test-time scaling.

We argue that the root cause of VLA deployment failures lies not only in action generation but also in the absence of action evaluation, i.e., the inability to anticipate the consequences of a proposed action before execution. VLAs are trained to imitate, not to evaluate: they produce locally plausible actions given the current observation[[4](https://arxiv.org/html/2607.03751#bib.bib41 "π0: A vision-language-action flow model for general robot control")], but receive no signal about what an action leads to (a successful grasp, a collision, or an irrecoverable state)[[10](https://arxiv.org/html/2607.03751#bib.bib10 "Causal confusion in imitation learning"), [9](https://arxiv.org/html/2607.03751#bib.bib24 "Exploring the limitations of behavior cloning for autonomous driving")], and no signal about whether a different action might yield a better outcome in the long term[[29](https://arxiv.org/html/2607.03751#bib.bib25 "Steering your generalists: improving robotic foundation models via value guidance"), [38](https://arxiv.org/html/2607.03751#bib.bib29 "Reinforcement learning: an introduction")]. Sec.[3](https://arxiv.org/html/2607.03751#S3 "3 Diagnosing the VLA Bottleneck: Generation or Evaluation? ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models") provides a diagnostic pass@k study to support this view: when a frozen VLA is allowed multiple independent attempts, success rates rise dramatically from 33% at pass@1 to 92% at pass@32. The analysis reveals that high-quality, task-completing actions already reside within the VLA’s output distribution; the model simply cannot distinguish them from mediocre or harmful alternatives it produces alongside. Beyond generation, evaluation is an equally critical yet overlooked bottleneck.

This diagnosis motivates a fundamentally different strategy for VLA’s policy improvement. Rather than rewriting the VLA’s parameters to generate better actions at the cost of generalization, we can equip it with a lightweight consequence evaluator that judges candidate actions by their predicted long-term outcomes. The VLA backbone remains entirely frozen, preserving its generalist capacity, while the evaluator provides the missing “look before you leap” capability: the ability to foresee which candidate action is most likely to lead to task success.

We draw inspiration from the Bitter Lesson[[37](https://arxiv.org/html/2607.03751#bib.bib18 "The bitter lesson")]: impactful advances in AI stem from methods that scale computation through search and learning. We propose SVA (S earch, V alue, and A ct), a simple three-stage framework that elicits policy improvement for frozen VLAs. In Search, we employ Monte-Carlo tree search (MCTS) in simulation to fully explore the VLA’s output distribution via principled look-ahead search, efficiently discovering diverse trajectories annotated with empirical returns. In Value, we distill the knowledge obtained from MCTS into a lightweight Q-value model that predicts the expected consequence of executing a candidate action. Built on a small VLM backbone, this consequence evaluator compresses the expensive search process into a fast-to-evaluate function that generalizes across states and tasks. In Act, the frozen VLA proposes N candidate actions, and the evaluator selects the one with the highest uncertainty-regularized Q-value. SVA provides a natural and tunable mechanism for test-time scaling: increasing N improves the probability of selecting a high-quality action, with inference latency scaling sub-linearly in N (see Sec.[5.3](https://arxiv.org/html/2607.03751#S5.SS3 "5.3 Scaling SVA at Test Time: Performance, Latency, and Cost-Effectiveness ‣ 5 Experiments ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models")).

In summary, our contributions are threefold:

*   •
Problem. We identify an evaluation bottleneck in frozen VLA policies: high-quality actions are often present in the output distribution, but single-shot generation cannot reliably pick them out.

*   •
Method. We propose a search-and-learning recipe that distills simulation-based tree search into a value model, enabling real-time action evaluation without simulator access at deployment.

*   •
Results. SVA delivers consistent gains across multiple manipulation benchmarks and VLA backbones, serving as a practical and effective alternative to costly policy fine-tuning; notably, a 9B VLA with SVA outperforms a 27B VLA by 7 points at 27% lower inference latency.

### 2 Related Work

VLA Models. A growing body of work adapts the success of LLMs/VLMs into embodied domains for generalist robotic control[[21](https://arxiv.org/html/2607.03751#bib.bib5 "SimpleVLA-RL: scaling VLA training via reinforcement learning")], with a diverse family of VLAs emerging, including RT-1/RT-2[[5](https://arxiv.org/html/2607.03751#bib.bib38 "RT-1: robotics transformer for real-world control at scale"), [55](https://arxiv.org/html/2607.03751#bib.bib21 "RT-2: vision-language-action models transfer web knowledge to robotic control")], OpenVLA[[18](https://arxiv.org/html/2607.03751#bib.bib22 "OpenVLA: an open-source vision-language-action model")], \pi-series[[4](https://arxiv.org/html/2607.03751#bib.bib41 "π0: A vision-language-action flow model for general robot control"), [14](https://arxiv.org/html/2607.03751#bib.bib48 "π0.6: A VLA that learns from experience"), [15](https://arxiv.org/html/2607.03751#bib.bib47 "π0.5: A vision-language-action model with open-world generalization")], GR00T[[3](https://arxiv.org/html/2607.03751#bib.bib49 "GR00T N1: an open foundation model for generalist humanoid robots")], and Octo[[39](https://arxiv.org/html/2607.03751#bib.bib42 "Octo: an open-source generalist robot policy")]. Despite this progress, VLAs remain markedly more fragile than their LLM/VLM counterparts, frequently failing on tasks that lie only modestly outside their training distribution[[22](https://arxiv.org/html/2607.03751#bib.bib33 "Evaluating real-world robot manipulation policies in simulation")]. This fragility motivates a large body of post-training research, mainly SFT on curated data[[31](https://arxiv.org/html/2607.03751#bib.bib50 "FAST: efficient action tokenization for vision-language-action models"), [42](https://arxiv.org/html/2607.03751#bib.bib23 "TinyVLA: towards fast, data-efficient vision-language-action models for robotic manipulation")] or RL with environment rewards[[44](https://arxiv.org/html/2607.03751#bib.bib7 "Self-improving vision-language-action models with data generation via residual RL"), [21](https://arxiv.org/html/2607.03751#bib.bib5 "SimpleVLA-RL: scaling VLA training via reinforcement learning"), [45](https://arxiv.org/html/2607.03751#bib.bib28 "Robot fine-tuning made easy: pre-training rewards and policies for autonomous real-world reinforcement learning")]. However, these methods share two drawbacks: updating billion-parameter backbones is computationally costly[[17](https://arxiv.org/html/2607.03751#bib.bib51 "Fine-tuning vision-language-action models: optimizing speed and success"), [52](https://arxiv.org/html/2607.03751#bib.bib8 "Sim2Real VLA: zero-shot generalization of synthesized skills to realistic manipulation")], and specialist fine-tuning often narrows generalist capabilities acquired during pretraining[[26](https://arxiv.org/html/2607.03751#bib.bib11 "LIBERO: benchmarking knowledge transfer for lifelong robot learning"), [11](https://arxiv.org/html/2607.03751#bib.bib6 "Actions as language: fine-tuning VLMs into VLAs without catastrophic forgetting")]. In contrast, SVA keeps the VLA backbone frozen and steers it via a lightweight external evaluator, avoiding costly backbone updates while preserving its generalist capabilities.

Action Evaluation for Robot Policies. This complementary line avoids modifying the backbone and learns a verifier to guide generation, which has proven effective in LLM reasoning, where outcome- and process-reward models[[25](https://arxiv.org/html/2607.03751#bib.bib2 "Let’s verify step by step"), [40](https://arxiv.org/html/2607.03751#bib.bib39 "Solving math word problems with process-and outcome-based feedback")] combined with search[[49](https://arxiv.org/html/2607.03751#bib.bib12 "Tree of thoughts: deliberate problem solving with large language models"), [41](https://arxiv.org/html/2607.03751#bib.bib14 "AlphaZero-like tree-search can guide large language model decoding and training"), [51](https://arxiv.org/html/2607.03751#bib.bib13 "ReST-MCTS∗: LLM self-training via process reward guided tree search")] boost a frozen generator. A growing body of work brings this idea to robot policies[[12](https://arxiv.org/html/2607.03751#bib.bib30 "Inner monologue: embodied reasoning through planning with language models"), [24](https://arxiv.org/html/2607.03751#bib.bib57 "Adaptive action chunking at inference-time for vision-language-action models"), [2](https://arxiv.org/html/2607.03751#bib.bib31 "ReVer: reasoning-guided verification for embodied agents")]. SayCan[[13](https://arxiv.org/html/2607.03751#bib.bib26 "Do as I can, not as I say: grounding language in robotic affordances")] grounds LLM-proposed plans by scoring candidate skills with an affordance function, V-GPS[[29](https://arxiv.org/html/2607.03751#bib.bib25 "Steering your generalists: improving robotic foundation models via value guidance")] learns a value function via offline RL pretraining on re-annotated trajectories, RoboMonkey[[19](https://arxiv.org/html/2607.03751#bib.bib27 "RoboMonkey: scaling test-time sampling and verification for vision-language-action models")] trains an action preference model using synthetic preference data, and V-VLAPS[[32](https://arxiv.org/html/2607.03751#bib.bib61 "V-VLAPS: value-guided planning for vision-language-action models")] learns a value model to guide tree search at deployment. Other methods train the action verifier by SFT on synthetic reasoning data in VeGAS[[34](https://arxiv.org/html/2607.03751#bib.bib58 "Think twice, act once: verifier-guided action selection for embodied agents")], offline RL on curated dataset in Hume[[36](https://arxiv.org/html/2607.03751#bib.bib52 "Hume: introducing system-2 thinking in visual-language-action model")], contrastive learning in CoVer-VLA[[20](https://arxiv.org/html/2607.03751#bib.bib59 "Scaling verification can be more effective than scaling policy learning for vision-language-action alignment")], probability matching in TACO[[48](https://arxiv.org/html/2607.03751#bib.bib55 "Steering vision-language-action models as anti-exploration: a test-time scaling approach")], or evolutionary diffusion in VLA-Pilot[[23](https://arxiv.org/html/2607.03751#bib.bib32 "Towards deploying VLA without fine-tuning: plug-and-play inference-time VLA policy steering via embodied evolutionary diffusion")], among others[[28](https://arxiv.org/html/2607.03751#bib.bib43 "Policy agnostic RL: offline RL and online RL fine-tuning of any class and backbone"), [43](https://arxiv.org/html/2607.03751#bib.bib53 "Do what you say: steering vision-language-action models via runtime reasoning-action alignment verification")]. Further, MG-Select[[16](https://arxiv.org/html/2607.03751#bib.bib9 "Verifier-free test-time sampling for vision-language-action models")] leverages the model’s internal properties to score actions without an external verifier. World model-based methods[[47](https://arxiv.org/html/2607.03751#bib.bib3 "Learning interactive real-world simulators"), [54](https://arxiv.org/html/2607.03751#bib.bib15 "RoboDreamer: learning compositional world models for robot imagination"), [53](https://arxiv.org/html/2607.03751#bib.bib16 "DINO-WM: world models on pre-trained visual features enable zero-shot planning"), [30](https://arxiv.org/html/2607.03751#bib.bib54 "Improving pre-trained vision-language-action policies with model-based search")] evaluate actions by rolling them out in a learned world model, but incur high inference cost and accumulated model error over long horizons. In contrast, SVA complements these approaches along three axes: it learns from MCTS rollouts rather than offline RL or synthetic data, producing consequence-aware signals rather than single-step preference, and obtains long-horizon estimates without costly world-model rollouts.

### 3 Diagnosing the VLA Bottleneck: Generation or Evaluation?

We argue that VLA failures arise not only from action generation but also from the absence of action evaluation. To this end, we conduct a diagnostic pass@k study to ask how often at least one out of k rollouts succeeds, probing the latent policy capability by isolating evaluation from generation.

Model, Benchmark, and Evaluation Protocol. We evaluate pass@k behavior of OpenVLA on Simpler and Libero, and \pi_{0.5} on RoboTwin. The three embodied benchmarks cover a range of task structures, including long-horizon object rearrangement, tabletop manipulation, and precise interaction tasks. For each task, we sample N=50 independent rollouts from the policy distribution with c successes and compute \mathrm{Pass}@k=\mathbb{E}\bigl[1-\binom{N-c}{k}/\binom{N}{k}\bigr], which estimates the probability that at least one of k independent attempts succeeds[[6](https://arxiv.org/html/2607.03751#bib.bib36 "Evaluating large language models trained on code")]. Fig.[2](https://arxiv.org/html/2607.03751#S3.F2 "Figure 2 ‣ 3 Diagnosing the VLA Bottleneck: Generation or Evaluation? ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models") shows the diagnostic results.

![Image 2: Refer to caption](https://arxiv.org/html/2607.03751v1/x2.png)

Figure 2: Pass@k results across embodied benchmarks.

Observation 1: Successful Behaviors Already Exist in the Output Distribution of Frozen VLAs. The probability of obtaining at least one success increases sharply with the number of attempts, with the same trend holding on every task. The average success rate rises from 33% at pass@1 to 92% at pass@32, an absolute gain of 58 points. Frozen VLAs, despite failing under single-shot execution due to stochastic sampling or compounding execution errors, assign non-trivial probability mass to successful behaviors within their output distribution. The deployment bottleneck thus does not lie in the absence of competent actions, but in the inability to identify them before execution. This reframes VLA’s failure mode as a deficiency in action evaluation rather than solely in action generation.

Observation 2: The Evaluation Bottleneck Is Most Severe on Tasks of Intermediate Difficulty. Already-easy tasks saturate quickly: Pick up Book/Soup and Sauce in Basket starts from 0.84/0.64 to 0.99/0.96 at only pass@4, leaving limited room for further gains. Tasks that exceed the base policy’s capability exhibit limited gains regardless of sampling budget: Stack Cubes only improves from 0.02 to 0.50 at pass@32. The practical value of action evaluation concentrates in the intermediate-difficulty regime, where the evaluator can efficiently translate latent competence into reliable execution: Turn Switch/Rotate QRCode start from 0.31/0.35 to near 1 at pass@32.

Observation 3: Diminishing Marginal Returns Suggest the Importance of Smarter Scaling Strategies. Although pass@k improves monotonically with k, the marginal gain shrinks rapidly as the sampling budget grows: \Delta_{1\rightarrow 2}\!=\!0.13,\Delta_{2\rightarrow 4}\!=\!0.13,\Delta_{4\rightarrow 8}\!=\!0.12,\Delta_{8\rightarrow 16}\!=\!0.10,\Delta_{16\rightarrow 32}\!=\!0.08. This diminishing-return pattern exposes inefficiency of naive independent sampling: while drawing more rollouts can surface better trajectories, the informational gain per sample shrinks rapidly even as environment-interaction cost grows linearly. A practical scaling strategy should therefore allocate additional computation more efficiently, such as evaluating promising actions before execution.

### 4 Look Before You Leap: Addressing the Evaluation Bottleneck with SVA

We consider an embodied task as a language-conditioned MDP \langle\mathcal{S},\mathcal{A},T,R,\gamma,\chi\rangle, where \mathcal{S}/\mathcal{A} is the state/action space, T(s^{\prime}|s,a) is the state transition function, R(s,a) is the reward function, \gamma is the discount factor, and \chi is the language space. At each step t, the agent receives a state s_{t}\in\mathcal{S} that may comprise an egocentric image, robot proprioception, or both, and selects action a_{t}\in\mathcal{A} according to a policy \pi(\cdot|s_{t};l) conditioned on the instruction l\in\chi that describes the task (e.g., open the door).

![Image 3: Refer to caption](https://arxiv.org/html/2607.03751v1/x3.png)

Figure 3: Overview of SVA.(a) Search: MCTS explores the frozen VLA’s policy distribution in simulation, collecting trajectories with empirical returns. (b) Value: A lightweight Q-model is distilled from the searched data to predict action consequences. (c) Act: The frozen VLA proposes N candidates and the Q-model selects the best one without simulator access.

We adopt an inference strategy to maximize the expected return of a frozen VLA policy \pi_{\theta}: at each step t, we draw a set of N candidate actions \{a^{(1)},\ldots,a^{(N)}\} from \pi_{\theta}(\cdot|s_{t};l), and select the action that maximizes a learned Q-function that predicts the expected consequence of candidate actions: a_{t}\!=\!\operatorname*{argmax}_{i}\hat{Q}_{\phi}(s_{t},a^{(i)};l). This formulation treats \pi_{\theta} as a fixed proposal distribution and redirects selected actions toward higher-return regions via \hat{Q}_{\phi}, enabling policy improvement by scaling test-time compute. Our setting is agnostic to the action granularity: a can denote an action chunk, a high-level skill (e.g., find the apple), or a low-level control primitive (e.g., a continuous end-effector displacement). Below, we sometimes omit the notation l for ease of reading.

#### 4.1 Search: Mining Evaluation Signals via MCTS

Recent work on RL for foundation models reinforces the view of the Bitter Lesson: the true value of RL lies not in parameter updates themselves, but in the _search_ process – rolling out policies, comparing outcomes, and assigning credit[[27](https://arxiv.org/html/2607.03751#bib.bib19 "On-policy distillation")]. We employ MCTS[[33](https://arxiv.org/html/2607.03751#bib.bib20 "Mastering the game of Go with deep neural networks and tree search")] to fully explore the VLA’s policy distribution, performing structured look-ahead search to discover high-informative trajectories.

Each edge (s,a) of the search tree stores an action-value Q(s,a), visit count n(s,a), and prior probability P(s,a). The tree is traversed starting from the root state with the following procedure:

Selection. Action a_{t} is selected using the PUCT (Predictor Upper Confidence bound for Trees) rule:

a_{t}=\operatorname*{argmax}_{a}\left(Q(s_{t},a)+c_{\text{puct}}\,P(s_{t},a)\frac{\sqrt{n(s_{t})}}{1+n(s_{t},a)}\right),(1)

so as to maximize action value plus a bonus that is proportional to the prior probability but decays with repeated visits to encourage exploration, and c_{\text{puct}} controls the exploration degree.

Expansion. When the traversal reaches a leaf node s_{L} at step L, the leaf node may be expanded. The policy \pi_{\theta} processes s_{L} just once to sample N candidate actions \{a^{(1)},\ldots,a^{(N)}\}; each candidate is added to the tree as a new edge, with the successor state as the corresponding new leaf node. The output probabilities are stored as prior probabilities P for each action a, P(s_{L},a)=\pi_{\theta}(a|s_{L}).

Evaluation. From the newly expanded leaf node s_{L}, we estimate its value by performing a simulation rollout. Specifically, we clone the simulator state at s_{L} and execute the policy \pi_{\theta} forward for up to D steps (or until task termination), accumulating the discounted return as G(s_{L})=\sum\nolimits_{i=0}^{D-1}\gamma^{i}r_{L+i}. This rollout provides a Monte-Carlo estimate of the long-term consequence of reaching s_{L} under \pi_{\theta}.

Backup. After obtaining the rollout return G(s_{L}), we propagate it back along the traversal path from the leaf to the root. For every edge (s,a) visited during the selection phase, the statistics are updated:

n(s,a)\leftarrow n(s,a)+1,\qquad Q(s,a)\leftarrow Q(s,a)+\frac{G(s_{L})-Q(s,a)}{n(s,a)},(2)

i.e., the visit count is incremented and the action value is updated as a running mean of all rollout returns that have passed through that edge. These updated statistics refine the PUCT scores in subsequent iterations, progressively biasing the search toward higher-return branches of the tree.

After several iterations of the selection-expansion-evaluation-backup loop under a finite budget, for each task we collect a few trajectories from diverse search episodes, including both successful and failed rollouts, providing valuable contrastive signals for learning relative action quality in Sec.[4.2](https://arxiv.org/html/2607.03751#S4.SS2 "4.2 Value: Learning a Deployable Consequence Evaluator ‣ 4 Look Before You Leap: Addressing the Evaluation Bottleneck with SVA ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models").

#### 4.2 Value: Learning a Deployable Consequence Evaluator

Having used MCTS to _search_ for informative trajectories, we now turn to the complementary pillar of the Bitter Lesson: _learning_. Rather than relying on expensive tree search at every deployment step, we distill the knowledge discovered by MCTS into a lightweight model \hat{Q}_{\phi} that predicts the expected consequence of candidate actions. This amortization serves two purposes: i) it compresses the rich contrastive signals produced by search into a compact, fast-to-evaluate function, enabling real-time action selection without simulation; and ii) because neural networks generalize across states, the learned \hat{Q}_{\phi} can transfer the credit-assignment insights obtained from searched trajectories to unseen situations/tasks, effectively extending the reach of search beyond its original computational budget.

The consequence evaluator \hat{Q}_{\phi}(s,a;l) is built on a lightweight VLM backbone (e.g., Qwen3.5-0.8B), augmented with LoRA adapters and an ensemble of small MLP value heads. We append a special <VALUE> token to a prompt template containing the textual instruction l, visual/proprioceptive inputs s, and the candidate action a. The hidden state positioned at this token after self-attention is fed into the ensemble of value heads. The ensemble mean is used as the predicted Q-value, while the standard deviation provides an uncertainty estimate. The model is optimized using a Smooth-L1 loss that provides stable gradients on large errors (like L1) and smooth optimization on small errors (like L2):

\mathcal{L}(\phi)=\begin{cases}\frac{1}{2}\epsilon^{2}&\text{if}|\epsilon|<1\\
|\epsilon|-\frac{1}{2}&\text{otherwise}\end{cases}\quad\quad\epsilon=\frac{1}{B}\sum_{i=1}^{B}\hat{Q}_{\phi}^{(i)}(s,a;l)-Q(s,a;l),(3)

where B is the number of value heads, and target values Q(s,a;l) are provided by the collected data in Sec.[4.1](https://arxiv.org/html/2607.03751#S4.SS1 "4.1 Search: Mining Evaluation Signals via MCTS ‣ 4 Look Before You Leap: Addressing the Evaluation Bottleneck with SVA ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models") and normalized by dataset statistics. Only value heads and LoRA adapters are fine-tuned.

#### 4.3 Act: Evaluation-Guided Action Selection at Test Time

At test time, we use the VLA policy \pi_{\theta} as a proposal distribution and the learned Q-model \hat{Q}_{\phi} as a verifier. The frozen \pi_{\theta}(\cdot|s_{t};l) generates N candidate actions \{a^{(1)},\ldots,a^{(N)}\}, where the empirical prior of each candidate is estimated by its sampling frequency:

p\left(a\mid s_{t}~;~l\right)=\frac{1}{N}\sum\nolimits_{i=1}^{N}\mathbb{I}\left[a^{(i)}=a\right].(4)

Finally, we select the candidate with the highest uncertainty-regularized Q-value:

a_{t}=\operatorname*{argmax}_{i\in\{1,\ldots,N\}}\left[\mu_{\phi}\left(s_{t},a^{(i)};l\right)-\lambda_{1}\sigma_{\phi}\left(s_{t},a^{(i)};l\right)+\lambda_{2}\log p\left(a^{(i)}\mid s_{t};l\right)\right],(5)

where \mu_{\phi}(\cdot)/\sigma_{\phi}(\cdot) is the mean/standard deviation of the Q-ensemble, and (\lambda_{1}, \lambda_{2}) are regularization coefficients. Action a_{t} is executed until completion, task termination, or invalid feedback.

Inference Latency. The best-of-N strategy nominally requires N VLA forward passes for action candidates and N Q-model passes for scoring. Since the lightweight Q-model (e.g., 0.8B) is far smaller than the VLA backbone (e.g., 7B), the scoring overhead is minimal. Generation cost is well below the naive N\times estimate for modern VLAs that decouple a heavy VLM backbone from a lightweight action expert (e.g., \pi_{0.5}): the VLM runs once per observation and only the small action expert is invoked N times, adding little overhead over single-shot inference (see Sec.[5.3](https://arxiv.org/html/2607.03751#S5.SS3 "5.3 Scaling SVA at Test Time: Performance, Latency, and Cost-Effectiveness ‣ 5 Experiments ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models")).

Scaling Behavior. The best-of-N strategy provides a natural knob for test-time compute scaling: increasing N improves the probability of selecting a high-quality action, which naturally supports adaptive compute allocation across tasks of varying difficulty. Experiments suggest that scaling test-time evaluation in SVA can be more cost-effective than scaling model size (see Sec.[5.3](https://arxiv.org/html/2607.03751#S5.SS3 "5.3 Scaling SVA at Test Time: Performance, Latency, and Cost-Effectiveness ‣ 5 Experiments ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models")).

### 5 Experiments

Benchmarks. We use three benchmarks spanning two categories: 1) embodied reasoning, including EB-Habitat and EB-Navigation from EmbodiedBench[[46](https://arxiv.org/html/2607.03751#bib.bib17 "EmbodiedBench: comprehensive benchmarking multi-modal large language models for vision-driven embodied agents")]; 2) robot manipulation, including SimplerEnv[[22](https://arxiv.org/html/2607.03751#bib.bib33 "Evaluating real-world robot manipulation policies in simulation")] on the WidowX platform and RoboTwin 2.0[[7](https://arxiv.org/html/2607.03751#bib.bib56 "RoboTwin 2.0: a scalable data generator and benchmark with strong domain randomization for robust bimanual robotic manipulation")]. See Appendix[B](https://arxiv.org/html/2607.03751#A2 "Appendix B Benchmarks ‣ Appendix ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models") for benchmark details and Appendix[C](https://arxiv.org/html/2607.03751#A3 "Appendix C On the Real-Robot Relevance of Our Simulation Study ‣ Appendix ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models") for their real-robot relevance. For each task, episodes are split into training and evaluation sets at a 3:2 ratio, and success rate is reported. See Appendix[D](https://arxiv.org/html/2607.03751#A4 "Appendix D Experimental Details of SVA ‣ Appendix ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models") for experiment details.

Baselines. For EmbodiedBench, we adopt five backbones as the base policy spanning proprietary (GPT-4o), open-source (Qwen3.5-4B/9B/27B), and lightweight (Gemma-4-E4B-it) families, probing SVA’s model-agnostic property across scales and architectures. For SimplerEnv and RoboTwin, we compare against OpenVLA as a reference generalist, \bm{\pi_{0}/\pi_{0.5}} as SOTA VLAs trained on large-scale real-robot data (ruling out the concern that gains merely compensate for a weak base), and \bm{\pi_{0}/\pi_{0.5}}+RoboMonkey as the most directly comparable test-time selection method, which reranks candidates via a preference model trained on synthetic data. Comparing against RoboMonkey under identical proposal distribution and candidate budget directly tests whether SVA yields stronger action evaluation than single-step preferences. See Appendix[E](https://arxiv.org/html/2607.03751#A5 "Appendix E Baselines ‣ Appendix ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models") for details.

#### 5.1 Main Results

###### EmbodiedBench.

Table[1](https://arxiv.org/html/2607.03751#S5.T1 "Table 1 ‣ EmbodiedBench. ‣ 5.1 Main Results ‣ 5 Experiments ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models") summarizes results on EB-Habitat and EB-Navigation. SVA consistently improves success rates across all five backbones, yielding average gains of +15.4 on EB-Habitat and +13.2 points on EB-Navigation. These improvements demonstrate that SVA serves as a model-agnostic test-time enhancement strategy that generalizes across scales and architectures. The most pronounced gains emerge on categories requiring long-horizon planning or visual grounding: Long Horizon (+23.34 on Qwen3.5-27B) and Common Sense (+28.33 on Qwen3.5-27B), where the base policy tends to commit prematurely to locally plausible but globally suboptimal actions. By scoring candidate actions with predicted long-term consequences, the Q-model effectively corrects these myopic preferences. The few minor regressions are confined to categories where the base policy already performs near ceiling, leaving little headroom for reranking.

Table 1: Success rates (%) on EmbodiedBench. Green/Red\Delta denotes gain/drop over the base policy.

Model Method EB-Habitat EB-Navigation
Base Common Sense Complex Instr.Spatial Rel.Visual App.Long Horizon\columncolor cyan!10 Avg.Base Common Sense Complex Instr.Visual App.Long Horizon\columncolor cyan!10 Avg.
GPT-4o Base 73.33 20.00 50.00 51.67 40.00 28.33\columncolor cyan!1043.89 47.22 30.56 51.39 38.89 44.44\columncolor cyan!1042.50
SVA 88.33 36.67 48.33 56.67 58.33 46.67\columncolor cyan!10 55.83 51.39 50.00 45.83 41.67 50.00\columncolor cyan!10 47.78
\Delta+15.00+16.67-1.67+5.00+18.33+18.34\columncolor cyan!10+11.94+4.17+19.44-5.56+2.78+5.56\columncolor cyan!10+5.28
Qwen3.5-4B Base 56.67 3.33 33.33 48.33 28.33 16.67\columncolor cyan!1031.11 45.83 38.89 31.94 31.94 25.00\columncolor cyan!1034.72
SVA 96.67 25.00 50.00 56.67 50.00 40.00\columncolor cyan!10 53.06 54.17 41.67 45.83 41.67 50.00\columncolor cyan!10 46.67
\Delta+40.00+21.67+16.67+8.34+21.67+23.33\columncolor cyan!10+21.95+8.34+2.78+13.89+9.73+25.00\columncolor cyan!10+11.95
Qwen3.5-9B Base 71.67 10.00 48.33 55.00 35.00 36.67\columncolor cyan!1042.78 40.28 37.50 40.28 30.56 47.22\columncolor cyan!1039.17
SVA 96.67 36.67 63.33 56.67 43.33 46.67\columncolor cyan!10 57.22 62.50 59.72 52.78 47.22 58.33\columncolor cyan!10 56.11
\Delta+25.00+26.67+15.00+1.67+8.33+10.00\columncolor cyan!10+14.44+22.22+22.22+12.50+16.66+11.11\columncolor cyan!10+16.94
Qwen3.5-27B Base 83.33 11.67 50.00 55.00 45.00 38.33\columncolor cyan!1047.22 40.28 51.39 44.44 37.50 16.67\columncolor cyan!1038.06
SVA 93.33 40.00 70.00 53.33 66.67 61.67\columncolor cyan!10 64.17 61.11 56.94 58.33 58.33 52.78\columncolor cyan!10 57.50
\Delta+10.00+28.33+20.00-1.67+21.67+23.34\columncolor cyan!10+16.95+20.83+5.55+13.89+20.83+36.11\columncolor cyan!10+19.44
Gemma-4-E4B-it Base 63.33 1.67 16.67 35.00 23.33 11.67\columncolor cyan!1025.28 37.50 25.00 38.89 30.56 11.11\columncolor cyan!1028.61
SVA 81.67 11.67 36.67 45.00 31.67 16.67\columncolor cyan!10 37.22 41.67 37.50 50.00 38.89 37.50\columncolor cyan!10 41.11
\Delta+18.34+10.00+20.00+10.00+8.34+5.00\columncolor cyan!10+11.94+4.17+12.50+11.11+8.33+26.39\columncolor cyan!10+12.50

![Image 4: Refer to caption](https://arxiv.org/html/2607.03751v1/x4.png)

Figure 4: Success rates on SimplerEnv and RoboTwin. Full results in Tables[7](https://arxiv.org/html/2607.03751#A7.T7 "Table 7 ‣ Appendix G Detailed VLA Manipulation Results ‣ Appendix ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models") and[8](https://arxiv.org/html/2607.03751#A7.T8 "Table 8 ‣ Appendix G Detailed VLA Manipulation Results ‣ Appendix ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models").

###### Manipulation.

Fig.[4](https://arxiv.org/html/2607.03751#S5.F4 "Figure 4 ‣ EmbodiedBench. ‣ 5.1 Main Results ‣ 5 Experiments ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models") examines whether SVA generalizes to manipulation policies requiring precise object interaction. On SimplerEnv, \pi_{0}+SVA achieves 50.7% average success rate, outperforming \pi_{0}+RoboMonkey without requiring any curated preference annotations. The largest gains concentrate on contact-rich multi-step tasks: Stack Cubes (63.9 vs. 37.5, +26.4) and Eggplant in Basket (83.3 vs. 69.4, +13.9), where selecting the correct grasp-then-place sequence is critical and the Q-model provides a clear discriminative signal among candidates. On RoboTwin, \pi_{0.5}+SVA improves the average from 36.0% to 43.5% (+7.5), with gains spanning diverse manipulation primitives: Turn Switch (+20.0), Press Stapler (+16.7), and Click Bell (+15.0). Also, SVA surpasses competitive RoboMonkey by +5.2 points, verifying the superiority of its action verifier. These tasks span distinct motor skills (rotation, pressing, tapping), indicating that the Q-evaluator transfers across manipulation modalities rather than overfitting to a single skill. These results confirm that SVA extends beyond embodied reasoning to fine-grained robot manipulation: the Q-model acts as a general-purpose action verifier whenever the base policy produces diverse candidates.

#### 5.2 Ablation Study

![Image 5: Refer to caption](https://arxiv.org/html/2607.03751v1/x5.png)

Figure 5: Ablation on EB-Navigation (Qwen3.5-9B). Full results in Table[9](https://arxiv.org/html/2607.03751#A8.T9 "Table 9 ‣ Appendix H Detailed Results on Ablation Study ‣ Appendix ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models").

To isolate each component’s contribution, we evaluate three ablations: 1)w/o MCTS, training the Q-model with only policy-rollout returns; 2)w/o Q-model, replacing value evaluation with majority voting; and 3)w/o multi-cand., querying a single candidate (pass-through). Results in Fig.[5](https://arxiv.org/html/2607.03751#S5.F5 "Figure 5 ‣ 5.2 Ablation Study ‣ 5 Experiments ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models") show that: i) Dropping MCTS (SVA vs. w/o MCTS) causes a moderate decline on EB-Navigation (56.11 vs. 50.83); ii) Removing the Q-model yields the largest drop (56.11\to 43.33 on EB-Navigation), confirming that an explicit value estimator is essential to distinguish among plausible but unequally effective candidates; iii) Eliminating multi-candidate selection reduces performance to the single-shot backbone (56.11\to 39.17). In summary, SVA outperforms all ablations by at least +5.28 avg., confirming that each component is necessary to the overall performance.

#### 5.3 Scaling SVA at Test Time: Performance, Latency, and Cost-Effectiveness

![Image 6: Refer to caption](https://arxiv.org/html/2607.03751v1/x6.png)

Figure 6: Scaling candidates N improves success rate while Q-model scoring adds negligible overhead versus proposal latency (Qwen3.5-9B).

SVA exhibits strong test-time scaling. We analyze SVA’s scaling behavior by sweeping the number of candidate actions N at deployment and measuring both success rate and inference latency. As shown in Fig.[6](https://arxiv.org/html/2607.03751#S5.F6 "Figure 6 ‣ 5.3 Scaling SVA at Test Time: Performance, Latency, and Cost-Effectiveness ‣ 5 Experiments ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models"), increasing N yields monotonic gains in success rate for SVA (9B), confirming our pass@k observation: the base policy already contains competent behaviors but lacks a reliable mechanism to identify them. SVA closes this gap via scaling action evaluation, turning otherwise wasted samples into consistent performance gains.

SVA’s inference latency grows sub-linearly in N. The evaluation latency is negligible compared to generation (\leq 1.33 s vs. up to 11.2 s at N{=}32), thanks to the lightweight Q-model design. The generation latency scales sub-linearly with N: each doubling of the candidate set (1{\to}2,2{\to}4,\ldots,16{\to}32) incurs only a 10.63\%–36.58\% increase in latency (26\% avg.), reflecting effective decoding within the VLA backbone. This sub-linear growth is what makes test-time scaling _practical_ for robotics, where real-time execution is non-negotiable.

Scaling test-time evaluation is more cost-effective than scaling model size. Single-shot Qwen3.5-27B reaches 46.7\% success at 10.2 s of inference latency. In contrast, Qwen3.5-9B + SVA reaches 51.1% at 5.87s with best-of-4 and 53.6% at 7.43s with best-of-8: a 7-point gain over the 27B baseline at 27% lower latency, using a 3\times smaller backbone. This result reframes action _evaluation_ at test time as a first-class lever for VLA, orthogonal to (and substantially cheaper than) data scaling and policy post-training. This is also consistent with test-time scaling findings in LLMs[[35](https://arxiv.org/html/2607.03751#bib.bib4 "Scaling LLM test-time compute optimally can be more effective than scaling parameters for reasoning")].

#### 5.4 Case Study

We provide a qualitative case study with the instruction: _On the sofa there’s an apple, but instead find a plate and move it to the brown table._ This task is challenging because the first clause introduces a salient but irrelevant object-location pair (apple-sofa), distracting from the true goal of moving the plate to the brown table. Full results are provided in Appendix[I](https://arxiv.org/html/2607.03751#A9 "Appendix I More Results on Case Study ‣ Appendix ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models"). As shown in Fig.[7](https://arxiv.org/html/2607.03751#A9.F7 "Figure 7 ‣ Case 2: spatial-relation grounding. ‣ Appendix I More Results on Case Study ‣ Appendix ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models"), the base policy is misled by the distractor, navigating to the sofa and then issuing repeated invalid recovery actions.

Table 2:  Action evaluation: lower prior but higher Q-value leads to correct selection. 

Candidate Prior Q-value Selected
Distractor-driven 0.4375-0.1282✗
\rowcolor lightBlue Goal-directed 0.0625 0.2948✓

In contrast, SVA selects a plate-centric plan and reaches the target table in just 4 steps with no invalid actions. The candidate scoring in Table[2](https://arxiv.org/html/2607.03751#S5.T2 "Table 2 ‣ 5.4 Case Study ‣ 5 Experiments ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models") explains this success: although the base policy assigns high likelihood to the distractor-driven sofa-first plan, the Q-network ranks the lower-prior plate-centric plan highest. The value model can override myopic proposal preferences via long-horizon consequence evaluation, improving robustness and reducing invalid behaviors.

### 6 Discussion and Limitations

We revisited VLA’s policy improvement from an under-explored angle: reframing VLA’s failure mode as an evaluation bottleneck rather than purely a generation one. Across embodied benchmarks, our method delivers consistent gains over diverse backbones, showing that scaling test-time evaluation can be more cost-effective than scaling model size. We hope these results encourage the community to view evaluation as a first-class lever for VLA improvement, orthogonal to data scaling and policy fine-tuning. Also, our method reveals several limitations, including decoupled tree search and value learning, reliance on resettable simulators, and sim-only evaluation (test on physical robots is the most pressing next step). See Appendix[A](https://arxiv.org/html/2607.03751#A1 "Appendix A Limitations and Future Work ‣ Appendix ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models") for details of limitations that we leave for future work.

##### Acknowledgments

If a paper is accepted, the final camera-ready version will (and probably should) include acknowledgments. All acknowledgments go at the end of the paper, including thanks to reviewers who gave useful comments, to colleagues who contributed to the ideas, and to funding agencies and corporate sponsors that provided financial support.

### References

*   [1]J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman, D. Almeida, J. Altenschmidt, S. Altman, S. Anadkat, et al. (2023)GPT-4 technical report. arXiv preprint arXiv:2303.08774. Cited by: [§1](https://arxiv.org/html/2607.03751#S1.p1.1 "1 Introduction ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models"). 
*   [2] (2026)ReVer: reasoning-guided verification for embodied agents. In ICRA 2026 Workshop: From Data to Decisions: VLA Pipelines for Real Robots, Cited by: [§2](https://arxiv.org/html/2607.03751#S2.p2.1 "2 Related Work ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models"). 
*   [3]J. Bjorck, F. Castañeda, N. Cherniadev, X. Da, R. Ding, L. Fan, Y. Fang, D. Fox, F. Hu, S. Huang, et al. (2025)GR00T N1: an open foundation model for generalist humanoid robots. arXiv preprint arXiv:2503.14734. Cited by: [§1](https://arxiv.org/html/2607.03751#S1.p1.1 "1 Introduction ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models"), [§2](https://arxiv.org/html/2607.03751#S2.p1.1 "2 Related Work ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models"). 
*   [4]K. Black, N. Brown, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, L. Groom, K. Hausman, B. Ichter, et al. (2024)\pi_{0}: A vision-language-action flow model for general robot control. arXiv preprint arXiv:2410.24164. Cited by: [Appendix C](https://arxiv.org/html/2607.03751#A3.SS0.SSS0.Px2.p1.2 "Our backbones are real-robot policies, not simulation-only ones. ‣ Appendix C On the Real-Robot Relevance of Our Simulation Study ‣ Appendix ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models"), [§1](https://arxiv.org/html/2607.03751#S1.p2.1 "1 Introduction ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models"), [§2](https://arxiv.org/html/2607.03751#S2.p1.1 "2 Related Work ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models"). 
*   [5]A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, J. Dabis, C. Finn, K. Gopalakrishnan, K. Hausman, A. Herzog, J. Hsu, et al. (2022)RT-1: robotics transformer for real-world control at scale. arXiv preprint arXiv:2212.06817. Cited by: [§1](https://arxiv.org/html/2607.03751#S1.p1.1 "1 Introduction ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models"), [§2](https://arxiv.org/html/2607.03751#S2.p1.1 "2 Related Work ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models"). 
*   [6]M. Chen, J. Tworek, H. Jun, Q. Yuan, H. P. D. O. Pinto, J. Kaplan, H. Edwards, Y. Burda, N. Joseph, G. Brockman, et al. (2021)Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374. Cited by: [§3](https://arxiv.org/html/2607.03751#S3.p2.5 "3 Diagnosing the VLA Bottleneck: Generation or Evaluation? ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models"). 
*   [7]T. Chen, Z. Chen, B. Chen, Z. Cai, Y. Liu, Z. Li, Q. Liang, X. Lin, Y. Ge, Z. Gu, et al. (2025)RoboTwin 2.0: a scalable data generator and benchmark with strong domain randomization for robust bimanual robotic manipulation. arXiv preprint arXiv:2506.18088. Cited by: [Appendix C](https://arxiv.org/html/2607.03751#A3.SS0.SSS0.Px1.p1.1 "Our benchmarks are explicitly designed for real-robot predictivity. ‣ Appendix C On the Real-Robot Relevance of Our Simulation Study ‣ Appendix ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models"), [§5](https://arxiv.org/html/2607.03751#S5.p1.1 "5 Experiments ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models"). 
*   [8]Y. Chen, S. Tian, S. Liu, Y. Zhou, H. Li, and D. Zhao (2025)ConRFT: a reinforced fine-tuning method for VLA models via consistency policy. arXiv preprint arXiv:2502.05450. Cited by: [§1](https://arxiv.org/html/2607.03751#S1.p1.1 "1 Introduction ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models"). 
*   [9]F. Codevilla, E. Santana, A. M. López, and A. Gaidon (2019)Exploring the limitations of behavior cloning for autonomous driving. In Proceedings of the IEEE/CVF International Conference on Computer Vision,  pp.9329–9338. Cited by: [§1](https://arxiv.org/html/2607.03751#S1.p2.1 "1 Introduction ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models"). 
*   [10]P. De Haan, D. Jayaraman, and S. Levine (2019)Causal confusion in imitation learning. In Advances in Neural Information Processing Systems, Vol. 32. Cited by: [§1](https://arxiv.org/html/2607.03751#S1.p2.1 "1 Introduction ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models"). 
*   [11]A. J. Hancock, X. Wu, L. Zha, O. Russakovsky, and A. Majumdar (2026)Actions as language: fine-tuning VLMs into VLAs without catastrophic forgetting. In International Conference on Learning Representations, Cited by: [§2](https://arxiv.org/html/2607.03751#S2.p1.1 "2 Related Work ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models"). 
*   [12]W. Huang, F. Xia, T. Xiao, H. Chan, J. Liang, P. Florence, A. Zeng, J. Tompson, I. Mordatch, Y. Chebotar, et al. (2023)Inner monologue: embodied reasoning through planning with language models. In Conference on Robot Learning,  pp.1769–1782. Cited by: [§2](https://arxiv.org/html/2607.03751#S2.p2.1 "2 Related Work ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models"). 
*   [13]B. Ichter, A. Brohan, Y. Chebotar, C. Finn, K. Hausman, et al. (2022)Do as I can, not as I say: grounding language in robotic affordances. In Annual Conference on Robot Learning, Cited by: [§2](https://arxiv.org/html/2607.03751#S2.p2.1 "2 Related Work ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models"). 
*   [14]P. Intelligence, A. Amin, R. Aniceto, A. Balakrishna, K. Black, K. Conley, G. Connors, J. Darpinian, K. Dhabalia, J. DiCarlo, et al. (2025)\pi_{0.6}: A VLA that learns from experience. arXiv preprint arXiv:2511.14759. Cited by: [§2](https://arxiv.org/html/2607.03751#S2.p1.1 "2 Related Work ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models"). 
*   [15]P. Intelligence, K. Black, N. Brown, J. Darpinian, K. Dhabalia, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, et al. (2025)\pi_{0.5}: A vision-language-action model with open-world generalization. arXiv preprint arXiv:2504.16054. Cited by: [Appendix C](https://arxiv.org/html/2607.03751#A3.SS0.SSS0.Px2.p1.2 "Our backbones are real-robot policies, not simulation-only ones. ‣ Appendix C On the Real-Robot Relevance of Our Simulation Study ‣ Appendix ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models"), [§1](https://arxiv.org/html/2607.03751#S1.p1.1 "1 Introduction ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models"), [§2](https://arxiv.org/html/2607.03751#S2.p1.1 "2 Related Work ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models"). 
*   [16]S. Jang, D. Kim, C. Kim, Y. Kim, and J. Shin (2026)Verifier-free test-time sampling for vision-language-action models. In The Fourteenth International Conference on Learning Representations, Cited by: [§2](https://arxiv.org/html/2607.03751#S2.p2.1 "2 Related Work ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models"). 
*   [17]M. J. Kim, C. Finn, and P. Liang (2025)Fine-tuning vision-language-action models: optimizing speed and success. arXiv preprint arXiv:2502.19645. Cited by: [§1](https://arxiv.org/html/2607.03751#S1.p1.1 "1 Introduction ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models"), [§2](https://arxiv.org/html/2607.03751#S2.p1.1 "2 Related Work ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models"). 
*   [18]M. J. Kim, K. Pertsch, S. Karamcheti, T. Xiao, A. Balakrishna, S. Nair, R. Rafailov, E. P. Foster, P. R. Sanketi, Q. Vuong, et al. (2024)OpenVLA: an open-source vision-language-action model. In Annual Conference on Robot Learning, Cited by: [§1](https://arxiv.org/html/2607.03751#S1.p1.1 "1 Introduction ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models"), [§2](https://arxiv.org/html/2607.03751#S2.p1.1 "2 Related Work ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models"). 
*   [19]J. Kwok, C. Agia, R. Sinha, M. Foutter, S. Li, I. Stoica, A. Mirhoseini, and M. Pavone (2025)RoboMonkey: scaling test-time sampling and verification for vision-language-action models. In Annual Conference on Robot Learning, Cited by: [§2](https://arxiv.org/html/2607.03751#S2.p2.1 "2 Related Work ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models"). 
*   [20]J. Kwok, X. Zhang, M. Xu, Y. Liu, A. Mirhoseini, C. Finn, and M. Pavone (2026)Scaling verification can be more effective than scaling policy learning for vision-language-action alignment. arXiv preprint arXiv:2602.12281. Cited by: [§2](https://arxiv.org/html/2607.03751#S2.p2.1 "2 Related Work ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models"). 
*   [21]H. Li, Y. Zuo, J. Yu, Y. Zhang, Z. Yang, K. Zhang, X. Zhu, Y. Zhang, T. Chen, G. Cui, et al. (2026)SimpleVLA-RL: scaling VLA training via reinforcement learning. In International Conference on Learning Representations, Cited by: [§2](https://arxiv.org/html/2607.03751#S2.p1.1 "2 Related Work ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models"). 
*   [22]X. Li, K. Hsu, J. Gu, O. Mees, K. Pertsch, H. R. Walke, C. Fu, I. Lunawat, I. Sieh, S. Kirmani, S. Levine, J. Wu, C. Finn, H. Su, Q. Vuong, and T. Xiao (2024)Evaluating real-world robot manipulation policies in simulation. In Annual Conference on Robot Learning, Vol. 270,  pp.3705–3728. Cited by: [Appendix C](https://arxiv.org/html/2607.03751#A3.SS0.SSS0.Px1.p1.1 "Our benchmarks are explicitly designed for real-robot predictivity. ‣ Appendix C On the Real-Robot Relevance of Our Simulation Study ‣ Appendix ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models"), [§1](https://arxiv.org/html/2607.03751#S1.p1.1 "1 Introduction ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models"), [§2](https://arxiv.org/html/2607.03751#S2.p1.1 "2 Related Work ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models"), [§5](https://arxiv.org/html/2607.03751#S5.p1.1 "5 Experiments ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models"). 
*   [23]Z. Li, J. Liu, Z. Dong, T. Teng, Q. Rouxel, D. Caldwell, and F. Chen (2026)Towards deploying VLA without fine-tuning: plug-and-play inference-time VLA policy steering via embodied evolutionary diffusion. IEEE Robotics and Automation Letters 11 (5),  pp.6234–6241. Cited by: [§2](https://arxiv.org/html/2607.03751#S2.p2.1 "2 Related Work ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models"). 
*   [24]Y. Liang, X. Wang, K. Wang, S. Wang, X. Peng, H. Chen, D. K. H. Chua, and P. Vadakkepat (2026)Adaptive action chunking at inference-time for vision-language-action models. arXiv preprint arXiv:2604.04161. Cited by: [§2](https://arxiv.org/html/2607.03751#S2.p2.1 "2 Related Work ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models"). 
*   [25]H. Lightman, V. Kosaraju, Y. Burda, H. Edwards, B. Baker, T. Lee, J. Leike, J. Schulman, I. Sutskever, and K. Cobbe (2024)Let’s verify step by step. In International Conference on Learning Representations, Vol. 2024,  pp.39578–39601. Cited by: [§2](https://arxiv.org/html/2607.03751#S2.p2.1 "2 Related Work ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models"). 
*   [26]B. Liu, Y. Zhu, C. Gao, Y. Feng, Q. Liu, Y. Zhu, and P. Stone (2023)LIBERO: benchmarking knowledge transfer for lifelong robot learning. In Advances in Neural Information Processing Systems, Vol. 36,  pp.44776–44791. Cited by: [§1](https://arxiv.org/html/2607.03751#S1.p1.1 "1 Introduction ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models"), [§2](https://arxiv.org/html/2607.03751#S2.p1.1 "2 Related Work ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models"). 
*   [27]K. Lu and T. M. Lab (2025)On-policy distillation. Thinking Machines Lab: Connectionism. Note: https://thinkingmachines.ai/blog/on-policy-distillation Cited by: [§4.1](https://arxiv.org/html/2607.03751#S4.SS1.p1.1 "4.1 Search: Mining Evaluation Signals via MCTS ‣ 4 Look Before You Leap: Addressing the Evaluation Bottleneck with SVA ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models"). 
*   [28]M. S. Mark, T. Gao, G. G. Sampaio, M. K. Srirama, A. Sharma, C. Finn, and A. Kumar (2024)Policy agnostic RL: offline RL and online RL fine-tuning of any class and backbone. arXiv preprint arXiv:2412.06685. Cited by: [§2](https://arxiv.org/html/2607.03751#S2.p2.1 "2 Related Work ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models"). 
*   [29]M. Nakamoto, O. Mees, A. Kumar, and S. Levine (2025)Steering your generalists: improving robotic foundation models via value guidance. In Annual Conference on Robot Learning,  pp.4996–5013. Cited by: [§1](https://arxiv.org/html/2607.03751#S1.p2.1 "1 Introduction ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models"), [§2](https://arxiv.org/html/2607.03751#S2.p2.1 "2 Related Work ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models"). 
*   [30]C. Neary, O. G. Younis, A. Kuramshin, O. Aslan, and G. Berseth (2025)Improving pre-trained vision-language-action policies with model-based search. arXiv preprint arXiv:2508.12211. Cited by: [§2](https://arxiv.org/html/2607.03751#S2.p2.1 "2 Related Work ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models"). 
*   [31]K. Pertsch, K. Stachowicz, B. Ichter, D. Driess, S. Nair, Q. Vuong, O. Mees, C. Finn, and S. Levine (2025)FAST: efficient action tokenization for vision-language-action models. arXiv preprint arXiv:2501.09747. Cited by: [§D.1](https://arxiv.org/html/2607.03751#A4.SS1.p2.1 "D.1 Q-Model Architecture ‣ Appendix D Experimental Details of SVA ‣ Appendix ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models"), [§1](https://arxiv.org/html/2607.03751#S1.p1.1 "1 Introduction ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models"), [§2](https://arxiv.org/html/2607.03751#S2.p1.1 "2 Related Work ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models"). 
*   [32]K. Ren, A. Salamatian, K. Pattison, and C. Neary (2026)V-VLAPS: value-guided planning for vision-language-action models. arXiv preprint arXiv:2601.00969. Cited by: [§2](https://arxiv.org/html/2607.03751#S2.p2.1 "2 Related Work ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models"). 
*   [33]D. Silver, A. Huang, C. J. Maddison, A. Guez, L. Sifre, G. Van Den Driessche, J. Schrittwieser, I. Antonoglou, V. Panneershelvam, M. Lanctot, et al. (2016)Mastering the game of Go with deep neural networks and tree search. Nature 529 (7587),  pp.484–489. Cited by: [§4.1](https://arxiv.org/html/2607.03751#S4.SS1.p1.1 "4.1 Search: Mining Evaluation Signals via MCTS ‣ 4 Look Before You Leap: Addressing the Evaluation Bottleneck with SVA ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models"). 
*   [34]N. Singhi, C. Bialas, S. Jauhri, V. Prasad, G. Chalvatzaki, M. Rohrbach, and A. Rohrbach (2026)Think twice, act once: verifier-guided action selection for embodied agents. arXiv preprint arXiv:2605.12620. Cited by: [§2](https://arxiv.org/html/2607.03751#S2.p2.1 "2 Related Work ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models"). 
*   [35]C. Snell, J. Lee, K. Xu, and A. Kumar (2025)Scaling LLM test-time compute optimally can be more effective than scaling parameters for reasoning. In International Conference on Learning Representations, Vol. 2025,  pp.10131–10165. Cited by: [§5.3](https://arxiv.org/html/2607.03751#S5.SS3.p3.3 "5.3 Scaling SVA at Test Time: Performance, Latency, and Cost-Effectiveness ‣ 5 Experiments ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models"). 
*   [36]H. Song, D. Qu, Y. Yao, Q. Chen, Q. Lv, Y. Tang, M. Shi, G. Ren, M. Yao, B. Zhao, et al. (2025)Hume: introducing system-2 thinking in visual-language-action model. arXiv preprint arXiv:2505.21432. Cited by: [§2](https://arxiv.org/html/2607.03751#S2.p2.1 "2 Related Work ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models"). 
*   [37]R. Sutton (2019)The bitter lesson. External Links: [Link](http://www.incompleteideas.net/IncIdeas/BitterLesson.html)Cited by: [§1](https://arxiv.org/html/2607.03751#S1.p4.3 "1 Introduction ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models"). 
*   [38]R. S. Sutton, A. G. Barto, et al. (1998)Reinforcement learning: an introduction. Vol. 1, MIT press Cambridge. Cited by: [§1](https://arxiv.org/html/2607.03751#S1.p2.1 "1 Introduction ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models"). 
*   [39]O. M. Team, D. Ghosh, H. Walke, K. Pertsch, K. Black, O. Mees, S. Dasari, J. Hejna, T. Kreiman, C. Xu, et al. (2024)Octo: an open-source generalist robot policy. arXiv preprint arXiv:2405.12213. Cited by: [§1](https://arxiv.org/html/2607.03751#S1.p1.1 "1 Introduction ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models"), [§2](https://arxiv.org/html/2607.03751#S2.p1.1 "2 Related Work ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models"). 
*   [40]J. Uesato, N. Kushman, R. Kumar, F. Song, N. Siegel, L. Wang, A. Creswell, G. Irving, and I. Higgins (2022)Solving math word problems with process-and outcome-based feedback. arXiv preprint arXiv:2211.14275. Cited by: [§2](https://arxiv.org/html/2607.03751#S2.p2.1 "2 Related Work ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models"). 
*   [41]Z. Wan, X. Feng, M. Wen, S. M. Mcaleer, Y. Wen, W. Zhang, and J. Wang (2024)AlphaZero-like tree-search can guide large language model decoding and training. In International Conference on Machine Learning, Vol. 235,  pp.49890–49920. Cited by: [§2](https://arxiv.org/html/2607.03751#S2.p2.1 "2 Related Work ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models"). 
*   [42]J. Wen, Y. Zhu, J. Li, M. Zhu, Z. Tang, K. Wu, Z. Xu, N. Liu, R. Cheng, C. Shen, et al. (2025)TinyVLA: towards fast, data-efficient vision-language-action models for robotic manipulation. IEEE Robotics and Automation Letters. Cited by: [§1](https://arxiv.org/html/2607.03751#S1.p1.1 "1 Introduction ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models"), [§2](https://arxiv.org/html/2607.03751#S2.p1.1 "2 Related Work ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models"). 
*   [43]Y. Wu, A. Li, T. Hermans, F. Ramos, A. Bajcsy, and C. PÃŠrez-D’Arpino (2025)Do what you say: steering vision-language-action models via runtime reasoning-action alignment verification. arXiv preprint arXiv:2510.16281. Cited by: [§2](https://arxiv.org/html/2607.03751#S2.p2.1 "2 Related Work ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models"). 
*   [44]W. Xiao, H. Lin, A. Peng, H. Xue, T. He, Z. Luo, Y. Xie, F. Hu, L. Fan, G. Shi, and Y. Zhu (2026)Self-improving vision-language-action models with data generation via residual RL. In International Conference on Learning Representations, Cited by: [§2](https://arxiv.org/html/2607.03751#S2.p1.1 "2 Related Work ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models"). 
*   [45]J. Yang, M. S. Mark, B. Vu, A. Sharma, J. Bohg, and C. Finn (2024)Robot fine-tuning made easy: pre-training rewards and policies for autonomous real-world reinforcement learning. In IEEE International Conference on Robotics and Automation,  pp.4804–4811. Cited by: [§2](https://arxiv.org/html/2607.03751#S2.p1.1 "2 Related Work ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models"). 
*   [46]R. Yang, H. Chen, J. Zhang, M. Zhao, C. Qian, K. Wang, Q. Wang, T. V. Koripella, M. Movahedi, M. Li, H. Ji, H. Zhang, and T. Zhang (2025)EmbodiedBench: comprehensive benchmarking multi-modal large language models for vision-driven embodied agents. In International Conference on Machine Learning, Vol. 267,  pp.70576–70631. Cited by: [§5](https://arxiv.org/html/2607.03751#S5.p1.1 "5 Experiments ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models"). 
*   [47]S. Yang, Y. Du, S. Ghasemipour, J. Tompson, L. Kaelbling, D. Schuurmans, and P. Abbeel (2024)Learning interactive real-world simulators. In International Conference on Learning Representations, Vol. 2024,  pp.45210–45234. Cited by: [§2](https://arxiv.org/html/2607.03751#S2.p2.1 "2 Related Work ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models"). 
*   [48]S. Yang, Y. Zhang, H. He, L. Pan, X. Li, C. Bai, and X. Li (2025)Steering vision-language-action models as anti-exploration: a test-time scaling approach. arXiv preprint arXiv:2512.02834. Cited by: [§2](https://arxiv.org/html/2607.03751#S2.p2.1 "2 Related Work ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models"). 
*   [49]S. Yao, D. Yu, J. Zhao, I. Shafran, T. Griffiths, Y. Cao, and K. Narasimhan (2023)Tree of thoughts: deliberate problem solving with large language models. In Advances in Neural Information Processing Systems, Vol. 36,  pp.11809–11822. Cited by: [§2](https://arxiv.org/html/2607.03751#S2.p2.1 "2 Related Work ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models"). 
*   [50]H. Zang, M. Wei, S. Xu, Y. Wu, Z. Guo, Y. Wang, H. Lin, L. Shi, Y. Xie, Z. Xu, et al. (2025)RLinf-VLA: a unified and efficient framework for reinforcement learning of vision-language-action models. arXiv preprint arXiv:2510.06710. Cited by: [§1](https://arxiv.org/html/2607.03751#S1.p1.1 "1 Introduction ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models"). 
*   [51]D. Zhang, S. Zhoubian, Z. Hu, Y. Yue, Y. Dong, and J. Tang (2024)ReST-MCTS∗: LLM self-training via process reward guided tree search. In Advances in Neural Information Processing Systems, Vol. 37,  pp.64735–64772. Cited by: [§2](https://arxiv.org/html/2607.03751#S2.p2.1 "2 Related Work ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models"). 
*   [52]R. Zhao, S. Xu, R. Jin, Y. Deng, Y. Tai, K. Jia, and G. Liu (2026)Sim2Real VLA: zero-shot generalization of synthesized skills to realistic manipulation. In International Conference on Learning Representations, Cited by: [§2](https://arxiv.org/html/2607.03751#S2.p1.1 "2 Related Work ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models"). 
*   [53]G. Zhou, H. Pan, Y. Lecun, and L. Pinto (2025)DINO-WM: world models on pre-trained visual features enable zero-shot planning. In International Conference on Machine Learning, Vol. 267,  pp.79115–79135. Cited by: [§2](https://arxiv.org/html/2607.03751#S2.p2.1 "2 Related Work ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models"). 
*   [54]S. Zhou, Y. Du, J. Chen, Y. Li, D. Yeung, and C. Gan (2024)RoboDreamer: learning compositional world models for robot imagination. In International Conference on Machine Learning, Vol. 235,  pp.61885–61896. Cited by: [§2](https://arxiv.org/html/2607.03751#S2.p2.1 "2 Related Work ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models"). 
*   [55]B. Zitkovich, T. Yu, S. Xu, P. Xu, T. Xiao, F. Xia, J. Wu, P. Wohlhart, S. Welker, A. Wahid, et al. (2023)RT-2: vision-language-action models transfer web knowledge to robotic control. In Conference on Robot Learning,  pp.2165–2183. Cited by: [§1](https://arxiv.org/html/2607.03751#S1.p1.1 "1 Introduction ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models"), [§2](https://arxiv.org/html/2607.03751#S2.p1.1 "2 Related Work ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models"). 

## Appendix

### Appendix A Limitations and Future Work

While SVA delivers consistent gains across embodied reasoning and manipulation benchmarks, several limitations remain that point to promising directions for future research.

First, our pipeline is deliberately staged with decoupled tree search and value learning, so search is blind to the evaluator being learned and the policy never benefits from improved values beyond test-time re-ranking, exploiting only a slice of what RL has to offer. A natural next step is to unify the stages into an online search-and-learning loop, where an up-to-date Q-model guides MCTS and on-policy MCTS rollouts continually refine the Q-model, unlocking more of RL’s potential while still touching the generalist backbone as gently as possible.

Second, the Search stage relies on a resettable simulator with a task-success signal; while this is standard for benchmarks such as EmbodiedBench and RoboTwin, it limits direct applicability to settings where high-fidelity simulation or reward functions are unavailable. A promising path forward is to replace, or supplement, simulator rollouts with learned world models or sparse human/auto-labeled outcomes, so that the same Search–Value–Act recipe can mine evaluation signals with improved sample efficiency in domains where ground-truth simulation is impractical.

Third, our evaluation is conducted entirely in simulation, and the calibration of the learned Q-model on physical robots remains untested; validating SVA on real hardware, possibly via sim-to-real co-training or lightweight online residual calibration of the evaluator, is the most pressing next step.

Taken together, these limitations chart a coherent research agenda: tightening the search–learning loop, relaxing the simulator assumption, and bridging the sim-to-real gap. We view SVA as an initial step toward scalable test-time reasoning for embodied agents, and we hope the Search–Value–Act paradigm provides a foundation upon which the community can build more general, scalable, deployable, and self-improving embodied systems.

### Appendix B Benchmarks

We evaluate SVA on EmbodiedBench and two VLA manipulation benchmarks. EmbodiedBench is used for the main experiments, while SimplerEnv and RoboTwin 2.0 are used to evaluate whether value-guided reranking generalizes to continuous-control manipulation settings.

###### EmbodiedBench.

We use two EmbodiedBench suites. EB-Habitat is a household rearrangement benchmark built on Habitat. The agent receives a first-person RGB observation and a natural-language instruction, and executes high-level discrete skills including navigation, pick, place, open, and close. We evaluate on the base, common_sense, complex_instruction, spatial_relationship, visual_appearance, and long_horizon splits, with a maximum episode length of 30 environment steps. EB-Navigation is an object-goal navigation benchmark built on AI2-THOR. The action space contains eight primitive actions: move forward, move backward, move right, move left, rotate right, rotate left, look up, and look down. We evaluate on the base, common_sense, complex_instruction, visual_appearance, and long_horizon splits, with a maximum episode length of 20 environment steps.

###### VLA manipulation benchmarks.

We additionally evaluate on SimplerEnv and RoboTwin 2.0. SimplerEnv uses ManiSkill2-based real-to-sim WidowX manipulation tasks. We evaluate four tasks: widowx_carrot_on_plate, widowx_stack_cube, widowx_spoon_on_towel, and widowx_put_eggplant_in_basket. Observations include the front RGB image, the language instruction, and a 7-D Bridge end-effector proprioceptive state. RoboTwin 2.0 evaluates bimanual manipulation in a SAPIEN-based simulator. We use 10 selected tasks with the aloha-agilex embodiment. Observations include head and wrist RGB images, joint states, and end-effector proprioception. Success is determined by the simulator-provided success signal.

Table 3: Benchmark settings used in our experiments.

Benchmark Simulator Observation Action type Tasks / splits
EB-Habitat Habitat RGB Discrete skills 6 splits
EB-Navigation AI2-THOR RGB 8 discrete actions 5 splits
SimplerEnv ManiSkill2 RGB + proprio.7-D WidowX chunks 4 tasks
RoboTwin 2.0 SAPIEN Multi-view RGB + proprio.14-D qpos chunks 10 tasks

### Appendix C On the Real-Robot Relevance of Our Simulation Study

All our experiments are conducted in simulation, but several deliberate choices make the conclusions directly informative for physical-robot deployment. We summarize the evidence below.

###### Our benchmarks are explicitly designed for real-robot predictivity.

SimplerEnv[[22](https://arxiv.org/html/2607.03751#bib.bib33 "Evaluating real-world robot manipulation policies in simulation")] is constructed and quantitatively validated as a real-world proxy for VLAs: its authors show that policy rankings on SimplerEnv match rankings on the corresponding physical Google Robot and WidowX setups across RT-1, RT-1-X, Octo, and OpenVLA, highlighting the potential of simulation-based approaches for evaluating generalist real-world manipulation policies in a scalable and reliable way. RoboTwin 2.0[[7](https://arxiv.org/html/2607.03751#bib.bib56 "RoboTwin 2.0: a scalable data generator and benchmark with strong domain randomization for robust bimanual robotic manipulation")] is a scalable simulation framework supporting five robotic arms (Franka, Piper, UR5, ARX-X5, and AlohaAgileX), specifically optimized for sim-to-real transfer via structured domain randomization along five axes (clutter, lighting, background, tabletop height, and language instructions) with its authors reporting promising results on real hardware. SVA’s gains are consistent across both benchmarks, making the conclusions directly informative for physical-robot deployment.

###### Our backbones are real-robot policies, not simulation-only ones.

\pi_{0} and \pi_{0.5} are state-of-the-art generalist VLAs trained on large-scale real-robot demonstrations and deployed on physical platforms by their original authors[[4](https://arxiv.org/html/2607.03751#bib.bib41 "π0: A vision-language-action flow model for general robot control"), [15](https://arxiv.org/html/2607.03751#bib.bib47 "π0.5: A vision-language-action model with open-world generalization")]. SVA keeps these backbones strictly frozen, so the action distribution we re-rank is exactly the one a real robot would produce. By construction, the evaluation bottleneck exposed by our pass@k analysis is a property of this real-data-grounded distribution, and the Q-model’s role in selecting among already-feasible candidates is identical on hardware.

###### SVA is simulator-free at deployment.

Unlike methods that require online tree search, environment resets, or dense reward signals at inference time, SVA invokes only the frozen VLA and the lightweight Q-model. Both consume the same RGB + proprioception + language inputs a real robot produces and emit actions in the robot’s native action space. The simulator is used solely during training of the Q-model, as a source of task-success labels for MCTS rollouts, and never as a source of privileged features that the deployment-time evaluator depends on. Crossing the sim-to-real boundary therefore reduces to the standard VLA observation-distribution shift, a challenge that is orthogonal to our contribution and is already targeted by the chosen backbones’ real-data pretraining.

###### Transferring SVA to a real robot requires no algorithmic change.

Because the Q-model is a feed-forward network over standard observations, deploying SVA on a physical robot requires either (a) using the simulator-trained Q-model zero-shot, or (b) lightly fine-tuning it on a small set of real-robot rollouts with the same MCTS-style return targets. We highlight a physical-robot replication of our SimplerEnv/RoboTwin results as the most pressing next step.

### Appendix D Experimental Details of SVA

#### D.1 Q-Model Architecture

The Q-model is initialized from Qwen/Qwen3.5-0.8B. We convert the backbone into a multimodal action-value estimator by adding a special <|VALUE|> token to the tokenizer. For each candidate action sequence, the input contains the current observation, language instruction, recent interaction history, and candidate action sequence. The hidden state at <|VALUE|> is used as the state-action representation.

For EmbodiedBench, candidate actions are discrete high-level skills and are serialized directly into the textual prompt. For SimplerEnv and RoboTwin, candidate actions are continuous robot control vectors. To represent these continuous action chunks in the language-model input space, we introduce the FAST[[31](https://arxiv.org/html/2607.03751#bib.bib50 "FAST: efficient action tokenization for vision-language-action models")] action tokenizer from physical-intelligence/fast. Given an action chunk, FAST converts the continuous action vectors into a sequence of discrete action tokens. We then map these tokens to newly added Qwen special tokens, bracketed by action boundary tokens, and append them to the prompt before <|VALUE|>. This design allows the same Qwen backbone to score both symbolic action sequences and continuous robot action chunks.

The backbone hidden dimension is 1024. We attach five bootstrapped scalar Q-heads. Each head is implemented as

\texttt{Dropout}(0.1)\rightarrow\texttt{Linear}(1024,512)\rightarrow\texttt{GELU}\rightarrow\texttt{Linear}(512,1).

The final Q-value is the ensemble mean, and the ensemble standard deviation is used as an uncertainty estimate during reranking. The five Q-heads contain 2,626,565 trainable parameters in total.

We fine-tune the backbone with LoRA and train the Q-heads jointly. LoRA is applied to q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, and down_proj. We use rank 16, \alpha=32, dropout 0.05, and do not train bias terms.

#### D.2 Search-Stage Data Collection

Offline supervision is collected using simulator-backed MCTS. Each sample corresponds to a visited tree edge and stores the instruction, observation, interaction history, candidate action sequence, MCTS-backed return target, visit count, and metadata. MCTS is used only on the training split and is not used during act-time evaluation. For the VLA manipulation benchmarks, we use open-source policy checkpoints as action proposers: the \pi_{0} proposer in SimplerEnv is initialized from petkopetkov/INTACT-pi0-finetune-bridge, and the \pi_{0.5} proposer in RoboTwin 2.0 is initialized from motus-robotics/pi0.5_robotwin2.

Table 4: Search-stage data collection settings.

Benchmark Proposer Cand.Sims.Depth\gamma PUCT Training split
EB-Habitat Qwen3.5 16 32 10 0.98 2.0 Episodes 1–30
EB-Navigation Qwen3.5 16 32 10 0.98 2.0 Episodes 1–36
SimplerEnv\pi_{0}32 256 15 0.99 4.0 15 seeds
RoboTwin 2.0\pi_{0.5}16 256 100 0.99 4.5 30 seeds

#### D.3 Value-Stage Training

The Q-model is trained by supervised regression to MCTS-backed returns. Target values are normalized using the mean and standard deviation of each training set. We train separate Q-models for EB-Habitat, EB-Navigation, SimplerEnv and RoboTwin 2.0. Both models are trained for 5 epochs with AdamW, bfloat16 mixed precision, batch size 8, learning rate 1\times 10^{-4}, and gradient clipping of 1.0. All training and evaluation experiments are conducted on NVIDIA RTX PRO 6000 GPUs.

#### D.4 Act-Stage Evaluation

At each decision step, the base policy samples multiple candidate action sequences or continuous action chunks. The Q-model scores all candidates in a batch, and the candidate with the highest reranking score is executed. The agent then replans until task success, invalid execution, or the episode budget is reached.

Table 5: Act-stage evaluation settings. EmbodiedBench uses sampling temperature 0.7.

Benchmark Cand.Horizon\lambda_{u}\lambda_{p}Evaluation split
EB-Habitat 32 10 0.1 0.1 Episodes 31–50
EB-Navigation 32 6 0.1 0.1 Episodes 37–60
SimplerEnv 16 4 0.1 0 9 held-out seeds
RoboTwin 2.0 16 32 0 0 20 held-out seeds

### Appendix E Baselines

We also evaluate a RoboMonkey-style candidate reranking baseline for comparison. The baseline follows the same candidate selection formulation but uses benchmark-specific verifiers.

For RoboTwin 2.0, we collect MCTS supervision in simulation. At each decision state, we sample a set of candidate action chunks and derive preference supervision from MCTS over these candidates. A Qwen3.5-0.8B-based scorer is then trained with a pairwise preference objective. During evaluation, the policy samples 16 candidate chunks, the learned scorer ranks them conditioned on the multi-view observation and language instruction, and the top-ranked chunk is executed. For scoring, 5 action proposals are generated by \pi_{0.5}, and the remaining proposals are sampled from a Gaussian distribution fitted to these policy outputs. The grasp state is handled separately and determined by majority voting.

For SimplerEnv, we use the released RoboMonkey verifier, robomonkey-vla/monkey-verifier-7b, as a frozen verifier. The policy samples multiple 7-DoF WidowX action chunks. For each chunk, we query the verifier on each action and use the mean verifier reward as the chunk score. For scoring, 5 action proposals are generated by \pi_{0}, and the remaining proposals are sampled from a Gaussian distribution fitted to the policy outputs, while the grasp state is determined separately by majority voting. The candidate with the highest mean score is selected for execution.

### Appendix F Detailed Pass@k Results

Table[6](https://arxiv.org/html/2607.03751#A6.T6 "Table 6 ‣ Appendix F Detailed Pass@k Results ‣ Appendix ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models") reports task-level Pass@k results on representative manipulation benchmarks. The results show a consistent gap between Pass@1 and larger k across all benchmarks. This suggests that base VLA models frequently sample successful or near-successful candidates, but their default likelihood ranking does not always select them.

Table 6: Task-level Pass@k results across manipulation benchmarks using popular VLA models.

Benchmark Task Pass@1 Pass@2 Pass@4 Pass@8 Pass@16 Pass@32
Libero Long(OpenVLA)Pick up Book 0.8440 0.9674 0.9981 1.0000 1.0000 1.0000
Soup and Sauce in Basket 0.6460 0.8467 0.9614 0.9970 1.0000 1.0000
Put Mug on Plate 0.4420 0.6600 0.8390 0.9383 0.9856 0.9996
Moka Pots on Stove 0.2640 0.4459 0.6682 0.8649 0.9763 0.9998
Simpler(OpenVLA)Spoon on Towel 0.1700 0.2571 0.3646 0.4949 0.6469 0.8087
Carrot on Plate 0.0740 0.1385 0.2453 0.3993 0.5936 0.8284
Stack Cubes 0.0260 0.0507 0.0966 0.1767 0.3057 0.5074
Eggplant in Basket 0.1080 0.2016 0.3541 0.5639 0.7873 0.9568
RoboTwin(\pi_{0.5})Turn Switch 0.3100 0.4950 0.6851 0.8248 0.9205 0.9858
Move Stapler Pad 0.1460 0.2651 0.4455 0.6665 0.8638 0.9787
Click Bell 0.6600 0.8078 0.8927 0.9377 0.9694 0.9958
Rotate QRCode 0.3560 0.5482 0.7346 0.8747 0.9588 0.9957

### Appendix G Detailed VLA Manipulation Results

Table[7](https://arxiv.org/html/2607.03751#A7.T7 "Table 7 ‣ Appendix G Detailed VLA Manipulation Results ‣ Appendix ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models") and Table[8](https://arxiv.org/html/2607.03751#A7.T8 "Table 8 ‣ Appendix G Detailed VLA Manipulation Results ‣ Appendix ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models") provide task-level results on SimplerEnv and RoboTwin 2.0. On SimplerEnv, SVA improves over the \pi_{0} baseline on manipulation tasks that require temporally extended object interaction, such as stacking cubes and placing the eggplant into the basket. On RoboTwin 2.0, SVA improves the average success rate and outperforms the base \pi_{0.5} policy on most tasks.

Table 7: Results on SimplerEnv with the WidowX platform. All values are success rates (%).

Task OpenVLA\pi_{0}\pi_{0}+RoboMonkey\pi_{0}+SVA
Spoon on Towel 29.6 23.6 29.6 27.8
Carrot on Plate 3.7 23.6 33.3 27.8
Stack Cubes 0.0 37.5 40.7 63.9
Eggplant in Basket 48.2 69.4 81.5 83.3
Average 20.4 38.5 46.3 50.7

Table 8: Results on RoboTwin 2.0. All values are success rates (%).

Task\bm{\pi_{0.5}}\bm{\pi_{0.5}}+RoboMonkey\bm{\pi_{0.5}}+SVA
Turn Switch 11.7 20.0 31.7
Move Stapler Pad 1.7 5.0 3.3
Click Bell 53.3 53.3 68.3
Rotate QRCode 41.7 43.3 40.0
Press Stapler 50.0 55.0 66.7
Click Alarmclock 78.3 76.7 88.3
Move Playingcard Away 48.3 50.0 50.0
Move Pillbottle Pad 21.7 21.7 26.7
Place Mouse Pad 28.3 28.3 35.0
Place Phone Stand 25.0 30.0 25.0
Average 36.0 38.3 43.5

### Appendix H Detailed Results on Ablation Study

Table[9](https://arxiv.org/html/2607.03751#A8.T9 "Table 9 ‣ Appendix H Detailed Results on Ablation Study ‣ Appendix ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models") provides the complete ablation results on EB-Navigation. The results further support the necessity of the three-stage SVA design.

Table 9: Detailed ablation results on EB-Navigation.

Method EB-Navigation
Base Common Sense Complex Instr.Visual App.Long Horizon\columncolor cyan!10 Avg.
SVA 62.50 59.72 52.78 47.22 58.33\columncolor cyan!10 56.11
w/o MCTS 50.00 56.94 43.06 47.22 56.94\columncolor cyan!1050.83
w/o Q-model 47.22 43.06 38.89 38.89 48.61\columncolor cyan!1043.33
w/o Multi-candidate 40.28 37.50 40.28 30.56 47.22\columncolor cyan!1039.17

### Appendix I More Results on Case Study

We provide additional qualitative results for two representative cases, each comparing the base policy with a successful SVA rollout.

###### Case 1: distractor-aware instruction following.

The instruction is: _On the sofa there’s an apple, but instead find a plate and move it to the brown table._ As shown in Fig.[7](https://arxiv.org/html/2607.03751#A9.F7 "Figure 7 ‣ Case 2: spatial-relation grounding. ‣ Appendix I More Results on Case Study ‣ Appendix ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models"), the base policy is distracted by the sofa clause and fails after repeated invalid actions, while SVA follows the plate-centric plan and completes the task.

###### Case 2: spatial-relation grounding.

We further examine the instruction: _Find a wrench and move it to the right of the sink._ As shown in Fig.[8](https://arxiv.org/html/2607.03751#A9.F8 "Figure 8 ‣ Case 2: spatial-relation grounding. ‣ Appendix I More Results on Case Study ‣ Appendix ‣ Look Before You Leap: Distilling Tree Search into Action Evaluation for Frozen VLA Models"), the base policy repeatedly navigates to the sofa and attempts to pick up the wrench, producing no task progress. In contrast, SVA selects the spatially correct sequence: navigate to the sink, pick up the wrench, move to the right counter, and place it there.

Across both cases, SVA improves robustness by ranking candidate action sequences according to predicted long-horizon return rather than proposal likelihood alone. This enables the policy to reject distractor-driven or spatially incorrect plans and choose actions that better satisfy the task constraints.

\columncolor white CASE 1 _On the sofa there’s an apple, but instead find a plate and move it to the brown table._
\columncolor lightRed Base Policy ✗Failure
![Image 7: Refer to caption](https://arxiv.org/html/2607.03751v1/Figs/case_study/case1/base_0.png)

Step 0: navigate to sofa![Image 8: Refer to caption](https://arxiv.org/html/2607.03751v1/Figs/case_study/case1/base_1.png)

Step 1: follow distractor![Image 9: Refer to caption](https://arxiv.org/html/2607.03751v1/Figs/case_study/case1/base_6.png)

Step 6: invalid recovery![Image 10: Refer to caption](https://arxiv.org/html/2607.03751v1/Figs/case_study/case1/base_25.png)

Step 25: failure
\columncolor lightBlue SVA (Ours) ✓Success
![Image 11: Refer to caption](https://arxiv.org/html/2607.03751v1/Figs/case_study/case1/sva_0.png)

Step 0: nav. to kitchen![Image 12: Refer to caption](https://arxiv.org/html/2607.03751v1/Figs/case_study/case1/sva_1.png)

Step 1: find plate![Image 13: Refer to caption](https://arxiv.org/html/2607.03751v1/Figs/case_study/case1/sva_3.png)

Step 3: move to target![Image 14: Refer to caption](https://arxiv.org/html/2607.03751v1/Figs/case_study/case1/sva_4.png)

Step 4: success

Figure 7: Case 1 – distractor-aware instruction following. The base policy follows the salient but irrelevant apple-on-sofa clause and fails after repeated invalid recovery actions. SVA selects a plate-centric plan and completes the task in 4 steps. 

\columncolor white CASE 2 _Find a wrench and move it to the right of the sink._
\columncolor lightRed Base Policy ✗Failure
![Image 15: Refer to caption](https://arxiv.org/html/2607.03751v1/Figs/case_study/case2/base_1.png)

Step 1: wrong search![Image 16: Refer to caption](https://arxiv.org/html/2607.03751v1/Figs/case_study/case2/base_2.png)

Step 2: invalid pickup![Image 17: Refer to caption](https://arxiv.org/html/2607.03751v1/Figs/case_study/case2/base_10.png)

Step 10: repeated failure![Image 18: Refer to caption](https://arxiv.org/html/2607.03751v1/Figs/case_study/case2/base_20.png)

Step 20: failure
\columncolor lightBlue SVA (Ours) ✓Success
![Image 19: Refer to caption](https://arxiv.org/html/2607.03751v1/Figs/case_study/case2/sva_1.png)

Step 1: nav. to sink![Image 20: Refer to caption](https://arxiv.org/html/2607.03751v1/Figs/case_study/case2/sva_2.png)

Step 2: pick wrench![Image 21: Refer to caption](https://arxiv.org/html/2607.03751v1/Figs/case_study/case2/sva_3.png)

Step 3: move right![Image 22: Refer to caption](https://arxiv.org/html/2607.03751v1/Figs/case_study/case2/sva_4.png)

Step 4: success

Figure 8: Case 2 – spatial-relation grounding. For the instruction _Find a wrench and move it to the right of the sink_, the base policy searches an incorrect location and issues repeated invalid pickup actions, while SVA grounds the spatial relation and completes the rearrangement.

