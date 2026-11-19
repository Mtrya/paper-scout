<!-- page 1 of 45 -->

arXiv:2610.39497v1 [cs.AI] 29 Oct 2026

# Does Scaling Toward AGI Depend on What System-1 Lets System-2 Execute?

**Qi Wang**<sup>1</sup>**, Yun Qu**<sup>1</sup>**, Lizhou Cai**<sup>1</sup>**, Heming Zou**<sup>1</sup>**, Haoang Chi**<sup>1</sup>**, Yiqin Lv**<sup>1</sup>**, Xiang Li**<sup>1</sup>, **Haiyan Feng**<sup>1</sup>**, Mengyu Ge**<sup>2</sup>**, Yize Geng**<sup>2</sup>**, Yuhang Jiang**<sup>1</sup>**, Yixiu Mao**<sup>1</sup>**, Xiangyang Ji**<sup>1†</sup>

<sup>1</sup>Department of Automation, Tsinghua University, Beijing, China

<sup>2</sup>Department of Computer Science, Tsinghua University, Beijing, China

Progress toward artificial general intelligence (AGI) has largely relied on scaling a resource-intensive executive, System-2, through larger models, more data and compute, longer reasoning, and more agents. We ask whether the next stage may come instead from co-designing it with System-1, a lightweight, semantically aware predictor that evaluates candidate interactions before execution, so that System-2 acts only on those worth their cost. Our answer is a qualified yes. We study conditional interaction, a predict–gate–execute–update loop: System-1 estimates each candidate’s utility, System-2 executes those admitted by an adaptive threshold, and the outcomes update System-1. The loop spans training, test-time reasoning, adaptive decision-making, and agentic tool use, with utility and cost defined by the application. Under ideal screening and equal execution costs, execution compute falls by a factor approaching $1 / \rho _ { u } ,$ where $\rho _ { u }$ is the density of utility-relevant interactions (cost-weighted when costs differ); prediction errors, exploration, and distribution shift reduce the saving. We bound selection regret by predictor error and the utility margin. These results motivate our central hypothesis: $\rho _ { u }$ is a measurable coordinate of scaling efficiency and makes informative compute a better description of scaling than total compute.

**Date:** October 27, 2026 **Correspondence:** †Prof. Xiangyang Ji at [xyji@tsinghua.edu.cn](mailto:xyji@tsinghua.edu.cn) **Project Website:** [https://cheemswang.github.io/blogs/sys1-sys2-codesign.html](https://cheemswang.github.io/blogs/sys1-sys2-codesign.html)

**THU-IDM**

[**www.thuidm.com**](https://www.thuidm.com/)

## 1 The Interaction Problem at Scale

Recent advances toward artificial general intelligence (AGI) (Morris et al., 2024) have largely focused on scaling model capacity, data volume, and computational resources (Sutton, 2019; Kaplan et al., 2020; Hoffmann et al., 2022), along with innovations in architectures, pretraining or post-training techniques, retrieval, and inference-time search. A significant portion of this investment has gone to a resource-intensive intelligent system, referred to as System-2, which handles reasoning, planning, decision-making, function calling, or even computer use. As frontier large models (Jaech et al., 2024; Guo et al., 2025), extend compute, and larger collectives of agents emerge, the potential of this executive system grows proportionally with resource availability. However, as systems become more capable, the costs of generating, evaluating, and acting on each additional interaction may become a constraining factor. This presents an alternative scaling challenge: not merely how many interactions a system can manage, but which interactions justify their associated costs.

An interaction encompasses any information exchange between an intelligent system and its data, computations, or environment. It gathers information at a cost: for a large language model (LLM), a generation episode in which a prompt produces tokens, such as a Chain-of-Thought (CoT) (Wei et al., 2022); for an embodied or reinforcement-learning agent, a trajectory of states, actions, and rewards; for a tool-using agent, a tool invocation that calls an external resource and integrates its result. Here, we treat interaction as the unit of analysis, recognizing that these operations are not necessarily equivalent in a physical or statistical sense; they share that each incurs resource commitments before observing the outcome and realizes utility only afterward. Even though reasoning models (Guo et al., 2025) and embodied agents (Kim et al., 2024) benefit from increased interactions, the relative returns from each interaction

1

<!-- page 2 of 45 -->

![Image block](doc:a7b5b7e/tier:basic/page:2/block:1)

Figure 1 An Illustrative Roadmap to AGI and its Beyond. As AI development progresses, broad capability resembles a summit that systems approach but may hit diminishing returns due to constraints in data, compute, and parameters. In Era 1 (2012–2020) progress came mainly from scaling model capacity, and in Era 2 (2020–2025) from scaling data and computation. For Era 3 (2026+), this work emphasizes a next-scaling lens: co-designing a cost-effective System-1 that prioritizes interactions by expected utility and a resource-intensive System-2 for complicated tasks. This emphasis becomes increasingly consequential for foundation models and embodied agents as they depend on generation, computation, tool use, and environmental interaction, and thus on the cost of each.

tend to diminish. For instance, tokens that fail to alter the relevant belief state, training rollouts that yield little policy-gradient signal under the selected objective (Yu et al., 2026), and environment steps that reiterate known information all consume valuable resources while providing minimal task-relevant learning signals.

Determining which interactions to execute is a challenge that System-2 cannot resolve autonomously because assessing an interaction’s value must cost considerably less than executing it. This necessitates the co-design of System-1 and System-2, and this work introduces the conditional interaction as a feasible co-design principle: a lightweight System-1 predicts the utility of potential interactions, the resource-intensive System-2 executes only those interactions that exceed a resource-aware threshold, and the resulting outcomes then serve to refine System-1. Because each system shapes the other, the two must be designed together, and this Perspective asks whether such co-design is a promising path toward AGI. Our central hypothesis posits that the density of utility-relevant interactions, denoted as $\rho _ { u }$ , rather than the sheer count of interactions alone, serves as a pivotal metric of scaling efficiency. The co-design of System-1 and System-2, as detailed below, indicates a promising approach for scenarios where interactions are costly, exhibit uneven value, and can be anticipated. This framework is based on the premise that low-cost experiments can be conducted and foreseen effectively.

**The Closing of the Accumulation Frontier.** The pursuit of accumulating additional data is increasingly difficult. Human-generated data are finite, since pretraining consolidates much of humanity’s recorded text into a single model and the stock of public text may be depleted within this decade (Villalobos et al., 2024), while high-quality data are scarcer still, as the subset that teaches most is only a fraction of that stock. Synthetic data shift the focus from

2

<!-- page 3 of 45 -->

quantity to coverage: they can partially replenish both (Xu et al., 2025), but the space of synthesizable interactions is combinatorially vast, e.g., robotic data alone would require covering scenes, objects, dynamics, lighting, and sensing, so the pertinent question becomes which synthetic interactions are worth their cost. Moreover, some interactions are inherently expensive: in scientific discovery, robotics, and specialized clinical and industrial settings, data are produced one experiment, actuation, or measurement at a time, at the pace of the underlying process, and no amount of computational power can accelerate a chemical reaction. Shifting the focus toward continual, post-deployment adaptation, we can find that self-evolution sharpens rather than alleviates the bottleneck: an agent that interacts indiscriminately exhausts its finite budget on experiences that yield minimal task-relevant learning, so its progress hinges on which interactions it selects, precisely the decision this Perspective examines.

**The vanishing marginal return of undirected interactions and complexity of search space.** Neural scaling laws (Kaplan et al., 2020; Hoffmann et al., 2022) illustrate why counting interactions is ineffective. Let C be cumulative compute, $\mathcal { L } ( C )$ the loss, and $\mathcal { P }   =   1 - \mathcal { L }$ a performance measure, e.g., any smooth $\mathcal { P }   =   f ( \mathcal { L } )$ with bounded $f ^ { \prime }$ yields similar conclusions. Under a power law $\mathcal { L } ( C ) \propto C ^ { - \alpha } , \alpha > 0$ , the marginal gain from one more unconditional interaction is

$$
\eta _ { \tt u n c } ( t ) \; = \; \frac { d \mathcal { P } } { d C } \Bigg | _ { t } \; = \; - \frac { d \mathcal { L } } { d C } \Bigg | _ { t } \; \propto \; \alpha   C ^ { - \alpha - 1 } ,\tag{1}
$$

which vanishes as compute grows. This empirical relation describes diminishing returns on a fixed landscape traversed by undifferentiated interactions; altering which interactions are executed can shift the curve itself, and this Perspective explores that degree of freedom. Information gain is a canonical utility but not a universal one (Theorems 3 and 5). The following two regimes make the gap between interaction count and interaction utility concrete, and show where the waste arises.

The digital regime: intensive computations, wasted rollouts. In reinforcement learning with verifiable rewards (RLVR), a reasoning model is post-trained on problems with automatically checkable answers, scoring each sampled thought chain (a rollout) with a low-cost verifier. Group Relative Policy Optimization (GRPO) (Shao et al., 2024; Guo et al., 2025) samples 𝐺 responses $\{ o _ { g } \} _ { g = 1 } ^ { G }$ per prompt 𝜏, scores each with reward $r _ { g } ,$ and forms group-relative advantages

$$
A _ { g } ^ { \mathrm { G R P O } } ( \tau ) = \frac { r _ { g } - \mu _ { r } } { \sigma _ { r } + \epsilon } , \quad \mu _ { r } = \frac { 1 } { G } \sum _ { g = 1 } ^ { G } r _ { g } , \quad \sigma _ { r } ^ { 2 } = \frac { 1 } { G } \sum _ { g = 1 } ^ { G } ( r _ { g } - \mu _ { r } ) ^ { 2 } ,\tag{2}
$$

which weight the per-token log-likelihood gradients in a clipped surrogate objective regularized toward a reference policy. When all responses receive the same reward $( r _ { g } \equiv \mu _ { r } )$ , every $A _ { g } = 0$ , rendering the prompt degenerate: it provides no relative policy-gradient signal while consuming 𝐺 rollouts. Only prompts of intermediate difficulty yield mixed outcomes, and a substantial fraction of the pool is degenerate (Yu et al., 2026; Qu et al., 2026a). Conditional interaction aids in predicting which prompts yield non-zero advantage, thereby optimizing the generation budget.

The physical regime: expensive rollouts in high-dimensional task spaces. In contrast to the digital regime’s cheap interactions, the physical regime is constrained by expensive rollouts: real-world interactions cannot be freely parallelized and consume tangible time tied to physical dynamics. A generalist agent trained by domain randomization performs across a simulation parameter space $\Xi \subseteq \mathbb { R } ^ { d }$ , where the rollouts needed for generalization grow with the effective dimensionality of Ξ and the complexity of its deployment scenarios. The qualitative insight is that, in high-dimensional spaces, which tasks are evaluated holds equal importance to how many are assessed.

**The super parrot hypothesis: AI for AI, and its efficiency issue.** Fig. 2 sketches a trend we call the super parrot hypothesis. The early foundation model first behaves as a “stochastic parrot” (Bender et al., 2021), imitating human text without grounded understanding (Stage 1, roughly 2020–2022); CoT prompting (Wei et al., 2022) and RLVR (Shao et al., 2024; Guo et al., 2025) then let it develop reasoning that humans can still follow (Stage 2, 2022–2025); and we hypothesize a Stage 3 in which it encodes information in self-coded representations and improves itself through self-evolution, beyond human interpretation. Viewed as a trajectory, this is a trend toward AI for AI: at each stage a larger share of what the system learns from is produced by AI itself, synthetic instructions (Xu et al., 2025), self-generated reasoning traces scored by automatic verifiers, and ultimately self-chosen experiments (Silver and

3

<!-- page 4 of 45 -->

![Image block](doc:a7b5b7e/tier:basic/page:4/block:1)

Figure 2 The Super Parrot Hypothesis. An illustrative, hypothesized trajectory rather than an established chronology: a conceptual progression of AI from imitation-based language learning (2020–2022), through interpretable reasoning enabled by chain-of-thought (CoT) and reinforcement learning with verifiable rewards (RLVR) (2022–2025), to a hypothesized future (2026+) of self-coded representations and self-evolution that may exceed human interpretability. As AI increasingly produces the data and feedback it learns from (AI for AI), deciding which self-generated interactions are worth executing becomes a central efficiency problem.

Sutton, 2025). The same trend exhibits the efficiency issue in sharper form. Every self-generated interaction is paid for with the system’s own compute, yet many of them, such as degenerated RLVR prompts, reasoning tokens that leave the answer unchanged, and self-proposed tasks that are already mastered, teach little to the model. Developing a super parrot therefore needs an internal judgment of value before execution, which is precisely the role this Perspective assigns to System-1.

All of the above discussions raise a fundamental Research Question: How can a System-2 be aware of an interaction’s value before execution?

**Our position: co-design the two systems, and the conditional interaction decides what the other pays for.** In response, we prioritize architecture over algorithm: give the executive a partner whose job is to judge, and design the two together. This work argues that a major part of the next advance toward AGI and beyond may come from co-designing System-1 and System-2 rather than from scaling System-2 alone: specifying, training, and scaling the two as one system, where System-1 decides what System-2 executes, which executive runs it, how much it may spend, and what it keeps in memory; System-2’s outcomes are System-1’s supervision; and each is sized with the other in view (Section 3). Integrating a cheap predictor with a costly executor introduces three decisions at scale: which prompts, tasks, or requests merit rollouts, trials, or extensive thought chains (compute); which samples, buffers, caches, and context entries should be preserved (storage); and how costly outcomes should adjust the inexpensive judgment that allocated them (feedback).

This work proposes conditional interaction as a coupling mechanism that allocates interactions based on predicted utility, explicitly optimizing to avoid those predicted to fall below a certain threshold. This perspective applies to

4

<!-- page 5 of 45 -->

![Image block](doc:a7b5b7e/tier:basic/page:5/block:1)

Figure 3 The design objective: a better Pareto frontier over cost, latency and task success. Each surface is the best task success probability attainable within a cost and latency budget. The co-designed pair would spend both only where they change the answer, so its frontier (green) is hypothesized to dominate: the same success more cheaply (red), or more of it at the same budget (blue). This expresses the co-design objective.

diverse applications, including online prompt selection in reinforcement learning from verifiable rewards (RLVR) (Qu et al., 2026a), task selection in embodied agents (Qu et al., 2025), model routing, and adaptive inference (Li et al., 2026) (Section 2.3). The same pattern applies in a deployed system, where the ledger records inference compute and real-world actions rather than gradient updates (Snell et al., 2024). The transfer focuses on the decision pattern and the scope is intentionally confined to situations with a reliable utility predictor, as this principle does not substitute exploration when the predictor is uncertain (Section 2.4).

$\rho _ { u } ;$ **an efficiency coordinate to scale limited resource.** Classical scaling laws express performance $\mathcal { P }$ through parameters 𝑁, data 𝐷, and compute C. The density of utility-relevant interactions $\rho _ { u } ,$ the fraction of candidate interactions whose utility 𝑢 clears a declared floor $u _ { \mathrm { m i n } }$ in Eq. (14), is not an additional resource but an efficiency coordinate $\mathcal { P }   =   f \big ( N , D , C ; \rho _ { u } \big )$ describing how well the three convert into capability, generated by the learning dynamics rather than supplied externally. For the canonical utility, Bayesian information gain, $\rho _ { u }$ is the informative interaction density $\rho _ { \mathrm { i n f o } } ,$ a special case of the general $\rho _ { u }$ in which every formal result is stated; every statement is nonetheless relative to a declared utility, since the densities of prompts with non-zero GRPO advantage, of tasks above an adaptation-risk threshold, and of queries whose answer improves with more compute are different quantities and are not interchangeable.

Uninformative interactions waste both computation and the data slots they occupy, so the effective resources under unconditional training are discounted as $\rho _ { u } D$ and $\rho _ { u } C$ . The compute discount is a direct accounting identity; $\rho _ { u } D$ is a proposed measure of utility-relevant data consumed, not a smaller available corpus, which a gate leaves unchanged. Two interpretations of this fraction must be kept apart. The pool fraction $\rho _ { u } ( \theta _ { t } )$ is a property of the candidate distribution at the current state, produced by the learning dynamics, and no gate can change it; the executed fraction is the share of interactions actually paid for that are informative. Unconditional training executes the pool, so the two coincide, while a perfect gate drives the executed fraction to one. Conditional interaction raises the executed fraction, so that at fixed 𝑁, 𝐷 and C more of the budget tends to buy utility-relevant interactions and, hypothetically, as a dynamic account of the exponent. Note that $\rho _ { u }$ depends on the policy state, it is not a free axis.

5

<!-- page 6 of 45 -->

**A potentially tighter Cost–QoS–Intelligence frontier.** Directing interactions by predicted utility can tighten the Pareto frontier over compute, data, and model complexity: For example, under the assumptions of Theorem 24, a target is probably reached with less compute, or a fixed budget yields more performance. Suppressing the uninformative fraction $1 - \rho _ { u }$ acts on every axis at once, since fewer rollouts and gradient reduce compute and data, and routing each task only to the System-2 models it needs cuts the capacity activated per task. The same holds at deployment: training-time gating lowers cost, test-time gating improves quality of service with lowers latency, and concentrating both on informative interactions raises task success per unit of either, so the co-designed pair aims at a frontier that improves on scaling System-2 alone on all three (See Fig. 3), the comprehensive co-design objective.

**What we claim, and with what confidence.** Throughout this work, the claim hierarchy is as follows. (i) Premise: future digital and robotic systems may be limited by interaction efficiency more than by parameters or data. (ii) Position: progress toward general intelligence should co-design a cheap System-1 with a costly System-2, coupled by a strategy that values interactions before execution and suppresses those predicted to be uninformative. (iii) Hypothesis: the density of utility-relevant interactions $\rho _ { u }   ( \rho _ { \mathrm { i n f o } }$ for information gain) is an important determinant of how efficiently resources become capability. (iv) Prediction: $\rho _ { u }$ decays as the system improves, and this decay may account for part of the diminishing returns and of the scaling exponent, so that scaling laws are better described in informative compute than in compute alone.

The rest of this Perspective proceeds as follows: formalizing the predict–gate–execute–update loop to implement the conditional interaction (Section 2); building and co-scaling the two subsystems (Section 3); deriving the loop’s screening and regret properties, and why a lightweight System-1 can provide them (Section 4); instantiating the framework across digital training, physical adaptation, test-time computation and agentic use (Sections 5 and 6); and formulating informative compute as a scaling hypothesis (Section 7).

## 2 Conditional Interaction as A Co-Design Strategy: Predict Before You Pay

Co-design requires an interface: a precise statement of what System-1 judges, what System-2 executes as a resource intensive operation, and what flows back. Conditional interaction is that interface. Both systems are defined by execution role rather than model family: System-1 is a lightweight, semantic-aware module that takes a candidate interaction as input, e.g., a prompt, task configuration, query, or tool call, with its context and returns a score rather than a solution, and it is consulted on every candidate at a cost $c _ { \mathrm { p r e d } } \ll c _ { \mathrm { e x e c } }$ . System-2 is the resource-demanding model or agent that performs the interaction, and is invoked only on the candidates System-1 certifies. The loop runs in both directions, making it co-design rather than a filter bolted onto a finished executive: System-1 shapes the data and effort System-2 receives, and System-2’s outcomes become System-1’s supervision. AlphaZero illustrates this, since its play comes not from deeper search alone but from a policy-value network and a tree search that trained each other (Silver et al., 2018): a more capable System-2 raises the intelligence of accomplishing tasks, while a well-calibrated System-1 determines how much of it is reached per unit of cost and latency (See Fig. 3).

**Static cascades versus adaptive co-design.** The pairing can be built in two ways (See Fig. 4). In a static cascade, a standalone fast model classifies each request and hands the hard minority to a large model; Jev, a commercial model described by its developers as a “System One” (TypeSafe AI, 2026), is a clear instance, returning calibrated probabilities over answers defined in advance (Osman, 2026). Such products illustrate commercial interest in fast, structured judgment models compatible with the System-1 role, although a single product does not establish an industry-wide convergence. A static cascade is, by construction, trained once, routes only at the inference layer, and does not receive System-2’s outcomes, which suffices when its predictions are stationary but fails here, because how informative an interaction is for the current System-2 changes with every update. In co-design, System-1 instead predicts utility such as difficulty or learning value relative to the current System-2; it acts in training as well as at inference, scheduling compute and storage such as rollout budgets, replay buffers, batches, key–value (KV) caches, context windows, rather than a single call path. System-2’s realized rewards, gradients, and outcomes recalibrate System-1 online (See Fig. 5). A cascade retrained online on its outcomes moves toward co-design, so the distinction is not the presence of two models but feedback and joint optimization.

6

<!-- page 7 of 45 -->

![Image block](doc:a7b5b7e/tier:basic/page:7/block:1)

Figure 4 Static cascades versus adaptive co-design. (a) A standalone fast model routes each request once and is never updated by the outcomes. (b) In co-design, System 1 schedules System $2 ^ { \circ } \mathrm { s }$ compute and storage, System $2 ^ { \circ } \mathrm { s }$ outcomes recalibrate System 1, and the two are optimized jointly, so System 1’s judgements track an executive that keeps changing. Cascades retrained online on their outcomes lie between the two.

## 2.1 Core Definition

**Definition 1** (Conditional Interaction Policy). Let I be the space of feasible interactions, $\mathcal { H } _ { t } = \{ ( i _ { s } , o _ { s } ) \} _ { s = 0 } ^ { t - 1 }$ <sup>the</sup> history of interaction–outcome pairs, and $\hat { u } : \mathcal { I } \times \mathcal { H } ^ { * } \rightarrow$ R a predicted utility. A conditional interaction policy is the selection rule

$$
\pi _ { \mathrm { C I } } ( i _ { t } \mid \mathcal { H } _ { 0 : t } ) = \frac { 1 } { Z _ { t } }   \exp \bigl ( \beta   \hat { u } ( i _ { t } \mid \mathcal { H } _ { 0 : t } ) \bigr )   \cdot   \mathbf { 1 } [ \hat { u } ( i _ { t } \mid \mathcal { H } _ { 0 : t } ) \geq \kappa _ { t } ]   ,\tag{3a}
$$

$$
Z _ { t } = \sum _ { i \in \mathcal { I } } \exp \bigl ( \beta   \hat { u } ( i \mid \mathcal { H } _ { 0 : t } ) \bigr )   \mathbf { 1 } [ \hat { u } ( i \mid \mathcal { H } _ { 0 : t } ) \geq \kappa _ { t } ]   ,\tag{3b}
$$

where $\beta > 0$ controls selection sharpness, $K _ { t }$ is a utility threshold enforcing a minimum predicted utility, and $Z _ { t }$ is a sum for finite $\mathcal { I } ,$ or an integral against a base measure on a continuous I to normalize $\pi _ { \mathrm { C I } }$ over the admissible set $\{ i : \hat { u } \geq \kappa _ { t } \}$ . When the admissible set is empty $( Z _ { t } = 0 )$ , the policy selects an explicit abstention action ∅ with probability one and executes nothing at step 𝑡; <sup>𝜋</sup>CI is thus a distribution on ${ \mathcal { I } } \cup \{ \varnothing \}$ . The score 𝑢ˆ is a predicted gross utility, which is the right ranking signal when execution costs are equal; when they differ, the policy is applied to the cost-adjusted score 𝑢˜ of Eq. (16). Score-based results (the greedy limit, the threshold rule, the regret bounds and Theorem 21) then apply to 𝑢˜ provided the true score is redefined consistently as $u - \lambda C$ , with relevance, error rates and regret all measured on that net score. Results on the IIE and on $\rho _ { u }$ keep their gross-utility definitions unless explicitly redefined; a gate that is accurate for the net score need not have the same error rates with respect to 𝑢.

Interpretation. The indicator in Eq. (3a) restricts the search space by suppressing candidates whose predicted utility falls too far below the current budget. At the same time, the exponential factor sets how sharply the policy concentrates on the best feasible candidates, and hence how much it explores among them. This work reserves gate for this mechanism in prose and uses selection policy to denote $\pi _ { \mathrm { C I } }$ . We call the resulting closed loop — predict, gate, execute, and fold the outcome back into the predictor — the predict-then-gate architecture (See Fig. 5); every instance shares it in this Perspective, whatever utility fills and whatever ledger it is scored against.

**From soft gating to greedy selection: limiting regimes of the policy.** The two parameters of Eq. (3) play distinct roles: the threshold $K _ { t }$ decides which interactions are admissible, and the inverse temperature $\beta$ decides how sharply the policy prefers the best admissible one. The threshold is a statement about utility, not about budget; a budget enters through a price, since in the master objective, i.e., Eq. (15), a multiplier 𝜆 on cost turns the threshold into the cost-dependent rule $\hat { u } ( i ) \geq \lambda C ( i )$ , and the adaptive update Eq. (22) is one way of setting $K _ { t }$ so that a target admission rate is met. Write $\hat { u } _ { t } ( i ) = \hat { u } \big ( i \mid \mathcal { H } _ { 0 : t } \big )$ , and let

$$
\mathcal { I } _ { t } ^ { \mathrm { a d m } } = \{ i \in \mathcal { I } : \hat { u } _ { t } ( i ) \geq \kappa _ { t } \} , \qquad \hat { u } _ { t } ^ { \mathrm { m a x } } = \operatorname* { m a x } _ { i \in \mathcal { I } } \hat { u } _ { t } ( i ) , \qquad \mathcal { I } _ { t } ^ { \star } = \mathop { \mathrm { a r g   m a x } } _ { i \in \mathcal { I } } \hat { u } _ { t } ( i )\tag{4}
$$

denote the admissible set, the largest predicted utility, and the set of maximizers (assumed finite and attained).

7

<!-- page 8 of 45 -->

![Image block](doc:a7b5b7e/tier:basic/page:8/block:1)

Figure 5 The predict-then-gate mechanism. System 1 scores every candidate at cost $c _ { \mathrm { p r e d } } ,$ and the threshold $K _ { t }$ admits to System 2 only those predicted to be informative. Suppressed candidates cost only $c _ { \mathrm { p r e d } } ;$ executed outcomes, together with a small audit sample of suppressed candidates, are logged and refit System 1 (Theorem 14).

**Proposition 2** (Degeneration to the greedy argmax). Let I be finite and $\kappa _ { t } \leq \hat { u } _ { t } ^ { \operatorname* { m a x } }$ , so that every maximizer is admissible. As $\beta \rightarrow \infty ,$

$$
\pi _ { \mathrm { C I } } ( i _ { t } \mid \mathcal { H } _ { 0 : t } ) \longrightarrow \frac { \mathbf { 1 } \big [ i _ { t } \in \mathcal { I } _ { t } ^ { \star } \big ] } { | \mathcal { I } _ { t } ^ { \star } | } ,\tag{5}
$$

the greedy rule $\begin{array} { r } { i _ { t } = \arg \operatorname* { m a x } _ { i } \hat { u } \big ( i \mid \mathcal { H } _ { 0 : t } \big ) } \end{array}$ with ties broken uniformly. If the maximizer $i _ { t } ^ { \star }$ is unique, with gap $\Delta _ { t } > 0$ to the best other admissible candidate, then

$$
1 - \pi _ { \mathrm { C I } } \big ( i _ { t } ^ { \star } \mid \mathcal { H } _ { 0 : t } \big ) \leq \big ( | \mathcal { I } _ { t } ^ { \mathrm { a d m } } | - 1 \big )   e ^ { - \beta \Delta _ { t } } ,\tag{6}
$$

so, when $| \mathcal { I } _ { t } ^ { \mathrm { a d m } } | \geq 2 ,   \beta \geq \Delta _ { t } ^ { - 1 } \ln \bigl ( ( | \mathcal { I } _ { t } ^ { \mathrm { a d m } } | - 1 ) / \delta \bigr )$ selects $i _ { t } ^ { \star }$ with probability at least $1 - \delta ;$ when $| \mathcal { I } _ { t } ^ { \mathrm { a d m } } |   =   1$ it is selected with probability one for every $\beta .$

Any $\kappa _ { t } \leq \hat { u } _ { t } ^ { \operatorname* { m a x } }$ gives the same greedy limit, whereas $\kappa _ { t } > \hat { u } _ { t } ^ { \mathrm { m a x } }$ empties the admissible set and triggers abstention; a small threshold alone, at finite $\beta ,$ yields not the argmax but the unconstrained Boltzmann policy 𝜋 ∝ $e ^ { \beta \hat { u } _ { t } }$ , and as $\beta \rightarrow 0$ the policy becomes a pure filter that admits candidates without ranking them.

**Definition 3** (Interaction Utility). As a running example we take the expected Bayesian surprise (Itti and Baldi, 2005): the utility of interaction 𝑖 at executive state $\theta _ { t }$ is the expected posterior-to-prior divergence it induces,

$$
u ( i ; \theta _ { t } ) \triangleq \mathbb { E } \Big [   D _ { \mathrm { K L } } \Big [ p \big ( \theta \mid \mathcal { H } _ { 0 : t } , i , o \big )   \Big \|   p \big ( \theta \mid \mathcal { H } _ { 0 : t } \big ) \Big ] \big | \mathcal { H } _ { 0 : t } , i   \Big ]   ,\tag{7}
$$

with the expectation over the predictive distribution of the outcome 𝑜. It answers: $i f I$ execute this interaction and observe its outcome, how much will my belief change? Interactions with predictable outcomes score near zero; utility measures the magnitude of the belief update, not difficulty or novelty. It is the conditional mutual information $I ( \theta ; o \mid i , \mathcal { H } _ { 0 : t } )$ , which, when the relevant entropies are finite, equals the expected reduction in the entropy of 𝜃. The predictor 𝑢ˆ in Eq. (3) estimates whatever utility fills this slot; information gain is one possible choice of $u ,$ not the universal definition of usefulness. Different contexts require different utilities:

$$
\begin{aligned} &u(i) = \left\{ \begin{aligned}\\ &I(\theta; o \mid i) & & ( exploration  /  experiment design ), \\&\mathbb{E}[\Delta J \mid i] & & ( expected improvement in ourinclusion learning ), \\& Vol (i) = \mathbb{E}[\Delta V \mid i] & & ( decision making: value of information ), \\&\Re_{ adapt }(i; \theta_t) & & ( robust domain randomization  /  metal learning risk ).\\ &\end{aligned} \right.\\ \end{aligned}\tag{8}
$$

Their maximizers generally vary case by case. Eq. (7) coincides with the first line (Theorem 4) and, combined with a cost, underlies the third; the fourth encodes adaptation risk, and the second is the objective that predictive task

8

<!-- page 9 of 45 -->

samplers estimate directly. Two conventions keep these lines consistent with the master objective of Section 2.3. First, every 𝑢 is a gross utility: cost never enters 𝑢 and is charged once, through the ledger C (𝑖) and its price 𝜆, so the value of information becomes net of cost only through that charge. Second, the fourth line is an acquisition score, not the training objective: a robust learner minimizes the tail risk CVaR<sub>𝜈</sub> of Eq. (64), and to do so it prioritizes the tasks with the largest predicted adaptation risk $\mathcal { R } _ { \mathrm { a d a p t } }$ of Eq. (63), the score maximized in Eq. (65). We do not merge these utilities. What they share is the predict-then-gate architecture, and the efficiency results below (screening gain, regret bounds, rescaling) hold for any of them, given the predictor’s ranking quality and the cost asymmetry.

**Remark 4** (Representation invariance). The KL divergence is invariant under invertible reparameterizations of 𝜃, and $u ( i ; \theta _ { t } ) = I ( \theta ; o \mid i , \mathcal { H } _ { 0 : t } )$ is also invariant under invertible transformations of 𝑜. Hence the utility, and the IIE of Eq. (9), do not depend on the chosen coordinates.

**Remark 5** (Scope of the utility). Eq. (7) measures belief change, not task success, descent direction, or robustness, and coincides with each only under conditions: (a) inference success, only under uniform outcome preferences (Theorem 20); (b) the post-training update direction, only through Fisher (natural-gradient) preconditioning, so an uncertain but noisy interaction need not help (Qu et al., 2026a; Zou et al., 2025); (c) robust adaptation, only when evaluated on the tail, as in the CVaR adaptation risk of Eqs. (63) and (64) in Section B (Qu et al., 2025; Wang et al., 2026). Conversely, an interaction that barely moves the posterior can still be valuable: it may confirm a belief, test robustness, improve calibration, avert a rare failure, or inform another agent. Information gain, learning progress, decision value and task improvement are therefore distinct quantities, and conditional interaction is always relative to whichever one is declared.

## 2.2 The Interaction Information Efficiency Metric

**Definition 6** (Interaction Information Efficiency). For a stream of 𝑇 screened candidates $i _ { 0 } , \ldots , i _ { T - 1 }$ , the Interaction Information Efficiency (IIE) is the utility obtained per unit of system-level cost,

$$
\mathrm { I I E } _ { T } = \frac { \sum _ { t = 0 } ^ { T - 1 } \mathbb { I } ( i _ { t } )   \mathbf { 1 } [ i _ { t } \mathrm { ~ e x e c u t e d } ] } { \sum _ { t = 0 } ^ { T - 1 } \big ( c _ { \mathrm { p r e d } } ( i _ { t } ) + C ( i _ { t } )   \mathbf { 1 } [ i _ { t } \mathrm { ~ e x e c u t e d } ] \big ) } ,\tag{9}
$$

where $\mathbb { I } \big ( i _ { t } \big ) = u \big ( i _ { t } ; \theta _ { t } \big )$ is the declared utility (the expected Bayesian surprise of Eq. (7) in the canonical case), $C ( i _ { t } )$ the execution cost, and $c _ { \mathrm { p r e d } } ( i _ { t } )$ the cost of screening the candidate, paid whether or not it is executed (zero for unconditional interaction); it is the full per-candidate screening cost, encoding plus scoring, which Theorem 14 splits as $c _ { \mathrm { s c r e e n } } = c _ { \mathrm { e m b e d } } + c _ { \mathrm { s c o r e } }$ . Predictor refitting and audit executions are amortized system costs, reported separately (Section 4.1). Because a ratio can be raised by obtaining very little at very low cost, IIE is always reported together with the yield, the utility obtained per screened candidate,

$$
Y _ { T } \; = \; \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \mathbb { I } ( i _ { t } )   { \bf 1 } [ i _ { t } \; \mathrm { e x e c u t e d } ] \; .\tag{10}
$$

Both are empirical estimates, over a finite stream, of population quantities for candidates $i \sim \mathcal { D } ;$ for the IIE,

$$
\mathrm { I I I E } _ { \mathrm { p o p } } = \frac { \mathbb { E } \big [ \mathbb { I } ( i )   \mathbf { 1 } [ i   \mathrm { e x e c u t e d } ] \big ] } { \mathbb { E } \big [ c _ { \mathrm { p r e d } } ( i ) + C ( i )   \mathbf { 1 } [ i   \mathrm { e x e c u t e d } ] \big ] } ,\tag{11}
$$

and $\mathrm { I I E } _ { T } \to \mathrm { I I E } _ { \mathrm { p o p } }$ almost surely for an i.i.d. stream, by the strong law of large numbers applied to numerator and denominator. A ratio of random sums is not the ratio of their expectations at finite 𝑇, so the results below are stated for $\mathrm { I I E } _ { \mathrm { p o p } } ;$ they hold exactly for a fixed pool when fractions and class means are taken over that pool.

**Notation:** $\rho _ { u }$ **in general,** $\rho _ { \mathrm { i n f o } }$ as a special case. Every proposition, theorem and corollary below is stated for a general declared utility 𝑢 and its density of utility-relevant interactions $\rho _ { u }$ , the fraction of candidates whose true utility clears a floor $u _ { \mathrm { m i n } }$ fixed in advance (Eq. (14)); candidates below the floor are called dead. The informative interaction density $\rho _ { \mathrm { i n f o } }$ is the special case in which 𝑢 is the expected information gain of Eq. (7): there “utility-relevant” reads

9

<!-- page 10 of 45 -->

“informative”, I is the expected Bayesian surprise, and utilities are measured in nats. No result below relies on that choice unless it says so.

Partition the candidate pool into the utility-relevant fraction $\rho _ { u } ,$ with mean utility $\overline { { \mathbb { I } } } _ { \mathrm { r e l } }$ and mean cost $\bar { C } _ { \mathrm { r e l } }$ , and the dead fraction $1 - \rho _ { u }$ , with mean utility $\bar { \mathbb { I } } _ { \mathrm { d e a d } }$ and mean cost $\bar { C } _ { \mathrm { d e a d } }$ . Pool averages are $\bar { \mathbb { I } } = \rho _ { u } \bar { \mathbb { I } } _ { \mathrm { r e l } } + ( 1 - \rho _ { u } ) \bar { \mathbb { I } } _ { \mathrm { d e a d } }$ and $\bar { C } = \rho _ { u } \bar { C } _ { \mathrm { r e l } } + ( 1 - \rho _ { u } ) \bar { C } _ { \mathrm { d e a d } }$ , and unconditional interaction attains $\mathrm { I I E } ^ { \mathrm { u n c } } = \bar { \mathbb { I } } / \bar { C }$

**Proposition 7** (IIE Gain). Assume dead candidates carry no utility $\begin{array} { r } { ( \bar { \mathbb { I } } _ { d e a d } = 0 , } \end{array}$ , as when $u _ { \mathrm { m i n } } \to 0 ^ { + }$ for a non-negative utility), and use the system-level ledger of Eq. (9), in its population form Eq. (11), which charges a predictor cost <sup>𝑐</sup>pred to every screened candidate.

**1.** (Perfect, free predictor.) If exactly the utility-relevant candidates are executed and $c _ { p r e d } = 0 ,$

$$
\frac { \mathrm { I I E } ^ { \mathrm { C I } } } { \mathrm { I I E } ^ { u n c } } = \frac { \bar { \mathbb { I } } _ { r e l } / \bar { C } _ { r e l } } { \bar { \mathbb { I } } / \bar { C } } = \frac { 1 } { \rho _ { u } } \cdot \frac { \bar { C } } { \bar { C } _ { r e l } } = 1 + \frac { 1 - \rho _ { u } } { \rho _ { u } } \cdot \frac { \bar { C } _ { d e a d } } { \bar { C } _ { r e l } } ,\tag{12}
$$

which equals $1 / \rho _ { u }$ when $\bar { C } _ { d e a d } = \bar { C } _ { r e l }$ and exceeds it if and only $i f \bar { C } _ { d e a d } > \bar { C } _ { r e l } .$

**2.** (Imperfect, costly predictor.) With class-conditional false-positive rate fp (the fraction of dead candidates admitted) and false-negative rate fn (the fraction of utility-relevant candidates suppressed),

$$
\frac { \mathrm { I I E } ^ { \mathrm { C I } } } { \mathrm { I I E } ^ { u n c } } = \frac { \bar { \mathbb { I } } _ { r e l } / \bar { C } _ { r e l } } { \bar { \mathbb { I } } / \bar { C } } \cdot \left( 1 + \frac { \mathrm { f p } \left( 1 - \rho _ { u } \right) \bar { C } _ { d e a d } + c _ { p r e d } } { \left( 1 - \mathrm { f n } \right) \rho _ { u }   \bar { C } _ { r e l } } \right) ^ { - 1 } ,\tag{13}
$$

which is non-increasing in each of fp, fn and $c _ { p r e d } ,$ strictly decreasing in any of them whose coefficient is positive, and recovers Eq. (12) as $\mathrm { f p } , c _ { p r e d } \rightarrow 0$

False negatives do not dilute the IIE ratio directly, since suppressed interactions are simply not paid for; they enter by spreading screening and false-positive costs over fewer utility-relevant executions and, more importantly, by reducing the yield Eq. (10) by the factor $1 - \mathrm { f n }$ . Efficiency and yield must therefore be reported together: an aggressive gate can raise IIE while starving the learner of signal. With $\rho _ { u } \ll 1$ , small error rates, and $c _ { \mathrm { p r e d } } \ll \bar { C } _ { \mathrm { r e l } }$ , the realized multiplier remains substantial, consistent with the several-fold rollout savings reported for predictive prompt sampling (Qu et al., 2026a), but these are ideal values, not guarantees. Under equal execution costs and perfect screening, $1 / \rho _ { u }$ is the benchmark for the IIE gain; with heterogeneous costs the ideal gain is Eq. (12) itself, which exceeds $1 / \rho _ { u }$ when dead interactions cost more than relevant ones. The zero-utility assumption on dead candidates matters once $u _ { \mathrm { m i n } } > 0 ;$ if they retain a mean utility $\bar { \mathbb { I } } _ { \mathrm { d e a d } } > 0$ , the first equality of Eq. (12) still holds, but the ideal ratio becomes $\begin{array} { r } { \frac { 1 } { \rho _ { u } } \frac { \bar { C } } { \bar { C } _ { \mathrm { r e l } } } \left( 1 - ( 1 - \rho _ { u } ) \bar { \mathbb { I } } _ { \mathrm { d e a d } } / \bar { \mathbb { I } } \right) } \end{array}$ , so the closed forms of Eq. (12) become upper bounds (Section A.2), and $1 / \rho _ { u }$ bounds the gain only when, in addition, $\bar { C } _ { \mathrm { d e a d } } \leq \bar { C } _ { \mathrm { r e l } }$

**Remark 8** (Cost-correlated errors). Eq. (13) assumes error rates independent of cost. If errors concentrate on expensive interactions, the false-positive cost becomes $\begin{array} { r } { \int \mathrm { f p } ( C )   C   d F _ { \mathrm { d e a d } } ( C ) > \mathrm { f p }   \bar { C } _ { \mathrm { d e a d } } } \end{array}$ , so validation should report cost-conditioned rates.

**Remark 9** (Identifiability of $\rho _ { u } ) ,$ . Fitting $\rho _ { u }$ to a loss curve is circular, since $\mathcal { L } = a ( \rho ( C ) C ) ^ { - \alpha ^ { \prime } }$ with $\rho ( C ) = C ^ { s }$ fits any exponent. We therefore measure it independently: for a declared utility 𝑢 and a threshold $u _ { \mathrm { m i n } } > 0$ fixed a priori,

$$
\rho _ { u } ( \theta _ { t } ) = \operatorname* { P r } _ { i \sim \mathcal { D } } \bigl [ u ( i ; \theta _ { t } ) \geq u _ { \operatorname* { m i n } } \bigr ] ,\tag{14}
$$

with 𝑢 the true utility estimated from executed interactions, not the predictor’s $\hat { u } _ { \phi } ; \rho _ { \mathrm { i n f o } }$ denotes $\rho _ { u }$ for the information gain utility, and densities under different utilities are not comparable. In RLVR the natural declared utility is the probability that a group of 𝐺 rollouts has mixed rewards, $\mu _ { \mathrm { a d v } } ( \tau ; \theta _ { t } ) = 1 - p _ { \tau } ^ { G } - ( 1 - p _ { \tau } ) ^ { G }$ with $p _ { \tau }$ the current pass rate; with $u _ { \mathrm { m i n } } \to 0 ^ { + } , \rho _ { u }$ becomes $\rho _ { \mathrm { a d v } }$ , the fraction of prompts that can yield a non-zero GRPO advantage. This is a proxy for useful policy-gradient signal, not for information gain: an all-fail group still informs a posterior over the prompt’s difficulty, and a mixed group can give a noisy update, so $\rho _ { \mathrm { a d v } }$ and $\rho _ { \mathrm { i n f o } }$ are distinct densities. In the physical regime $\rho _ { u }$ is the fraction of tasks whose adaptation risk exceeds $u _ { \mathrm { m i n } }$ . Because $\rho _ { u }$ depends on $( u , u _ { \mathrm { m i n } } )$

10

<!-- page 11 of 45 -->

the meaningful object is its relative decay $\rho _ { u } ( \theta _ { t } ) / \rho _ { u } ( \theta _ { 0 } )$ . We propose its agreement across thresholds and across proxies such as the expected gradient norm or the learnability score $\partial _ { t } \mathbb { E } [ r \mid \tau , \theta _ { t } ]$ as a robustness diagnostic, not a theoretical invariance: tail probabilities at different thresholds need not decay proportionally, and gradient norm and learning progress are not equivalent. Measurement also needs a protocol. Because the gate censors what is observed (Section 7.2), $\rho _ { u }$ is estimated on an audit pool drawn from D independently of the gate, with repeated rollouts per candidate, confidence intervals, and re-estimation as $\theta _ { t }$ changes.

## 2.3 A Master Objective and Its Special Cases

Conditional interaction provides a common predict-then-gate architecture for online prompt selection in LLM post-training, active domain randomization for Sim2Real transfer, and adaptive experimental design in AI for science. The unification is one of architecture, not of objective. What elevates the family from heuristics to a scaling lens is (i) suppression as a first-class objective, (ii) predictive amortization, i.e., estimating utility before execution, and (iii) generality across interaction types, cost structures, policy representations, and reward topologies. On its own, this is a family resemblance; to formalize the claim, we state one optimization problem and obtain each technique by instantiating its variables, including the utility that fills the gating slot.

**Definition 10** (Master Conditional-Interaction Objective). Let I be a finite menu of candidates with predicted utility $\hat { u } ( i )$ and cost $C ( i ) > 0$ , and let $x ( i ) \in [ 0 , 1 ]$ be the probability of executing candidate 𝑖. The master objective maximizes expected executed utility under a cost budget Γ:

$$
x ^ { \star } \; = \; \underset { x \in [ 0 , 1 ] ^ { \mathcal { I } } } { \arg \operatorname* { m a x } } \; \sum _ { i \in \mathcal { I } } x ( i )   \hat { u } ( i ) \qquad \mathrm { s . t . } \qquad \sum _ { i \in \mathcal { I } } x ( i )   C ( i ) \leq \Gamma .\tag{15}
$$

This is a fractional knapsack. For a multiplier $\lambda \geq 0$ its Lagrangian $\begin{array} { r } { \sum _ { i } x ( i ) \big ( \hat { u } ( i ) - \lambda C ( i ) \big ) } \end{array}$ is maximized by the threshold rule $x ^ { \star } ( i ) = \mathbf { 1 } [ \hat { u } ( i ) \geq \lambda C ( i ) ]$ , i.e. by admitting candidates in order of value per unit cost $\hat { u } ( i ) / C ( i )$ , with 𝜆 fixed by the budget (at most one candidate is fractional at the optimum). The threshold rule is the gate of Eq. (3) with the cost-dependent threshold $\lambda C ( i )$ , which reduces to a common $\kappa _ { t } = \lambda \bar { C }$ when costs are equal. Adding an entropy regularizer $\beta ^ { - 1 } \textstyle \sum _ { i } H _ { b } ( x ( i ) )$ softens the gate into $x ^ { \star } ( i ) = \sigma \big ( \beta \big ( \hat { u } ( i ) - \lambda C ( i ) \big ) \big )$ , with $\sigma$ the logistic function, of which Eq. (3) applied to the cost-adjusted score is the normalized sampling analogue. A cardinality constraint $\textstyle \sum _ { i } x ( i ) = B$ with integral 𝑥 gives the top-𝐵 batch problem, which differs from independent thresholding only when utilities interact (redundancy among co-selected candidates), captured by the diversity term of Eq. (47).

**Gross utility, net value, and three selection rules.** With the cost-adjusted (net) score

$$
\tilde { u } \big ( i \big ) \; = \; \hat { u } \big ( i \big ) - \lambda   C \big ( i \big ) ,\tag{16}
$$

three rules must be kept apart when costs differ. (a) The budgeted batch rule of Eq. (15) admits every candidate with $\tilde { u } ( i ) \geq 0 ,$ , i.e. with value per unit cost $\hat { u } ( i ) / C ( i ) \geq \lambda .$ (b) The single-draw net-value rule selects arg $\operatorname* { m a x } _ { i } \tilde { u } ( i )$ , as in Eq. (33). (c) The policy of Eq. (3) applied to the raw score 𝑢ˆ selects arg max<sub>𝑖</sub> $\hat { u } ( i )$ in its greedy limit (Theorem 2). When costs are equal the three rank candidates identically, by 𝑢ˆ, although (a) selects a set and (b) and (c) a single candidate. When they differ, (c) can contradict (a) and (b): the candidate with the largest raw utility need not have the largest net value or value per unit cost. Applying Eq. (3) to 𝑢˜ with $\kappa _ { t } = 0$ removes the conflict: its admissible set is exactly the solution of (a), up to the one fractional item, and its greedy limit is (b). We therefore use the raw score only when costs are equal, and 𝑢˜ otherwise.

The techniques discussed here arise by choosing I, 𝑢ˆ, and C. In **agentic interaction selection** (Sections 5, 6.1 and 6.3), I is the task or prompt pool, the reasoning increments available for a query, or the tool/think/ask menu; 𝑢ˆ is adaptation risk, prompt informativeness, or expected reduction in answer uncertainty; and C is rollout cost, inference compute, or a composite of latency, money, tokens, and risk, so that Eq. (15) becomes risk-prioritized sampling, adaptive halting, or the agentic gate. In the **orchestration of System-2 executives** (Section 3), System-1 interprets a task semantically before any of it is executed and can therefore also decide who should execute it and how much they may spend: I is the pool of available System-2 models or agents together with their compute budgets, 𝑢ˆ their

11

<!-- page 12 of 45 -->

predicted contribution, and C their inference cost, so Eq. (15) is a joint routing and budget-allocation rule; token-level mixture-of-experts layers and dynamic depth are the same decision at finer granularity. Eq. (15) captures the selection structure these methods share, not the interchangeability of their learning dynamics or guarantees.

## 2.4 Scope: When Gating Is and Is Not Justified

The principle applies only where interactions are individually costly, heterogeneous in informativeness, and predictable before execution by a model cheaper than execution itself. Gating has no guaranteed advantage when candidates are homogeneous in utility, when predicting their utility costs about as much as executing them, or when execution is cheap. Dense pretraining, minibatch stochastic gradient descent over independent, identically distributed data, and static supervised learning often come close to these conditions, but need not: a fixed corpus can hold examples whose value differs and is predictable from cheap features, which data-selection methods exploit. A gain observed there calls for an accounting of where it comes from, not a refutation of the principle. Within that regime, gating is justified when a predictor exists whose cost is well below execution cost and whose error $\varepsilon _ { \phi }$ is small relative to the utility gaps it must resolve (Theorem 18).

It should be relaxed, executing uncertain or even predicted-uninformative interactions, when: (1) the predictor is poorly calibrated or off-distribution, since its blind spots are where it wrongly suppresses; (2) interactions carry option or exploratory value not captured by one-step utility; (3) prediction costs about as much as execution, voiding Theorem 14; (4) false negatives are catastrophic (safety-critical control, irreversible actions); or (5) the utility is misspecified, so that the predictor systematically suppresses the very interactions that would reveal the misspecification.

In such regimes the right object is an exploratory value of information, e.g. an upper-confidence or Thompson criterion, replacing 𝑢ˆ in Eq. (15) with $\hat { u } + c   \sigma _ { \phi }$ , which recovers pure exploitation as $c \rightarrow 0$ and uncertainty-seeking exploration, dominated by $\sigma _ { \phi } , \mathrm { a s } \; c \to \infty$ . That limit favours the most uncertain candidates rather than sampling the pool uniformly; undirected exploration needs a separate mixture with a uniform policy, as in Section 7.2; the uncertainty-aware acquisition functionals of Eqs. (65) and (62) (Section B) are the mechanism by which the principle avoids over-pruning. Conditional interaction is a statement about how to spend a trustworthy informativeness signal, not a claim that one always exists. Regimes (1) and (5) are the most insidious, since the predictor’s errors concentrate precisely where it has no data and the interactions that would reveal them are never executed; the audit sampling described after Theorem 14 is the minimal safeguard, and Section 7.2 treats this censoring formally.

## 3 Co-Designing the Two Systems: Architecture and Co-Scaling

Section 2 specified what System-1 decides and how its decisions are scored. Once the predictor exists, three questions follow: how should it be trained, how accurate must it be, and how should its capacity scale with the executor? We address them in turn.

## 3.1 The Two Systems as an Operator Decomposition

Let 𝜃 and $\phi$ be the executive and predictive parameters. Define

$$
\begin{array} { r } { \mathcal { T } _ { \mathrm { e x c c } } ( \theta , \phi ; \mathcal { H } ) = \Big ( \theta - \eta _ { \theta }   \mathbb { E } _ { i \sim \pi _ { \mathrm { C I } } ( \cdot | \mathcal { H } ; \phi ) } \big [ \nabla _ { \theta } \mathcal { L } ( \theta , i ) \big ] + \xi _ { \theta } , \quad \phi \Big ) , } \end{array}\tag{17}
$$

$$
\mathcal { T } _ { \mathrm { p r e d } } ( \theta , \phi ;   \mathcal { H } ) = \Big ( \theta , \quad \phi - \eta _ { \phi }   \nabla _ { \phi }   \mathbb { E } _ { i \in \mathcal { H } } \big [ d ( \hat { u } _ { \phi } ( i ; \mathcal { H } ) ,   u ( i ; \theta ) ) \big ] + \xi _ { \phi } \Big ) ,\tag{18}
$$

where 𝑑 is a strictly proper scoring rule (e.g. squared error or log loss) and $\xi _ { \theta } , \xi _ { \phi }$ are stochastic-gradient noise. $\mathcal { T } _ { \mathrm { e x e c } }$ improves the policy along gradients sampled through the gate; $\mathcal { T } _ { \mathrm { p r e d } }$ fits the predictor to realized utilities. Each operator maps the pair $( \theta , \phi )$ to a pair and updates one component, so their composition is well defined: System-1 is refitted first, and System-2 then steps along gradients sampled through the refitted gate.

12

<!-- page 13 of 45 -->

**Definition 11** (Conditional-Interaction Operator Decomposition). A conditional-interaction architecture is any learning system whose update can be written as

$$
\begin{array} { r } { \big ( \theta _ { t + 1 } , \phi _ { t + 1 } \big ) = \big ( \mathcal { T } _ { \tt e x e c } \circ \mathcal { T } _ { \tt p r e d } \big ) \big ( \theta _ { t } , \phi _ { t } ; \mathcal { H } _ { 0 : t } \big ) , } \end{array}\tag{19}
$$

with $\pi _ { \mathrm { C I } }$ satisfying Theorem 1 for $\hat { u } = \hat { u } _ { \phi }$

This decomposition is the load-bearing object of co-design: neither operator is defined without the other. It is not a specific algorithm but a class of algorithms, indexed by latent space, predictor family, and scoring rule, of which the predictive-sampling systems cited in Section 5.1 are special cases. The two operators run on different clocks: System-1 (𝜙) adapts quickly to realized utilities, while System-2 (𝜃) improves slowly along the gated gradients. Because better predictions change which interactions are executed, and hence the supervision System-1 receives, one may worry that the coupled loop diverges; two-timescale stochastic approximation (Borkar and Borkar, 2008) gives conditions under which such coupled updates converge, and MPTS trains the two systems jointly in this way in practice (Wang et al., 2026), but a convergence guarantee for the general architecture remains open.

**Co-evolution: how the two systems learn together.** Co-evolution concerns how the two systems learn together, $\left( \theta _ { t } , \phi _ { t } \right) \rightarrow \left( \theta _ { t + 1 } , \phi _ { t + 1 } \right)$ , as when an agent’s execution harness and the policy running inside it are evolved jointly (Chen et al., 2026), and is dynamical; co-scaling (Section 3.2) concerns how System-1’s optimal capacity responds to System-2’s, and is a comparative-statics problem. As a heuristic reference point for the former, one may expect the predictor’s error to decrease with the amount of executed history and the quality of supervision,

$$
\varepsilon _ { \phi } ^ { ( t ) } ~ \lesssim ~ \frac { c ( \theta _ { t } ) } { \sqrt { | \mathcal { H } _ { 0 : t } | } } ,\tag{20}
$$

where $c ( \theta _ { t } )$ would decrease as a stronger policy yields cleaner outcomes, and a better predictor in turn makes selection sharper. We write the familiar $O ( | \mathcal { H } _ { 0 : t } | ^ { - 1 / 2 } )$ form, borrowed from independent, identically distributed sampling, only as a benchmark: under adaptive, censored collection the rate need not hold, because the predictor’s training data are gathered by a changing policy and censored by the gate itself (Section 7.2), and a stronger System-2 can make utilities harder rather than easier to predict. Where the loop does work, it resembles self-play along the informativeness axis rather than adversarial reward, and the separation of timescales is what keeps it stable.

**Ranking by the posterior mean: using a predictor of given quality.** Let 𝑈 be the unknown utility of a candidate and $\hat { U }$ the predictor’s output. Intuitively, the predictor should compress history into just enough information about 𝑈: too expressive a predictor wastes parameters and erodes the amortization benefit, while too coarse a predictor cannot discriminate informative from uninformative interactions. An information-bottleneck (IB) objective trades complexity against informativeness,

![Image block](doc:a7b5b7e/tier:basic/page:13/block:9)

(21)

Whatever representation $\hat { U }$ is learned, the question of how to use it has a clean answer.

**Proposition 12** (Rank by the Posterior Mean). Assume equal costs, a budget of 𝐵 executions, and that each candidate’s utility depends on the predictor outputs only through its own, $\mathbb { E } [ U ( i ) \mid { \hat { U } } ] = \mathbb { E } [ U ( i ) \mid { \hat { U } } ( i ) ]$ . Then, among all selection rules that see only the predictor outputs $\hat { U } ,$ executing the 𝐵 candidates with the largest posterior mean $\mathbb { E } [ U \mid { \hat { U } } ]$ maximizes expected executed utility and hence the IIE. Setting $K _ { t }$ to the 𝐵-th largest posterior mean makes the admissible set of Eq. (3) exactly that top-𝐵 set, up to ties at the 𝐵-th value.

The proposition is modest: it says nothing about which $\hat { U }$ to learn, which is what Eq. (21) addresses. Its practical content is that the calibrated mean of the predictor, not an uncalibrated score, is the right ranking signal.

13

<!-- page 14 of 45 -->

## 3.2 Co-Scaling: System-1’s Size as System-2 Grows

Current practice sizes System-2 and treats any selector as an afterthought. Eq. (13) suggests this is incomplete: a larger System-1 lowers its error rates but raises $c _ { \mathrm { p r e d } }$ , and where the balance lies depends on how expensive System-2 is. Let System-1 have a size 𝑠 from a totally ordered set $S _ { 1 }$ (parameters, depth, or context), with error rates fp(𝑠), fn(𝑠) and scoring cost $c _ { \mathrm { p r e d } } ( s )$ , and let scaling System-2 multiply every execution cost by a factor $k \geq 1$ , so that $\bar { C } _ { \mathrm { r e l } } \mapsto k \bar { C } _ { \mathrm { r e l } }$ and $\bar { C } _ { \mathrm { d e a d } } \stackrel { \cdot } { \mapsto } k \bar { C } _ { \mathrm { d e a d } }$

**Proposition 13** (Co-Scaling Under Fixed Error Curves). Assume the setting of Theorem 7(ii); that $\rho _ { u } ,$ fp(𝑠) and $\mathrm { f n } ( s ) < 1$ do not depend on 𝑘; that $c _ { p r e d } ( s ) / ( 1 - \mathrm { f n } ( s ) )$ is non-decreasing in 𝑠; and that the IIE ratio $E q . ~ ( I 3 )$ attains its maximum over $S _ { 1 }$ for every $k ,$ on a set $S ^ { \star } ( k )$ of optimal System-1 sizes. Then $( i ) \; S ^ { \star }$ is non-decreasing in 𝑘 in the strong set order: $i f k ^ { \prime } > k ,   s \in S ^ { \star } ( k )$ and $s ^ { \prime } \in S ^ { \star } ( k ^ { \prime } )$ , then max $( s , s ^ { \prime } ) \in S ^ { \star } ( k ^ { \prime } )$ and $\operatorname* { m i n } ( s , s ^ { \prime } ) \in S ^ { \star } ( k )$ , so the largest and the smallest optimal sizes, where they exist, are non-decreasing in $k ;$ and (ii) the maximal IIE ratio is non-decreasing in 𝑘.

The mechanism the proof isolates is that scaling System-2 dilutes the predictor’s own cost, $c _ { \mathrm { p r e d } } / k$ relative to execution, but not the cost of its mistakes, which grows with System-2 itself, so accuracy becomes relatively cheaper to buy. Under these assumptions, a System-1 that is optimal for one System-2 remains optimal or becomes too small, never too large, as System-2’s cost scale grows; the proposition does not say how large the resulting loss is. The result is comparative statics under fixed error curves: it does not show that growing System-1 helps an evolving executive, nor at what rate System-1 should grow, and both remain experimental hypotheses. Its assumptions do most of the work, and it does not imply the same monotonicity when scaling System-2 changes the prediction problem itself, as happens whenever a stronger executive changes the distribution of utilities, the separation between useful and dead interactions, or the cost structure. The statement is also qualitative: it fixes the direction of co-scaling, not its rate, since fp(𝑠), fn(𝑠) and $c _ { \mathrm { p r e d } } ( s )$ are empirical curves. Measuring them, and hence a compute-optimal ratio between the two systems in the spirit of the parameter-to-data ratio of Hoffmann et al. (2022), is the co-design counterpart of a scaling law (Section 7).

## 3.3 Example of Architectural Implementation

The decomposition is realized by four domain-agnostic components, summarized by four verbs (See Fig. 6). Represent: candidates $\{ i _ { k } \} _ { k = 1 } ^ { n }$ are embedded by $e : \mathcal { I } \to \mathbb { R } ^ { d _ { e } } ,   \mathrm { e . g }$ ., frozen language-model encoders in the digital domain; graph or transformer encoders over the structure of the task’s Markov decision process in the physical one. Predict (System-1): a lightweight $f _ { \phi } : \mathbb { R } ^ { d _ { e } } \times \mathcal { H } ^ { * } \to \mathbb { R } \times \mathbb { R } _ { + }$ outputs $( \hat { \mu } _ { k } , \hat { \sigma } _ { k } )$ , trained online on executed outcomes, with $d _ { \phi } \ll d _ { \theta }$ keeping $c _ { \mathrm { p r e d } }$ ≪ $c _ { \mathrm { e x e c } } ;$ System-1 may itself be a small language model when semantic understanding is required, but its defining property is computational role rather than model family. Gate: the selection policy <sup>𝜋</sup>CI produces an execution mask $\mathbf { z } _ { t } \in \{ 0 , 1 \} ^ { n }$ , and when the number executed is set by the threshold rather than fixed, $K _ { t }$ tracks a target admission rate $\rho ^ { * }$ by stochastic approximation,

$$
\kappa _ { t + 1 } = \kappa _ { t } + \alpha _ { \kappa } \left( \frac { n _ { t } ^ { \mathrm { a d m } } } { n } - \rho ^ { * } \right) , \qquad n _ { t } ^ { \mathrm { a d m } } = \big | \{ i : \hat { u } _ { \phi } ( i ) \geq \kappa _ { t } \} \big | ,\tag{22}
$$

which treats the threshold as a dual variable enforcing that rate; when costs differ, $\tilde { u } _ { \phi }$ of Eq. (16) replaces $\hat { u } _ { \phi }$ in $n _ { t } ^ { \mathrm { a d m } }$ and the equal-cost form is shown. Learn: executed interactions are verified, recorded in $\mathcal { H } _ { t }$ , and used for one step of $\mathcal { T } _ { \mathrm { p r e d } }$ , with a decay factor on old history trading stability against tracking non-stationarity. These four components are the operational form of Theorem 11, and the ranking rule of Theorem 12 tells the gate how to use the predictor’s output.

**Three interfaces of co-design.** Beyond the gate, co-design fixes three interfaces. The knowledge interface determines what is distilled from System-2 into System-1: where System-2 repeatedly solves a class of candidates, the solution itself, not only its value, can be amortized into System-1, as expert iteration does (Anthony et al., 2017), so that System-1 answers routine cases outright and escalates the rest. The representation interface determines what System-1

14

<!-- page 15 of 45 -->

![Image block](doc:a7b5b7e/tier:basic/page:15/block:1)

Figure 6 Architectural realization: represent, predict, gate, learn. Every candidate is embedded 1 and scored cheaply by System 1 2 . The gate 3 admits only candidates whose score clears the threshold $K _ { t }   ,$ which a controller adjusts to hold a target admission rate $\rho ^ { * } \left( \mathrm { E q . } \left( 2 2 \right) \right)$ . Admitted candidates are executed by System 2; suppressed ones cost only $c _ { \mathrm { p r e d } } ,$ and a small audit sample of them is executed anyway so that the predictor’s misses can be detected. Executed outcomes are logged and refit System 1 4 , while System 2 learns from the same outcomes. Dotted arcs mark the two other interfaces of co-design: what is distilled from System 2 into System 1, and which of System 2’s representations System 1 may access.

observes: using System-2’s own features buys accuracy at the price of sharing its blind spots, whereas encoding the candidate independently keeps the two error profiles uncorrelated, and conditioning System-1 on context lets it adapt to a new task, scenario or user without retraining System-2. The feedback interface determines which System-2 outcomes supervise System-1 — realized rewards, gradients, verifier verdicts, or an audit sample of suppressed candidates — and thereby fixes which of its own errors System-1 can ever observe (Section 7.2). Co-design is therefore not merely a size-matching problem. The same position also makes System-1 a natural place for oversight: because it sees every candidate before System-2 acts, cost, irreversibility and safety constraints (Eq. (53)) can be imposed before a capable executive commits.

**Gates within gates: hierarchical extensions.** The same gate can operate at several temporal and representational scales at once. For a hierarchy of levels $\ell \in \{ 1 , \ldots , L _ { h } \}$ (motor, skill, plan), each with its own predictor and threshold,

$$
\pi _ { \mathrm { C I } } ^ { ( \ell ) }   \left( i _ { t } ^ { ( \ell ) } \mid \mathcal { H } _ { 0 : t } ^ { ( \ell ) } \right) \; \propto \; \exp   \left( \beta _ { \ell }   \hat { u } _ { \phi } ^ { ( \ell ) } \right) \mathbf { 1 }   \left[ \hat { u } _ { \phi } ^ { ( \ell ) } \geq \kappa _ { t } ^ { ( \ell ) } \right] ,\tag{23}
$$

with each level gating access to the one below; each $\pi _ { \mathrm { C I } } ^ { ( \ell ) }$ is normalized over its own admissible set and abstains when that set is empty, as in Theorem 1. Analysing the joint dynamics of such a multi-level hierarchy is left open.

## 4 What Prediction Buys, and Why a Small System-1 Can Work

Sections 2 and 3 defined the gate and the architecture built around it. Co-design pays only if System-1 is both much cheaper than System-2 and accurate enough to steer it; this section asks when both hold. Section 4.1 states the core formal results: what prediction buys, in time and memory, and the regret a cheap predictor risks. The remaining subsections ask why a cheap System-1 can make a useful prediction about an expensive interaction at all, from three complementary views. These connections are interpretive rather than separate contributions; each supplies insight and, under its own assumptions, a guarantee, but none is a derivation of the framework. Information-theoretically (Section 4.2), the quantity gated on is expected information gain net of cost, with epistemic action selection as its limiting case. Generatively (Section 4.3), an interaction is a transition of a world or policy model, and System-1 is an amortized projection of that model onto a single scalar, which is why valuation is cheaper than simulation. Control-theoretically (Section 4.4), executing only what the gate admits is receding-horizon control of a learner whose own dynamics are still being learned.

15

<!-- page 16 of 45 -->

## 4.1 Core Results: What Prediction Buys and What It Risks

Predictive sampling has two structural advantages over any scheme that must execute interactions to evaluate them, both following from the fact that $\hat { u } _ { \phi }$ scores a candidate from a cheap embedding without performing it: one in time and one, conditionally, in memory. Against them stands one risk, that a cheap predictor selects the wrong interactions, which the end of this subsection bounds.

**Screening efficiency.** A reactive filter must execute an interaction before discovering that it was uninformative, paying full cost on the dead fraction. Predictive sampling replaces this with a forward pass of a small model, as in recent predictive prompt and task samplers (Qu et al., 2026a; Mao et al., 2026; Wen et al., 2026a).

**Proposition 14** (Screening Gain Relative to Exhaustive Execute-then-Filter Evaluation). Let a pool of 𝑛 candidates contain a fraction $\rho _ { u }$ of utility-relevant ones, with execution cost $c _ { e x e c } ,$ screening cost $c _ { s c r e e n } = c _ { e m b e d }   +   c _ { s c o r e }$ (encoding the candidate and scoring it; this is the $c _ { p r e d }$ of Eq. (9)), and $r = c _ { s c r e e n } / c _ { e x e c }$ . With a perfect screen, acquiring the utility-relevant subset costs

$$
\frac { C _ { \mathrm { C I } } } { C _ { r e a c t } } = \rho _ { u } + r\tag{24}
$$

of what exhaustive execute-then-filter evaluation spends $( C _ { r e a c t } = n   c _ { e x e c } ,$ , the baseline that must execute every candidate to find all relevant ones), a speedup approaching $1 / \rho _ { u } \; a s \; r \rightarrow 0 ,$ , provided a candidate’s utility class cannot be known without executing or scoring it.

Under the shorthand of Section 4.3 for a dense transformer the costs are concrete: evaluating a pool of 𝑛 candidates by rollout to obtain 𝐵 utility-relevant ones, versus screening them predictively, costs

$$
C _ { \mathrm { u n c } } = \Theta \big ( n   G   L   d _ { \mathrm { m o d e l } } ^ { 2 } \big ) ,\tag{25}
$$

$$
C _ { \mathrm { C I } } = \underbrace { \Theta ( n   c _ { \mathrm { e m b e d } } ) } _ { \mathrm { e n c o d i n g } } + \underbrace { \Theta ( n   d _ { \phi } ^ { 2 } ) } _ { \mathrm { p r e d i c t i o n } } + \underbrace { \Theta ( B   G   L   d _ { \mathrm { m o d e l } } ^ { 2 } ) } _ { \mathrm { e x e c u t i o n } } ,\tag{26}
$$

with $d _ { \phi } \ll d _ { \mathrm { m o d e l } }$ , a cost ratio (now including encoding) of $B / n + \Theta \big ( ( c _ { \mathrm { e m b e d } } + d _ { \phi } ^ { 2 } ) / ( G L   d _ { \mathrm { m o d e l } } ^ { 2 } ) \big )$ : Eq. (24) with $\rho _ { u } = B / n$ and $r = \Theta \big ( ( c _ { \mathrm { e m b e d } } + d _ { \phi } ^ { 2 } ) / ( G L   d _ { \mathrm { m o d e l } } ^ { 2 } ) \big )$ . Sparse or fused kernels change the constants, not the argument. The encoding term need not be small. A frozen language-model encoder applied to a prompt of length 𝐿<sub>𝑥</sub> costs $c _ { \mathrm { e m b e d } } = \Theta ( L _ { x } d _ { \mathrm { e n c } } ^ { 2 } + L _ { x } ^ { 2 } d _ { \mathrm { e n c } } )$ , negligible next to 𝐺 rollouts of length 𝐿 when $L _ { x } d _ { \mathrm { e n c } } ^ { 2 } \ll G L d _ { \mathrm { m o d e l } } ^ { 2 }$ and paid once per candidate when embeddings are cached across the rounds of a static pool; for short rollouts, or when candidates must be re-encoded every round, it can dominate 𝑟. End to end, executing an audit fraction 𝑎 of suppressed candidates adds $a ( 1 - \rho _ { u } )$ to Eq. (24), refitting the predictor adds $c _ { \mathrm { t r a i n } } / ( n   c _ { \mathrm { e x e c } } )$ per round, and screening errors move the ratio as in Eq. (13); Eq. (24) is the perfect-screen ceiling, not a realized speedup. The error rates separating the realized multiplier from this ceiling are measurable: periodically executing a small audit sample of suppressed candidates estimates fn, and the executed survivors estimate fp; reporting these, and their cost-conditioned versions (Theorem 8), turns Eq. (13) into a measured quantity.

**Memory scaling.** When evaluation requires real interaction, the candidate pool may also be capped by the memory for in-flight rollouts (KV cache for 𝐺 length-𝐿 generations, replay buffers, simulator state). A predictively scored candidate contributes only an embedding $e ( \tau ) \in \mathbb { R } ^ { d _ { e } }$ , and embeddings can be streamed.

**Proposition 15** (Peak Rollout Memory Under Synchronous Evaluation). To select the top-𝐵 of 𝑛 candidates, synchronous reactive evaluation needs peak memory $M _ { r e a c t } ( n )$ , whereas predictive scoring streamed in micro-batches of size 𝑏 needs $M _ { \mathrm { C I } } ( n )$ , where

$$
M _ { r e a c t } ( n ) = \Theta \big ( n   G   L   d _ { m o d e l } \big ) ,\tag{27}
$$

$$
M _ { \mathrm { C I } } ( n ) = \Theta \big ( B   G   L   d _ { m o d e l } + b   d _ { e } + B \big ) ,\tag{28}
$$

independent of 𝑛; the pool is then limited only by screening time $\Theta ( n   c _ { p r e d } )$ and score storage.

16

<!-- page 17 of 45 -->

**Corollary 16** (Larger Pools Without Rollout Memory). The best true utility $U _ { ( 1 ) } ^ { n }$ among 𝑛 i.i.d. candidates is non-decreasing in 𝑛: it converges to the essential supremum if utility is bounded, and grows as $n ^ { \xi }$ (in probability, up to a slowly varying factor) $\mathit { i f }   \mathrm { P r } ( U > u ) = u ^ { - 1 / \xi } \ell ( u )$ . Under predictive sampling this gain costs no additional rollout memory (Eq. (28)), provided the predictor ranks the top of the pool correctly.

The memory advantage is conditional on the synchronous premise: a reactive filter that streams executions and keeps only scores also has 𝑛-independent memory, but pays $\Theta ( n   c _ { \mathrm { e x e c } } )$ time, so the advantage then reduces to the screening gain. More generally, System-1 allocates storage as well as compute — which sample groups stay in a replay buffer, which examples enter a capacity-limited batch (Zou et al., 2025), which demonstrations occupy a context window (Wen et al., 2026b), how much KV cache a request receives — and the rule of Eq. (15) applies unchanged, with memory in place of compute as the cost.

**Selection regret.** These savings help only if the cheap predictor selects nearly the right interactions.

**Definition 17** (Conditional Regret). With $\mathcal { B } _ { t } ^ { * }$ the oracle’s top-𝐵 set by true utility $u _ { t } = u \bigl ( \cdot ; \theta _ { t } \bigr )$ in the round-𝑡 pool $\mathcal { I } _ { t }$ and $\hat { \mathcal { B } } _ { t }$ the set selected by $\pi _ { \mathrm { C I } }$ at round 𝑡,

$$
\mathcal { R } _ { T } ( \pi _ { \mathrm { C I } } , \pi ^ { * } ) = \sum _ { t = 0 } ^ { T - 1 } \Bigl [ \sum _ { i \in \mathcal { B } _ { t } ^ { * } } u _ { t } ( i ) - \sum _ { i \in \hat { \mathcal { B } } _ { t } } u _ { t } ( i ) \Bigr ]   .\tag{29}
$$

**Theorem 18** (Regret Bound). Assume (i) selection is the deterministic limit of the fixed-size batch rule Eq. (47) with $\mu = 0$ and A the predicted utility 𝑢ˆ, the cardinality-constrained form of the master objective Eq. (15), so each round retains the top-𝐵 set by 𝑢ˆ; (ii) $\varepsilon _ { \phi } = \operatorname { \operatorname* { s u p } } _ { 0 \leq t < T } \operatorname * { s u p } _ { i \in \mathcal { I } _ { t } } \left| \hat { u } _ { t } \big ( i \big ) - u _ { t } \big ( i \big ) \right|$ , a time-uniform error, with $\hat { u } _ { t } \big ( i \big ) = \hat { u } _ { \phi _ { t } } \big ( i \mid \mathcal { H } _ { 0 : t } \big ) ;$ and $\begin{array} { r } { ( i i i ) \; \mathrm { g a p } _ { \operatorname* { m i n } } = \operatorname* { m i n } _ { t } \operatorname* { m i n } _ { i \in \mathcal { B } _ { t } ^ { * } , \; j \in \mathcal { I } _ { t } \backslash \mathcal { B } _ { t } ^ { * } } \bigl ( u _ { t } ( i ) - u _ { t } ( j ) \bigr ) > 0 . } \end{array}$ . Then

$$
\mathcal { R } _ { T } ( \pi _ { \mathrm { C I } } , \pi ^ { * } ) \; \leq \; 2   T   B   \varepsilon _ { \phi }   \cdot   { \bf 1 } \big [ 2 \varepsilon _ { \phi } \geq \mathrm { g a p } _ { \mathrm { m i n } } \big ] \; ;\tag{30}
$$

in particular, the selection is exactly the oracle’s whenever $2 \varepsilon _ { \phi } < \mathrm { g a p } _ { \mathrm { m i n } } .$ More generally, with round-specific errors $\varepsilon _ { t } = \operatorname { s u p } _ { i \in \mathcal { I } _ { t } } \left| \hat { u } _ { t } \big ( i \big ) - u _ { t } \big ( i \big ) \right|$ and gaps gap<sub>𝑡</sub>, $\begin{array} { r } { \mathcal { R } _ { T } \leq \sum _ { t < T } 2 B \varepsilon _ { t }   \mathbf { 1 } \big [ 2 \varepsilon _ { t } \geq \mathrm { g a p } _ { t } \big ] } \end{array}$

**Corollary 19** (High-Probability Regret). Suppose each round’s pool has at most 𝑛 candidates; for every round 𝑡 and every candidate outside a blind-spot set $O _ { t } \subseteq \mathcal { I } _ { t }$ with $| O _ { t } | \leq n _ { \mathrm { o o d } }$ , the error $\hat { u } _ { t } ( i ) - u _ { t } ( i )$ is zero-mean and 𝜎 -sub-Gaussian, while errors on $O _ { t }$ are unrestricted; and the utility takes values in an interval of length $u _ { \mathrm { m a x } }$ (e.g. $u \in [ 0 , u _ { \operatorname* { m a x } } ] )$ . Then, with probability at least $1 - \delta$ and $\varepsilon _ { \phi } ^ { \mathrm { e f f } } = \sigma _ { \phi } \sqrt { 2 \ln ( 2 n T / \delta ) }$

$$
\mathcal { R } _ { T } ( \pi _ { \mathrm { C I } } , \pi ^ { * } ) \leq 2   T   B   \varepsilon _ { \phi } ^ { \mathrm { e f f } }   { \bf 1 } \left[ 2 \varepsilon _ { \phi } ^ { \mathrm { e f f } } \geq \mathrm { g a p } _ { \mathrm { m i n } } \right] + T   \operatorname* { m i n } ( B , n _ { \mathrm { o o d } } )   u _ { \mathrm { m a x } } .\tag{31}
$$

Interpretation. The batch rule of assumption (i) is the cardinality-constrained counterpart of the single-draw greedy limit (Theorem 2), not a consequence of it. For a fixed positive selection margin, regret is exactly zero once the predictor is accurate relative to the separation of interactions $( 2 \varepsilon _ { \phi } < \mathrm { g a p } _ { \mathrm { m i n } } ) ;$ regret is measured in the units of the declared utility, and $\mathtt { g a p } _ { \operatorname* { m i n } }$ enters not as a denominator but as the threshold deciding whether any misranking can occur, with the bound degrading gracefully, linearly in $\varepsilon _ { \phi }$ and 𝐵, once $2 \varepsilon _ { \phi } \geq \operatorname { g a p } _ { \operatorname* { m i n } } .$ . In continuous interaction spaces the global gap may vanish, and $\mathrm { g a p _ { m i n } }$ should then be interpreted as the margin at the $B { \mathrm { - t h } }$ order statistic. The uniform error and strictly positive gap of Theorem 18 are thus the clean idealization that exposes the mechanism; Theorem 19 is the version intended for realistic predictors, and its second term makes explicit that blind spots, not average error, dominate once the predictor is otherwise accurate. The theorem does not imply a self-improving loop: better predictions reduce regret and more informative interactions improve the policy, but a stronger System-2 can make its own utilities harder to predict (Sections 3 and 7.2).

17

<!-- page 18 of 45 -->

## 4.2 Understanding Through the Lens of Active Inference

**Information net of cost: the utility of an interaction.** Conditional interaction is resource allocation under uncertainty: at each decision point the system chooses whether to spend compute, environment, or inference resources on an interaction whose value is not yet known. Let 𝜃 be the latent state of knowledge (model parameters, latent task representations, or beliefs). Executing $i \in { \cal I }$ given history $\mathcal { H } _ { t }$ yields an outcome 𝑜 and posterior $p ( \theta \mid \mathcal { H } _ { t } , i , o )$ . The expected information gain is

$$
\mathcal { U } _ { \mathrm { I G } } ( i \mid \mathcal { H } _ { t } ) = \mathbb { E } _ { o \sim p ( o \mid i , \mathcal { H } _ { t } ) } \Big [ D _ { \mathrm { K L } } \Big [ p ( \theta \mid \mathcal { H } _ { t } , i , o ) \left\|   p ( \theta \mid \mathcal { H } _ { t } ) \right] \Big ] = I ( \theta ; o \mid i , \mathcal { H } _ { t } ) ,\tag{32}
$$

the utility of Theorem 3. Bayesian surprise is the realized, per-outcome divergence measured after execution; expected information gain is its expectation under the predictive distribution, which must be estimated before execution and is what the gate acts on. When outcomes are drawn from the same predictive model, realized surprises on executed interactions are therefore unbiased targets for the pre-execution estimate, which licenses the bootstrapping loop of Section 3. Net of cost,

$$
\mathcal { U } ( i \mid \mathcal { H } _ { t } ) = \mathcal { U } _ { \mathrm { I G } } ( i \mid \mathcal { H } _ { t } ) - \lambda   C ( i ) , \quad i _ { t } ^ { * } = \underset { i \in \mathcal { I } } { \arg \max } \mathcal { U } ( i \mid \mathcal { H } _ { t } ) ,\tag{33}
$$

with $\lambda > 0$ pricing information against resources; this is the $\beta \to \infty$ limit of Eq. (3) applied to the cost-adjusted score of Eq. (16), rule (b) of Section 2.3, and agrees with the raw-score limit only when costs are equal.

A scalar price treats cost as one-dimensional and fungible, but in deployment latency, money, context occupancy, and safety or irreversibility are not interchangeable and cannot be collapsed into one currency without losing the constraints that matter. The framework accommodates this directly: 𝜆 becomes a vector of prices, $K _ { t }$ a vector-valued threshold, and the gate a constrained rather than scalar comparison, where a hard deadline or a billing cap weights its component differently from a safety budget. The agentic ledger of Eq. (53) carries exactly these terms; the scalar form is the special case where they share a currency.

**Amortized variational estimation of utility.** The exact posterior is unavailable, so a lightweight predictive subsystem (System-1; in the task-sampling literature, a risk prediction model (Wang et al., 2026; Wen et al., 2026a), or a Bayesian filter over a prompt’s solving dynamics (Mao et al., 2026)) maintains a variational approximation $q _ { \phi } ( \theta \mid \mathcal { H } _ { t } )$ fitted by minimizing the variational free energy

$$
\mathcal { F } ( \phi ) = D _ { \mathrm { K L } } \Big [ q _ { \phi } ( \theta \mid \mathcal { H } _ { t } ) \Big \|   p ( \theta ) \Big ] - \mathbb { E } _ { q _ { \phi } } \Big [ \mathrm { l n }   p ( \mathcal { H } _ { t } \mid \theta ) \Big ] = - \mathrm { E L B O } ( \phi ) \geq - \ln p ( \mathcal { H } _ { t } ) .\tag{34}
$$

Minimizing $\mathcal { F }$ improves the approximate posterior from which utilities are computed, $\hat { u } _ { \phi } \approx \mathcal { U } _ { \mathrm { I G } } ;$ it does not by itself guarantee better utility estimates or better rankings, since a closer posterior approximation need not order candidates more accurately (compare the remark after Theorem 12). The approximation is substantial: a full Bayesian posterior over a modern network is intractable, so $\hat { u } _ { \phi }$ is at best an amortized estimate of Eq. (7), and the gap may be large – that gap is the predictor bottleneck on which every gain of the co-design rests, and Theorem 18 quantifies how it propagates to decisions. In practice the predictor learns the scalar functional directly (Section 4.3) rather than the posterior it summarizes. Unlike a literal free-energy agent, Eq. (34) is not the selection objective (that is Eq. (33)); it only keeps the predictor calibrated.

**Epistemic action selection: the active-inference limit.** In active inference an agent minimizes expected free energy (EFE) (Friston, 2010), which has a pragmatic and an epistemic term. Under uniform outcome preferences and with cost handled separately, the expected-free-energy criterion reduces to epistemic action selection based on expected information gain. Conditional interaction adds an explicit pre-execution cost gate and a learned predictor of that quantity.

**Proposition 20** (Active-Inference Limit). Write the expected free energy of interaction 𝑖 as

$$
\begin{array} { r } { G ( i ) = \underbrace { - \: \mathbb { E } _ { q _ { \phi } ( o \mid i ) } \big [ \mathrm { l n } \: p ( o \mid C ) \big ] } _ { p r a g m a i c \: v a l u e } - \underbrace { \mathbb { E } _ { q _ { \phi } ( o \mid i ) } \big [ D _ { \mathrm { K L } } \left[ q _ { \phi } ( \theta \mid o , i ) \: \middle \| \: q _ { \phi } ( \theta \mid i ) \right] \big ] } _ { e p i s t e m i c \: v a l u e } , } \end{array}\tag{35}
$$

18

<!-- page 19 of 45 -->

with preferences $p ( o \mid C )$ over a finite outcome space $O .$ If preferences are uniform and cost is accounted separately through C (𝑖), then

$$
\mathop { \arg \operatorname* { m i n } } _ { i \in \mathcal { I } } G ( i ) = \mathop { \arg \operatorname* { m a x } } _ { i \in \mathcal { I } } I _ { q _ { \phi } } ( \theta ; o \mid i ) = \mathop { \arg \operatorname* { m a x } } _ { i \in \mathcal { I } } \mathcal { U } _ { \mathrm { I G } } ^ { q _ { \phi } } ( i \mid \mathcal { H } _ { t } ) ,\tag{36}
$$

where $\mathcal { U } _ { \mathrm { I G } } ^ { q _ { \phi } }$ is $E q .$ (32) evaluated under the variational model $q _ { \phi } ,$ , and coincides with $\mathcal { U } _ { \mathrm { I G } }$ when $q _ { \phi }$ is the exact posterior and predictive. Under non-uniform preferences the two criteria diverge.

The identity is standard active-inference theory and we claim no novelty for it. Expected-free-energy decompositions differ across that literature; we adopt the pragmatic-plus-epistemic form of Eq. (35) and interpret the result as a correspondence, not a foundational derivation: it locates conditional interaction as the cost-gated, predictively amortized special case of epistemic action selection, applicable whenever information gain can be estimated more cheaply than the interaction can be executed.

## 4.3 Understanding Through the Lens of Decision-Making

Every interaction is coupled to the world through a generative transition kernel

$$
o \sim p _ { \psi } ( o \mid i ,   s _ { t } )   , \qquad s _ { t + 1 } = T ( s _ { t } ,   i ,   o )   ,\tag{37}
$$

where $s _ { t }$ is the state the interaction acts on, 𝑜 its outcome, $p _ { \psi }$ the observation/transition model (environment dynamics in the physical regime; the policy’s own generation in the digital regime), and 𝑇 the update folding the outcome into the next state. The uncertain parameter 𝜃 and the state $s _ { t }$ may coincide (the policy is both uncertain and evolving) or differ (a robot’s configuration versus its skill parameters).

In the digital regime it is essential to distinguish the interaction, the costly generation of a sequence $y = ( y ^ { 1 } , \ldots , y ^ { L } )$ from context $x ,$

$$
\pi _ { \theta } \big ( y \mid x \big ) = \prod _ { \ell = 1 } ^ { L } \pi _ { \theta } \Big ( y ^ { \ell }   \big |   x ,   y ^ { < \ell } \Big )   ,\tag{38}
$$

which costs 𝐿 sequential forward passes (with key–value caching, roughly linear in 𝐿, with a per-token cost set by the model’s depth, width and attention and feed-forward operations; we write $\Theta ( L   d _ { \mathrm { m o d e l } } ^ { 2 } )$ as shorthand, absorbing depth and context-length terms; this is the width-dominated regime $L \lesssim d _ { \mathrm { m o d e l } }$ , and with full attention each token also attends over the accumulated context, adding $\Theta ( L ^ { 2 } d _ { \mathrm { m o d e l } } )$ per sequence, which key–value caching does not remove), from the outcome, the verdict of a verifier $\mathcal { V }$ applied to 𝑦 (does the CoT solve the problem, do the unit tests pass):

$$
o = { \mathcal { V } } ( y ) , \qquad p _ { \psi } ( o \mid i , s _ { t } ) = \sum _ { y : { \mathcal { V } } ( y ) = o } \pi _ { \theta } ( y \mid x ) .\tag{39}
$$

The kernel is the generation distribution pushed through the verifier: the verdict $o$ (often a scalar reward, correct/incorrect, or a score) is what updates the agent’s beliefs, and the tokens 𝑦 are merely the costly medium through which it is obtained. The cost is paid on all 𝐿 tokens, whereas the information resides in $o ,$ which may be a single bit.

A prompt on which every rollout returns the same verdict yields zero GRPO advantage and hence no group-relative policy-gradient signal despite its full cost. This is a statement about the update, not about belief change: an all-fail group can still inform a posterior over the prompt’s difficulty, and the gradient depends on the sequences through their log-likelihoods, not only on the verdict. The verdict channel models where the cost and the information sit, while the RLVR gate scores the declared proxy $u _ { \mathrm { a d v } }$ of Theorem 9 rather than information gain. The predictor’s task is to forecast the verdict’s declared utility before generating the sequence, which is why $c _ { \mathrm { p r e d } } \ll c _ { \mathrm { e x e c } } ;   \mathrm { F i g . } ~ 7$ shows the pipeline and the point at which System-1 cuts in.

The physical regime is identical with the verifier replaced by the environment’s reward and the sequence by a trajectory with probability $\textstyle \prod _ { k } \pi _ { \theta } { \bigl ( } a _ { k } \mid s _ { k } { \bigr ) }   p { \bigl ( } s _ { k + 1 } \mid s _ { k } , a _ { k } { \bigr ) }$ , whose factors run over physical time steps that cannot be parallelized.

19

<!-- page 20 of 45 -->

![Image block](doc:a7b5b7e/tier:basic/page:20/block:1)

Figure 7 The generative model, and where System 1 cuts in. The cost is paid in full on generating the sequence $y ,$ while the information lies in the verdict $\boldsymbol { o } = { } ^ { c } \boldsymbol { V } ( \boldsymbol { y } )$ , often a single bit. System 1 forecasts the belief change that verdict will cause before 𝑦 exists, and the gate generates only if the forecast clears $\kappa _ { t } ;$ suppressed candidates cost only $c _ { \mathrm { p r e d } } .$ Realized verdicts refit System 1. With information gain as the utility the forecast is the expected Bayesian surprise; for any other declared utility the pipeline forecasts that utility instead.

**One kernel, two interpretations: dynamics and decision.** Viewed forward in time, Eq. (37) is a dynamical-system transition. Viewed through the agent’s posterior, it defines a decision under uncertainty, since the utility is a functional of $p _ { \psi }$

$$
\begin{array} { r } { u ( i ; \mathcal { H } _ { t } ) \; = \; \mathbb { E } _ { o \sim p _ { \psi } ( o \mid i , s _ { t } ) } \Big [ D _ { \mathrm { K L } } \Big [ p ( \theta \mid \mathcal { H } _ { t } , i , o ) \Big \| p ( \theta \mid \mathcal { H } _ { t } ) \Big ] \Big ] . } \end{array}\tag{40}
$$

The gate thus chooses among transitions by the posterior change each is predicted to cause. System-1 is a cheap surrogate estimating Eq. (40) without rolling $p _ { \psi }$ forward; System-2 is the true transition, paid for only when the surrogate certifies it. Unlike a world model, which learns $p _ { \psi }$ to simulate outcomes (re-incurring much of the cost the gate avoids), the predictor learns only this scalar functional. The bottleneck is therefore one of valuation, not simulation.

**Valuation without simulation: the predictor as a decision-focused surrogate.** This interpretation clarifies what $\hat { u } _ { \phi }$ is: an amortized, decision-focused projection of the generative model, enough of $p _ { \psi }$ to rank transitions by their epistemic value, and no more. That is why the predictor can be small relative to the executive (Theorem 14), and why its failures (Sections 2.4 and 7.2) are failures to value transitions rather than to simulate them. It also motivates the control-theoretic interpretation that follows: once interactions are transitions of a system with a generative model, choosing them to optimize a long-horizon objective is a control problem, and model predictive control is its standard closed-loop solution.

## 4.4 Understanding Through the Lens of Control Theory

**Receding-horizon control of a learning plant.** Conditional interaction is structurally analogous to model predictive control (MPC) of a non-stationary plant whose state is the policy parameter. The analogy is not an equivalence (MPC selects continuous actions; conditional interaction selects subsets of interactions), but it holds component by component.

(1) Model. The plant state is $\theta _ { t } ;$ the predictor Φ forecasts how an interaction batch $\mathcal { B } _ { t }$ moves $\mathrm { i t } ,$

$$
\begin{array} { r } { \hat { \theta } _ { t + h } = \Phi _ { h } ( \theta _ { t } , \mathcal { B } _ { t } , \ldots ) , \qquad \theta _ { t + 1 } = \theta _ { t } - \eta   \nabla _ { \theta } \mathcal { L } ( \theta _ { t } , \mathcal { B } _ { t } ) + \xi _ { t } , } \end{array}\tag{41}
$$

where the right-hand equation gives the true dynamics with gradient noise $\xi _ { t }$ of covariance ∝ $| \mathcal { B } _ { t } | ^ { - 1 }$

(2) Prediction. System-1 forecasts the evaluation risk $R ( \tau _ { \mathrm { e v a l } } ; \hat { \theta } _ { t + h } ) ,   h = 1 , \ldots , H$ , without executing any interaction (Eq. (46)).

20

<!-- page 21 of 45 -->

![Image block](doc:a7b5b7e/tier:basic/page:21/block:1)

**Figure 8 The closed loop as receding-horizon control.** System 1 is refitted 1 , scores the pool 2 , and solves a budgeted selection 3 ; only then does System 2 execute, and only the first batch 4 , whose outcomes re-enter the history 5 . Steps 1 – 3 execute nothing in the environment, but refitting, scoring and solving the budgeted selection still cost compute; the loop’s cost is dominated by execution 4 only when those steps are cheap enough, which is a design requirement (Theorem 14) rather than a consequence of their executing nothing.

(3) Receding-horizon planning. The control law minimizes discounted predicted risk over admissible batches,

$$
\mathcal { B } _ { t } ^ { * } = \underset { \mathcal { B } \subseteq \mathcal { D } ,   | \mathcal { B } | = B } { \arg \min } \sum _ { h = 1 } ^ { H } \gamma ^ { h }   \mathbb { E } \big [ R ( \tau _ { \mathrm { e v a l } } ; \hat { \theta } _ { t + h } ) \big ]   ;\tag{42}
$$

𝐻 = 1 is a one-step, risk-based analogue of the greedy gate of Eq. (3). It recovers that gate exactly when a candidate’s gross utility is its expected reduction of evaluation risk, costs are equal or priced into the score as in Eq. (16), and the effect of a batch decomposes into candidate-level contributions, while 𝐻 > 1 accounts for how today’s selection shapes tomorrow’s reachable states.

(4) Execution. Only the first batch is executed and the rest re-planned, which makes the loop robust to model error in distant horizons.

(5) Update. Executed outcomes $y _ { t }$ update the model, $p ( \Phi \mid \mathcal { H } _ { 0 : t + 1 } ) \propto p ( y _ { t } \mid \mathcal { H } _ { 0 : t } , \Phi )   p ( \Phi \mid \mathcal { H } _ { 0 : t } )   ( \mathrm { E q . }   ( 4 5 ) ) .$

Unlike textbook MPC, plant and controller co-evolve: the state $\theta _ { t }$ that the controller steers is the same object whose dynamics the model Φ is still learning. Stability is therefore not classical; it rests on System-1 adapting faster than System-2 (Section 3), and the closed-loop procedure below is this MPC cycle instantiated.

**Index policies and optimal stopping.** Define the value of executing 𝑖 as

$$
V ( i , \mathcal { H } _ { 0 : t } ) = \mathbb { E } \big [ \Delta \mathcal { P } ( i ) \mid \mathcal { H } _ { 0 : t } \big ] - C ( i ) ,\tag{43}
$$

and execute 𝑖 iff $V \geq V _ { \emptyset } ,$ , the value of deferring. Here and in Eq. (52), C (𝑖) is the priced cost, in the units of the utility: $\lambda   C _ { \mathrm { r a w } } ( i )$ for a scalar ledger, or the composite price of Eq. (53); $V _ { \emptyset }$ is on the same scale. Under a discounted, independent-arm bandit approximation with a retirement option, the optimal policy is an index rule: execute while the Gittins index $\mathfrak { G } ( i , \mathcal { H } _ { 0 : t } )$ (Gittins, 1979) exceeds the retirement value $K _ { t }$

$$
\pi _ { \mathrm { C I } } ^ { * } ( i ) = \mathbf { 1 } [ \mathfrak { G } ( i , \mathcal { H } _ { 0 : t } ) \geq \kappa _ { t } ] \; .\tag{44}
$$

The Gittins guarantee holds only under those bandit assumptions. Conditional interaction is in general non-stationary (the state $\theta _ { t }$ drifts) and its interactions are dependent (they share one evolving policy), so Eq. (44) motivates the threshold form of Eq. (3) rather than certifying it. With that qualification, the principle inherits the information-theoretic interpretation of active inference (Theorem 20) and the resource-allocation interpretation of index policies as two limiting analyses, each valid under its own assumptions.

**Predict, select, execute, refit: the closed loop.** Fig. 8 shows the resulting pipeline. The MPC components are realized by four operations. Model update: the predictor’s parameters are fitted online by maximizing the conditional

21

<!-- page 22 of 45 -->

log-likelihood of the observed history (the streaming counterpart of Eq. (34)), in practice by a few stochastic gradient steps per round:

$$
\operatorname* { m a x } _ { \phi } \; \mathcal { L } _ { \mathrm { M L } } ( \phi ) \; : = \; \sum _ { t } \ln p _ { \phi } \big ( \mathcal { H } _ { t } \mid \mathcal { H } _ { 0 : t - 1 } \big ) .\tag{45}
$$

Prediction: for each candidate $\tau _ { i } , i = 1 , \ldots , n ,$ , the predictor returns the predicted mean and uncertainty of the learning signal $\ell ,$

$$
p _ { \phi } \big ( \ell \mid \tau _ { i } , \mathcal { H } _ { 0 : t } ; \theta _ { t } \big ) \; \longrightarrow \; \big \{ m ( \ell _ { i } ) ,   \sigma ( \ell _ { i } ) \big \} _ { i = 1 } ^ { n } ,\tag{46}
$$

at a cost independent of the cost of executing $\tau _ { i } ,$ since the predictor operates on embeddings.

Planning and execution: the estimates feed a budgeted subset selection, the fixed-size restriction of Eq. (15),

$$
\operatorname* { m a x } _ { \mathbf { z } _ { t } \in \{ 0 , 1 \} ^ { n } } \sum _ { i = 1 } ^ { n } z _ { t } ^ { ( i ) }   \mathcal { R } ( \boldsymbol { \ell } _ { t } ^ { ( i ) } ) + \mu   \mathcal { S } \Big ( \big \{ \boldsymbol { \tau } _ { t } ^ { ( i ) } : z _ { t } ^ { ( i ) } = 1 \big \} \Big ) , \qquad \mathrm { s . t . } \| \mathbf { z } _ { t } \| _ { 1 } = B ,\tag{47}
$$

where $\mathcal { A }$ aggregates $m ( \ell _ { i } )$ and $\sigma ( \ell _ { i } )$ and S rewards diversity among co-selected interactions. If $\mathcal { A } \geq 0 ,   \mu \geq 0 ,$ and $S$ is monotone submodular, the objective is monotone submodular and greedy selection attains a $( 1 - 1 / e )$ approximation. The condition matters: a diversity term that sums pairwise distances, as in Eq. (65), is supermodular, since a new item adds more pairs to a larger set, so the guarantee does not cover it. Monotone submodular choices include facility-location coverage and log det $\left( I + K _ { S } \right)$ for a positive semidefinite similarity kernel $K .$

Feedback: executed outcomes are appended to the history,

$$
\mathcal { H } _ { 0 : t + 1 } \; \leftarrow \; \mathcal { H } _ { 0 : t } \; \cup \; \Big \{ \big ( \tau _ { t } ^ { ( i ) } ,   \ell _ { t } ^ { ( i ) } \big )   :   z _ { t } ^ { ( i ) } = 1 \Big \} ,\tag{48}
$$

and fed back into Eq. (45) on the next round. The update is asymmetric: only executed interactions enter $\mathcal { H } ,$ so the predictor is trained on a biased sample of its own choosing, and the validity of the procedure rests on the timescale separation discussed in Section 3; the statistical consequences of this censoring are discussed in Section 7.2. Together, the four operations realize a complete closed-loop instance of the principle in which the predictive subsystem and the executive policy co-adapt through their shared history. Model predictive task sampling and its in-context extension run this loop over tasks (Wang et al., 2026; Wen et al., 2026a), and recent agentic RL methods carry it inside long-horizon trajectories, allocating rollouts to shared prefixes and propagating terminal rewards back through the resulting trees (Zou et al., 2026b,a).

## 5 Conditional Interaction Across Digital and Physical Domains

Reasoning-model fine-tuning and embodied-agent training are instances of one abstraction, distinguished by state space, cost structure, and verifier. In both, conditional interaction learns a predictor of value from past outcomes and executes the interactions whose predicted value clears the threshold $K _ { t }$ of Theorem 1. The two instantiations share one predicament — finding out whether a task is worth training on often costs almost as much as training on it — while differing in why. Table 1 summarizes the correspondence together with the test-time and agentic settings.

## 5.1 Digital and Physical Instantiations in the Literature

**Digital: from filtering after generation to predicting before it.** In RLVR only prompts whose sampled responses disagree carry a group-relative advantage, and that band drifts as the policy improves (Qu et al., 2026b). Early methods filter after generating: dynamic sampling in DAPO discards prompts whose groups are uniformly correct or wrong (Yu et al., 2026), and online difficulty filtering keeps prompts of intermediate pass rate (Bae et al., 2026), both paying in full for the rollouts that reveal a prompt to be uninformative — the execute-then-filter baseline of Theorem 14. Later methods predict before generating: GRESO skips prompts that their reward history marks as likely zero-variance (Zheng et al., 2026); MoPPS tracks each prompt’s success rate with an online Bayesian posterior and approaches an evaluation-intensive dynamic-sampling baseline with far fewer rollouts (Qu et al., 2026a); DPS, BOTS and PCL model

22

<!-- page 23 of 45 -->

prompt difficulty with a hidden-Markov process, Thompson sampling and an on-policy value model, respectively (Mao et al., 2026; Shen et al., 2026; Gao et al., 2026); and GPS trains one small generative predictor on the shared optimization history to select prompts for a much larger reasoning model, improving training efficiency and test-time compute allocation (Qu et al., 2026c). The same move extends beyond prompts, to online batch selection in supervised fine-tuning (Zou et al., 2025), to demonstration selection for in-context learning, where a small trained model makes a single edit to the retrieved demonstrations instead of iteratively scoring or searching over demonstration subsets (Wen et al., 2026b), to rollout allocation across tasks and trajectory prefixes in agentic RL (Zou et al., 2026b), and to critic-free, tree-aware credit assignment for long-horizon agents (Zou et al., 2026a). These results illustrate rather than validate the framework.

**Physical: from randomizing to predicting adaptation risk.** For embodied agents, judging a task configuration means acting on it, on hardware or in a high-fidelity simulator, at the pace of physics. Domain randomization trains across wide ranges of simulated appearance and dynamics (Tobin et al., 2017), with an explicit assessment of transfer (Muratore et al., 2018); because uniform randomization spends most trajectories on mastered configurations, later work prioritizes, through active and automatic domain randomization (Mehta et al., 2020; Akkaya et al., 2019) and through environment design and level replay driven by regret or learning potential (Dennis et al., 2020; Jiang et al., 2021), though still by executing configurations to score them. When robustness matters, the target is the adverse tail of post-adaptation risk, measured by the conditional value-at-risk (Rockafellar et al., 2000) and pursued in robust meta-learning (Finn et al., 2017; Wang et al., 2023; Lv et al., 2024), risk-averse planning (Carpin et al., 2016) and safety-critical scenario generation (Rempe et al., 2022). Locating that tail requires evaluating many tasks; MPTS removes the evaluation by learning a risk predictive model from the optimization history (Wang et al., 2026), and its successors add posterior sampling and diversity (Qu et al., 2025) and in-context risk prediction (Wen et al., 2026a), yielding curriculum-like sampling under a calibrated predictor.

**What the two literatures share.** Evaluation requires generating in the digital regime and acting in the physical one, yet both literatures converge on the same loop: System-1 scores a candidate from cheap features and history, System-2 executes only what clears the score, and System-2’s outcomes refit System-1. They differ in utility (mixed outcomes versus tail adaptation risk; Theorem 5) and in the conserved resource (compute versus actuation); their acquisition functions are collected in Section B. Unlike post-hoc data valuation by influence functions or data Shapley (Koh and Liang, 2017; Ghorbani and Zou, 2019), and unlike active learning, experimental design and bandit index policies over a static pool (Krishnakumar, 2007; Chaloner and Verdinelli, 1995; Gittins, 1979), these methods value candidates before execution and track a moving learner. They also extend earlier fast–slow pairings, such as expert iteration and AlphaZero (Anthony et al., 2017; Silver et al., 2018) and speculative decoding (Leviathan et al., 2023), from one game or one decoding step to training and deployment.

## 6 Conditional Interaction at Test Time

The framework so far concerns training. The same decision problem appears at inference, but with a different state, utility and cost ledger. Nothing in Theorem 1 and Eq. (9) requires the training setting: the rule gates an interaction on predicted utility net of cost, whether the resource is a training rollout or an inference-time forward pass. We argue that test-time scaling and adaptive decision-making, usually studied separately, can be cast as test-time instances of conditional interaction. Prospective metacognitive control of test-time compute is an active line of work in its own right (Nazi and Dipta, 2026; Cao et al., 2026); the contribution here is not to discover it but to place it in the same ledger as training. Co-design therefore does not end at deployment: the System-1 that shaped training goes on to ration System-2’s inference effort.

At deployment 𝜃 is fixed; what varies, and costs, is the computation and the number of real-world actions spent per output. An interaction is now an increment of inference: a reasoning step, an extra sample, a tool or retrieval call, or a physical action. The executive state is $b _ { t } = p ( a \mid q , \mathcal { H } _ { 0 : t } )$ , a predictive distribution over candidate answers or latent solution states 𝑎 for query 𝑞, which we call the belief state for convenience without requiring a literal Bayesian

23

<!-- page 24 of 45 -->

posterior. The utility of an increment is

$$
\mathbb { I } _ { \mathsf { t t } } ( i _ { t } ) = \mathbb { E } \big [   D _ { \mathrm { K L } } \Big [ b _ { t + 1 }   \bigg \|   b _ { t }   \Big ] \big | \mathcal { H } _ { 0 : t } ,   i _ { t }   \big ] ,\tag{49}
$$

and its cost is the inference compute or actuation it requires, so Theorem 1 applies verbatim with $\hat { u } _ { \phi }$ predicting Eq. (49) and $K _ { t }$ as a stopping threshold. Eq. (49) measures epistemic movement, not improvement: a changed answer is not necessarily a better one, so movement of $b _ { t }$ should be interpreted as computational sensitivity, and accuracy must be checked separately. The test-time hypothesis is that this sensitivity predicts where additional computation is useful. Only the ledger changes — parameter belief change becomes answer belief change, and training compute becomes inference compute — and that substitution is the entire test-time claim.

## 6.1 Face I: Efficient Test-Time Scaling

Test-time scaling improves a fixed model by spending more inference per query (longer CoT, repeated sampling with aggregation, tree search, tool calls). This spend is highly heterogeneous across queries: easy ones are solved by the first short trajectory and the hardest by no feasible budget, so extra reasoning moves $b _ { t }$ only in an intermediate band, the analogue of the partially solved prompts of Section 5.1. Uniform test-time scaling therefore inherits the vanishing marginal efficiency of Eq. (1), now in inference cost. Conditional interaction continues to spend only while the predicted change of increment $c _ { t }$ clears the threshold:

$$
\mathrm { h a l t \; a t \; s t e p \; } t ^ { \star } \; = \; \operatorname* { m i n } \bigl \{ t : \hat { u } _ { \phi } ( c _ { t } \mid \mathcal { H } _ { 0 : t } ) < \kappa _ { t } \bigr \} ,\tag{50}
$$

the optimal-stopping form of Eqs. (43) and (44) applied at inference. Familiar techniques are choices of increment and predictor: adaptive-length reasoning and early exit gate one more decoding step, adaptive self-consistency one more sample, budget-aware tree search one more node expansion. All suppress increments predicted to leave $b _ { t }$ unchanged, so one proposition covers them, with one caveat. Eq. (49) values the next increment, and an increment can leave $b _ { t }$ unchanged while enabling later ones that do change it, as when an intermediate thought pays off only several steps on. A gate that halts on the next-step utility can therefore stop before such a delayed gain. The guarantee below is stated for the continuation utility

$$
u _ { \mathrm { c o n t } } ( q ) = \mathbb { E } \Big [   D _ { \mathrm { K L } } \Big [ b _ { C }   \bigg \|   b _ { 1 } \Big ] \Big | q , \mathcal { H } _ { 0 : 1 }   \Big ] ,\tag{51}
$$

the expected belief change from the first increment to the full budget 𝐶, which System-1 must predict after the probing increment and whose prediction cost is charged as $c _ { \mathrm { p r e d } }$ . Predicting the value of the next token, of a whole continuation, and of the final answer are different tasks, and the second and third are generally harder. Eq. (50) with the next-step utility is covered only under a no-delayed-gain condition: a query whose next increment leaves $b _ { t }$ unchanged in expectation gains nothing from any later increment. Two stopping policies must also be kept apart. The one-probe gate spends one increment, predicts Eq. (51), and then stops or runs to the full budget; it is the policy Theorem 22 covers. A sequential halting policy, Eq. (50), re-decides at every step and should score the continuation value from the current state, $u _ { \mathrm { c o n t } , t } ( q ) = \mathbb { E } [ D _ { \mathrm { K L } } \bigg [ b _ { C } \bigg \| b _ { t } \bigg ] \mid q , \mathcal { H } _ { 0 : t } ] ;$ a halt at step 𝑡 then loses accuracy only on a false negative, so by a union bound its accuracy loss is at most $\textstyle \sum _ { t } \delta _ { \mathrm { F N } , t }$ , the sum of the per-step false-negative masses, and its cost must charge a prediction at every step. We do not develop that guarantee further. Its error term, and that of Theorem 23, rests on the following margin condition, which turns a score error into a misclassification rate.

Lemma 21 (Misclassification Under a Margin Condition). Let candidates $i \sim \mathcal { D }$ be utility-relevant when $u ( i ) \geq u _ { \mathrm { m i n } } ,$ let the gate admit 𝑖 when $\begin{array} { r } { \hat { u } ( i ) \geq u _ { \mathrm { m i n } } , } \end{array}$ , and suppose $| \hat { u } ( i ) - u ( i ) | \leq \varepsilon   f$ or every 𝑖. More generally, let the gate admit 𝑖 when $\begin{array} { r } { \hat { u } ( i ) \geq \kappa . } \end{array}$ Every misclassified candidate, admitted while dead or suppressed while relevant, then lies in the band $\mathcal { B } _ { \varepsilon } = \left\{ i : \kappa - \varepsilon \leq u \big ( i \big ) < u _ { \operatorname* { m i n } } \right\} \cup \left\{ i : u _ { \operatorname* { m i n } } \leq u \big ( i \big ) < \kappa + \varepsilon \right\}$ , so the misclassified masses satisfy $\delta _ { \mathrm { F P } } + \delta _ { \mathrm { F N } } \leq \mathrm { P r } _ { i \sim \mathcal { D } } ( \mathcal { B } _ { \varepsilon } )$ With $\kappa   =   u _ { \mathrm { m i n } } ,$ if the true utility satisfies the margin condition $\operatorname* { P r } _ { i \sim \mathcal { D } } \bigl ( \lvert u \bigl ( i \bigr ) - u _ { \operatorname* { m i n } } \rvert   \leq   \varepsilon \bigr )   \leq$ 𝑚 𝜀 with constant 𝑚, then the gate misclassifies, by admitting a dead candidate or suppressing a relevant one, at most a fraction 𝑚 𝜀 of candidates.

A density of 𝑢 bounded by $m / 2$ on $\left[ u _ { \operatorname* { m i n } } - \varepsilon , u _ { \operatorname* { m i n } } \right.$ + 𝜀] gives the condition. The band, not the margin condition, is the general object, and atoms are the case that matters in practice. At test time many queries have continuation utility

24

<!-- page 25 of 45 -->

exactly zero, and relevance means $u > 0   ( u _ { \mathrm { m i n } } \to 0 ^ { + } ) ;$ ; the whole atom then lies within 𝜀 of the floor, and the margin condition with $\kappa = u _ { \operatorname* { m i n } }$ fails. Placing the gate threshold above the error, $\kappa > \varepsilon ,$ excludes the atom: a query with $u = 0$ has $\hat { u } \leq \varepsilon < \kappa$ and is never admitted, so $\delta _ { \mathrm { F P } } = 0$ and $\delta _ { \mathrm { F N } } \leq \operatorname* { P r } \bigl ( 0 < u < \kappa + \varepsilon \bigr )$ , the mass of small but nonzero utilities. Whether that mass is small is a property of the task distribution, to be measured on an audit pool, not an assumption to take for granted.

**Proposition 22** (Conditional Test-Time Cost Bound Under a Halting Margin). Let $\mathcal { R } ^ { \mathfrak { t } }$ be the queries on which the full budget is more accurate in expectation than one increment, with density $\rho _ { \mathrm { a c c } } ^ { \mathrm { t f } }$ and let the gate threshold a declared test-time utility whose relevant set contains $\mathcal { R } ^ { \mathfrak { t } } ,$ , with density $\rho _ { u } ^ { \mathrm { t t } } \geq \rho _ { \mathrm { a c c } } ^ { \mathrm { t t } }$ . Let unconditional scaling spend 𝐶 unit increments per query, let the answer be a function of the belief state, and let the selection policy be the one-probe gate: it spends one probing increment plus $c _ { p r e d }$ and then halts or continues to the full budget, with 𝛿<sub>FN</sub> and 𝛿<sub>FP</sub> the fractions of all queries that are relevant but halted and dead but continued. Then the gated policy loses at most $\delta _ { \mathrm { F N } }$ accuracy while spending at most a fraction

$$
\rho _ { u } ^ { \mathrm { t t } } + \frac { 1 + c _ { p r e d } } { C } + \delta _ { \mathrm { F P } }
$$

of the unconditional compute, a multiplier approaching $1 / \rho _ { u } ^ { \mathrm { t t } }$ as $C \rightarrow$ ∞ and $\delta _ { \mathrm { F P } } \rightarrow 0 .$ . By Theorem 21, $\delta _ { \mathrm { F N } } + \delta _ { \mathrm { F P } } \leq$ Pr $( \mathcal { B } _ { \varepsilon _ { \phi } } )$ for a predictor with error at most $\varepsilon _ { \phi }$ , which is at most 𝑚 $\varepsilon _ { \phi }$ under the margin condition. The tightest choice is the accuracy-gain utility itself, for which $\rho _ { u } ^ { \mathrm { t t } } = \rho _ { \mathrm { a c c } } ^ { \mathrm { t t } }$ . The continuation utility of Eq. (51) with $u _ { \mathrm { m i n } } \to 0 ^ { + }$ is also admissible: if its value is zero then $b _ { C } = b _ { 1 }$ almost surely, so the answer, and with it the accuracy, is unchanged; its gate should use $\kappa > \varepsilon _ { \phi } ,$ , so that the atom of zero-utility queries is never admitted. With density $\rho _ { \mathrm { c o n t } } ^ { \mathrm { t t } } \geq \rho _ { \mathrm { a c c } } ^ { \mathrm { t t } }$ it gives the weaker multiplier $1 / \rho _ { \mathrm { c o n t } } ^ { \mathrm { t t } }$ . The next-step utility of Eq. (49) is admissible only under the no-delayed-gain condition.

The proposition is the inference-time counterpart of Eq. (12), with the per-query halting decision in place of per-batch selection, and only $\rho _ { u } ^ { \uparrows }$ has to be re-estimated. Its status needs care: the multiplier follows given a small informative fraction and an accurate halting predictor, and unlike the training side neither premise has been measured, making this the one place where our efficiency claims lack even indirect support. The falsification protocol is direct: measure $\rho _ { \mathrm { a c c } } ^ { \mathrm { t f } }$ as the fraction of queries whose answer becomes correct when the budget is raised from a short default to the full one, run unconditional scaling at full budget for reference accuracy, then run the one-probe gate and record accuracy and mean compute. A larger accuracy loss than the false-negative mass $\delta _ { \mathrm { F N } }$ measured on an audit sample, or a failure to realize the saving, refutes the test-time claim.

## 6.2 Face II: Efficient Adaptive Decision-Making

For systems acting in the world, the increments are external actions (a manipulation, a sensor query, a tool call, a clarifying question), each with a real cost and the capacity to change beliefs. The efficient policy acts when predicted value of information net of cost is positive, i.e. Eq. (43) promoted to an online rule:

$$
\mathsf { e x e c u t e } \; i _ { t } \quad \Longleftrightarrow \quad \mathbb { E } \big [ \Delta \mathcal { P } ( i _ { t } ) \mid \mathcal { H } _ { 0 : t } \big ] - C ( i _ { t } ) \geq V _ { \emptyset } ,\tag{52}
$$

with $\Delta \mathcal { P } ( i _ { t } )$ the expected improvement in task return, $C ( i _ { t } )$ the priced cost defined after Eq. (43), and $V _ { \emptyset }$ the value of deferring or taking the cheap default, adapted online as in Eq. (22). Reducing task uncertainty is a different utility, not a second reading of $\Delta \mathcal { P } ;$ it instantiates the same rule with $\Delta \mathcal { P }$ replaced by the expected entropy reduction of the task posterior. Selective tool use, active perception, and interrupt-or-defer control are Eq. (52) with different $i _ { t }$ and utilities: a tool call scored by answer-entropy reduction, a sensor movement or question scored by world-model belief change, a re-plan triggered when predicted plan value degrades. The adaptation-risk predictor that prioritized training tasks in Section 5.1 becomes, at deployment, the value-of-information predictor that gates actions: efficient adaptive decision-making can be cast as conditional interaction with the world as the executive.

25

<!-- page 26 of 45 -->

| Schema element | Digital (training) | Physical (training) | Test time | Agentic |
| --- | --- | --- | --- | --- |
| Interaction i (and execution) Executive state Verifier / feedback Utility I(i) | $\tau \to \mathrm { C o T }$ prompt generation policy parameters θ reward checker / scorer mixed-outcome proxy $u _ { \mathrm { a d v } }$ | task config. ξ → env. rollout policy parameters θ adaptation loss predicted adaptation risk | reasoning step / sample $b _ { t }$ answer belief state consistency / halting signal accuracy gain (or belief change) | tool / think / ask action answer / plan belief state $b _ { t }$ tool result / task outcome value of information |

Table 1 One decision pattern under four cost ledgers. Execution is part of each interaction (first row), not a separate stage. The selection pattern is shared across columns; the utilities and learning dynamics are not, and the conserved resource differs: gradients, actuation, inference compute, or the composite cost of Eq. (53).

## 6.3 Selective Agentic Tool and Computer Use

An agent that plans, retrieves, calls tools, asks questions and operates a computer faces Eq. (52) against a composite cost. At step 𝑡 it holds an answer/plan belief state $b _ { t }$ and chooses from a heterogeneous menu (think, retrieve or call a tool, ask, delegate, commit), each action costing

$$
C ( i _ { t } ) = \lambda ^ { \top } c ( i _ { t } ) + \lambda _ { \mathrm { r i s k } } \mathrm { P r } [ \mathrm { i r r e c o v e r a b l e } \mid i _ { t } ] ,\tag{53}
$$

where $c ( i _ { t } ) = \big ( c _ { \mathrm { l a t } } ( i _ { t } ) ,   c _ { \mathrm { \mathfrak { S } } } ( i _ { t } ) ,   c _ { \mathrm { \mathfrak { t o k } } } ( i _ { t } ) \big )$ collects latency, metered spend and context tokens, $\lambda \geq 0$ prices them, and 𝜆<sub>risk</sub> prices an irrecoverable side effect. In the co-designed pair, System-1 (typically a verifier head or small router) admits $i _ { t }$ only when Eq. (52) holds for this cost, with $V _ { \emptyset }$ the value of committing the current best answer, and System-2 is the agent that executes it. Tool use and computer use differ only in the size of the space to search: a tool call is one typed request among a few, whereas a computer-use episode branches at each of tens of steps, every one costing a screenshot, a model call and unparallelizable time.

**Proposition 23** (Conditional Agentic Cost Bound for Separable Actions). Suppose (a) omitting any set of decisionirrelevant actions leaves the answer unchanged; (b) the gate thresholds a predicted relevance score with error at most $\varepsilon _ { \phi } ,$ and on every task the action scores satisfy the margin condition of Theorem 21 with constant 𝑚, so that in expectation at most a fraction 𝑚 $\varepsilon _ { \phi }$ of the actions are misclassified; and (c) prediction costs a fraction 𝑟 of an average action. Consider tasks with 𝑁 candidate actions in which no action costs more than $R _ { c }$ times the task’s average action cost. If utility-relevant actions carry a share $\rho _ { u } ^ { C }$ of the expected call-everything cost (the cost-weighted density of utility-relevant actions, which reduces to their fraction $\rho _ { u }$ when all actions cost the same), the conditional agent’s accuracy is within 𝑁𝑚 $\varepsilon _ { \phi }$ of call-everything accuracy, at expected cost

$$
\mathbb { E } [ C _ { \mathrm { C I } } ] \leq \left( \rho _ { u } ^ { C } + r + m   \varepsilon _ { \phi }   R _ { c } \right) \mathbb { E } [ C _ { a l l } ] .
$$

The proposition is therefore a separable-action special case. Assumption (a) fails on long interface trajectories, where a dozen individually uninformative clicks may be jointly needed to reach the screen that matters, and whenever a locally uninformative action, such as navigating, authenticating or opening a file, is a prerequisite for a useful one; there utility belongs to the sub-trajectory or state transition, valued by a state-dependent value of information, and the unit to score is a sub-trajectory, and since the accuracy loss grows linearly in 𝑁, the per-action error must shrink with the horizon. Pure exploration (Section 2.4) and mandated actions such as safety logging lie outside the gate. Across Sections 6.1 and 6.2 and this subsection the predict–gate–execute pattern is shared while utilities and costs are not (Table 1), so a System-1 calibrated against System-2 during training can serve as the inference-time router, with deployment outcomes flowing back as its training data.

26

<!-- page 27 of 45 -->

## 7 A Testable Scaling Hypothesis, and Open Problems

Co-design makes two kinds of prediction: what the pair should achieve as it scales, and where it can fail. We state the first as a testable hypothesis about informative compute, then turn to the main obstacle — System-1 is supervised only by the interactions it lets System-2 execute — and to open problems. It helps to separate three kinds of claim. Proved under stated assumptions: the IIE accounting (Theorem 7), the screening and memory results (Theorems 14 and 15), the regret bounds (Theorems 18 and 19), the test-time and agentic cost bounds (Theorems 22 and 23), co-scaling under fixed error curves (Theorem 13) and horizontal rescaling (Theorem 24). Supported by application-specific evidence: the savings of predictive prompt and task sampling (Section 5.1). Proposed hypotheses: a co-designed frontier that dominates on cost, quality of service and task success at once (Fig. 3), the decay of $\rho _ { u }$ (Theorem 25), and a changed scaling exponent.

## 7.1 The Scaling Hypothesis: Informative Compute

Standard scaling laws express loss as a power law in total compute, $\mathcal { L } ( C ) = ( C _ { 0 } / C ) ^ { \alpha _ { C } }$ , with

$$
C _ { \mathrm { i n f o } } ( C ) \; = \; \int _ { 0 } ^ {   C } \rho _ { u } ^ { \mathrm { e x e c } } ( c )   d c .\tag{54}
$$

the informative compute absorbed along a run, where $\rho _ { u } ^ { \mathrm { e x e c } } ( c )$ is the executed fraction at cumulative compute $c ,$ the share of compute then spent on utility-relevant interactions. Under unconditional training it equals the cost-weighted pool density

$$
\rho _ { u } ^ { C } ( \theta _ { c } ) \; = \; \frac { \mathbb { E } _ { i \sim \mathcal { D } } \big [ C ( i )   \mathbf { 1 } [ u ( i ; \theta _ { c } ) \geq u _ { \operatorname* { m i n } } ] \big ] } { \mathbb { E } _ { i \sim \mathcal { D } } [ C ( i ) ] } ,\tag{55}
$$

the compute share of relevant interactions, as in Theorem $2 3 ;$ it reduces to the interaction share $\rho _ { u } ( \theta _ { c } )$ only when execution costs are equal, and can differ substantially otherwise. Under a perfect gate it is one. With equal costs, as Theorem 24 assumes, the two runs of Theorem 24 accumulate informative compute at rates $\rho _ { u }$ and 1. The scaling hypothesis below is stated for $\rho _ { u }$ in this equal-cost case; with heterogeneous costs, $\rho _ { u } ^ { C }$ takes its place. The framework predicts more than a constant-factor gain: if the fraction of utility-relevant interactions changes as a learner becomes more capable, total compute is not the only meaningful coordinate of scaling, and the question is whether performance curves become more comparable when plotted against Eq. (54). Two hypotheses answer differently and are experimentally distinguishable: the conservative one is that gating produces only a horizontal rescaling, the stronger one that it also alters the trajectory of interaction density and thereby the effective exponent.

**Proposition 24** (Horizontal Rescaling of the Scaling Curve). Suppose (a) unconditional training follows a loss curve $\mathcal { L } _ { \mathrm { b a s e } } ( C )$ on a pool whose density of utility-relevant interactions is a constant $\rho _ { u } \in ( 0 , 1 ] ,$ ; (b) dead interactions leave the learner unchanged; (c) progress depends only on the utility-relevant interactions executed, whose distribution and order the selection policy preserves; and (d) the selection policy is perfect, prediction is free, and executions cost the same. Then

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { C I } } \big ( C \big ) \; = \; \mathcal { L } _ { \mathrm { b a s e } } \big ( C / \rho _ { u } \big )   , } \end{array}\tag{56}
$$

i.e. on a log-compute axis the conditional curve is the unconditional one shifted left by ln $( 1 / \rho _ { u } )$

This is a rescaling identity, valid for any shape of $\mathcal { L } _ { \mathrm { b a s e } } ;$ base: it is the precise content of the $1 / \rho _ { u }$ multiplier used throughout. Assumption (a) fixes $\rho _ { u } ,$ , which the next hypothesis denies in general; when the pool fraction drifts, the constant factor becomes the time-varying reparameterization Eq. (54), and that integral is what the curve-collapse test below plots against.

**Hypothesis 25** (Informative Interaction Decay). After an initial transient, $\rho _ { u } ( \theta _ { t } )$ is non-increasing as the system improves: mastered interactions leave the utility-relevant set, so $\rho _ { u }$ tracks the shrinking measure of reachable but not-yet-mastered interactions.

27

<!-- page 28 of 45 -->

The hypothesis is empirical and can fail, since new capabilities can expose new informative tasks and an improved predictor changes what is selected. Its logical direction matters, and it rests on an assumption that the definition of $\rho _ { u }$ does not supply. Nothing in Eq. (14) makes relevant interactions equally useful: two prompts with non-zero advantage can produce updates of very different size and direction. We therefore assume constant conditional progress: over the regime studied, a unit of utility-relevant compute reduces the expected loss by the same amount 𝜂. Under that assumption unconditional learning is the absorption of informative compute, $d \mathcal { L } / d C = - \eta   \rho _ { u } ( \theta )$ , and an observed power law corresponds to one particular decay, $\rho _ { u } = k   \mathcal { L } ^ { 1 + 1 / \alpha }$ with $\alpha _ { C } = \alpha$ , but it does not follow that $\rho _ { u }$ produces the scaling law. The construction establishes consistency, not derivation: an independently measured trajectory must satisfy it, so slower decay means a larger exponent. Gradient noise, intrinsic dimension and loss curvature generate power-law-like curves too; $\rho _ { u }$ is simply the most actionable, since gating intervenes on it directly (Theorem 9).

Against informative compute measured as in Eq. (14), the null implied by Theorem 24 predicts that the two curves coincide, the alternative that the gated curve is steeper even after rescaling. A strict improvement would not by itself establish Theorem 25, since different optimizer dynamics, lower gradient variance, a selection-induced curriculum or reduced redundancy could all produce it; attributing it to $\rho _ { u }$ requires measuring $\rho _ { u }$ along both runs and testing whether the difference in exponents is explained by the difference in those trajectories. The savings reported in Section 5.1 are consistent with the null.

## 7.2 Selective Supervision: The Gate Censors Its Own Training Data

The most consequential technical limitation is structural: once System-1 decides which interactions are executed, it also decides which interactions it is trained on. With $p _ { \mathrm { c a n d } } ( i )$ the candidate distribution and $e ( i ) = \operatorname* { P r } [ i ]$ executed $[ i , \mathcal { H } _ { 0 : t } ]$ the gate’s execution probability, the predictor is trained on

$$
p _ { \mathrm { t r a i n } } ( i ) \propto p _ { \mathrm { c a n d } } ( i )   e ( i ) \neq p _ { \mathrm { c a n d } } ( i ) ,\tag{57}
$$

so labels are missing not at random: whether an outcome is observed depends on the predictor’s own estimate of it, and in the worst case a candidate scored as low-value is suppressed, its outcome never observed, and the score never corrected. This is the selective-labels problem of algorithmic decision-making (Lakkaraju et al., 2017), the positivity problem of causal inference, and, under shifting distributions, the label-querying problem of active self-training (Zhou et al., 2022b,a). Three features make it tractable rather than hopeless. Positivity can be designed in, but not by temperature: the indicator in Eq. (3) gives every candidate with $\hat { u } < \kappa _ { t }$ probability zero at every finite $\beta ,$ so a soft gate only randomizes within the admissible set. Positivity needs an explicit component that reaches the suppressed region: an exploration mixture $\pi _ { \epsilon } = \left( 1 - \epsilon \right) \pi _ { \mathrm { C I } } + \epsilon   \pi _ { 0 }$ with $\pi _ { 0 } ( i ) > 0$ on the whole pool, which gives $e ( i ) \geq \epsilon   \pi _ { 0 } ( i )$ per draw, or independent audit sampling of suppressed candidates at a known rate $a ,$ which gives $e ( i ) \geq a$ Either way $e ( i ) \geq e _ { \mathrm { m i n } } > 0$ The propensities are known, because the gate assigns them, so weighting the predictor’s loss by $1 / e ( i )$ (Horvitz and Thompson, 1952), or combining that weighting with the predictor’s own regression in a doubly robust estimator (Dudík et al., 2011), gives unbiased estimates of its error on the full candidate distribution. And the audit sample of suppressed candidates (Section 4.1) is exactly such a positivity-preserving exploration stream, provided its rate is fixed independently of the predictor’s scores. What remains open is quantitative: how large $e _ { \mathrm { m i n } }$ must be for the predictor’s error to shrink while the policy and the candidate distribution drift, how the variance of inverse-propensity weights trades against the screening gain, and whether any rate like $\mathrm { E q . }$ (20) survives adaptive, censored collection. Of all the problems the framework raises, we regard this as the one most in need of theory.

## 7.3 Open Problems and Outlook

The decay profile of $\rho _ { u } ( \theta _ { t } )$ has not yet been measured, and doing so is the cleanest single test of the framework; it would also test a mechanistic account of emergence, since a sub-skill whose $\rho _ { u }$ crosses a usable threshold abruptly should appear abruptly. We regard this as the most actionable contribution of this Perspective.

Several questions stay open. Because aggressive gating discards informative interactions while permissive gating readmits dead ones, efficiency is maximized at an interior admission rate $\rho ^ { * } ( \varepsilon _ { \phi } )$ , whose characterization sits inside a

28

<!-- page 29 of 45 -->

non-stationary bandit problem (Auer et al., 2002), as does its inference-time counterpart, whether a halting predictor can reach the accuracy Theorem 22 requires without its own forward pass eroding the savings. Theorem 13 fixes the direction in which System-1 should grow with System-2, and sweeping System-1’s size at several System-2 sizes while measuring fp, fn and $c _ { \mathrm { p r e d } }$ would give the compute-optimal split between them. And the framework is single-learner: in agent societies, self-play and debate, informativeness is a property of joint beliefs, suggesting a relational utility and an informative fraction defined on an interaction graph, since an interaction uninformative to its initiator may inform an observer. We regard this last as the most consequential boundary of the framework.

Conditional interaction finally resonates, without being equivalent, with the enactive view of cognition urged for AI by Rafiee and Sutton (2026), building on Varela et al. (2017) and the “era of experience” (Silver and Sutton, 2025): perception as selective activity, in which an agent acts in order to reveal information. Our utility nonetheless answers to an externally given task and cost ledger rather than the agent’s own self-maintenance, so we formalize the selective-engagement facet alone. Within that scope, routing tasks to System-2 executives (Section 2.3) extends the gate from data to model selection and Section 6 adds the inference boundary, so a fully conditional system, whose data, computation, inference effort and external actions are all gated by a co-designed System-1, is a promising architecture for resource-efficient intelligence at scale, and the argument strengthens with capability if both Theorem 25 and Theorem 13 hold. Beyond human-level performance (Mulgan, 2016), human data and curricula would no longer say what is worth learning, and a system would have to choose its own experiments; its System-1 would then decide what a superhuman System-2 learns and does, making the gate a major source of efficiency.

## 8 Conclusion

This work poses the perspective that the next stage of progress toward AGI may come from the co-design of System-1 and System-2: a cheap System-1 that values each candidate interaction before it is paid for, and a costly System-2 that executes only what clears the gate, each trained on the other’s outputs. Conditional interaction, the predict–gate–execute–update loop that couples them, applies alike to training rollouts, test-time reasoning and agentic actions, and under explicit assumptions yields screening, memory, regret and co-scaling results. These results motivate the scaling hypothesis of Section 7. Our answer to the research question is therefore a qualified yes: the co-design is promising wherever interactions are costly, unequal in value and predictable in advance. The decisive next step is measurement: quantify $\rho _ { u } ,$ compare scaling curves at matched informative compute, and test whether co-design changes more than a constant factor.

## References

Ilge Akkaya, Marcin Andrychowicz, Maciek Chociej, Mateusz Litwin, Bob McGrew, Arthur Petron, Alex Paino, Matthias Plappert, Glenn Powell, Raphael Ribas, et al. Solving rubik’s cube with a robot hand. arXiv preprint arXiv:1910.07113, 2019.

Thomas Anthony, Zheng Tian, and David Barber. Thinking fast and slow with deep learning and tree search. Advances in neural information processing systems, 30, 2017.

Peter Auer, Nicolo Cesa-Bianchi, and Paul Fischer. Finite-time analysis of the multiarmed bandit problem. Machine learning, 47 (2):235–256, 2002.

Sanghwan Bae, Jiwoo Hong, Min Young Lee, Hanbyul Kim, JeongYeon Nam, and Donghyun Kwak. Online difficulty filtering for reasoning oriented reinforcement learning. In Proceedings of the 19th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers), pages 700–719, 2026.

Emily M Bender, Timnit Gebru, Angelina McMillan-Major, and Shmargaret Shmitchell. On the dangers of stochastic parrots: Can language models be too big? In Proceedings of the 2021 ACM conference on fairness, accountability, and transparency, pages 610–623, 2021.

Vivek S Borkar and Vivek S Borkar. Stochastic approximation: a dynamical systems viewpoint, volume 2. Springer, 2008.

Qi Cao, Yufan Wang, Peijia Qin, Shuhao Zhang, and Pengtao Xie. Llms know when they know, but do not act on it: A metacognitive harness for test-time scaling. arXiv preprint arXiv:2605.14186, 2026.

29

<!-- page 30 of 45 -->

Stefano Carpin, Yin-Lam Chow, and Marco Pavone. Risk aversion in finite markov decision processes using total cost criteria and average value at risk. In 2016 ieee international conference on robotics and automation (icra), pages 335–342. IEEE, 2016.

Kathryn Chaloner and Isabella Verdinelli. Bayesian experimental design: A review. Statistical science, pages 273–304, 1995.

Mingju Chen, Can Lv, Guibin Zhang, Heng Chang, and Shiji Zhou. Harnessforge: Joint harness and policy evolution for adaptive agent systems. arXiv preprint arXiv:2606.01779, 2026.

Michael Dennis, Natasha Jaques, Eugene Vinitsky, Alexandre Bayen, Stuart Russell, Andrew Critch, and Sergey Levine. Emergent complexity and zero-shot transfer via unsupervised environment design. Advances in neural information processing systems, 33: 13049–13061, 2020.

Miroslav Dudík, John Langford, and Lihong Li. Doubly robust policy evaluation and learning. arXiv preprint arXiv:1103.4601, 2011.

Chelsea Finn, Pieter Abbeel, and Sergey Levine. Model-agnostic meta-learning for fast adaptation of deep networks. In International conference on machine learning, pages 1126–1135. PMLR, 2017.

Karl Friston. The free-energy principle: a unified brain theory? Nature reviews neuroscience, 11(2):127–138, 2010.

Zhaolin Gao, Joongwon Kim, Wen Sun, Thorsten Joachims, Sid Wang, Richard Yuanzhe Pang, and Liang Tan. Prompt curriculum learning for efficient llm post-training. In International Conference on Learning Representations, volume 2026, pages 58614–58652, 2026.

Amirata Ghorbani and James Zou. Data shapley: Equitable valuation of data for machine learning. In International conference on machine learning, pages 2242–2251. PMLR, 2019.

John C Gittins. Bandit processes and dynamic allocation indices. Journal of the Royal Statistical Society Series B: Statistical Methodology, 41(2):148–164, 1979.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. Deepseek-r1 incentivizes reasoning in llms through reinforcement learning. Nature, 645(8081):633–638, 2025.

Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. Training compute-optimal large language models. arXiv preprint arXiv:2203.15556, 2022.

Daniel G Horvitz and Donovan J Thompson. A generalization of sampling without replacement from a finite universe. Journal of the American statistical Association, 47(260):663–685, 1952.

L. Itti and P. Baldi. Bayesian surprise attracts human attention. Advances in Neural Information Processing Systems, 18, 2005.

Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al. Openai o1 system card. arXiv preprint arXiv:2412.16720, 2024.

Minqi Jiang, Edward Grefenstette, and Tim Rocktäschel. Prioritized level replay. In International Conference on Machine Learning, pages 4940–4950. PMLR, 2021.

Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020.

Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, et al. Openvla: An open-source vision-language-action model. arXiv preprint arXiv:2406.09246, 2024.

Pang Wei Koh and Percy Liang. Understanding black-box predictions via influence functions. In International conference on machine learning, pages 1885–1894. Pmlr, 2017.

Anita Krishnakumar. Active learning literature survey. Tech. rep., Technical reports, University of California, Santa Cruz, 42, 2007.

Himabindu Lakkaraju, Jon Kleinberg, Jure Leskovec, Jens Ludwig, and Sendhil Mullainathan. The selective labels problem: Evaluating algorithmic predictions in the presence of unobservables. In Proceedings of the 23rd ACM SIGKDD international conference on knowledge discovery and data mining, pages 275–284, 2017.

Yaniv Leviathan, Matan Kalman, and Yossi Matias. Fast inference from transformers via speculative decoding. In International conference on machine learning, pages 19274–19286. PMLR, 2023.

Zhenyu Li, Guanlin Wu, Cheems Wang, and Yongqiang Zhao. Limited reasoning space: The cage of long-horizon reasoning in llms. arXiv preprint arXiv:2602.19281, 2026.

Yiqin Lv, Qi Wang, Dong Liang, and Zheng Xie. Theoretical investigations and practical enhancements on tail task risk minimization in meta learning. Advances In Neural Information Processing Systems, 37:82921–82961, 2024.

30

<!-- page 31 of 45 -->

Yixiu Mao, Yun Qu, Qi Wang, Heming Zou, and Xiangyang Ji. Dynamics-predictive sampling for active rl finetuning of large reasoning models. In International Conference on Learning Representations, volume 2026, pages 25242–25270, 2026.

Bhairav Mehta, Manfred Diaz, Florian Golemo, Christopher J Pal, and Liam Paull. Active domain randomization. In Conference on robot learning, pages 1162–1176. PMLR, 2020.

Meredith Ringel Morris, Jascha Sohl-Dickstein, Noah Fiedel, Tris Warkentin, Allan Dafoe, Aleksandra Faust, Clement Farabet, and Shane Legg. Position: Levels of agi for operationalizing progress on the path to agi. In Forty-first International Conference on Machine Learning, 2024.

Tim Mulgan. Superintelligence: Paths, dangers, strategies, 2016.

Fabio Muratore, Felix Treede, Michael Gienger, and Jan Peters. Domain randomization for simulation-based policy optimization with transferability assessment. In Conference on Robot Learning, pages 700–713. PMLR, 2018.

Zabir Al Nazi and Shubhashis Roy Dipta. Triage: Evaluating prospective metacognitive control in llms under resource constraints. arXiv preprint arXiv:2605.13414, 2026.

Maddy Osman. What is Jev (2026)? TypeSafe AI’s System One model. DigitalOcean, updated September 24, 2026; latency figures attributed to TypeSafe’s own evaluations, 2026. URL [https://www.digitalocean.com/resources/articles/what-is-jev](https://www.digitalocean.com/resources/articles/what-is-jev).

Yun Qu, Qi Cheems Wang, Yixiu Mao, Yiqin Lv, and Xiangyang Ji. Fast and robust: Task sampling with posterior and diversity synergies for adaptive decision-makers in randomized environments. arXiv preprint arXiv:2504.19139, 2025.

Yun Qu, Qi Wang, Yixiu Mao, Vincent Tao Hu, Björn Ommer, and Xiangyang Ji. Can prompt difficulty be online predicted for accelerating rl finetuning of reasoning models? In Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 1, pages 1240–1250, 2026a.

Yun Qu, Qi Wang, Yixiu Mao, Heming Zou, Yuhang Jiang, Yingyue Li, Wutong Xu, Lizhou Cai, Weijie Liu, Clive Bai, et al. Listwise policy optimization: Group-based rlvr as target-projection on the llm response simplex. arXiv preprint arXiv:2605.06139, 2026b.

Yun Qu, Qi Wang, Yixiu Mao, Heming Zou, Yuhang Jiang, Weijie Liu, Clive Bai, Kai Yang, Yangkun Chen, Saiyong Yang, et al. Small generalizable prompt predictive models can steer efficient rl post-training of large reasoning models. arXiv preprint arXiv:2602.01970, 2026c.

Banafsheh Rafiee and Richard Sutton. Toward enactive artificial intelligence. arXiv preprint arXiv:2605.24238, 2026.

Davis Rempe, Jonah Philion, Leonidas J Guibas, Sanja Fidler, and Or Litany. Generating useful accident-prone driving scenarios via a learned traffic prior. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 17284–17294. IEEE, 2022.

R Tyrrell Rockafellar, Stanislav Uryasev, et al. Optimization of conditional value-at-risk. Journal of risk, 2:21–42, 2000.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Qianli Shen, Daoyuan Chen, Yilun Huang, Zhenqing Ling, Yaliang Li, Bolin Ding, and Jingren Zhou. Bots: A unified framework for bayesian online task selection in llm reinforcement finetuning. In International Conference on Learning Representations, volume 2026, pages 99871–99911, 2026.

David Silver and Richard S Sutton. Welcome to the era of experience. Google AI, 1(11):6, 2025.

David Silver, Thomas Hubert, Julian Schrittwieser, Ioannis Antonoglou, Matthew Lai, Arthur Guez, Marc Lanctot, Laurent Sifre, Dharshan Kumaran, Thore Graepel, et al. A general reinforcement learning algorithm that masters chess, shogi, and go through self-play. Science, 362(6419):1140–1144, 2018.

Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. Scaling llm test-time compute optimally can be more effective than scaling model parameters. arXiv preprint arXiv:2408.03314, 2024.

Richard Sutton. The bitter lesson. Incomplete Ideas (blog), 13(1):38, 2019.

Naftali Tishby, Fernando C Pereira, and William Bialek. The information bottleneck method. arXiv preprint physics/0004057, 2000.

Josh Tobin, Rachel Fong, Alex Ray, Jonas Schneider, Wojciech Zaremba, and Pieter Abbeel. Domain randomization for transferring deep neural networks from simulation to the real world. In 2017 IEEE/RSJ international conference on intelligent robots and systems (IROS), pages 23–30. IEEE, 2017.

31

<!-- page 32 of 45 -->

TypeSafe AI. System One. TypeSafe documentation; describes Jev as “the first System One model”, 2026. URL [https://docs.typesafe.ai/concepts/system-one](https://docs.typesafe.ai/concepts/system-one). Accessed October 2026.

Francisco J Varela, Evan Thompson, and Eleanor Rosch. The embodied mind: Cognitive science and human experience. The MIT press, 2017.

Pablo Villalobos, Anson Ho, Jaime Sevilla, Tamay Besiroglu, Lennart Heim, and Marius Hobbhahn. Position: Will we run out of data? limits of llm scaling based on human-generated data. In Forty-first International Conference on Machine Learning, 2024.

Qi Wang, Yiqin Lv, Zheng Xie, Jincai Huang, et al. A simple yet effective strategy to robustify the meta learning paradigm. Advances in Neural Information Processing Systems, 36:12897–12928, 2023.

Qi Cheems Wang, Zehao Xiao, Yixiu Mao, Yun Qu, Jiayi Shen, Yiqin Lv, and Xiangyang Ji. Model predictive task sampling for efficient and robust adaptation. Nature Communications, 2026.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.

Jiarong Wen, Qi Tao, Kaiyu Zhang, Yun Qu, Lizhou Cai, Heming Zou, Yuhang Jiang, and Qi Wang. Bridging risk approximation gaps in model predictive task sampling via in-context modeling. arXiv preprint arXiv:2610.14913, 2026a.

Jiarong Wen, Qi Wang, Yun Qu, Yixiu Mao, Heming Zou, Haoang Chi, Lizhou Cai, Yiqin Lv, Kaiyu Zhang, Yuhang Jiang, et al. You only edit once: Incentivizing in-context capability of llms via local demonstration refinement. arXiv preprint arXiv:2609.33609, 2026b.

Zhangchen Xu, Fengqing Jiang, Luyao Niu, Yuntian Deng, Radha Poovendran, Yejin Choi, and Bill Yuchen Lin. Magpie: Alignment data synthesis from scratch by prompting aligned llms with nothing. In International Conference on Learning Representations, volume 2025, pages 76346–76382, 2025.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. Advances in Neural Information Processing Systems, 38:113222–113244, 2026.

Haizhong Zheng, Yang Zhou, Brian Bartoldson, Bhavya Kailkhura, Fan Lai, Jiawei Zhao, and Beidi Chen. Act only when it pays: Efficient reinforcement learning for llm reasoning via selective rollouts. Advances in Neural Information Processing Systems, 38: 124321–124346, 2026.

Shiji Zhou, Lianzhe Wang, Shanghang Zhang, Zhi Wang, and Wenwu Zhu. Active gradual domain adaptation: Dataset and approach. IEEE Transactions on Multimedia, 24:1210–1220, 2022a.

Shiji Zhou, Han Zhao, Shanghang Zhang, Lianzhe Wang, Heng Chang, Zhi Wang, and Wenwu Zhu. Online continual adaptation with active self-training. In International conference on artificial intelligence and statistics, pages 8852–8883. PMLR, 2022b.

Heming Zou, Yixiu Mao, Yun Qu, Qi Wang, and Xiangyang Ji. Utility-diversity aware online batch selection for llm supervised fine-tuning. arXiv preprint arXiv:2510.16882, 2025.

Heming Zou, Qi Wang, Yixiu Mao, Yun Qu, Yuyi Guo, Zeyuan Liu, Chenglin Li, Weijie Liu, Saiyong Yang, and Xiangyang Ji. Tree-aware posterior distillation for long-horizon agentic reinforcement learning. arXiv preprint arXiv:2610.23181, 2026a.

Heming Zou, Qi Wang, Yun Qu, Yuhang Jiang, Lizhou Cai, Yixiu Mao, Ru Peng, Xin Xu, Weijie Liu, Kai Yang, et al. Trace: A unified rollout budget allocation framework for efficient agentic reinforcement learning. arXiv preprint arXiv:2606.11119, 2026b.

32

<!-- page 33 of 45 -->

## A Proofs

Each subsection below restates a result exactly as it appears in the main text, keeping its original number, and proves it step by step. All results are proved for a general declared utility 𝑢 and its density of utility-relevant interactions $\rho _ { u }$ (Eq. (14)); the informative interaction density $\rho _ { \mathrm { i n f o } }$ is the special case in which 𝑢 is the expected information gain, and every proof below specializes to it by reading “utility-relevant” as “informative”. We write $u ( i ) = u ( i ; \theta _ { t } )$ for the true utility of interaction 𝑖 and $\hat { u } ( i ) = \hat { u } _ { \phi } ( i \mid \mathcal { H } _ { 0 : t } )$ for its prediction, suppressing arguments not in play. Expectations are over stochastic gradients, predictor outputs, and the interaction stream unless noted. Together, these results make precise what co-design buys and what it risks.

## A.1 Proof of Theorem 2

**Theorem 2 (restated).** Let I be finite and $\kappa _ { t } \leq \hat { u } _ { t } ^ { \operatorname* { m a x } }$ , so that every maximizer is admissible. $A s   \beta \to \infty ,$

$$
\pi _ { \mathrm { C I } } ( i _ { t } \mid \mathcal { H } _ { 0 : t } ) \longrightarrow \frac { \mathbf { 1 } \big [ i _ { t } \in \mathcal { I } _ { t } ^ { \star } \big ] } { | \mathcal { I } _ { t } ^ { \star } | } ,\tag{5}
$$

the greedy rule $\begin{array} { r } { i _ { t } = \arg \operatorname* { m a x } _ { i } \hat { u } ( i \mid \mathcal { H } _ { 0 : t } ) } \end{array}$ with ties broken uniformly. If the maximizer $i _ { t } ^ { \star }$ is unique, with gap $\Delta _ { t } > 0$ to the best other admissible candidate, then

$$
1 - \pi _ { \mathrm { C I } } \big ( i _ { t } ^ { \star } \mid \mathcal { H } _ { 0 : t } \big ) \leq \big ( | \mathcal { I } _ { t } ^ { \mathrm { a d m } } | - 1 \big )   e ^ { - \beta \Delta _ { t } } ,\tag{6}
$$

so, when $| \mathcal { I } _ { t } ^ { \mathrm { a d m } } | \geq 2 ,   \beta \geq \Delta _ { t } ^ { - 1 } \ln \bigl ( ( | \mathcal { I } _ { t } ^ { \mathrm { a d m } } | - 1 ) / \delta \bigr )$ selects $i _ { t } ^ { \star }$ with probability at least $1 - \delta ;$ when $| \mathcal { I } _ { t } ^ { \mathrm { a d m } } |   =   1$ it is selected with probability one for every $\beta .$

Proof. **Step 1 (Restrict the policy to the admissible set).** The indicator in Eq. (3) vanishes outside $\varUpsilon _ { t } ^ { \mathrm { a d m } }$ , so for every $i \in { \cal I }$

$$
\pi _ { \mathrm { C I } } \big ( i \mid \mathcal { H } _ { 0 : t } \big ) = \frac { \mathbf { 1 } \big [ i \in \mathcal { I } _ { t } ^ { \mathrm { a d m } } \big ] e ^ { \beta \hat { u } _ { t } ( i ) } } { \sum _ { j \in \mathcal { I } _ { t } ^ { \mathrm { a d m } } } e ^ { \beta \hat { u } _ { t } ( j ) } } .
$$

Because $\kappa _ { t }   \leq   \hat { u } _ { t } ^ { \operatorname* { m a x } }$ , every maximizer satisfies $\hat { u } _ { t } \big ( i \big )   =   \hat { u } _ { t } ^ { \mathrm { m a x } }   \geq   \kappa _ { t }$ , hence $\mathcal { I } _ { t } ^ { \star } \subseteq \mathcal { I } _ { t } ^ { \mathrm { a d m } }$ , the denominator is at least $e ^ { \beta \hat { u } _ { t } ^ { \mathrm { m a x } } } > 0 ,$ and $Z _ { t } > 0$

**Step 2 (Normalize by the largest term).** Define the utility deficit $d _ { t } \big ( i \big ) = \hat { u } _ { t } ^ { \operatorname* { m a x } } - \hat { u } _ { t } \big ( i \big ) \geq 0 ,$ , which is zero exactly when $i \in  { \mathbb { T } _ {   t } } ^ { \star }$ . Multiplying numerator and denominator by $e ^ { - \beta \hat { u } _ { t } ^ { \mathrm { m a x } } }$

$$
\pi _ { \mathrm { C I } } \big ( i \mid \mathcal { H } _ { 0 : t } \big ) = \frac { \mathbf { 1 } \big [ i \in \mathcal { I } _ { t } ^ { \mathrm { a d m } } \big ] e ^ { - \beta d _ { t } ( i ) } } { \sum _ { j \in \mathcal { I } _ { t } ^ { \mathrm { a d m } } } e ^ { - \beta d _ { t } ( j ) } } .
$$

**Step 3 (Limit of the denominator).** Split the denominator into maximizers and the rest:

$$
\sum _ { j \in \mathcal { I } _ { t } ^ { \mathrm { a d m } } } e ^ { - \beta d _ { t } ( j ) } = \underbrace { \sum _ { j \in \mathcal { I } _ { t } ^ { \star } } 1 + \sum _ { j \in \mathcal { I } _ { t } ^ { \mathrm { a d m } } \setminus \mathcal { I } _ { t } ^ { \star } } e ^ { - \beta d _ { t } ( j ) } } _ { = | \mathcal { I } _ { t } ^ { \star } | } .
$$

Every term of the second sum has $d _ { t } ( j ) > 0$ , so it tends to 0 as $\beta \rightarrow \infty ;$ the sum is finite because I is finite. The denominator therefore converges to $| { { { \cal I } _ { t } } ^ { \star } } |$

**Step 4 (Limit of each probability).** For $i \in \mathcal { I } _ { t } ^ { \star }$ the numerator equals 1, so $\pi _ { \mathrm { C I } } ( i ) \to 1 / | \mathcal { I } _ { t } ^ { \star } |$ . For $i \in \mathcal { I } _ { t } ^ { \mathrm { a d m } } \setminus \mathcal { I } _ { t } ^ { \star }$ the numerator $e ^ { - \beta d _ { t } ( i ) } \to 0 , \mathrm { s o } \; \pi _ { \mathrm { C I } } ( i ) \to 0 .$ . For $i \notin \mathcal { I } _ { t } ^ { \mathrm { a d m } } , \pi _ { \mathrm { C I } } ( i ) = 0$ for every 𝛽. Together these give Eq. (5).

33

<!-- page 34 of 45 -->

**Step 5 (Rate for a unique maximizer).** Let $\mathcal { I } _ { t } ^ { \star }   =   \{ i _ { t } ^ { \star } \}$ and $\begin{array} { r } { S _ { \beta }   =   \sum _ { j \in \mathcal { I } _ { t } ^ { \mathrm { a d m } } \backslash \left\{ i _ { t } ^ { \star } \right\} } e ^ { - \beta d _ { t } ( j ) } } \end{array}$ . By Step $2,   \pi_{ CI }(i_{t}^{\star})   =$ $1 / ( 1 + S _ { \beta } )$ . Each of the $| \mathcal { I } _ { t } ^ { \mathrm { a d m } } | - 1$ terms of $S _ { \beta }$ has $d _ { t } ( j ) \geq \Delta _ { t }$ , so $S _ { \beta } \leq ( | \mathcal { I } _ { t } ^ { \mathrm { a d m } } | - 1 ) e ^ { - \beta \Delta _ { t } }$ . Hence

$$
1 - \pi _ { \mathrm { C I } } \big ( i _ { t } ^ { \star } \big ) = \frac { S _ { \beta } } { 1 + S _ { \beta } } \leq S _ { \beta } \leq \big ( | \mathcal { I } _ { t } ^ { \mathrm { a d m } } | - 1 \big ) e ^ { - \beta \Delta _ { t } } ,
$$

which is Eq. (6).

**Step 6 (Sufficient inverse temperature).** The right-hand side of Eq. (6) is at most 𝛿 if and only if $e ^ { - \beta \Delta _ { t } } \leq \delta / ( | \mathcal { I } _ { t } ^ { \mathrm { a d m } } | { - 1 } )$ i.e. $\beta \geq \Delta _ { t } ^ { - 1 } \ln \bigl ( ( | \mathcal { I } _ { t } ^ { \mathrm { a d m } } | - 1 ) / \delta \bigr ) . \mathrm { ~ ( I f ~ } | \mathcal { I } _ { t } ^ { \mathrm { a d m } } | = 1$ the policy is already deterministic for every $\beta . )$ □

## A.2 Proof of Theorem 7

**Theorem 7 (restated).** Assume dead candidates carry no utility $\begin{array} { r } { ( \bar { \mathbb { I } } _ { d e a d } = 0 , } \end{array}$ , as when $u _ { \mathrm { m i n } } \to 0 ^ { + }$ for a non-negative utility), and use the system-level ledger of Eq. (9), in its population form Eq. (11), which charges a predictor cost $c _ { p r e d }$ to every screened candidate.

**1.** (Perfect, free predictor.) If exactly the utility-relevant candidates are executed and $c _ { p r e d } = 0 ,$

$$
\frac { \mathrm { I I E } ^ { \mathrm { C I } } } { \mathrm { I I E } ^ { u n c } } = \frac { \bar { \mathbb { I } } _ { r e l } / \bar { C } _ { r e l } } { \bar { \mathbb { I } } / \bar { C } } = \frac { 1 } { \rho _ { u } } \cdot \frac { \bar { C } } { \bar { C } _ { r e l } } = 1 + \frac { 1 - \rho _ { u } } { \rho _ { u } } \cdot \frac { \bar { C } _ { d e a d } } { \bar { C } _ { r e l } } ,\tag{12}
$$

which equals $1 / \rho _ { u }$ when $\bar { C } _ { d e a d } = \bar { C } _ { r e l }$ and exceeds it if and only $i f \bar { C } _ { d e a d } > \bar { C } _ { r e l } .$

**2.** (Imperfect, costly predictor.) With class-conditional false-positive rate fp (the fraction of dead candidates admitted) and false-negative rate fn (the fraction of utility-relevant candidates suppressed),

$$
\frac { \mathrm { I I E } ^ { \mathrm { C I } } } { \mathrm { I I E } ^ { u n c } } = \frac { \bar { \mathbb { I } } _ { r e l } / \bar { C } _ { r e l } } { \bar { \mathbb { I } } / \bar { C } } \cdot \left( 1 + \frac { \mathrm { f p } \left( 1 - \rho _ { u } \right) \bar { C } _ { d e a d } + c _ { p r e d } } { \left( 1 - \mathrm { f n } \right) \rho _ { u }   \bar { C } _ { r e l } } \right) ^ { - 1 } ,\tag{13}
$$

which is non-increasing in each of fp, fn and $c _ { p r e d } ,$ strictly decreasing in any of them whose coefficient is positive, and recovers Eq. (12) as $\mathrm { f p } , c _ { p r e d } \rightarrow 0$

Proof. **Step 1 (Per-candidate accounting).** We work with the population efficiency of Eq. (11), the ratio of the expected utility per screened candidate to the expected cost per screened candidate; the finite-stream $\mathrm { I I E } _ { T }$ converges to it almost surely for an i.i.d. stream, and for a fixed pool the identities below hold exactly with empirical fractions and class means. We normalize everything per candidate. Recall $\bar { \mathbb { I } } = \rho _ { u } \bar { \mathbb { I } } _ { \mathrm { r e l } } + \big ( 1 - \rho _ { u } \big ) \bar { \mathbb { I } } _ { \mathrm { d e a d } }$ and $\bar { C } = \rho _ { u } \bar { C } _ { \mathrm { r e l } } + ( 1 - \rho _ { u } ) \bar { C } _ { \mathrm { d e a d } }$ and the clean case sets $\bar { \mathbb { I } } _ { \mathrm { d e a d } } = 0$

34

**Step 2 (Unconditional interaction).** Unconditional interaction executes every candidate, so its utility and cost per candidate are Ī and ${ \bar { C } } ,$ and $\mathrm { I I E ^ { u n c } } = \bar { \mathbb { I } } / \bar { C }$

**Step 3 (Utility under the gate).** The gate executes a fraction 1 − fn of the utility-relevant candidates (mass $( 1 - \mathrm { f n } ) \rho _ { u } )$ and a fraction fp of the dead ones (mass $\operatorname { f p } ( 1 - \rho _ { u } ) )$ . In the clean case the latter add no utility, so the utility per candidate is

$$
\mathbb { S } _ { \mathrm { C I } } = \left( 1 - \mathrm { f n } \right) \rho _ { u }   \bar { \mathbb { I } } _ { \mathrm { r e l } } .
$$

**Step 4 (Cost under the gate).** Executed utility-relevant candidates cost $\bar { C } _ { \mathrm { r e l } }$ each, executed dead candidates $\bar { C } _ { \mathrm { d e a d } }$ each, and every candidate is scored at $c _ { \mathrm { p r e d } } ,$ , so

$$
\mathbb { C } _ { \mathrm { C I } } = \left( 1 - \mathrm { f n } \right) \rho _ { u }   \bar { C } _ { \mathrm { r e l } } + \mathrm { f p } \left( 1 - \rho _ { u } \right) \bar { C } _ { \mathrm { d e a d } } + c _ { \mathrm { p r e d } } .
$$

<!-- page 35 of 45 -->

**Step 5 (Factor out the utility-relevant cost).** Dividing $\mathbb { S } _ { \mathrm { C I } }$ and $\mathbb { C } _ { \mathrm { C I } }$ by $\mathrm { ( 1 - f n ) } \rho _ { u } \bar { C } _ { \mathrm { r e l } }$ (assume fn < 1),

$$
\mathrm { I I E } ^ { \mathrm { C I } } = \frac { \mathbb { S } _ { \mathrm { C I } } } { \mathbb { C } _ { \mathrm { C I } } } = \frac { \bar { \mathbb { I } } _ { \mathrm { r e l } } } { \bar { C } _ { \mathrm { r e l } } } \big ( 1 + g \big ) ^ { - 1 } , \qquad g = \frac { \mathrm { f p }   ( 1 - \rho _ { u } )   \bar { C } _ { \mathrm { d e a d } } + c _ { \mathrm { p r e d } } } { ( 1 - \mathrm { f n } )   \rho _ { u }   \bar { C } _ { \mathrm { r e l } } } .
$$

Dividing by $\mathrm { { I I E ^ { u n c } } }$ from Step 2 gives Eq. (13).

**Step 6 (Monotonicity).** Write the ratio as $K ( 1 + g ) ^ { - 1 }$ with $K = \big ( \bar { \mathbb { I } } _ { \mathrm { r e l } } / \bar { C } _ { \mathrm { r e l } } \big ) / \big ( \bar { \mathbb { I } } / \bar { C } \big )$ independent of the error rates. The map $g \mapsto ( 1 + g ) ^ { - 1 }$ is decreasing, and 𝑔 is non-decreasing in fp (coefficient $( 1 - \rho _ { u } ) \bar { C } _ { \mathrm { d e a d } } \geq 0 )$ , in $c _ { \mathrm { p r e d } }$ (coefficient positive), and in fn (through $( 1 - \mathrm { f n } ) ^ { - 1 }$ , which increases on [0, 1)). Hence the ratio is non-increasing in each, and strictly decreasing whenever the corresponding coefficient is positive.

**Step 7 (Ideal case).** With $\mathrm { f } \mathtt { p } = c _ { \mathrm { p r e d } } = 0 ,   g = 0$ and the ratio equals 𝐾, the first equality of Eq. (12). Substituting $\bar { \mathbb { I } } = \rho _ { u } \bar { \mathbb { I } } _ { \mathrm { r e l } }$

$$
K = \frac { \bar { \mathbb { I } } _ { \mathrm { r e l } } } { \bar { C } _ { \mathrm { r e l } } } \cdot \frac { \bar { C } } { \rho _ { u } \bar { \mathbb { I } } _ { \mathrm { r e l } } } = \frac { 1 } { \rho _ { u } } \cdot \frac { \bar { C } } { \bar { C } _ { \mathrm { r e l } } } ,
$$

the second equality. Substituting $\bar { C } ,$

$$
\frac { 1 } { \rho _ { u } } \cdot \frac { \rho _ { u } \bar { C } _ { \mathrm { r e l } } + ( 1 - \rho _ { u } ) \bar { C } _ { \mathrm { d e a d } } } { \bar { C } _ { \mathrm { r e l } } } = 1 + \frac { 1 - \rho _ { u } } { \rho _ { u } } \cdot \frac { \bar { C } _ { \mathrm { d e a d } } } { \bar { C } _ { \mathrm { r e l } } } ,
$$

the third equality.

**Step 8 (Comparison with** $1 / \rho _ { u } )$ . Subtracting, $\begin{array} { r } { K - \frac { 1 } { \rho _ { u } } = \frac { 1 - \rho _ { u } } { \rho _ { u } } \big ( \frac { \bar { C } _ { \mathrm { d e a d } } } { \bar { C } _ { \mathrm { r e l } } } - 1 \big ) } \end{array}$ , which for $\rho _ { u } \in ( 0 , 1 )$ is zero if and only if $\bar { C } _ { \mathrm { d e a d } } = \bar { C } _ { \mathrm { r e l } }$ and positive if and only if $\bar { C } _ { \mathrm { d e a d } } > \bar { C } _ { \mathrm { r e l } }$

**Step 9 (Dead candidates with positive utility).** If the zero-utility assumption is dropped, $\bar { \mathbb { I } } _ { \mathrm { d e a d } } \geq 0 ,$ , the perfect free predictor still has $\mathrm { H E } ^ { \mathrm { C I } }   =   \rho _ { u } \bar { \mathbb { I } } _ { \mathrm { r e l } } / ( \rho _ { u } \bar { C } _ { \mathrm { r e l } } )$ , so the first equality of Eq. (12) holds unchanged. Since $\rho _ { u } \bar { \mathbb { I } } _ { \mathrm { r e l } } =$ $\bar { \mathbb { I } } - ( 1 - \rho _ { u } ) \bar { \mathbb { I } } _ { \mathrm { d e a d } }$

$$
K = \frac { \bar { C } } { \bar { C } _ { \mathrm { r e l } } } \cdot \frac { \bar { \mathbb { I } } _ { \mathrm { r e l } } } { \bar { \mathbb { I } } } = \frac { 1 } { \rho _ { u } } \cdot \frac { \bar { C } } { \bar { C } _ { \mathrm { r e l } } } \bigg ( 1 - \frac { ( 1 - \rho _ { u } )   \bar { \mathbb { I } } _ { \mathrm { d c a d } } } { \bar { \mathbb { I } } } \bigg ) \leq \frac { 1 } { \rho _ { u } } \cdot \frac { \bar { C } } { \bar { C } _ { \mathrm { r e l } } } ,
$$

with equality if and only $\mathrm { i f } \; \bar { \mathbb { I } } _ { \mathrm { d e a d } } = 0 \; ( \mathrm { f o r } \; \rho _ { u } < 1 )$ . The closed forms of Eq. (12) are then upper bounds; this is the relevant case when the floor $u _ { \mathrm { m i n } }$ of Eq. (14) is strictly positive. □

Remarks. False negatives affect the ratio only through $g ;$ the IIE is utility per unit cost, and suppressed utility-relevant candidates are neither paid for nor counted. Their main effect is on yield, $\mathbb { S } _ { \mathrm { C I } } ,$ , which falls by the factor 1 − fn and determines how large a pool is needed to reach a target. In the notation of Eq. (10), $\mathbb { S } _ { \mathrm { C I } }$ is the expected yield. Setting $c _ { \mathrm { p r e d } } = 0$ recovers execution-only efficiency.

## A.3 Proof of Theorem 14

**Theorem 14 (restated).** Let a pool of 𝑛 candidates contain a fraction $\rho _ { u }$ of utility-relevant ones, with execution cost 𝑐<sub>exec</sub>, screening cost $c _ { s c r e e n } = c _ { e m b e d } + c _ { s c o r e }$ (encoding the candidate and scoring it; this is the $c _ { p r e d }$ of Eq. (9)), and $r = c _ { s c r e e n } / c _ { e x e c } .$ . With a perfect screen, acquiring the utility-relevant subset costs

$$
\frac { C _ { \mathrm { C I } } } { C _ { r e a c t } } = \rho _ { u } + r\tag{24}
$$

35

of what exhaustive execute-then-filter evaluation spends $( C _ { r e a c t } = n   c _ { e x e c } ,$ the baseline that must execute every candidate to find all relevant ones), a speedup approaching $1 / \rho _ { u }$ as $r \rightarrow 0 ,$ , provided a candidate’s utility class cannot be known without executing or scoring it.

<!-- page 36 of 45 -->

Proof. **Step 1 (Information model).** We assume that a candidate’s utility class (relevant or dead) cannot be learned without either executing it (reactive evaluation) or scoring it with the predictor (conditional interaction), and that screening costs $c _ { \mathrm { s c r e e n } } = c _ { \mathrm { e m b e d } } + c _ { \mathrm { p r e d } } ;$ the predictor head costs $c _ { \mathrm { s c o r e } }   =   \Theta ( d _ { \phi } ^ { 2 } )$ because it acts on a fixed-size embedding, and encoding the candidate costs $c _ { \mathrm { e m b e d } } ,$ zero when embeddings are precomputed and cached; neither depends on the rollout length 𝐿 or the group size 𝐺.

**Step 2 (Reactive cost and its lower bound).** A reactive scheme has no signal about an unexecuted candidate. If it skipped a candidate, it could not tell whether that candidate is utility-relevant, so it could not guarantee to acquire the full utility-relevant subset. It must therefore execute all 𝑛 candidates:

$$
C _ { \mathrm { r e a c t } } = n   c _ { \mathrm { e x e c } } = \Theta ( n   G   L   d _ { \mathrm { m o d e l } } ^ { 2 } ) ,
$$

which matches Eq. (25). Under the information model of Step 1, no execute-then-filter scheme that must guarantee the full utility-relevant subset can spend less; schemes that sample adaptively, use side information or accept losses fall outside this baseline.

**Step 3 (Conditional cost).** Conditional interaction encodes and scores all 𝑛 candidates, at cost 𝑛 $c _ { \mathrm { s c r e e n } }$ . A perfect screen then executes exactly the $\rho _ { u } n$ utility-relevant candidates, at cost $\rho _ { u } n   c _ { \mathrm { e x e c } }$ . Hence

$$
C _ { \mathrm { C I } } = n   c _ { \mathrm { s c r e e n } } + \rho _ { u }   n   c _ { \mathrm { e x e c } } ,
$$

which matches Eq. (26) with $B = \rho _ { u } n$

**Step 4 (Ratio and speedup).** Dividing Step 3 by Step 2,

$$
\frac { C _ { \mathrm { C I } } } { C _ { \mathrm { r e a c t } } } = \frac { n   c _ { \mathrm { s c r e e n } } + \rho _ { u } n   c _ { \mathrm { e x e c } } } { n   c _ { \mathrm { e x e c } } } = \rho _ { u } + \frac { c _ { \mathrm { s c r e e n } } } { c _ { \mathrm { e x e c } } } = \rho _ { u } + r .
$$

The speedup is the reciprocal $1 / ( \rho _ { u } + r )$ , which increases to $1 / \rho _ { u }$ as $r \rightarrow 0$

## A.4 Proof of Theorem 15 and Theorem 16

**Theorem 15 (restated).** To select the top-𝐵 of 𝑛 candidates, synchronous reactive evaluation needs peak memory $M _ { r e a c t } ( n )$ , whereas predictive scoring streamed in micro-batches of size 𝑏 needs $M _ { \mathrm { C I } } ( n )$ , where

$$
M _ { r e a c t } ( n ) = \Theta \big ( n   G   L   d _ { m o d e l } \big ) ,\tag{27}
$$

$$
M _ { \mathrm { C I } } ( n ) = \Theta \big ( B   G   L   d _ { m o d e l } + b   d _ { e } + B \big ) ,\tag{28}
$$

independent of 𝑛; the pool is then limited only by screening time $\Theta ( n   c _ { p r e d } )$ and score storage.

Proof. **Step 1 (Assumptions).** (i) Synchronous reactive evaluation keeps each candidate’s in-flight state resident until all outcomes can be compared; this state is $\Theta ( G L d _ { \mathrm { m o d e l } } )$ per candidate (the KV cache of 𝐺 length-𝐿 generations, or trajectory and simulator state). (ii) Predictive scoring needs only the embedding $e ( \tau ) \in \mathbb { R } ^ { d _ { e } }$ and returns a scalar. (iii) Scoring has no side effects.

**Step 2 (Reactive peak memory).** By (i), at the moment outcomes are compared all 𝑛 states are resident, so the peak memory is $n \cdot \Theta ( G L d _ { \mathrm { m o d e l } } )$ , giving Eq. (27).

**Step 3 (Reactive pool cap).** Requiring $n   \Theta ( G L d _ { \mathrm { m o d e l } } ) \leq M   \mathrm { g i v e s }   n \leq n _ { \mathrm { m a x } } ^ { \mathrm { r e a c t } } = \Theta ( M / ( G L d _ { \mathrm { m o d e l } } ) )$

**Step 4 (Predictive peak memory).** Process candidates in micro-batches of size 𝑏. By (ii) and (iii), each embedding can be discarded once its scalar score is recorded, so at most 𝑏 embeddings are resident at once: $\Theta ( b   d _ { e } )$ . The scores feed a running top-𝐵 structure: Θ(𝐵). The only large state is that of the 𝐵 executed survivors: $\Theta ( B G L d _ { \mathrm { m o d e l } } )$ Summing the three terms gives Eq. (28), and none of them depends on 𝑛.

**Step 5 (What still depends on** 𝑛**).** Screening all candidates takes time $\Theta ( n   c _ { \mathrm { p r e d } } )$ , and storing every score takes Θ(𝑛) memory (or Θ(𝐵) if only the running top-𝐵 is kept). These, not rollout memory, bound the pool. □

36

<!-- page 37 of 45 -->

Remark (streaming reactive evaluation). If assumption (i) is dropped, a reactive filter can also execute candidates in micro-batches and keep only their scores, giving 𝑛-independent peak memory at $\Theta ( n   c _ { \mathrm { e x e c } } )$ time. The memory advantage is thus genuine for synchronous pipelines (e.g. group-based advantage estimation holding all 𝐺 rollouts of a prompt together, or batched simulators), and otherwise reduces to the time advantage of Theorem 14.

**Theorem 16 (restated).** The best true utility $U _ { ( 1 ) } ^ { n }$ among 𝑛 i.i.d. candidates is non-decreasing in 𝑛: it converges to the essential supremum if utility is bounded, and grows as $n ^ { \xi }$ (in probability, up to a slowly varying factor) if $\operatorname* { P r } ( U > u ) = u ^ { - 1 / \xi } \ell ( u )$ . Under predictive sampling this gain costs no additional rollout memory (Eq. (28)), provided the predictor ranks the top of the pool correctly.

Proof. Let $U _ { 1 } , U _ { 2 } , \ldots$ . be i.i.d. candidate utilities and $U _ { ( 1 ) } ^ { n } = \operatorname* { m a x } _ { i \leq n } U _ { i } .$

**Step 1 (Monotonicity).** Adding a candidate cannot lower the maximum: $U _ { ( 1 ) } ^ { n + 1 } = \operatorname* { m a x } ( U _ { ( 1 ) } ^ { n } , U _ { n + 1 } ) \geq U _ { ( 1 ) } ^ { n }$ pointwise, so $U _ { ( 1 ) } ^ { n }$ is non-decreasing in 𝑛, pointwise and hence stochastically.

**Step 2 (Bounded utility).** Let $u ^ { + }$ be the essential supremum. For any $s > 0 ,   p _ { s } = \operatorname* { P r } ( U \leq u ^ { + } - s ) < 1$ by definition of $u ^ { + }$ , and independence gives $\operatorname* { P r } ( U _ { ( 1 ) } ^ { n } \leq u ^ { + } - s ) = p _ { s } ^ {   n } \to 0$ . Hence $U _ { ( 1 ) } ^ { n } \rightarrow u ^ { + }$ in probability, and almost surely because the sequence is monotone.

**Step 3 (Heavy upper tail).** Suppose $\operatorname* { P r } ( U \; > \; u ) \; = \; u ^ { - 1 / \xi } \ell ( u )$ with ℓ slowly varying, and choose $a _ { n }$ with $\Pr(U > a_n) = 1/n;$ then $a _ { n }$ is regularly varying in 𝑛 with index 𝜉. For 𝑥 > 0,

$$
\begin{array} { r } { \operatorname* { P r } \bigl ( U _ { ( 1 ) } ^ { n } \leq x   a _ { n } \bigr ) = \bigl ( 1 - \operatorname* { P r } ( U > x   a _ { n } ) \bigr ) ^ { n } = \biggl ( 1 - \frac { x ^ { - 1 / \xi } \ell ( x a _ { n } ) } { n   \ell ( a _ { n } ) } \biggr ) ^ { n } \longrightarrow \exp \bigl ( - x ^ { - 1 / \xi } \bigr ) , } \end{array}
$$

using $\ell ( x a _ { n } ) / \ell ( a _ { n } ) \to 1$ . This is the Fréchet law, so $U _ { ( 1 ) } ^ { n } / a _ { n }$ converges in distribution and $U _ { ( 1 ) } ^ { n }$ grows as $n ^ { \xi }$ up to a slowly varying factor, in probability. For $\xi > 1$ the mean of 𝑈 is infinite (for $\xi = 1$ it may be, depending on ℓ), so the growth statement is made in probability, not in expectation.

**Step 4 (Cost of realizing the growth).** By Theorem 15, enlarging the pool leaves peak rollout memory at Eq. (28); by Theorem 14, each extra candidate adds only $c _ { \mathrm { p r e d } }$ . The growth of Steps 2 and 3 concerns the best true utility in the pool; it is realized by the selection only if the predictor ranks the top of the pool correctly, which is quantified by Theorems 18 and 19. □

## A.5 Proof of Theorem 20

**Theorem 20 (restated).** Write the expected free energy of interaction 𝑖 as

$$
\begin{array} { r } { G ( i ) = \underbrace { - \: \mathbb { E } _ { q _ { \phi } ( o \mid i ) } \big [ \mathrm { l n } \: p ( o \mid C ) \big ] } _ { p r a g m a i c \: v a l u e } - \underbrace { \mathbb { E } _ { q _ { \phi } ( o \mid i ) } \big [ D _ { \mathrm { K L } } \left[ q _ { \phi } ( \theta \mid o , i ) \: \middle \| \: q _ { \phi } ( \theta \mid i ) \right] \big ] } _ { e p i s t e m i c \: v a l u e } , } \end{array}\tag{35}
$$

with preferences $p ( o \mid C )$ over a finite outcome space O. If preferences are uniform and cost is accounted separately through $C ( i )$ , then

$$
\mathop { \arg \operatorname* { m i n } } _ { i \in \mathcal { I } } G ( i ) = \mathop { \arg \operatorname* { m a x } } _ { i \in \mathcal { I } } I _ { q _ { \phi } } ( \theta ; o \mid i ) = \mathop { \arg \operatorname* { m a x } } _ { i \in \mathcal { I } } \mathcal { U } _ { \mathrm { I G } } ^ { q _ { \phi } } ( i \mid \mathcal { H } _ { t } ) ,\tag{36}
$$

where $\mathcal { U } _ { \mathrm { I G } } ^ { q _ { \phi } }$ is Eq. (32) evaluated under the variational model $q _ { \phi } ,$ , and coincides with $\mathcal { U } _ { \mathrm { I G } }$ when $q _ { \phi }$ is the exact posterior and predictive. Under non-uniform preferences the two criteria diverge.

Proof. **Step 1 (Well-posedness).** Assume $q _ { \phi } ( \theta \mid o , i )$ is absolutely continuous with respect to $q _ { \phi } ( \theta \mid i )$ for every 𝑜 and 𝑖, so all divergences below are finite.

37

<!-- page 38 of 45 -->

38

**Step 2 (The pragmatic term is constant).** Under uniform preferences $p ( o \mid C ) = 1 / | O |$ , so − ln $p ( o \mid C ) = \ln | O |$ for every 𝑜, and its expectation under any $q _ { \phi } ( o \mid i )$ is ln |O |, independent of 𝑖.

**Step 3 (Reduce to the epistemic term).** Subtracting a constant does not change a minimizer, and minimizing the negative of a quantity is maximizing it, so by Eq. (35)

$$
\begin{align*}\underset{i}{\arg\min}   G(i) &= \underset{i}{\arg\min} \Big( - \mathbb{E}_{q_{\phi}(o \mid i)} \Big[ D_{\mathrm{KL}} \Big[ q_{\phi}(\theta \mid o, i) \left\|   q_{\phi}(\theta \mid i) \right] \Big] \Big) \\&= \underset{i}{\arg\max}   \mathbb{E}_{q_{\phi}(o \mid i)} \Big[ D_{\mathrm{KL}} \Big[ q_{\phi}(\theta \mid o, i) \left\|   q_{\phi}(\theta \mid i) \right] \Big].\end{align*}\tag{58}
$$

**Step 4 (The epistemic term is a mutual information).** Write the divergence as an expectation and use Bayes’ rule, $q _ { \phi } ( \theta \mid o , i ) / q _ { \phi } ( \theta \mid i ) = q _ { \phi } ( \theta , o \mid i ) / [ q _ { \phi } ( \theta \mid i )   q _ { \phi } ( o \mid i ) ]$ :

$$
\mathbb { E } _ { q _ { \phi } ( o \mid i ) } \biggl [ \mathbb { E } _ { q _ { \phi } ( \theta \mid o , i ) } \ln \frac { q _ { \phi } ( \theta \mid o , i ) } { q _ { \phi } ( \theta \mid i ) } \biggr ] = \mathbb { E } _ { q _ { \phi } ( \theta , o \mid i ) } \biggl [ \ln \frac { q _ { \phi } ( \theta , o \mid i ) } { q _ { \phi } ( \theta \mid i )   q _ { \phi } ( o \mid i ) } \biggr ] = I _ { q _ { \phi } } ( \theta ; o \mid i ) .\tag{59}
$$

**Step 5 (Identify the utility).** By definition, $\mathcal { U } _ { \mathrm { I G } } ^ { q _ { \phi } } ( i \mid \mathcal { H } _ { t } )$ is the expected posterior-to-prior divergence of Eq. (32) with $p$ replaced by $q _ { \phi } ,$ which by Eq. (59) equals $I _ { q _ { \phi } } ( \theta ; o \mid i )$ . Combining Steps 3 and 4 gives Eq. (36). If $q _ { \phi }$ is the exact posterior and predictive, $\mathcal { U } _ { \mathrm { I G } } ^ { q _ { \phi } } = \mathcal { U } _ { \mathrm { I G } } ;$ otherwise the maximizer is that of the variational surrogate, not necessarily of the true information gain.

**Step 6 (Non-uniform preferences).** If $p ( o \mid C )$ is not uniform, the pragmatic term depends on 𝑖 through $q _ { \phi } ( o \mid i )$ , it no longer drops out in Step 3, and the minimizer of 𝐺 trades information gain against preference satisfaction. □

## A.6 Proof of Theorem 18 and Theorem 19

**Theorem 18 (restated).** Assume (i) selection is the deterministic limit of the fixed-size batch rule $E q . ~ ( 4 7 )$ with $\mu = 0$ and A the predicted utility 𝑢ˆ, the cardinality-constrained form of the master objective Eq. (15), so each round retains the top-𝐵 set by $\hat { u } ;   ( i i ) \; \varepsilon _ { \phi } = \operatorname { s u p } _ { 0 \leq t < T } \operatorname { s u p } _ { i \in \mathcal { I } _ { t } } | \hat { u } _ { t } ( i ) - u _ { t } ( i ) | ,$ , a time-uniform error, with $\hat { u } _ { t } ( i ) = \hat { u } _ { \phi _ { t } } ( i \mid \mathcal { H } _ { 0 : t } ) ,$ ; and (iii) $\begin{array} { r } { \mathrm { g a p } _ { \operatorname* { m i n } } = \operatorname* { m i n } _ { t } \operatorname* { m i n } _ { i \in \mathcal { B } _ { t } ^ { * } , j \in \mathcal { I } _ { t } \backslash \mathcal { B } _ { t } ^ { * } } ( u _ { t } ( i ) - u _ { t } ( j ) ) > 0 . } \end{array}$ . Then

$$
\mathcal { R } _ { T } ( \pi _ { \mathrm { C I } } , \pi ^ { * } ) \; \leq \; 2   T   B   \varepsilon _ { \phi }   \cdot   { \bf 1 } \big [ 2 \varepsilon _ { \phi } \geq \mathrm { g a p } _ { \mathrm { m i n } } \big ] \; ;\tag{30}
$$

in particular, the selection is exactly the oracle’s whenever $2 \varepsilon _ { \phi } < \mathrm { g a p } _ { \mathrm { m i n } } .$ . More generally, with round-specific errors $\varepsilon _ { t } = \operatorname { s u p } _ { i \in \mathcal { I } _ { t } } \left| \hat { u } _ { t } \big ( i \big ) - u _ { t } \big ( i \big ) \right|$ and gaps gap<sub>𝑡</sub>, $\begin{array} { r } { \mathcal { R } _ { T } \leq \sum _ { t < T } 2 B \varepsilon _ { t }   \mathbf { 1 } \big [ 2 \varepsilon _ { t } \geq \mathrm { g a p } _ { t } \big ] } \end{array}$

Proof. **Step 1 (Per-round regret).** By Eq. (29), $\begin{array} { r } { \mathcal { R } _ { T }   =   \sum _ { t = 0 } ^ { T - 1 } r _ { t } } \end{array}$ with $\begin{array} { r } { r _ { t }   =   \sum _ { i \in \mathcal { B } _ { t } ^ { * } } u _ { t } \big ( i \big )   -   \sum _ { i \in \hat { \mathcal { B } } _ { t } } u _ { t } \big ( i \big ) } \end{array}$ . It suffices to bound each $r _ { t }$ . Fix a round 𝑡 and, in Steps 2–6, write $u = u _ { t }$ and $\hat { u } = \hat { u } _ { t }$ ; by the time-uniform assumption (ii), $| \hat { u } ( i ) - u ( i ) | \leq \varepsilon _ { \phi }$ for every $i \in { \cal I } _ { t }$

**Step 2 (Only swapped items contribute).** Write $\mathcal { B } _ { t } ^ { * } \setminus \hat { \mathcal { B } } _ { t }   =   \{ i _ { 1 } , \ldots , i _ { m } \}$ (oracle items missed) and $\hat { \mathcal { B } } _ { t } \setminus \mathcal { B } _ { t } ^ { * } =$ $\{ j _ { 1 } , \ldots , j _ { m } \}$ (wrong items selected). Both sets have the same size $m \leq B$ because both selections have size 𝐵. Items in $\mathcal { B } _ { t } ^ { * } \cap \hat { \mathcal { B } } _ { t }$ cancel, so

$$
r _ { t } = \sum _ { k = 1 } ^ { m } \Bigl [ u \bigl ( i _ { k } \bigr ) - u \bigl ( j _ { k } \bigr ) \Bigr ]\tag{60}
$$

for any pairing of the two sets.

**Step 3 (Ranking inequality).** $\hat { \mathcal { B } } _ { t }$ is the top-𝐵 set by 𝑢ˆ and $i _ { k } \notin \hat { \mathcal { B } } _ { t }$ , so every selected item scores at least as high: $\hat { u } \big ( j _ { k } \big ) \geq \hat { u } \big ( i _ { k } \big )$ for all 𝑘.

<!-- page 39 of 45 -->

**Step 4 (Each swap costs at most** $2 \varepsilon _ { \phi } ) .$ **.** By (ii), $u \big ( i _ { k } \big ) \leq \hat { u } \big ( i _ { k } \big ) + \varepsilon _ { \phi }$ and $u \big ( j _ { k } \big ) \geq \hat { u } \big ( j _ { k } \big ) - \varepsilon _ { \phi }$ . Subtracting and using Step 3,

$$
u ( i _ { k } ) - u ( j _ { k } ) \leq \hat { u } ( i _ { k } ) - \hat { u } ( j _ { k } ) + 2 \varepsilon _ { \phi } \leq 2 \varepsilon _ { \phi } .\tag{61}
$$

**Step 5 (A swap needs a large error).** Since $i _ { k } \in \mathcal { B } _ { t } ^ { * }$ and $j _ { k } \notin \mathcal { B } _ { t } ^ { * }$ , (iii) gives $u \big ( i _ { k } \big ) - u \big ( j _ { k } \big ) \geq \mathrm { g a p } _ { \mathrm { m i n } }$ . Combined with Eq. (61), a swap is possible only if $\mathtt { g a p } _ { \operatorname* { m i n } } \leq 2 \varepsilon _ { \phi }$ . Hence if $2 \varepsilon _ { \phi } < \mathrm { g a p } _ { \mathrm { m i n } } ,$ then $m = 0 , \hat { \mathcal { B } } _ { t } = \mathcal { B } _ { t } ^ { * }$ , and $r _ { t } = 0$

**Step 6 (Per-round bound).** Combining Steps 2, 4 and 5 $\mathbf { \nabla } _ { \mathbf { \nabla } } \mathbf { \nabla } _ { \mathbf { \nabla } } r _ { t } \leq m \cdot 2 \boldsymbol { \varepsilon } _ { \phi } \cdot \mathbf { 1 } \left[ 2 \boldsymbol { \varepsilon } _ { \phi } \geq \mathrm { g a p } _ { \operatorname* { m i n } } \right] \leq 2 B \boldsymbol { \varepsilon } _ { \phi }   \mathbf { 1 } \left[ 2 \boldsymbol { \varepsilon } _ { \phi } \geq \mathrm { g a p } _ { \operatorname* { m i n } } \right]$

**Step 7 (Sum over rounds).** Summing Step 6 over $t = 0 , \ldots , T - 1$ gives Eq. (30). Steps $2 { - } 6$ use the error and gap of round 𝑡 only, so replacing $\varepsilon _ { \phi }$ and $\mathrm { g a p _ { m i n } }$ by $\varepsilon _ { t }$ and gap𝑡 gives $r _ { t } \leq 2 B \varepsilon _ { t } \mathbf { 1 } \big [ 2 \varepsilon _ { t } \geq \mathrm { g a p } _ { t } \big ]$ and the round-specific bound. Both sides are in the units of the declared utility (nats when 𝑢 is information gain); $\mathtt { g a p } _ { \operatorname* { m i n } }$ enters only through the indicator. □

**Theorem 19 (restated).** Suppose each round’s pool has at most 𝑛 candidates; for every round 𝑡 and every candidate outside a blind-spot set $O _ { t } \subseteq \mathcal { I } _ { t }$ with $| O _ { t } | \leq n _ { \mathrm { o o d } } ,$ , the error $\hat { u } _ { t } \big ( i \big ) - u _ { t } \big ( i \big )$ is zero-mean and 𝜎 -sub-Gaussian, while errors on $O _ { t }$ are unrestricted; and the utility takes values in an interval of length $u _ { \operatorname* { m a x } } \; ( e . g . \; u \in [ 0 , u _ { \operatorname* { m a x } } ] )$ . Then, with probability at least $1 - \delta$ and $\varepsilon _ { \phi } ^ { \mathrm { e f f } } = \sigma _ { \phi } \sqrt { 2 \ln ( 2 n T / \delta ) }$

$$
\mathcal { R } _ { T } ( \pi _ { \mathrm { C I } } , \pi ^ { * } ) \leq 2   T   B   \varepsilon _ { \phi } ^ { \mathrm { e f f } }   { \bf 1 } \Big [ 2 \varepsilon _ { \phi } ^ { \mathrm { e f f } } \geq \mathrm { g a p } _ { \mathrm { m i n } } \Big ] + T   \mathrm { m i n } ( B , n _ { \mathrm { o o d } } )   u _ { \mathrm { m a x } } .\tag{31}
$$

Proof. **Step 1 (Tail bound for one candidate).** A zero-mean $\sigma _ { \phi }$ -sub-Gaussian error satisfies $\operatorname* { P r } \bigl ( | \hat { u } _ { t } ( i ) - u _ { t } ( i ) | >$ $s ) \leq 2 e ^ { - s ^ { 2 } / ( 2 \sigma _ { \phi } ^ { 2 } ) }$ for all $s > 0$

**Step 2 (Union bound over cand**<strong><u>idates and</u></strong> **rounds).** There are at most 𝑛𝑇 candidate–round pairs outside the blind spots $O _ { t }$ . Setting $s = \varepsilon _ { \phi } ^ { \mathrm { e f f } } = \sigma _ { \phi } \sqrt { 2 \ln ( 2 n T / \delta ) }$ makes each tail probability at most $\delta / ( n T )$ , so with probability at least $1 - \delta$ every error outside the blind spots in every round is at most $\varepsilon _ { \phi } ^ { \mathrm { e f f } }$ . The rest of the proof works on this event.

**Step 3 (Classify swapped pairs).** In round 𝑡, pair the swapped items as in Step 2 of the proof of Theorem 18. Call a pair clean if neither of its items lies in $O _ { t }$ and contaminated otherwise.

**Step 4 (Clean pairs).** Steps 3–5 of the proof of Theorem 18 apply verbatim with $\varepsilon _ { \phi } ^ { \mathrm { e f f } }$ in place of $\varepsilon _ { \phi } \colon$ each clean pair costs at most $2 \varepsilon _ { \phi } ^ { \mathrm { e f f } }$ and exists only if $2 \varepsilon _ { \phi } ^ { \mathrm { e f f } }   \geq   \mathrm { g a p } _ { \mathrm { m i n } }$ . There are at most 𝐵 of them, contributing at most $2 B \varepsilon _ { \phi } ^ { \mathrm { e f f } } \mathbf { 1 } \Big [ 2 \varepsilon _ { \phi } ^ { \mathrm { e f f } } \geq \mathrm { g a p } _ { \mathrm { m i n } } \Big ]$

**Step 5 (Contaminated pairs).** Each item appears in at most one pair, so at most $n _ { \mathrm { o o d } }$ pairs contain a blind-spot item, and at most 𝐵 pairs exist in total; hence there are at most min $( B , n _ { \mathrm { o o d } } )$ contaminated pairs. Since the utility ranges over an interval of length $u _ { \mathrm { m a x } }$ , each costs $u \big ( i _ { k } \big ) - u \big ( j _ { k } \big ) \leq u _ { \mathrm { m a x } }$

**Step 6 (Sum over rounds).** Adding Steps 4 and 5 and summing over 𝑇 rounds gives Eq. (31) on the event of Step 2. □

Remark. The blind-spot term is stated as a count rather than a pool fraction deliberately: selection by 𝑢ˆ preferentially admits candidates whose utility is overestimated, so a single blind spot with unbounded overestimation is selected with certainty, and a bound of the form $B   \delta _ { \mathrm { o o d } } u _ { \mathrm { m a x } }$ , with $\delta _ { \mathrm { o o d } }$ the blind-spot fraction, would not hold in general. If utilities are tied at the 𝐵-th position $( \mathrm { g a p } _ { \mathrm { m i n } } = 0 )$ , the indicators are always active and the bounds degrade linearly in the error.

39

<!-- page 40 of 45 -->

## A.7 Proof of Theorem 21

**Theorem 21 (restated).** Let candidates $i \sim \mathcal { D }$ be utility-relevant when $u ( i )   \geq   u _ { \mathrm { m i n } } ,$ , let the gate admit 𝑖 when $\begin{array} { r } { \hat { u } ( i ) \geq u _ { \mathrm { m i n } } , } \end{array}$ and suppose $\left| \hat { u } \big ( i \big ) - u \big ( i \big ) \right| \leq \varepsilon   f _ { } { o r }$ every 𝑖. More generally, let the gate admit 𝑖 when $\hat { u } ( i ) \geq \kappa .$ Every misclassified candidate, admitted while dead or suppressed while relevant, then lies in the band $\mathcal { B } _ { \varepsilon } = \left\{ i : \kappa - \varepsilon \leq \right.$ $u ( i ) < u _ { \operatorname* { m i n } } \} \cup \{ i : u _ { \operatorname* { m i n } } \leq u ( i ) < \kappa + \varepsilon \}$ , so the misclassified masses satisfy 𝛿FP $+ \delta _ { \mathrm { F N } } \leq \mathrm { P r } _ { i \sim \mathcal { D } } ( \mathcal { B } _ { \varepsilon } )$ . With $\kappa = u _ { \mathrm { m i n } } ,$ if the true utility satisfies the margin condition $\operatorname* { P r } _ { i \sim \mathcal { D } } \bigl ( | u ( i ) - u _ { \operatorname* { m i n } } |   \leq   \varepsilon \bigr )   \leq   m   \varepsilon$ with constant 𝑚, then the gate misclassifies, by admitting a dead candidate or suppressing a relevant one, at most a fraction 𝑚 𝜀 of candidates.

Proof. Let the gate admit 𝑖 when $\hat { u } ( i ) \geq \kappa ;$ the lemma’s first statement is the case $\kappa = u _ { \mathrm { m i n } }$

**Step 1 (A false negative lies in the upper part of the band).** A relevant candidate is suppressed when $u ( i ) \geq u _ { \operatorname* { m i n } }$ and $\begin{array} { r } { \hat { u } ( i ) < \kappa . } \end{array}$ Then $u ( i ) \leq \hat { u } ( i ) + \varepsilon < \kappa + \varepsilon ,   \mathrm { s o }   u ( i ) \in [ u _ { \mathrm { m i n } } , \kappa + \varepsilon )$

**Step 2 (A false positive lies in the lower part of the band).** A dead candidate is admitted when $u ( i ) < u _ { \mathrm { m i n } }$ and $\begin{array} { r } { \hat { u } ( i ) \geq \kappa . } \end{array}$ Then $u ( i ) \geq \hat { u } ( i ) - \varepsilon \geq \kappa - \varepsilon , \mathrm { ~ s o ~ } u ( i ) \in [ \kappa - \varepsilon , u _ { \mathrm { m i n } } )$

**Step 3 (Count).** Every misclassified candidate therefore lies in ${ \mathcal { B } } _ { \varepsilon } ,$ so $\delta _ { \mathrm { F P } } + \delta _ { \mathrm { F N } }   \leq   \mathrm { P r } _ { i \sim \mathcal { D } } ( \mathcal { B } _ { \varepsilon } )$ . With $\kappa = u _ { \mathrm { m i n } } ,$ $\mathcal { B } _ { \varepsilon } \subseteq \{ i : | u ( i ) - u _ { \operatorname* { m i n } } | \leq \varepsilon \}$ , whose probability under D is at most 𝑚𝜀 by the margin condition. □

## A.8 Proof of Theorem 22 and Theorem 23

**Theorem 22 (restated).** Let $\mathcal { R } ^ { \mathfrak { t } }$ be the queries on which the full budget is more accurate in expectation than one increment, with density $\rho _ { \mathrm { a c c } } ^ { \mathrm { t t } }$ , and let the gate threshold a declared test-time utility whose relevant set contains $\mathcal { R } ^ { \mathfrak { t } }$ with density $\rho _ { u } ^ { \mathrm { t t } } \geq \rho _ { \mathrm { a c c } } ^ { \mathrm { t t } } .$ . Let unconditional scaling spend 𝐶 unit increments per query, let the answer be a function of the belief state, and let the selection policy be the one-probe gate: it spends one probing increment plus $c _ { p r e d }$ and then halts or continues to the full budget, with 𝛿<sub>FN</sub> and 𝛿<sub>FP</sub> the fractions of all queries that are relevant but halted and dead but continued. Then the gated policy loses at most $\delta _ { \mathrm { F N } }$ accuracy while spending at most a fraction

$$
\rho _ { u } ^ { \mathrm { t t } } + \frac { 1 + c _ { p r e d } } { C } + \delta _ { \mathrm { F P } }
$$

of the unconditional compute, a multiplier approaching $1 / \rho _ { u } ^ { \mathrm { t t } }$ as $C \rightarrow$ ∞ and $\delta _ { \mathrm { F P } } \rightarrow 0 .$ By Theorem $2 I ,   \delta _ { \mathrm { F N } } + \delta _ { \mathrm { F P } } \leq$ P $\mathbf { r } ( \mathcal { B } _ { \varepsilon _ { \phi } } )$ for a predictor with error at most $\varepsilon _ { \phi } ,$ , which is at most $m \varepsilon _ { \phi }$ under the margin condition. The tightest choice is the accuracy-gain utility itself, for which $\rho _ { u } ^ { \mathrm { t t } } = \rho _ { \mathrm { a c c } } ^ { \mathrm { t t } }$ . The continuation utility of Eq. (51) with $u _ { \mathrm { m i n } } \to 0 ^ { + }$ is also admissible: if its value is zero then $b _ { C } = b _ { 1 }$ almost surely, so the answer, and with it the accuracy, is unchanged; its gate should use $\kappa > \varepsilon _ { \phi } ,$ , so that the atom of zero-utility queries is never admitted. With density $\rho _ { \mathrm { c o n t } } ^ { \mathrm { t t } } \geq \rho _ { \mathrm { a c c } } ^ { \mathrm { t t } }$ it gives the weaker multiplier $1 / \rho _ { \mathrm { c o n t } } ^ { \mathrm { t t } }$ . The next-step utility of Eq. (49) is admissible only under the no-delayed-gain condition.

Proof. **Step 1 (Assumptions made explicit).** (i) Unconditional scaling spends 𝐶 unit-cost increments on every query. (ii) Every query outside the relevant set 𝑅 of the declared utility lies outside ${ \mathcal { R } } ^ { \mathfrak { t } } .$ , so its expected accuracy after the first increment is at least that after 𝐶 increments. For the continuation utility of Eq. (51) with $u _ { \mathrm { m i n } } \to 0 ^ { + }$ this holds because a query outside 𝑅 has $\mathbb { E } \big [ D _ { \mathrm { K L } } \bigg [ b _ { C } \bigg \| b _ { 1 } \bigg ] \big ] = 0 ;$ the divergence is non-negative, so $b _ { C } = b _ { 1 }$ almost surely, and since the answer is a function of the belief state, the answer and its expected accuracy are unchanged. For the next-step utility of Eq. (49) the same conclusion needs the no-delayed-gain condition, because a zero next-step change does not rule out later changes. (iii) The gated policy spends one probing increment, pays $c _ { \mathrm { p r e d } }$ for its prediction, and then either halts or continues to the full budget 𝐶. (iv) Per-query accuracy lies in [0, 1].

**Step 2 (Partition of queries).** Let $\rho = \rho _ { u } ^ { \mathrm { t t } }$ . Each query falls into one of four classes: relevant and continued (fraction $\rho - \delta _ { \mathrm { F N } } )$ , relevant but halted (false negatives, $\delta _ { \mathrm { F N } } )$ , dead and halted $( 1 - \rho - \delta _ { \mathrm { F P } } )$ , and dead but continued (false

40

<!-- page 41 of 45 -->

positives, 𝛿<sub>FP</sub>); these are unconditional masses of the whole query pool, not the class-conditional rates fp, fn of Theorem 7. By Theorem 21, $\delta _ { \mathrm { F N } } + \delta _ { \mathrm { F P } } \leq \operatorname* { P r } \bigl ( \mathcal { B } _ { \varepsilon _ { \phi } } \bigr )$ , which is at most $m \varepsilon _ { \phi }$ under the margin condition.

**Step 3 (Accuracy class by class).** Continued queries, relevant or not, receive the full budget and give exactly the unconditional answer. Correctly halted dead queries are, by (ii), at least as accurate as under the full budget. Only false negatives can lose accuracy, and by (iv) each loses at most one unit. Hence the accuracy loss is at most 𝛿<sub>FN</sub>.

**Step 4 (Compute).** Every query costs $c _ { \mathrm { p r e d } }$ for prediction, plus 𝐶 if continued or 1 if halted. The continued fraction is $\rho - \delta _ { \mathrm { F N } } + \delta _ { \mathrm { F P } } \leq \rho + \delta _ { \mathrm { F P } }$ and the halted fraction is at most 1, so the expected cost per query is at most

$$
\left( \rho + \delta _ { \mathrm { F P } } \right) C + 1 + c _ { \mathrm { p r e d } } .
$$

**Step 5 (Compute fraction).** Dividing by the unconditional cost 𝐶 gives the upper bound $\begin{array} { r } { \rho + \frac { 1 + c _ { \mathrm { p r e d } } } { C } + } \end{array}$ 𝛿<sub>FP</sub> on the compute fraction stated in the proposition, which tends to $\rho$ as $C \rightarrow$ ∞ and $\delta _ { \mathrm { F P } } \rightarrow 0 ;$ the multiplier then approaches $1 / \rho$ □

**Theorem 23 (restated).** Suppose (a) omitting any set of decision-irrelevant actions leaves the answer unchanged; (b) the gate thresholds a predicted relevance score with error at most $\varepsilon _ { \phi } ,$ , and on every task the action scores satisfy the margin condition of Theorem 21 with constant 𝑚, so that in expectation at most a fraction 𝑚 $\varepsilon _ { \phi }$ of the actions are misclassified; and (c) prediction costs a fraction 𝑟 of an average action. Consider tasks with 𝑁 candidate actions in which no action costs more than $R _ { c }$ times the task’s average action cost. If utility-relevant actions carry a share $\rho _ { u } ^ { C }$ of the expected call-everything cost (the cost-weighted density of utility-relevant actions, which reduces to their fraction $\rho _ { u }$ when all actions cost the same), the conditional agent’s accuracy is within 𝑁𝑚 $\varepsilon _ { \phi }$ of call-everything accuracy, at expected cost

$$
\mathbb { E } [ C _ { \mathrm { C I } } ] \leq \left( \rho _ { u } ^ { C } + r + m   \varepsilon _ { \phi }   R _ { c } \right) \mathbb { E } [ C _ { a l l } ] .
$$

Proof. **Step 1 (Setup).** Consider a task with 𝑁 candidate actions with costs $c _ { 1 } , \dots , c _ { N } \leq c _ { \operatorname* { m a x } }$ and total $\begin{array} { r } { C _ { \mathrm { a l l } } = \sum _ { k } c _ { k } } \end{array}$ so the average action costs $\bar { c } = C _ { \mathrm { a l l } } / N$ and, by assumption, $c _ { \mathrm { m a x } } \leq R _ { c }   \bar { c } .$ . The predictor costs 𝑟𝑐¯ per action, $r C _ { \mathrm { a l l } }$ in total.

**Step 2 (Accuracy).** Let E be the set of actions the conditional agent executes, and $\mathcal { E } _ { \mathrm { r e l } }$ the set of utility-relevant actions. If there is no false negative, $\mathcal { E } \supseteq \mathcal { E } _ { \mathrm { r e l } } ,$ , so E differs from the call-everything set only by omitting some decision-irrelevant actions, and by (a) the committed answer is unchanged. The answer can therefore differ only on the event that at least one false negative occurs. Conditional on the task, by (b) and Markov’s inequality, this event has probability at most E[#false negatives] $\leq m \varepsilon _ { \phi } N$ , which bounds the accuracy loss; averaging over tasks preserves the bound. The loss grows with the task length 𝑁.

**Step 3 (Cost of one task).** The conditional agent pays the predictor on every action $( r C _ { \mathrm { a l l } } )$ , the full cost on executed relevant actions (at most the task’s relevant cost $\begin{array} { r } { C _ { \mathrm { r e l } } = \sum _ { k \in \mathcal { E } _ { \mathrm { r e l } } } c _ { k } ) } \end{array}$ , and the full cost on false positives, each at most $c _ { \mathrm { m a x } } \leq R _ { c } \bar { c }$ . Conditional on the task, by (b) the expected number of false positives is at most $m \varepsilon _ { \phi } N$ , so

$$
\mathbb { E } \big [ C _ { \mathrm { C I } } \mid \mathrm { t a s k } \big ] \leq r C _ { \mathrm { a l l } } + C _ { \mathrm { r e l } } + m \varepsilon _ { \phi } N R _ { c } \bar { c } = r C _ { \mathrm { a l l } } + C _ { \mathrm { r e l } } + m \varepsilon _ { \phi } R _ { c }   C _ { \mathrm { a l l } } .
$$

**Step 4 (Take expectations over tasks).** By definition of the cost-weighted density, $\mathbb { E } [ C _ { \mathrm { r e l } } ] = \rho _ { u } ^ { C }   \mathbb { E } [ C _ { \mathrm { a l l } } ]$ . Averaging Step 3 over tasks therefore gives

41

$$
\mathbb { E } [ C _ { \mathrm { C I } } ] \leq \left( \rho _ { u } ^ { C } + r + m \varepsilon _ { \phi } R _ { c } \right) \mathbb { E } [ C _ { \mathrm { a l l } } ] ,
$$

the stated bound with every constant explicit. When all actions cost the same, $C _ { \mathrm { r e l } } / C _ { \mathrm { a l l } }$ is the fraction of relevant actions and $\rho _ { u } ^ { C }$ reduces to $\rho _ { u } .$ □

Remark. Separability is essential. If two individually irrelevant actions are jointly relevant (for instance, two retrievals whose combination answers the question), suppressing both can change the answer, and the accuracy statement fails.

<!-- page 42 of 45 -->

## A.9 Proof of Theorem 12

**Theorem 12 (restated).** Assume equal costs, a budget of 𝐵 executions, and that each candidate’s utility depends on the predictor outputs only through its own, $\mathbb { E } [ U ( i ) \mid { \hat { U } } ] = \mathbb { E } [ U ( i ) \mid { \hat { U } } ( i ) ]$ . Then, among all selection rules that see only the predictor outputs ${ \hat { U } } ,$ executing the 𝐵 candidates with the largest posterior mean $\mathbb { E } [ U \mid { \hat { U } } ]$ maximizes expected executed utility and hence the IIE. Setting $K _ { t }$ to the 𝐵-th largest posterior mean makes the admissible set of $E q .$ . (3) exactly that top-𝐵 set, up to ties at the 𝐵-th value.

Proof. **Step 1 (Admissible rules).** A selection rule that depends on the candidates only through their predictor outputs chooses a set 𝑆 of size 𝐵 that is a function of $\hat { U } = ( \hat { U } ( 1 ) , \dots , \hat { U } ( n ) )$ (and possibly of independent randomization).

**Step 2 (IIE reduces to executed utility).** With equal costs $\bar { C } ,$ a constant scoring cost $c _ { \mathrm { p r e d } }$ for each of the 𝑛 screened candidates, and exactly $B$ executions, the IIE denominator in Eq. (9) equals $n   c _ { \mathrm { p r e d } } + B \bar { C }$ for every such $S ,$ so maximizing IIE is equivalent to maximizing $\mathbb { E } \big [ \textstyle \sum _ { i \in S } U ( i ) \big ]$

**Step 3 (Tower property).** Because 𝑆 is determined by $\hat { U } .$

$$
\mathbb { E } \Bigl [ \sum _ { i \in S } U ( i ) \Bigr ] = \mathbb { E } \Bigl [ \left. \mathbb { E } \Bigl [ \sum _ { i \in S } U ( i ) \right| \hat { U } \Bigr ] \Bigr ] = \mathbb { E } \Bigl [ \sum _ { i \in S } \mathbb { E } \bigl [ U ( i ) | \hat { U } \bigr ] \Bigr ] .
$$

**Step 4 (Locality).** Assume that each candidate’s utility depends on the predictor outputs only through its own output, $\operatorname { i . e . } \mathbb { E } [ U ( i ) \mid { \hat { U } } ] = \mathbb { E } [ U ( i ) \mid { \hat { U } } ( i ) ] = : \mu ( { \hat { U } } ( i ) )$

**Step 5 (Pointwise optimization).** For each realization of ${ \hat { U } } ,$ the inner sum $\textstyle \sum _ { i \in S } \mu ( { \hat { U } } ( i ) )$ over sets of size 𝐵 is maximized by the 𝐵 largest values of $\mu ( \hat { U } ( i ) )$ : exchanging any selected item for an unselected one with a larger value increases the sum. The top-𝐵 rule is therefore optimal for every realization, hence in expectation, and randomization cannot help.

**Step 6 (Realization by the gate).** Setting $\hat { u } = \mu \big ( \hat { U } \big ) = \mathbb { E } \big [ U \big | \hat { U } \big ]$ and $K _ { t }$ to the 𝐵-th largest of these values makes the admissible set $\varUpsilon _ { t } ^ { \mathrm { a d m } }$ of $\operatorname { E q . }$ (4) exactly the top-𝐵 set of Step 5 (when several candidates tie at the 𝐵-th value, the admissible set contains all of them and ties are broken arbitrarily, which leaves the executed utility unchanged), so executing the admissible set realizes the optimum. This is the thresholding half of $$\mathrm { E q . ~ ( 3 ) } ;$$ the sampling half is irrelevant here, since the whole admissible set is executed, and the batch rule is the cardinality-constrained form of Eq. (47) rather than the single-draw limit of Theorem 2. □

Remark (relation to the information bottleneck). Stationary points of Eq. (21) satisfy the self-consistent IB equations (Tishby et al., 2000), $q ( \hat { U } \mid \mathcal { H } ) \propto q ( \hat { U } ) \exp \bigl ( - \beta _ { \mathrm { I B } }   D _ { \mathrm { K L } } \Bigl [ p ( U \mid \mathcal { H } ) \left\|   p ( U \mid \hat { U } ) \right] \bigr )$ . Stationarity does not imply that $\hat { U }$ is the most informative feature of its complexity, and a higher $I ( \hat { U } ; U )$ does not by itself guarantee better top-𝐵 selection. The proposition therefore makes no claim about which $\hat { U }$ is optimal, only about how to use a given one.

## A.10 Proof of Theorem 13

**Theorem 13 (restated).** Assume the setting of Theorem 7(ii); that $\rho _ { u } ,   \mathrm { f p } ( s )$ and $\mathrm { f n } ( s ) < 1$ do not depend on $k ;$ that $c _ { p r e d } ( s ) / ( 1 - \mathrm { f n } ( s ) )$ is non-decreasing in 𝑠; and that the IIE ratio Eq. (13) attains its maximum over $S _ { 1 }$ for every $k ,$ on a set $S ^ { \star } ( k )$ of optimal System-1 sizes. Then $( i ) \; S ^ { \star }$ is non-decreasing in 𝑘 in the strong set order: $\mathit { i f } \: k ^ { \prime } > k ,$ $s \in S ^ { \star } ( k )$ and $s ^ { \prime } \in S ^ { \star } ( k ^ { \prime } )$ , then max $( s , s ^ { \prime } ) \in S ^ { \star } ( k ^ { \prime } )$ and min $( s , s ^ { \prime } ) \in S ^ { \star } ( k )$ , so the largest and the smallest optimal sizes, where they exist, are non-decreasing in $k ;$ and (ii) the maximal IIE ratio is non-decreasing in 𝑘.

42

<!-- page 43 of 45 -->

Proof. **Step 1 (The ratio in terms of** 𝑠 and 𝑘). Write $\bar { C } _ { \mathrm { r e l } } , \bar { C } _ { \mathrm { d e a d } }$ for the execution costs at $k = 1$ . After scaling System 2 they become $k \bar { C } _ { \mathrm { r e l } }$ and $k \bar { C } _ { \mathrm { d e a d } } ,$ so the pool mean C¯ also scales by 𝑘. The first factor of Eq. (13), $( \bar { \mathbb { I } } _ { \mathrm { r e l } } / \bar { C } _ { \mathrm { r e l } } ) / ( \bar { \mathbb { I } } / \bar { C } ) = \bar { \mathbb { I } } _ { \mathrm { r e l } } \bar { C } / ( \bar { \mathbb { I } }   \bar { C } _ { \mathrm { r e l } } )$ , is a ratio of two costs and is therefore unchanged by 𝑘; it does not depend on 𝑠 either, and we call it $K > 0 .$ Dividing numerator and denominator of the overhead term by 𝑘,

$$
\frac { \mathrm { I I E } ^ { \mathrm { C I } } } { \mathrm { I I E } ^ { \mathrm { u n c } } } ( s , k ) = \frac { K } { 1 + h ( s , k ) } , \qquad h ( s , k ) = a ( s ) + \frac { b ( s ) } { k } ,
$$

with

$$
a ( s ) = \frac { \mathrm { f p } ( s ) \left( 1 - \rho _ { u } \right) \bar { C } _ { \mathrm { d c a d } } } { \left( 1 - \mathrm { f n } ( s ) \right) \rho _ { u } \bar { C } _ { \mathrm { r e l } } } , \qquad b ( s ) = \frac { c _ { \mathrm { p r e d } } ( s ) } { \left( 1 - \mathrm { f n } ( s ) \right) \rho _ { u } \bar { C } _ { \mathrm { r e l } } } .
$$

By assumption neither 𝑎 nor 𝑏 depends on 𝑘, and $b \geq 0 .$

**Step 2 (Reduction to minimizing** ℎ**).** Since $x \mapsto K / ( 1 + x )$ is strictly decreasing on $x \geq 0$ , maximizing the IIE ratio over 𝑠 is equivalent to minimizing $h ( \cdot , k )$ , and $\begin{array} { r } { S ^ { \star } ( k ) = \arg \operatorname* { m i n } _ { s \in \mathcal { S } _ { \mathrm { l } } } h ( s , k ) } \end{array}$ . Put $t = 1 / k$ , so that increasing 𝑘 means decreasing 𝑡, and $h = a ( s ) + b ( s )   t$

**Step 3 (Increasing differences).** The factor $\rho _ { u } \bar { C } _ { \mathrm { r e l } }$ is a positive constant, so 𝑏 is non-decreasing in 𝑠 by the assumption on $c _ { \mathrm { p r e d } } / ( 1 - \mathrm { f n } )$ . Hence for $s ^ { \prime } \geq s$ the difference

$$
h ( s ^ { \prime } , t ) - h ( s , t ) \; = \; a ( s ^ { \prime } ) - a ( s ) + \left( b ( s ^ { \prime } ) - b ( s ) \right) t
$$

is non-decreasing in 𝑡.

**Step 4 (Strong set order).** Let $k ^ { \prime } > k , \mathrm { i . e . } t ^ { \prime } = 1 / k ^ { \prime } < t = 1 / k$ , and take $s \in S ^ { \star } ( k ) ,   s ^ { \prime } \in S ^ { \star } ( k ^ { \prime } )$ . If $s \leq s ^ { \prime }$ then max $( s , s ^ { \prime } )   =   s ^ { \prime }$ and min $( s , s ^ { \prime } )   =   s$ and there is nothing to prove, so suppose $s ^ { \prime } \; < \; s$ . Optimality of 𝑠 at 𝑡 gives $h ( s , t ) \leq h ( s ^ { \prime } , t )$ . By Step 3 applied to the pair $s ^ { \prime } < s ,$

$$
h ( s , t ^ { \prime } ) - h ( s ^ { \prime } , t ^ { \prime } ) \; \leq \; h ( s , t ) - h ( s ^ { \prime } , t ) \; \leq \; 0 ,
$$

so $s = \max(s, s')$ is also optimal at $t ^ { \prime } , \mathrm { i . e . } s \in S ^ { \star } ( k ^ { \prime } )$ . Symmetrically, optimality of $s ^ { \prime }$ at $t ^ { \prime }$ gives $h ( s ^ { \prime } , t ^ { \prime } ) \leq h ( s , t ^ { \prime } )$ and Step 3 gives

$$
h(s',t) - h(s,t) \; = \; - \big (h(s,t) - h(s',t)\big ) \; \leq \; - \big (h(s,t') - h(s',t')\big ) \; = \; h(s',t') - h(s,t') \; \leq \; 0,
$$

so $s ^ { \prime } = \min ( s , s ^ { \prime } )$ is optimal at 𝑡, i.e. $s ^ { \prime }   \in   S ^ { \star } ( k )$ . This is (i). For the extreme points: if $\bar { s }   =   \operatorname* { m a x } S ^ { \star } ( k )$ exists, taking $s = \bar { s }$ and any $s ^ { \prime } \in S ^ { \star } ( k ^ { \prime } )$ shows $\operatorname* { m a x } ( \bar { s } , s ^ { \prime } ) \in S ^ { \star } ( k ^ { \prime } )$ , so every upper bound of $S ^ { \star } ( k ^ { \prime } )$ is at least 𝑠¯; likewise min $S ^ { \star } ( k ) \leq \operatorname* { m i n } S ^ { \star } ( k ^ { \prime } )$ whenever both minima exist.

43

**Step 5 (The optimal ratio).** For each fixed $s , h ( s , k ) = a ( s ) + b ( s ) / k$ is non-increasing in 𝑘 because $b ( s ) \geq 0 . \mathrm { ~ A ~ }$ pointwise minimum of non-increasing functions is non-increasing, so $h ^ { \star } ( k ) = \operatorname* { m i n } _ { s } h ( s , k )$ is non-increasing in 𝑘, and the maximal ratio $K / ( 1 + h ^ { \star } ( k ) )$ is non-decreasing in 𝑘. This is (ii). □

Remark. The monotonicity assumption on $c _ { \mathrm { p r e d } } / ( 1 - \mathrm { f n } )$ is mild: 1 − fn lies in (0, 1] and can rise at most to one, while the scoring cost grows roughly in proportion to System 1’s size. If it fails, the conclusion can fail too, since a larger predictor both more accurate and relatively cheaper per recalled utility-relevant candidate could become optimal only at small 𝑘. The assumption that error rates do not depend on 𝑘 is the substantive one, discussed after the proposition.

## A.11 Proof of Theorem 24

**Theorem 24 (restated).** Suppose (a) unconditional training follows a loss curve $\mathcal { L } _ { \mathrm { b a s e } } ( C )$ on a pool whose density of utility-relevant interactions is a constant $\rho _ { u } \in ( 0 , 1 ]$ ; (b) dead interactions leave the learner unchanged; (c) progress depends only on the utility-relevant interactions executed, whose distribution and order the selection policy preserves; and (d) the selection policy is perfect, prediction is free, and executions cost the same. Then

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { C I } } ( C ) \; = \; \mathcal { L } _ { \mathrm { b a s e } } ( C / \rho _ { u } )   , } \end{array}\tag{56}
$$

i.e. on a log-compute axis the conditional curve is the unconditional one shifted left by ln( $( 1 / \rho _ { u } )$

<!-- page 44 of 45 -->

Proof. **Step 1 (Unconditional informative compute).** By (a) and (d), each unit of compute executes candidates of equal cost, a fraction $\rho _ { u }$ of which are utility-relevant. Unconditional training at compute $C ^ { \prime }$ therefore spends informative compute $\rho _ { u } C ^ { \prime }$ (the integral of Eq. (54) with $\rho _ { u } ^ { \mathrm { e x e c } } \equiv \rho _ { u } )$

**Step 2 (Conditional informative compute).** By (d), the gate executes only utility-relevant candidates at no screening cost, so conditional training at compute C spends informative compute C $( \rho _ { u } ^ { \mathrm { e x e c } } \equiv 1 )$

**Step 3 (Only informative compute matters).** By (b), executing a dead candidate leaves the learner unchanged, and by (c) the learner’s state depends only on the utility-relevant interactions executed, whose distribution and order are the same in both regimes. Two runs that have executed the same informative compute are therefore in the same state and have the same loss.

**Step 4 (Match the runs).** By Steps 1–3, $\mathcal { L } _ { \mathrm { C I } } ( C ) = \mathcal { L } _ { \mathrm { b a s e } } ( C ^ { \prime } )$ whenever $\rho _ { u } C ^ { \prime } = C , \mathrm { i . e . } C ^ { \prime } = C / \rho _ { u }$ , which is Eq. (56).

**Step 5 (Log-axis interpretation).** On a logarithmic compute axis, ln $( C / \rho _ { u } ) = \ln C + \ln ( 1 / \rho _ { u } )$ , so the conditional curve is the unconditional one shifted left by $\ln ( 1 / \rho _ { u } )$ . No property of $\mathcal { L } _ { \mathrm { b a s e } }$ beyond being a function of compute was used, so the identity holds for any curve shape. □

## B Acquisition Functions for the Digital and Physical Instantiations

This appendix collects the acquisition functions behind the methods discussed in Section 5.1, and shows that they share one cost-normalized form: the scalar on which System-1 gates System-2.

**Digital: scoring a prompt before generation.** Given the prompt and the model’s recent history, System-1 estimates how the outcomes of a prompt 𝜏 would be distributed and scores it by how much of that distribution falls in the informative, mixed-outcome band,

$$
\mathcal { A } _ { \mathrm { d i g i t a l } } ( \tau ) = \mathbb { E } _ { z \sim \pi _ { t } ( \cdot | \tau ) } \big [ w ( z ) \big ] ,\tag{62}
$$

with 𝑧 summarizing how the prompt would resolve under the current policy and 𝑤 large on the mixed-outcome band. The predictive prompt selectors of Section 5.1 are of this form.

**Physical: adaptation risk and its tail.** The quantity worth predicting for a task configuration is how badly the agent would still do on it after adapting to it, the adaptation risk

$$
\begin{array} { r } { \mathcal { R } _ { \mathsf { a d a p t } } ( \tau ; \theta ) = \mathbb { E } _ { \xi \sim p ( \xi | \tau ) } \Big [ \ell \big ( \theta - \eta _ { \mathrm { i n } } \nabla _ { \theta } \mathcal { L } ( \theta , \tau , \xi ) ,   \tau _ { \mathrm { t a r g e t } } \big ) \Big ] \; , } \end{array}\tag{63}
$$

and concentrating training on its adverse tail is minimizing a conditional value-at-risk (Rockafellar et al., 2000) at level $\nu \in [ 0 , 1 )$ ,

$$
\operatorname* { m i n } _ { \theta } \; \operatorname { C V a R } _ { \nu } \bigl [ \ell ( \theta _ { \mathrm { a d a p t e d } } , \tau ) \bigr ] \; = \; \operatorname* { m i n } _ { \theta } \; \frac { 1 } { 1 - \nu } \int _ { \nu } ^ { 1 } F _ { \ell } ^ { - 1 } ( v ) \; d v ,\tag{64}
$$

which reduces to the average case at $\nu = 0$ . MPTS and PDTS (Wang et al., 2026; Qu et al., 2025) estimate this risk from a task embedding and the optimization history. MPTS scores each candidate optimistically by $\hat { \mathcal { R } } _ { \mathrm { a d a p t } } ( \tau ; \theta _ { t } ) + \gamma   \sigma _ { \phi } ( \tau )$ PDTS replaces this upper-confidence score with one posterior sample $\tilde { \mathcal { R } } _ { \mathrm { a d a p t } } ( \tau ; \theta _ { t } )$ per candidate and selects the whole batch at once, trading summed risk against the diversity of the subset:

$$
\mathcal { B } _ { t } ^ { \star } \; = \; \underset { \mathcal { B } \subseteq \hat { \mathcal { T } } _ { t } , \; | \mathcal { B } | = B } { \arg \operatorname* { m a x } } \; \sum _ { \tau \in \mathcal { B } } \tilde { \mathcal { R } } _ { \mathrm { a d a p t } } ( \tau ; \theta _ { t } ) \; + \; \mu \operatorname { D i v } ( \mathcal { B } ) ,\tag{65}
$$

44

where $\hat { \mathcal { T } } _ { t }$ is the candidate pool and Div(B) grows with the pairwise distances $d ( \tau , \tau ^ { \prime } )$ between selected tasks. The problem is NP-hard and is solved approximately; when Div is a sum of pairwise distances it is supermodular, so the $( 1 - 1 / e )$ greedy guarantee of Section 4.4 does not apply, and a coverage or log-determinant diversity term would restore it; it is the physical instance of the budgeted subset selection of Eq. (47).

<!-- page 45 of 45 -->

**Remark 26** (Common acquisition form). The per-prompt score of Eq. (62) and the per-task risk summed in Eq. (65) are both local approximations of the cost-normalized expected utility

$$
\mathcal { A } ^ { * } ( i \mid \mathcal { H } _ { 0 : t } ) = \frac { 1 } { C ( i ) }   \mathbb { E } _ { q _ { \phi } } \big [   u \big ( i ; \theta _ { t } \big )   \big ] ,\tag{66}
$$

with the mixed-outcome proxy, locally a Fisher-weighted gradient norm (Steps 3–4 below), in the digital case and predicted tail adaptation risk in the physical case; the diversity term captures diminishing returns from similar interactions. They share the form, not the scalar (Theorem 5).

## B.1 Derivation for Theorem 26

This is a heuristic derivation under local (Laplace) approximations, not a theorem. Only Steps 2 and 7 are exact, and only within the Gaussian models they state. The rest is a local analogy: a gradient step is not a Bayesian posterior update, a large gradient can reflect noise or a harmful direction rather than useful learning, and the connection to the acquisition functions of the cited methods is a motivation, not a derivation of them.

**Step 1 (Local model).** Assume the pre-interaction belief is approximately Gaussian, $\mathcal { N } ( \theta _ { t } , \mathcal { F } _ { t } ^ { - 1 } )$ , with precision $\mathcal { F } _ { t }$ (the accumulated Fisher information), and that an interaction moves the mean by a small step $\Delta \theta ( i ) = \eta   g ( i )$ , where $g ( i )$ is the score of the observed learning signal.

**Step 2 (Quadratic information gain).** For two Gaussians with the same covariance $\mathcal { F } _ { t } ^ { - 1 }$ and means differing by Δ𝜃, the KL divergence is exactly $\textstyle \frac { 1 } { 2 } \Delta \theta ^ { \top } \mathcal { F } _ { t } \Delta \theta$ . A single observation contracts the covariance only slightly, so to leading order

$$
\begin{array} { r } { D _ { \mathrm { K L } } \Big [ p \big ( \theta \mid \mathcal { H } _ { 0 : t } , i , o \big )   \Big \|   p \big ( \theta \mid \mathcal { H } _ { 0 : t } \big ) \Big ] \approx \frac { 1 } { 2 }   \Delta \theta ( i ) ^ { \top } \mathcal { F } _ { t }   \Delta \theta ( i ) . } \end{array}\tag{67}
$$

**Step 3 (Expected utility).** Substituting $\Delta \theta ( i )   =   \eta   g ( i )$ and taking expectations under $q _ { \phi }$ gives $\mathbb { E } _ { q _ { \phi } } \big [ u \big ( i ; \theta _ { t } \big ) \big ]   \approx$ $\begin{array} { r } { \frac { \eta ^ { 2 } } { 2 } \mathbb { E } _ { q _ { \phi } } \big [ g ( i ) ^ { \top } \mathcal { F } _ { t }   g ( i ) \big ] = \frac { \eta ^ { 2 } } { 2 } \mathbb { E } _ { q _ { \phi } }   \| g ( i ) \| _ { \mathcal { F } _ { t } } ^ { 2 } } \end{array}$

**Step 4 (Digital case).** If the score depends on the prompt only through the latent informativeness $z ,$ then conditioning on 𝑧 and dividing by cost gives $\mathcal { A } ^ { * } = \mathbb { E } _ { z \sim \pi _ { t } ( \cdot | \tau ) } \left[ w ( z ) \right]$ with $w ( z ) \propto \mathbb { E } [ \| g \| _ { \mathcal { F } _ { t } } ^ { 2 } \mid z ] / C$ . This is Eq. (62); 𝑤 is large exactly where outcomes are mixed, and the three-state weight $\mathbf { 1 } [ z = \mathrm { { \tt p a r t i a l } } ]$ is its binarization.

**Step 5 (Physical case).** Linearizing the post-adaptation loss of $\mathrm { E q }$ . (63) around $\theta _ { t }$ , its expectation is to leading order the predicted risk $\hat { \mathcal { R } } _ { \mathrm { a d a p t } } ( \tau ; \theta _ { t } )$ , and the spread of this linear predictor under $q _ { \phi }$ is $\sigma _ { \phi } ( \tau )$ . An optimistic (upper-confidence) estimate of the risk is then $\hat { \mathcal { R } } _ { \mathrm { a d a p t } } + \gamma \sigma _ { \phi }$ , the per-task score of MPTS, and a posterior sample $\tilde { \mathcal { R } } _ { \mathrm { a d a p t } }$ , as summed in Eq. (65), is its randomized counterpart; with roughly constant simulator cost, dividing by C does not change the ranking.

**Step 6 (Where the diversity term does not come from).** For a batch, the summed update $\textstyle \sum _ { i } \Delta \theta ( i )$ in Eq. (67) produces cross terms $\eta ^ { 2 } g ( i ) ^ { \top } \mathcal { F } _ { t }   g ( j )$ , which are positive when updates align. The mean-shift approximation therefore does not produce a redundancy penalty.

**Step 7 (Where it does come from).** Redundancy comes from covariance contraction, which Step 2 neglected. In the linear-Gaussian model $o _ { i } = g ( i ) ^ { \top } \theta + \varepsilon _ { i }$ with $\varepsilon _ { i } \sim \mathcal { N } ( 0 , \sigma ^ { 2 } )$ conditionally independent given 𝜃, the information gained by adding 𝑖 after a set 𝑆 is

$$
\begin{array} { r } { I \big ( \theta ; o _ { i } \mid o _ { S } \big ) = \frac { 1 } { 2 } \ln \bigl ( 1 + \sigma ^ { - 2 } g ( i ) ^ { \top } \Sigma _ { S }   g ( i ) \bigr ) , } \end{array}
$$

where $\Sigma _ { S }$ is the posterior covariance after 𝑆. Since $\Sigma _ { S }$ shrinks along the directions $g ( j ) , j \in S ,$ already probed, this marginal gain falls as $g ( i )$ aligns with the selected interactions. Information gain is thus submodular in the executed set of this linear-Gaussian model; nothing here shows that the task utility of a generic language model or embodied agent is submodular, and the subset diversity term $\mu \operatorname { D i v } ( { \mathcal { B } } )$ in Eq. (65) is a cheap proxy for this diminishing return, though a pairwise-distance sum is not itself submodular.

45