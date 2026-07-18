Title: BadWAM: When World-Action Models Dream Right but Act Wrong

URL Source: https://arxiv.org/html/2607.15207

Markdown Content:
Qi Li 1 Xingyi Yang 2 Xinchao Wang 1†

1 National University of Singapore 2 The Hong Kong Polytechnic University 

liqi@u.nus.edu xinchao@nus.edu.sg

†Corresponding author. 

[GitHub](https://github.com/LiQiiiii/BadWAM)[Homepage](https://liqiiiii.github.io/BadWAM)[HF Mirror](https://hf-mirror.com/collections/LIQIIIII/badwam)

###### Abstract

World-action models (WAMs) are emerging as a promising foundation for embodied control: rather than predicting actions alone, they learn representations that couple action generation with future world prediction. This coupling is often viewed as a source of robustness, interpretability, and safety, as a robot’s action can in principle be checked against its imagined future. In this paper, we show that this assumption is fragile. We introduce _BadWAM_, a unified framework for modeling and evaluating _World-Action Drift Attacks_: a new class of WAM-specific adversarial attacks that use small visual perturbations to break the alignment between what a WAM imagines and what it executes. BadWAM characterizes this attack surface along two natural criteria: attack strength and stealthiness. When the adversary prioritizes disruption, BadWAM instantiates an _action-only adversarial attack_, which directly drives the model toward task-failing actions. When the adversary additionally prioritizes stealth, BadWAM instantiates an _imagination-preserving adversarial attack_, which seeks to induce harmful action shifts while keeping the model’s predicted future close to its clean imagination. Together, these two attacks capture a spectrum of WAM-specific failures: from overt action hijacking to stealthier cases where the model appears to imagine a plausible future but executes a desynchronized action. We evaluate BadWAM across different variants of WAMs. Results show that our attacks substantially reduce task success rates under closed-loop execution. For example, our action-only attack reduces the model performance from 96.5% to 43.1% success. The results of our imagination-preserving attack further exposes a WAM-specific vulnerability: moderate future-preserving regularization can maintain strong attack performance while reducing future imagination drift.

## 1 Introduction

Robotic foundation models are rapidly moving from action prediction toward _world-action modeling_[kim2025openvla, black2024pi_0, pi0.7, zitkovich2023rt, yuan2026fastwam, shen2026world]. A world-action model (WAM) does not merely map observations and language instructions to low-level controls; it also learns representations that couple action generation with future world prediction[yuan2026fastwam, shen2026world]. This design is attractive for embodied AI. Future prediction can provide richer physical representations, enable planning or verification, and offer an intuitive safety signal: if a robot can imagine the consequences of its actions, then unsafe or inconsistent behavior might be detected before execution[shen2026world, yuan2026fastwam, li2026vision].

Recent WAMs instantiate this idea in different ways. Some models explicitly generate future observations or latent futures before action prediction, while others use video modeling as an auxiliary training objective and skip explicit future generation at test time[yuan2026fastwam]. Newer WAM variants further extend this paradigm toward object-addressable manipulation, mobile manipulation, and visual-tactile contact-rich control[liu2026oawam, chen2026abot, tian2026vtwam]. Across these designs, the central promise is the same: coupling action with world prediction should make robot behavior more physically grounded. This promise also motivates emerging WAM safety mechanisms that inspect imagined futures, latent rollouts, or visual trajectories to detect unsafe plans[liu2026jailwam, chen2026attackingtrusted].

This paper begins with a basic security question: _are WAMs robust to adversarially induced execution failures?_ We find that the answer is no. Small visual perturbations can substantially reduce task success across WAM variants. The deeper issue is specific to WAMs: _action generation and future imagination can fail asynchronously._ Even when the imagined future remains plausible, the action output need not remain aligned with it. We show that an attacker can desynchronize a WAM’s action from its future prediction. In other words, the model may still produce plausible future imaginations, yet execute actions that cause the task to fail. This failure mode is qualitatively different from standard adversarial attacks on image classifiers, visual world models, or reactive robot policies[szegedy2014intriguing, goodfellow2015explaining, madry2018towards, shen2026badworld]. The attacker is not merely changing a label or shifting an action vector; it is exploiting the interface between two coupled capabilities: imagining and acting.

We call this vulnerability _World-Action Drift Attack_. Rather than treating action and imagination as independent outputs, BadWAM focuses on their alignment. A small visual perturbation can push the model toward a task-failing action while leaving its imagined future visually plausible. This creates a dangerous gap: a robot may appear to imagine a reasonable future, yet execute an action that is no longer aligned with that imagination. In such cases, future prediction alone is not a reliable safety signal.

We instantiate this idea in _BadWAM_, a unified framework for modeling and evaluating world-action drift attacks under black-box access to a deployed WAM. BadWAM characterizes the attack surface along two natural criteria: attack strength and stealthiness. When the adversary prioritizes disruption, BadWAM instantiates an _action-only adversarial attack_, which drives the model toward task-failing actions using only queryable model outputs. This attack captures the overt end of the threat spectrum, where the adversary aims to maximize execution failure. When the adversary additionally prioritizes stealth, BadWAM instantiates an _imagination-preserving adversarial attack_. This attack also operates through black-box queries, but optimizes for harmful action shifts while keeping the model’s predicted future close to its clean imagination. Intuitively, this creates a stealthier failure: the robot appears to imagine a plausible future, but the action it executes has been desynchronized from that future. Together, they expose a spectrum of black-box WAM failures, from overt action hijacking to stealthy imagination-preserving failure.

We evaluate BadWAM on LIBERO[liu2023libero] and RoboTwin[chen2025robotwin]. Our experiments show that WAMs are vulnerable across variants. The action-only attack substantially reduces closed-loop task success, and the imagination-preserving attack exposes a stronger WAM-specific risk: moderate future-preserving regularization can maintain strong attack performance while reducing imagination drift. In our experiments, the action-only attack reduces task success from 96.5% to 43.1%. The imagination-preserving attack reduces task success while keeping future predictions close to clean rollouts. Our analysis further reveals where the attacks operate. The perturbations primarily affect continuous action channels and later portions of the action horizon, rather than merely flipping gripper commands. Episode-level statistics show that failed episodes tend to have larger action shifts, while future-imagination shifts can remain bounded. These findings suggest that WAM failures are not simply caused by visually destroying the model’s imagination. Instead, the attack exploits a decoupling between what the model imagines and what it executes.

This work makes the following contributions:

*   •
We identify _World-Action Drift Attack_ as a WAM-specific attack surface, where action outputs can be adversarially hijacked while the model’s predicted futures remain visually plausible.

*   •
We design _BadWAM_, a unified framework with two attack instantiations: _action-only adversarial attacks_ for maximizing execution disruption, and _imagination-preserving adversarial attacks_ for causing failures while preserving plausible predicted futures.

*   •
We evaluate BadWAM on closed-loop robot manipulation tasks, showing substantial task degradation under bounded visual perturbations. We also provide detailed analyses of attack dynamics, action-channel shifts, horizon-level effects, imagination preservation, and future-weight trade-offs, offering insight into why WAMs fail under world-action drift attacks.

## 2 Related Work

### 2.1 World Action Models

World-action models have recently emerged as a predictive-action paradigm for embodied AI[shen2026world]. Unlike conventional visuomotor policies that directly map observations and instructions to actions, WAMs expose future prediction as part of the action-generation process. Let o_{t} denote the current visual observation, g denote a language instruction or goal, and a_{t:t+H-1} denote an action chunk over horizon H. A conventional reactive policy can be abstracted as

a_{t:t+H-1}\sim\pi_{\theta}(o_{t},g).(1)

By contrast, a WAM additionally models future world states, such as latent future states z_{t+1:t+K} or decoded future frames v_{t+1:t+K}. Abstractly, its world-prediction module W_{\theta} first imagines a future, and its action module A_{\theta} then produces an action sequence conditioned on that imagination:

z_{t+1:t+K}\sim W_{\theta}(o_{t},g),\quad a_{t:t+H-1}\sim A_{\theta}(o_{t},g,z_{t+1:t+K}).(2)

In fully joint formulations, the model may instead learn a joint distribution over future states and actions:

(z_{t+1:t+K},a_{t:t+H-1})\sim p_{\theta}(z_{t+1:t+K},a_{t:t+H-1}\mid o_{t},g).(3)

In all cases, the defining feature is that action prediction is coupled, explicitly or implicitly, with a learned model of how the world may evolve.

This coupling can appear in multiple architectural forms. Some WAMs first imagine a future and then infer actions from that imagined future. Others jointly predict actions and future representations in a shared sequence model. Still others use future prediction primarily as a training signal, so that the model benefits from world modeling even if explicit future generation is not used at inference time[yuan2026fastwam]. Fast-WAM studies this design space and shows that video modeling during training can provide substantial benefits even when test-time future generation is removed[yuan2026fastwam]. This observation is important for security: even when a model does not expose future videos at deployment time, the learned action representation may still inherit vulnerabilities from the underlying world-action coupling.

Recent WAM variants further broaden the design space. OA-WAM introduces object-addressable world representations for robust manipulation[liu2026oawam]; ABot-M0.5 extends WAMs to unified mobility-and-manipulation control with action-space disentanglement and train-test consistency mechanisms[chen2026abot]; and VT-WAM incorporates visual-tactile prediction for contact-rich manipulation[tian2026vtwam]. These systems illustrate the rapid expansion of WAMs from visual manipulation to richer embodied settings. As WAMs become more capable and more deeply integrated into robot control stacks, understanding their security properties becomes increasingly urgent and meaningful.

Prior WAM work usually treats future prediction as a beneficial representation, planning interface, or safety signal[liu2026jailwam, chen2026attackingtrusted]. A natural hope is that an imagined future can be used to check whether a planned action is physically plausible or safe. Formally, one might imagine a monitor M that inspects the predicted future:

\mathrm{safe}=M(z_{t+1:t+K},g).(4)

However, such a monitor only observes the imagined future, not necessarily the alignment between the future and the action that will be executed. This leaves open a WAM-specific security question: can an adversary perturb the observation so that the action changes substantially while the imagined future remains apparently plausible? BadWAM formalizes and evaluates this gap in Section[4](https://arxiv.org/html/2607.15207#S4 "4 BadWAM ‣ BadWAM: When World-Action Models Dream Right but Act Wrong").

### 2.2 Attacks against Embodied AI Systems

Adversarial examples were first studied extensively in image classification, where small input perturbations can cause large changes in model predictions[szegedy2014intriguing, goodfellow2015explaining, madry2018towards]. Subsequent work extended these attacks to black-box and query-limited settings, including gradient-free and score-based estimation methods[chen2017zoo, ilyas2018blackbox, uesato2018adversarial]. Physical-world attacks further showed that adversarial patterns can remain effective under real-world transformations, for example through adversarial patches or robust physical perturbations[brown2017adversarialpatch, eykholt2018robust]. BadWAM builds on this broader adversarial robustness literature but targets a different object: not a classifier label or static perception output, but the closed-loop action behavior of a WAM.

Attacks on embodied AI systems are particularly concerning because model failures can translate into physical consequences. Prior work has studied attacks on perception modules, reinforcement-learning agents, and embodied policies, including perturbations that alter observations, timing, or environmental cues[lin2017tactics, gleave2019adversarial]. These attacks typically target reactive decision-making. For a reactive policy \pi_{\theta}, an observation attack can be written as

\max_{\|\delta\|_{\infty}\leq\epsilon}D\!\left(\pi_{\theta}(o_{t}+\delta,g),\pi_{\theta}(o_{t},g)\right),(5)

or, in closed-loop evaluation, as minimizing task reward or success. BadWAM naturally generalizes this view to predictive-action models whose outputs include both actions and imagined futures. The relevant question is no longer only whether the action changes, but whether the action changes while the imagination remains plausible.

Recent work has begun to examine attacks and safety risks for world models and WAMs. BadWorld attacks visual world models by perturbing context images to degrade future rollouts under uncertain controls[shen2026badworld]. Physical-conditioned attacks study how perturbing condition channels in generative world models can induce semantic or decision-level distortions[zhang2026physcondwma]. JailWAM proposes a safety evaluation framework for WAM jailbreaks in robot control, focusing on unsafe or destructive behaviors represented through visual trajectories[liu2026jailwam]. These works highlight that predictive models in embodied systems introduce new attack surfaces beyond conventional action policies.

Closest to our setting is recent work on oracle-level integrity attacks against imagine-then-act world models[chen2026attackingtrusted]. That work studies VLA-style and world-model-based systems, where an imagined future is consumed by downstream oracles such as safety gates, MPC planners, or imagine-then-check verifiers. It shows that corrupting the trusted imagination can compromise systems that rely on future predictions. BadWAM studies a complementary failure mode. Rather than attacking the imagined future as the primary object, we target the alignment between imagination and action in deployed WAM controllers. We show that action and imagination can be adversarially desynchronized: the predicted future may remain close to the clean prediction while the executed action changes enough to cause task failure.

This distinction is central to our threat model. If a safety monitor trusts plausible imagined futures, then a purely imagination-based check may miss an attack that preserves future predictions while hijacking actions. BadWAM therefore treats the alignment between action and imagination as the security-critical object. Our action-only adversarial attack prioritizes task disruption, while our imagination-preserving adversarial attack explicitly studies the trade-off between attack strength and stealthiness.

## 3 Threat Model

We study inference-time attacks against deployed WAM-based robot policies. Following the notation introduced above, the robot observes o_{t} and receives an instruction or goal g at each replanning step. The WAM then outputs an action chunk a_{t:t+H-1} and, depending on the model interface, may also expose an imagined future in latent form z_{t+1:t+K} or decoded video form v_{t+1:t+K}. The robot executes part of the predicted action chunk, observes the environment again, and repeats this process in closed loop.

Adversarial capability. The adversary can perturb the visual observation before it is processed by the WAM. We write the perturbed observation as

\tilde{o}_{t}=\mathrm{clip}(o_{t}+\delta_{t}),\quad\|\delta_{t}\|_{\infty}\leq\epsilon.(6)

The perturbation is bounded so that the attacked observation remains visually close to the clean observation. For multi-camera inputs, we treat o_{t} as the concatenation of all camera observations and apply the same norm constraint to the full visual input. The adversary does not modify the language instruction g, the robot state, the environment dynamics, the model parameters, or the training data.

We consider two access levels. In the _action-only_ setting, the adversary can query the WAM and observe only its output action chunk. This captures black-box access to a deployed policy API or robot inference stack. In the _imagination-visible_ setting, the adversary can additionally observe the model’s imagined future, either as latent rollouts or decoded future video. This captures systems where future prediction is exposed for planning, debugging, monitoring, or imagine-then-check safety verification[chen2026attackingtrusted, Zhao-RSS-24]. In both settings, the adversary does not require gradients or white-box access to model weights.

Adversarial objective. The adversary’s primary goal is untargeted task failure: the robot should fail to complete the instructed task under closed-loop execution. The attack is successful if a perturbation causes the robot to execute a sequence of actions that reduces task success, even though the instruction and environment remain unchanged.

![Image 1: Refer to caption](https://arxiv.org/html/2607.15207v1/x1.png)

Figure 1:  Empirical motivation for world-action adversarial attacks. Failed episodes tend to have larger action shifts, while predicted-future shifts overlap across successful and failed executions. This motivates attacking the alignment between action and imagination. 

BadWAM additionally studies stealthiness. A non-stealthy attack may simply maximize action disruption. A stealthier attack should degrade execution while keeping the WAM’s imagined future close to its clean prediction. This is important because future prediction is often treated as a safety signal: a monitor may inspect the imagined future and accept the action if the future appears plausible[chen2026attackingtrusted]. BadWAM targets precisely this gap between what the WAM imagines and what it executes.

Non-goals. We do not study training-time poisoning, model extraction, reward hacking, prompt injection, or attacks that physically modify the robot’s environment. We also do not assume that the attacker can directly command the robot or alter the controller after actions are produced. Our focus is narrower: small bounded observation perturbations that exploit the world-action interface of WAMs at inference time.

![Image 2: Refer to caption](https://arxiv.org/html/2607.15207v1/x2.png)

Figure 2:  Action-only adversarial attack produces structured action shifts. Across three WAM variants, the attack primarily perturbs continuous action channels and specific portions of the action horizon, rather than acting as uniform output noise. 

![Image 3: Refer to caption](https://arxiv.org/html/2607.15207v1/x3.png)

Figure 3: Overview of BadWAM. BadWAM injects a small visual perturbation into model observations and performs query-based online search over a frozen WAM. The optimized trigger disrupts the action prediction pathway while preserving or minimally altering visual rollout predictions, leading to world-action adversarial attacks during closed-loop execution. 

## 4 BadWAM

BadWAM is a unified framework for modeling and evaluating world-action drift attacks. As shown in Figure[3](https://arxiv.org/html/2607.15207#S3.F3 "Figure 3 ‣ 3 Threat Model ‣ BadWAM: When World-Action Models Dream Right but Act Wrong"). The framework treats a WAM as a queryable predictive-action system and optimizes bounded visual perturbations at each replanning step. Its two attack instantiations correspond to two points on the attack-strength/stealthiness spectrum: an _action-only adversarial attack_ that prioritizes execution failure, and an _imagination-preserving adversarial attack_ that additionally constrains predicted-future drift.

### 4.1 Design Motivation

BadWAM is motivated by a simple empirical observation: action and imagination are coupled in WAMs, but they are not inseparable. In our closed-loop traces, task failures tend to exhibit larger action shifts than successful episodes, while the corresponding imagination shifts can substantially overlap. This suggests that the security-critical object is not only the action or the imagination alone, but their alignment. Figure[1](https://arxiv.org/html/2607.15207#S3.F1 "Figure 1 ‣ 3 Threat Model ‣ BadWAM: When World-Action Models Dream Right but Act Wrong") visualizes this phenomenon under our imagination-preserving attack. This motivation is also consistent with recent diagnostic evidence that WAM action heads can rely on future-predictive latents in a task-phase-dependent manner[mishra2026temporalratio]. BadWAM asks a complementary security question: whether this coupling can be adversarially utilized.

We therefore formulate BadWAM around two distances. D_{\mathrm{act}} measures the deviation between clean and attacked action chunks, and D_{\mathrm{img}} measures the drift between clean and attacked imagination. These distances are interface-level quantities: they do not require a particular WAM architecture, only access to the corresponding outputs. Conceptually, a stealthy adversarial attack seeks large action deviation under bounded visual perturbations while keeping imagination drift below a monitorable threshold:

\max_{\|\delta\|_{\infty}\leq\epsilon}D_{\mathrm{act}}(a^{\delta},a)\quad\mathrm{s.t.}\;D_{\mathrm{img}}(z^{\delta},z)\leq\tau_{\mathrm{img}}.(7)

Equation[7](https://arxiv.org/html/2607.15207#S4.E7 "In 4.1 Design Motivation ‣ 4 BadWAM ‣ BadWAM: When World-Action Models Dream Right but Act Wrong") defines the core BadWAM attack surface. The two attacks below instantiate different relaxations of this objective.

### 4.2 Action-Only Adversarial Attack

The action-only attack captures the high-strength end of BadWAM. It treats the WAM as a queryable controller and focuses purely on execution disruption. Rather than enforcing any constraint on the imagined future, it searches for a bounded visual perturbation that maximizes the deviation between clean and attacked action chunks:

\delta_{t}^{\star}=\arg\max_{\|\delta_{t}\|_{\infty}\leq\epsilon}D_{\mathrm{act}}\left(a_{t:t+H-1}^{\delta},a_{t:t+H-1}\right).(8)

The clean action chunk is used only as a reference; the attacker does not need ground-truth expert actions or task success labels during optimization. This makes the attack applicable to deployed policy interfaces where the observable output is the action command sent to the robot controller.

Although simple, this objective is already highly disruptive. In our closed-loop experiments, the action-only attack reduces one WAM variant from 96.5% to 43.1% task success. We therefore use it as the high-strength endpoint of BadWAM. At the same time, the attack is not merely unstructured output corruption: the induced shifts are unevenly distributed across both action channels and temporal horizon segments. For example, Figure[2](https://arxiv.org/html/2607.15207#S3.F2 "Figure 2 ‣ 3 Threat Model ‣ BadWAM: When World-Action Models Dream Right but Act Wrong") shows that the perturbations concentrate on task-relevant channels such as translation and gripper commands, while their temporal concentration varies across WAM variants. This motivates logging channel-level and horizon-level action statistics in our evaluation.

![Image 4: Refer to caption](https://arxiv.org/html/2607.15207v1/x4.png)

(a)Imagined future without preservation.

![Image 5: Refer to caption](https://arxiv.org/html/2607.15207v1/x5.png)

(b)Imagined future with preservation.

Figure 4:  Qualitative illustration of the imagination-preserving objective on IDM WAM. Each panel compares predicted futures under clean and adversarial observations at selected future steps. abs diff denotes the absolute pixel difference between clean and adversarial predictions, amplified by 8\times for visibility. Both variants induce action-space failure, but the preservation term keeps the adversarial future more consistent with the clean imagination. 

### 4.3 Imagination-Preserving Adversarial Attack

The action-only objective provides no control over what happens to the imagined future. For WAM deployments that expose imagined futures to planners, monitors, or imagine-then-check safety gates, this is insufficient: a stealthier attack should degrade execution while preserving the model’s clean imagination. BadWAM therefore instantiates an imagination-preserving attack by solving the Lagrangian relaxation of Equation[7](https://arxiv.org/html/2607.15207#S4.E7 "In 4.1 Design Motivation ‣ 4 BadWAM ‣ BadWAM: When World-Action Models Dream Right but Act Wrong"):

\displaystyle\delta_{t}^{\star}=\arg\max_{\|\delta_{t}\|_{\infty}\leq\epsilon}\displaystyle D_{\mathrm{act}}(a_{t:t+H-1}^{\delta},a_{t:t+H-1})(9)
\displaystyle-\lambda D_{\mathrm{img}}(z_{t+1:t+K}^{\delta},z_{t+1:t+K}).

The coefficient \lambda controls the trade-off between attack-strength and stealthiness. When \lambda=0, the objective reduces to pure action disruption. As \lambda increases, the optimizer is discouraged from changing the imagined future, making the attack more stealthy but potentially less damaging.

When decoded future videos are available, we instantiate the imagination distance as an average frame-level distance:

D_{\mathrm{img}}(v_{t+1:t+K}^{\delta},v_{t+1:t+K})=\frac{1}{K}\sum_{k=1}^{K}d_{\mathrm{frame}}(v_{t+k}^{\delta},v_{t+k}).(10)

The same objective can be applied to latent futures by replacing the frame-level distance with a latent-space distance. In either case, the attack targets a WAM-specific failure: the robot may appear to imagine a plausible future while executing a desynchronized action.

### 4.4 Query-Based Online Optimization

BadWAM does not backpropagate through the WAM or access model parameters. At each replanning step, the attacker treats the WAM as a queryable input-output system and optimizes the attack objective with zeroth-order finite-difference queries. Concretely, for a current perturbation \delta_{t}, the attacker samples random directions u_{i} and evaluates the scalar objective under positively and negatively perturbed observations. This gives the estimator

\widehat{\nabla}J(\delta_{t})=\frac{1}{m}\sum_{i=1}^{m}\frac{J(\delta_{t}+cu_{i})-J(\delta_{t}-cu_{i})}{2c}u_{i},(11)

where J denotes either the action-only objective or the imagination-preserving objective, c is the finite-difference radius, and m is the number of sampled directions. The perturbation is then updated and projected back into the feasible \ell_{\infty} ball:

\delta_{t}\leftarrow\Pi_{[-\epsilon,\epsilon]}\left(\delta_{t}+\eta\widehat{\nabla}J(\delta_{t})\right).(12)

This optimization is zeroth-order because it only uses objective values computed from WAM outputs; it does not require gradients, model weights, or training data. In our implementation, the random directions follow a simultaneous perturbation strategy[spall1992multivariate, uesato2018adversarial]. The same optimizer supports both BadWAM instantiations by changing only the scalar objective J: the action-only attack scores action deviation, while the imagination-preserving attack scores action deviation together with future-prediction consistency.

For each replanning step, BadWAM keeps the best perturbation found within the query budget. The final adversarial observation is then used to produce the action chunk executed by the robot. This online design avoids training a separate attack model and naturally adapts to the current observation, instruction, and robot state.

Qualitative Example Figure[4](https://arxiv.org/html/2607.15207#S4.F4 "Figure 4 ‣ 4.2 Action-Only Adversarial Attack ‣ 4 BadWAM ‣ BadWAM: When World-Action Models Dream Right but Act Wrong") qualitatively illustrates the role of the preservation term. Both variants can induce action failure, but adding preservation keeps the predicted future closer to the clean imagination. This is precisely the WAM-specific failure mode BadWAM targets: the model’s imagination remains plausible while the executed action is desynchronized.

### 4.5 Closed-Loop Execution and Measurements

BadWAM is evaluated under closed-loop execution. The attack is recomputed at each replanning step, the WAM outputs an attacked action chunk, and the robot executes the selected actions before the next replan. This setting is more faithful than single-step action comparison because small action shifts may compound over time and cause task failure only after multiple replans.

We report three groups of measurements. First, we measure task-level attack effectiveness using closed-loop success rate and induced failures relative to clean execution. Second, we measure action disruption using action distance, action-channel statistics, and horizon-level action shifts. Third, for imagination-preserving attacks, we measure stealthiness using predicted-future distance and adversarial score. Together, these metrics separate two questions: whether the robot fails, and whether the failure is visible through the WAM’s imagined future.

## 5 Evaluation

We organize the evaluation around six research questions:

Table 1:  Main closed-loop attack results on full sweeps with \epsilon=0.06 and a search budget of 8 optimization iterations per replanning step. Gray values indicate the absolute reduction from clean task success. 

![Image 6: Refer to caption](https://arxiv.org/html/2607.15207v1/x6.png)

Figure 5:  Task-level failure profile under attack. Unlike Table[1](https://arxiv.org/html/2607.15207#S5.T1 "Table 1 ‣ 5 Evaluation ‣ BadWAM: When World-Action Models Dream Right but Act Wrong"), which reports aggregate success, this figure shows how attacked tasks distribute across success-rate bins. Lower bins indicate tasks that are consistently broken rather than merely slightly degraded. 

*   RQ1
Attack effectiveness. How much can bounded observation perturbations reduce closed-loop WAM reliability?

*   RQ2
Systematicity. Are the failures concentrated in a few tasks, or do they persist across suites and repeated trials?

*   RQ3
Failure mechanism. What do BadWAM failures look like during execution, and how does the query-based search produce them?

*   RQ4
Stealth under matched strength. Under matched attack strength, what does imagination preservation buy?

*   RQ5
Sensitivity and efficiency. How do future preservation, perturbation size, and query budget trade off?

*   RQ6
Transferability and defenses. Do the perturbations transfer across WAM variants, and how far do simple non-adaptive defenses go?

### 5.1 Experimental Setup

Benchmarks. Our main experiments use two language-conditioned robot manipulation benchmarks: LIBERO[liu2023libero] and RoboTwin[chen2025robotwin]. LIBERO provides four suites-Spatial, Object, Goal, and Long-horizon-for closed-loop tabletop manipulation. We evaluate all four LIBERO suites. RoboTwin provides a larger-scale dual-arm manipulation benchmark with diverse long-horizon tasks and richer visual observations. Unless otherwise stated, each LIBERO main-result cell uses the full LIBERO sweep with 20 trials per task, and each RoboTwin main-result cell uses the corresponding RoboTwin full evaluation protocol. For more expensive studies, such as transfer, defenses, and ablations, we use a balanced LIBERO subset with three tasks per suite and 10 trials per task.

![Image 7: Refer to caption](https://arxiv.org/html/2607.15207v1/x7.png)

Figure 6:  Comparison with random perturbations on full LIBERO dataset. All methods use the same \ell_{\infty} budget. Lower task success indicates a stronger attack. 

WAM policies. We evaluate three WAM-style controllers that expose different interfaces[yuan2026fastwam, shen2026world]. The action-only WAM maps the current observation and instruction directly to an action sequence. The joint WAM predicts future visual states and actions jointly. The IDM WAM first constructs a future-imagination representation and then decodes actions from it. This model set separates action-only policies from policies whose imagined futures are also query-visible. All evaluations are closed-loop: the policy is repeatedly queried during execution, and BadWAM may perturb the current observation at each replan. For the action-only variant on LIBERO and RobotWin, we use the officially released checkpoints. For the other two LIBERO variants, we train them on 8 H100 GPUs with the default settings[yuan2026fastwam]. For the RobotWin variants, we train them on 8 H100 GPUs for 50k steps.

![Image 8: Refer to caption](https://arxiv.org/html/2607.15207v1/x8.png)

Figure 7:  Mean pass@k across trials on different LIBERO suites. BadWAM lowers success throughout the trial budget instead of only causing isolated unlucky failures. 

Attacks. We evaluate two objectives. The _action-only adversarial attack_ maximizes the shift in the predicted action sequence under an \ell_{\infty}-bounded observation perturbation. It requires only action outputs and therefore applies to all three policies. The _imagination-preserving adversarial attack_ additionally penalizes changes in the predicted future, so it applies to WAMs whose future predictions are query-visible. Unless otherwise stated, both attacks perturb the full image with \epsilon=0.06. The default search budget is 8 optimization iterations per replan; with the clean reference query, this gives 17 WAM forward queries per attacked replan.

Metrics. Our primary metric is closed-loop task success rate; lower success under attack indicates a stronger attack. For a benchmark with T tasks and N_{i} evaluation episodes for task i, we compute

\mathrm{Success}=\frac{1}{T}\sum_{i=1}^{T}\frac{1}{N_{i}}\sum_{j=1}^{N_{i}}\mathbb{I}\!\left[s_{i,j}=1\right],(13)

where s_{i,j} indicates whether episode j of task i succeeds. We also report the induced success drop. For mechanism analysis, we use action distance D_{\mathrm{act}}, predicted-future distance D_{\mathrm{img}}, and the normalized decoupling score from Section[4](https://arxiv.org/html/2607.15207#S4 "4 BadWAM ‣ BadWAM: When World-Action Models Dream Right but Act Wrong"). For repeated-trial analysis, we report pass@k: for each task, pass@k is the fraction of successful executions among its first k trials, averaged across tasks.

### 5.2 BadWAM Reliably Induces Task Failures

To answer RQ1, Table[1](https://arxiv.org/html/2607.15207#S5.T1 "Table 1 ‣ 5 Evaluation ‣ BadWAM: When World-Action Models Dream Right but Act Wrong") reports the main closed-loop results on LIBERO and RoboTwin. Clean WAM policies are strong. For example, in LIBERO, the action-only WAM reaches 96.5% task success, while the joint and IDM WAMs reach 96.7% and 100.0%. BadWAM substantially reduces these rates. On the action-only WAM, the action-only attack lowers success to 43.1%, a 53.4% drop. On the joint WAM, action-only attacks lowers success to 61.5%, while imagination-preserving attacks lowers success to 63.0%. On the IDM WAM, the corresponding attacked success rates are 66.1% and 67.0%. The main message is that WAMs are not protected simply because action generation is coupled with future prediction. BadWAM causes high-performing systems to fail under small, bounded perturbations. Moreover, the imagination-preserving attack remains close to the action-only attack even though it explicitly suppresses predicted-future drift. This indicates that the imagination pathway can be preserved while the action pathway is still shifted toward task failure.

Figure[5](https://arxiv.org/html/2607.15207#S5.F5 "Figure 5 ‣ 5 Evaluation ‣ BadWAM: When World-Action Models Dream Right but Act Wrong") complements the aggregate table with a task-level failure profile. For the action-only WAM, 42.5% of tasks fall into the 0-25% success bin under attack. For the joint and IDM WAMs, 17.5-25.0% of tasks fall into this lowest-success bin, depending on the objective. This distributional view is important for safety: a moderate average can hide tasks that become almost unusable.

![Image 9: Refer to caption](https://arxiv.org/html/2607.15207v1/x9.png)

Figure 8:  Per-suite success rates on LIBERO. The attack is especially damaging on spatial and long-horizon tasks, while object-centric tasks remain comparatively more robust. 

![Image 10: Refer to caption](https://arxiv.org/html/2607.15207v1/x10.png)

Figure 9:  Qualitative comparison of clean and attacked rollouts on a LIBERO-10 task. Each row shows synchronized third-person and wrist-camera observations over time. The clean WAM completes the sequential manipulation task, while the attacked WAM initially behaves plausibly but gradually drifts, knocks over objects, fails to grasp the second object, and eventually fails. 

Sanity check against white-box attacks and random perturbations. To verify that BadWAM is not merely exploiting generic visual fragility, we compare it with two references in Figure[6](https://arxiv.org/html/2607.15207#S5.F6 "Figure 6 ‣ 5.1 Experimental Setup ‣ 5 Evaluation ‣ BadWAM: When World-Action Models Dream Right but Act Wrong"). First, we use a model-agnostic random uniform perturbation baseline under the same \ell_{\infty} budget. Second, we include a stronger white-box reference that backpropagates through the WAM to optimize an \ell_{\infty}-bounded input perturbation with a differentiable action-output surrogate; after the perturbation is found, the model is still evaluated with the standard closed-loop inference procedure. This white-box setting is outside BadWAM’s black-box threat model, but serves as an upper-bound reference for how much stronger attacks can become with gradient access.

![Image 11: Refer to caption](https://arxiv.org/html/2607.15207v1/x11.png)

Figure 10:  Per-replan action shifts persist across execution and accumulate over time, with failed episodes showing substantially larger cumulative shifts than successful ones. Action shift is measured between clean and attacked action chunks at each replan; shaded regions denote 95% confidence intervals. 

Random perturbations are much weaker than targeted attacks: on Joint WAM and IDM WAM, random noise leaves task success at 71.0% and 75.2%, while BadWAM reduces success to 61.5%/63.0% and 66.1%/68.1% for action-only and imagination-preserving objectives, respectively. The white-box reference further lowers success to 49.2% and 52.8%, showing that stronger access can amplify the vulnerability. Together, these results suggest that BadWAM failures are caused by targeted world-action attacks rather than arbitrary image corruption, and that our black-box attacks leave room below the white-box upper bound.

### 5.3 Failures Are Systematic Across Suites and Trials

To answer RQ2, we next test whether the attack is concentrated in a few task types or persists across the benchmark. Figure[8](https://arxiv.org/html/2607.15207#S5.F8 "Figure 8 ‣ 5.2 BadWAM Reliably Induces Task Failures ‣ 5 Evaluation ‣ BadWAM: When World-Action Models Dream Right but Act Wrong") breaks success down by LIBERO suite. The largest drops occur on tasks requiring precise spatial control or long-horizon execution. For the action-only WAM, Spatial success falls from 96.5% to 16.0%, Goal success falls from 97.0% to 40.0%, and Long-horizon success falls from 92.5% to 23.5%. Object tasks remain comparatively robust at 93.0%. The joint and IDM WAMs show the same qualitative pattern, although the degradation is less extreme. For example, the joint WAM under action-only attacks reaches 52.0% success on Spatial and 42.0% on Long-horizon, but remains at 92.5% on Object.

This suite-level structure suggests that BadWAM is not simply destroying visual perception uniformly. It is most effective when small action deviations can compound through geometry, contact timing, or sequential dependencies. Object-centric tasks often tolerate moderate drift as long as the relevant affordance remains visible and reachable. Spatial and long-horizon tasks leave less margin for a wrong approach direction, grasp position, or gripper timing.

We also test whether the success drop survives repeated trials. Figure[7](https://arxiv.org/html/2607.15207#S5.F7 "Figure 7 ‣ 5.1 Experimental Setup ‣ 5 Evaluation ‣ BadWAM: When World-Action Models Dream Right but Act Wrong") plots pass@k as k grows from 1 to 20. If BadWAM only exploited a few unlucky seeds, the attacked curves would move back toward the clean curves as more trials are averaged. Instead, the separation persists across the full trial budget. The clean action-only WAM stays close to 1.0, while the attacked action-only WAM remains around 0.40-0.43. The joint and IDM WAMs also remain consistently below their clean references under both attack objectives.

### 5.4 How Do BadWAM Failures Emerge?

We next answer RQ3 by examining representative executions and the optimization traces that produce them. Figure[9](https://arxiv.org/html/2607.15207#S5.F9 "Figure 9 ‣ 5.2 BadWAM Reliably Induces Task Failures ‣ 5 Evaluation ‣ BadWAM: When World-Action Models Dream Right but Act Wrong") shows a representative failure. The clean WAM completes the sequential manipulation task by grasping and placing the required objects in order. The attacked WAM, however, does not immediately collapse into random motion. It initially follows a plausible strategy and interacts with the right region, but its actions gradually drift away from the task intent. As the rollout progresses, the robot knocks over nearby objects, fails to stably grasp the second object, and eventually fails.

This example highlights why closed-loop evaluation is necessary. The perturbation need not make the observation obviously nonsensical or cause an obviously catastrophic first action. Instead, it can introduce structured deviations that remain locally plausible but become physically consequential over the horizon. Figure[10](https://arxiv.org/html/2607.15207#S5.F10 "Figure 10 ‣ 5.2 BadWAM Reliably Induces Task Failures ‣ 5 Evaluation ‣ BadWAM: When World-Action Models Dream Right but Act Wrong") confirms this behavior at the dataset level: per-replan action shifts persist across execution and accumulate over time. Failed episodes accumulate substantially larger action shifts than successful ones, suggesting that BadWAM failures often emerge from progressive closed-loop drift rather than a single isolated bad action.

Figure[11](https://arxiv.org/html/2607.15207#S5.F11 "Figure 11 ‣ 5.4 How Do BadWAM Failures Emerge? ‣ 5 Evaluation ‣ BadWAM: When World-Action Models Dream Right but Act Wrong") analyzes the search process behind these failures. Across query-optimization iterations, the best objective score increases and the best action distance increases with it. The future-video distance changes more mildly relative to the action shift. Thus, the attack is not simply random visual noise; across replans, the optimizer repeatedly finds bounded perturbations that move the action output while keeping future-prediction drift under pressure from the preservation term.

![Image 12: Refer to caption](https://arxiv.org/html/2607.15207v1/x12.png)

Figure 11:  Search dynamics for the imagination-preserving attack. Each panel uses a metric-specific y-axis range and reports mean \pm 95% confidence interval across replans. The query-based optimizer consistently improves the objective and increases action deviation, while the future-video distance changes much less in relative terms. 

![Image 13: Refer to caption](https://arxiv.org/html/2607.15207v1/x13.png)

Figure 12:  Matched-strength stealth trade-off. The imagination-preserving objective produces consistently smaller predicted-future shifts under a comparable input perturbation budget, revealing a stealthier failure mode than the action-only objective. Shaded regions denote 95% confidence intervals over replans. 

We then answer RQ4 by comparing the two objectives under matched perturbation and query budgets. Figure[12](https://arxiv.org/html/2607.15207#S5.F12 "Figure 12 ‣ 5.4 How Do BadWAM Failures Emerge? ‣ 5 Evaluation ‣ BadWAM: When World-Action Models Dream Right but Act Wrong") isolates this trade-off under matched attack resources. The two objectives use the same perturbation bound, query budget, and closed-loop protocol. Their under-attack success rates are close, but the imagination-preserving objective reduces average predicted-future distance from 14.01 to 13.04 while keeping the input perturbation budget essentially matched. Across the 40 LIBERO tasks in this run, its predicted-future distance is lower on 39 tasks. This supports the core BadWAM claim: a WAM can be pushed toward failure while its imagined future remains closer to the clean prediction than a purely action-only objective would allow.

These dynamics are also useful diagnostically. If action distance rises together with a large future-distance increase, the attack becomes less stealthy. If future distance stays low but action distance does not improve, the objective is over-regularized. The default configuration lies between these regimes: it induces a damaging action shift while retaining a comparatively stable predicted future.

### 5.5 Ablations: Future Preservation, Perturbation Budget, and Runtime

To answer RQ5, we ablate the future-preserving weight, perturbation budget, and query budget. Figure[13](https://arxiv.org/html/2607.15207#S5.F13 "Figure 13 ‣ 5.5 Ablations: Future Preservation, Perturbation Budget, and Runtime ‣ 5 Evaluation ‣ BadWAM: When World-Action Models Dream Right but Act Wrong") varies the future-preserving weight \lambda in the imagination-preserving objective. When \lambda=0, the optimizer does not penalize predicted-future changes. As \lambda increases, future distance generally decreases, showing that the preservation term actively shapes the search rather than merely changing the reported metric. For the joint WAM, increasing \lambda from 0 to 0.015 reduces D_{\mathrm{img}} from 14.70 to 14.34 and lowers task success from 61.7% to 56.7%. For the IDM WAM, the same setting reduces D_{\mathrm{img}} from 15.36 to 15.13 and lowers success from 55.0% to 51.7%. At larger \lambda, future distance continues to trend downward, but action distance and attack strength begin to weaken. Thus, \lambda controls a real action-imagination trade-off.

Figure[14](https://arxiv.org/html/2607.15207#S5.F14 "Figure 14 ‣ 5.5 Ablations: Future Preservation, Perturbation Budget, and Runtime ‣ 5 Evaluation ‣ BadWAM: When World-Action Models Dream Right but Act Wrong") studies the sensitivity of the imagination-preserving attack to perturbation size and query budget. The perturbation budget has the clearest effect. With \epsilon=0.01, task success remains high for both models, at 95.8% for the joint WAM and 98.3% for the IDM WAM. At the default \epsilon=0.06, success drops to 56.7% and 51.7% on the balanced subset. At \epsilon=0.20, both models collapse to 0.0% success. Attack strength therefore scales strongly with the allowed visual perturbation budget.

On the other hand, the query-budget sweep exposes a non-monotonic runtime-effectiveness frontier. Moving from budget 1 to 32 increases mean per-replan optimization time from 2.54s to 27.84s on the joint WAM and from 2.71s to 30.57s on the IDM WAM. Increasing the budget from 1 to 16 improves attack effectiveness, reducing task success from 64.2% to 56.7% for the joint WAM and from 60.8% to 51.7% for the IDM WAM. However, budget 32 does not further reduce closed-loop success despite its higher per-replan optimization cost. This reflects the gap between the local per-replan objective optimized by BadWAM and the long-horizon closed-loop task outcome: larger local action shifts do not always translate into more task failures after environment feedback and subsequent replanning. We therefore use a moderate default budget, which captures most of the attack benefit while avoiding unnecessary runtime overhead. These measurements come from an analysis-oriented implementation that saves extensive statistics, so they should be read as prototype costs rather than optimized real-time numbers.

![Image 14: Refer to caption](https://arxiv.org/html/2607.15207v1/x14.png)

Figure 13:  Ablation on future-preserving weight \lambda. Moderate future preservation can improve the action-future tradeoff, while excessive preservation weakens action manipulation. 

![Image 15: Refer to caption](https://arxiv.org/html/2607.15207v1/x15.png)

Figure 14:  Efficiency and budget sensitivity of BadWAM. Increasing the query budget B raises the per-replan optimization cost, but also gives the optimizer more opportunity to find stronger perturbations. The perturbation-budget study further shows that larger \epsilon consistently improves attack effectiveness. Together, the curves expose the practical tradeoff among attack strength, stealthiness, and runtime. 

### 5.6 Transferability Across WAM Variants

Here, we first answer the transfer part of RQ6 by testing whether BadWAM perturbations are tied to a single source model. The defense part of RQ6 is answered in Section[5.7](https://arxiv.org/html/2607.15207#S5.SS7 "5.7 Non-Adaptive Defenses and Detection ‣ 5 Evaluation ‣ BadWAM: When World-Action Models Dream Right but Act Wrong"). Table[2](https://arxiv.org/html/2607.15207#S5.T2 "Table 2 ‣ 5.7 Non-Adaptive Defenses and Detection ‣ 5 Evaluation ‣ BadWAM: When World-Action Models Dream Right but Act Wrong") optimizes perturbations on one WAM and evaluates the attacked closed-loop rollout on a different target WAM. The transferred attacks remain effective. For action-only attacks, transferred task success falls to 64.2% for Action-only WAM \rightarrow Joint WAM, 59.2% for Joint WAM \rightarrow IDM WAM, and 61.7% for IDM WAM \rightarrow Joint WAM. These are 32.5%-40.8% drops relative to the target clean references.

Imagination-preserving attacks also transfer between the joint and IDM WAMs. Joint WAM \rightarrow IDM WAM transfer lowers target success from 100.0% to 60.8%, while IDM WAM \rightarrow Joint WAM transfer lowers target success from 96.7% to 63.3%. The source-side future distances remain moderate, with D_{\mathrm{img}}^{\mathrm{src}}=14.86 and 14.32. Thus, BadWAM does not simply overfit to one action head. It exploits observation-space directions that affect action generation across related WAM variants.

### 5.7 Non-Adaptive Defenses and Detection

We then answer the defense part of RQ6 with simple non-adaptive preprocessing and detection baselines. Table[3](https://arxiv.org/html/2607.15207#S5.T3 "Table 3 ‣ 5.7 Non-Adaptive Defenses and Detection ‣ 5 Evaluation ‣ BadWAM: When World-Action Models Dream Right but Act Wrong") reports preprocessing defenses against the imagination-preserving attack on the balanced LIBERO subset mentioned in Section[5.1](https://arxiv.org/html/2607.15207#S5.SS1 "5.1 Experimental Setup ‣ 5 Evaluation ‣ BadWAM: When World-Action Models Dream Right but Act Wrong"). We test three lightweight transformations: Gaussian blur smooths the input image before querying the WAM, JPEG-noise ensemble averages predictions over JPEG-compressed/noisy views, and resize-crop ensemble averages predictions over random resized crops. Gaussian blur and JPEG-noise ensembles recover much of the attacked success while keeping clean performance high. For example, JPEG-noise raises attacked success to 89.2% on the joint WAM and 90.0% on the IDM WAM. By contrast, resize-crop ensembles are not operationally useful: they reduce clean success to nearly 50% and still leave the attacked policies mostly unusable.

These results should not be read as complete defenses. They mainly show that the current attack is non-adaptive to these transformations; an adaptive attacker could optimize through the same preprocessing or use expectation over transformations. We also evaluate an augmentation-consistency detector, which flags replans whose outputs are inconsistent under a JPEG-augmented view. Figure[15](https://arxiv.org/html/2607.15207#S5.F15 "Figure 15 ‣ 5.8 What Do These Results Imply for WAM Safety? ‣ 5 Evaluation ‣ BadWAM: When World-Action Models Dream Right but Act Wrong") shows moderate AUROC, but at 5% false-positive rate it detects only 13.4% of attacked joint-WAM replans and 21.4% of attacked IDM-WAM replans.

Table 2:  Transferability of BadWAM attacks across WAM variants on the LIBERO subset. The attacker optimizes the perturbation on the source WAM and evaluates the closed-loop task success on the target WAM. Lower transferred task success indicates stronger transfer. D_{\mathrm{img}}^{\mathrm{src}} reports the mean source-side imagination distance for imagination-preserving attacks. \dagger denotes matched/subset clean references. 

Table 3:  Results of some non-adaptive defense baselines on LIBERO. Higher attack success indicates better robustness against the imagination-preserving attack. \dagger denotes matched/subset clean references. 

### 5.8 What Do These Results Imply for WAM Safety?

Across RQ1-RQ6, the results point to a subtle failure mode for imagination-based safety checks. A natural WAM monitor might inspect predicted futures and accept an action if the future looks plausible. BadWAM shows that this criterion is incomplete. The relevant security property is not plausibility of the imagined future in isolation, but synchronization between the imagined future and the action that will actually be executed. In the matched-strength run, the imagination-preserving objective lowers predicted-future drift on 39 out of 40 LIBERO tasks while still inducing substantial closed-loop failures. A monitor that only scores visual future plausibility could therefore be satisfied even after the action channel has shifted toward failure.

This also explains why the attack is WAM-specific. For a conventional action-only policy, an adversarial perturbation can only be assessed through its effect on the action or final task outcome. For a WAM, the attacker can exploit a gap between two interfaces: the future-imagination interface and the action interface. The action-only BadWAM objective targets the high-strength endpoint of this spectrum; the imagination-preserving objective targets a stealthier endpoint. The latter is especially relevant for systems that expose predicted futures to planners, runtime monitors, or human supervisors. Even when the predicted future remains visually close to the clean prediction, the executed action chunk may no longer be the action that would realize that future.

The defense experiments reinforce the same lesson. Preprocessing can be a useful diagnostic, but it is not a principled solution to WAM safety. A practical defense must preserve clean control while checking whether action and imagination remain mutually consistent over the closed-loop trajectory. The low-recall detector in Figure[15](https://arxiv.org/html/2607.15207#S5.F15 "Figure 15 ‣ 5.8 What Do These Results Imply for WAM Safety? ‣ 5 Evaluation ‣ BadWAM: When World-Action Models Dream Right but Act Wrong") further shows that moderate AUROC is insufficient for robot safety: runtime monitors must operate at very low false-positive rates, where recall is hardest to maintain.

![Image 16: Refer to caption](https://arxiv.org/html/2607.15207v1/x16.png)

![Image 17: Refer to caption](https://arxiv.org/html/2607.15207v1/x17.png)

Figure 15:  Augmentation-consistency detection is insufficient against imagination-preserving attacks. 

Finally, the transfer results suggest that the vulnerability is not merely an implementation artifact. Perturbations optimized on one WAM variant remain damaging on another, indicating shared observation-space directions that affect action generation. A WAM safety evaluation should therefore go beyond clean and attacked success rates. It should also report action distance, predicted-future distance, input perceptibility, horizon-level shifts, and failure distributions across task families. Together, these metrics distinguish overt action hijacking from stealthier imagination-preserving failures, and test whether a defense restores action-imagination synchronization rather than masking one perturbation pattern.

### 5.9 Summary of Findings

The evaluation supports six conclusions. First, BadWAM substantially lowers closed-loop success across WAM variants, with around 30%-50% drops. Second, the failures are systematic: they persist across repeated trials and are strongest on spatial and long-horizon tasks. Third, the attack induces structured closed-loop failures rather than immediate random behavior. Fourth, imagination preservation creates a measurable stealth trade-off: it keeps predicted futures closer to clean rollouts while retaining comparable perturbation budgets and substantial attack strength. Fifth, BadWAM exposes a tunable sensitivity frontier: attack strength depends strongly on perturbation size, future-preserving weight, and query budget. Sixth, the vulnerability is not isolated to one model: attacks transfer across WAM variants, while simple detection baselines miss most attacks at practical false-positive rates.

Together, these results show that WAM safety cannot be reduced to asking whether the imagined future looks plausible. The more security-relevant question is whether the model’s imagined future and selected action remain synchronized under adversarial observation perturbations.

## 6 Conclusion

World-action models promise a tighter connection between perception, prediction, and control by coupling action generation with imagined futures. This paper shows that this coupling can itself become an attack surface. We introduce BadWAM, a black-box framework for world-action drift attacks that induces task failures through small observation perturbations. Beyond action-only attacks, BadWAM exposes a more WAM-specific failure mode: action-imagination decoupling, where the executed actions are shifted while its predicted future remains comparatively stable. Across multiple WAM variants, BadWAM substantially reduces task success under bounded perturbations. Our analyses show that these failures are not captured by any single metric: task success, action shifts, predicted-future drift, perturbation budget, horizon structure, and transfer behavior each reveal different aspects of the vulnerability. The matched-strength comparison further shows that imagination-preserving attacks can maintain similar attack strength while reducing future drift, suggesting that future-based monitors alone may be overconfident. These results highlight the need to evaluate WAM safety as a synchronization problem between what a model imagines and what it executes. We hope BadWAM provides both a concrete attack framework and a diagnostic protocol for building safer world-action models.

## References

Appendix

## Appendix A Details of Experiment Setup

### A.1 Full and Subset Evaluation Protocols

Table[4](https://arxiv.org/html/2607.15207#A1.T4 "Table 4 ‣ A.1 Full and Subset Evaluation Protocols ‣ Appendix A Details of Experiment Setup ‣ BadWAM: When World-Action Models Dream Right but Act Wrong") summarizes which results are currently reported on the full benchmark sweep and which results use the balanced LIBERO subset. The full LIBERO sweep contains all four suites—Spatial, Object, Goal, and Long-horizon—with 10 tasks per suite and 20 trials per task, resulting in 800 closed-loop episodes per model–attack pair. For expensive diagnostic studies, we use a balanced subset with three tasks per suite (task IDs 0, 4, and 9 in each suite) and 10 trials per task, resulting in 120 episodes per model–attack pair. We use the subset only for analysis-oriented experiments whose purpose is to compare trends under controlled settings. We plan to rerun these subset studies on the full LIBERO sweep and update the corresponding numbers in the final version.

Table 4:  Evaluation scope of the reported experiments. Full LIBERO denotes 40 tasks with 20 trials per task. The balanced subset denotes 12 tasks with 10 trials per task. 

The subset protocol is designed to preserve suite diversity while keeping expensive studies tractable. In particular, each subset experiment contains representatives from spatial reasoning, object manipulation, goal-conditioned manipulation, and long-horizon manipulation. We keep the same subset across future-preserving-weight ablations, perturbation-budget ablations, query-budget ablations, transfer experiments, and defense baselines, so that comparisons within these analyses are matched by task distribution.

### A.2 Training Details

For the action-only WAM, we use the official FastWAM checkpoints[yuan2026fastwam]. The joint WAM and IDM WAM variants used in our LIBERO experiments are trained from the FastWAM release using the default training configuration. For RoboTwin, we use the same FastWAM training recipe and train the joint and IDM variants for 50,000 optimization steps on 8 H100 GPUs.

Table[5](https://arxiv.org/html/2607.15207#A1.T5 "Table 5 ‣ A.2 Training Details ‣ Appendix A Details of Experiment Setup ‣ BadWAM: When World-Action Models Dream Right but Act Wrong") lists the main training settings. For LIBERO, the model uses two cameras, namely the third-person view and the wrist view. Each camera is resized to 224\times 224, and the two camera streams are concatenated horizontally. For RoboTwin, the model uses three cameras: a high camera and two wrist cameras. Each camera is resized to 240\times 320, and the resulting video representation follows the FastWAM RoboTwin configuration. Both benchmarks use 33-frame training windows with an action-video frequency ratio of 4, corresponding to 32 action steps and 9 video frames.

Table 5:  Training configuration for the WAM variants trained in this work. We follow the FastWAM default configuration unless otherwise stated. Batch size is per GPU. 

The model backbone and tokenizer follow FastWAM. Specifically, the configuration uses Wan-AI/Wan2.2-TI2V-5B as the video backbone and Wan-AI/Wan2.1-T2V-1.3B as the tokenizer model. We keep the default FastWAM action and video schedulers, each with 1000 training timesteps and shift value 5.0. We do not modify the FastWAM architecture when training the joint and IDM variants.

### A.3 Attack Hyperparameters and Query Accounting

Unless otherwise stated, BadWAM perturbs the full preprocessed visual input under an \ell_{\infty} bound of \epsilon=0.06. The perturbation is applied online at each replanning step during closed-loop execution. The default zeroth-order finite-difference search uses 16 perturbation queries plus one clean reference query per attacked replan, for a total of 17 WAM forward queries per attacked replan. Internally, the 16 perturbation queries are organized as 8 paired finite-difference updates. Each update samples one random Rademacher direction, evaluates positively and negatively perturbed inputs, and updates the perturbation by a projected signed step. We use a finite-difference radius of 0.02 and a search step size of 0.02.

For the action-only adversarial attack, the future-preserving coefficient is set to zero and the objective only maximizes clean–attacked action deviation. For the imagination-preserving adversarial attack, the objective additionally penalizes predicted-future drift. The default full-sweep setting uses \lambda=0.015 for the future-preserving coefficient. The matched-strength analysis in Figure[12](https://arxiv.org/html/2607.15207#S5.F12 "Figure 12 ‣ 5.4 How Do BadWAM Failures Emerge? ‣ 5 Evaluation ‣ BadWAM: When World-Action Models Dream Right but Act Wrong") uses a larger future-preserving coefficient when comparing the two objectives under similar attack strength, so that the effect of the preservation term is easier to isolate.

### A.4 White-Box Gradient-Access Reference

Figure[6](https://arxiv.org/html/2607.15207#S5.F6 "Figure 6 ‣ 5.1 Experimental Setup ‣ 5 Evaluation ‣ BadWAM: When World-Action Models Dream Right but Act Wrong") includes a white-box gradient-access reference to contextualize the strength of our query-based attacks. This reference is not part of the main BadWAM threat model. BadWAM assumes query access and does not require gradients or model parameters. The white-box reference instead asks how much stronger the attack can become if the adversary can backpropagate through the WAM’s action output.

At the beginning of an episode, the attacker optimizes one bounded input perturbation using projected gradient ascent. The objective maximizes the deviation between the clean action chunk and the perturbed action chunk. The perturbation is initialized randomly inside the \ell_{\infty} ball, updated for 16 gradient steps with step size 0.01, and projected back to \|\delta\|_{\infty}\leq 0.06 after each step. After the perturbation is found, the policy is evaluated with the standard closed-loop inference procedure. No model weights are modified, and the environment dynamics are not differentiated through.

This reference serves two purposes. First, it separates generic visual fragility from targeted action manipulation: random perturbations under the same budget are substantially weaker than targeted attacks. Second, it shows that gradient access can further amplify the vulnerability, which suggests that the query-based BadWAM attacks are not saturating the worst-case attack strength. Since this reference uses stronger access than BadWAM, we treat it as a diagnostic upper-bound reference rather than as the primary threat model.

### A.5 Random Perturbation Baseline

The random perturbation baseline uses the same \ell_{\infty} budget as BadWAM but removes the optimization objective. For each attacked observation, it samples an independent uniform perturbation from [-\epsilon,\epsilon] and clips the perturbed input to the valid model input range. This baseline is model-agnostic and does not use action outputs, future predictions, gradients, or task feedback. Its role is to test whether task failures are caused merely by arbitrary visual corruption. The results in Figure[6](https://arxiv.org/html/2607.15207#S5.F6 "Figure 6 ‣ 5.1 Experimental Setup ‣ 5 Evaluation ‣ BadWAM: When World-Action Models Dream Right but Act Wrong") show that optimized world-action drift attacks are consistently stronger than random perturbations under the same visual budget.

### A.6 Evaluation Metrics

Our primary metric is closed-loop task success. Lower success under attack indicates a stronger attack. In addition to success rate, we report action distance, predicted-future distance, and input perturbation statistics. The action distance is computed as the mean \ell_{2} deviation between clean and attacked action chunks. The predicted-future distance is computed analogously on the future representation or decoded future-video tensor available through the WAM interface. The decoupling score is the ratio between action distance and predicted-future distance, with a small numerical constant in the denominator for stability. A high decoupling score indicates that the attack obtains a larger action shift per unit of predicted-future drift.

For repeated-trial evaluation, we use pass@k. For a fixed task, pass@k is the fraction of successful executions among the first k trials. We then average this quantity across tasks. This metric shows whether an attack only causes isolated unlucky failures or consistently lowers reliability as more trials are considered.

