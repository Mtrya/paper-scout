Title: Hallucination in World Models is Predictable and Preventable

URL Source: https://arxiv.org/html/2606.27326

Published Time: Fri, 26 Jun 2026 01:04:35 GMT

Markdown Content:
###### Abstract

## 1 Introduction

Modern generative world models render strikingly realistic, action-controllable futures across diverse environments (Alonso et al., [2024](https://arxiv.org/html/2606.27326#bib.bib34 "Diffusion for world modeling: visual details matter in atari"); Valevski et al., [2024](https://arxiv.org/html/2606.27326#bib.bib36 "Diffusion models are real-time game engines"); Hafner et al., [2025](https://arxiv.org/html/2606.27326#bib.bib30 "Training agents inside of scalable world models"); Parker-Holder et al., [2025](https://arxiv.org/html/2606.27326#bib.bib39 "Genie 3: a new frontier for world models")). However, the rollouts they produce frequently _hallucinate_: they remain visually fluent and superficially plausible while drifting away from the ground-truth dynamics (Janner et al., [2019](https://arxiv.org/html/2606.27326#bib.bib4 "When to trust your model: model-based policy optimization")). The term is borrowed from the language modeling literature (Ji et al., [2023](https://arxiv.org/html/2606.27326#bib.bib43 "Survey of hallucination in natural language generation"); Huang et al., [2025](https://arxiv.org/html/2606.27326#bib.bib42 "A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions")), where it typically denotes generation of factually incorrect text; analogous failures have also been studied in image (Li et al., [2023](https://arxiv.org/html/2606.27326#bib.bib45 "Evaluating object hallucination in large vision-language models")) and video generation (Huang et al., [2024](https://arxiv.org/html/2606.27326#bib.bib44 "VBench: comprehensive benchmark suite for video generative models")). In a world model, the failure is arguably more consequential: hallucinated trajectories are fed directly into downstream planners and policies (Schrittwieser et al., [2020](https://arxiv.org/html/2606.27326#bib.bib5 "Mastering atari, go, chess and shogi by planning with a learned model"); Hafner et al., [2023](https://arxiv.org/html/2606.27326#bib.bib9 "Mastering diverse domains through world models"); Hansen et al., [2024](https://arxiv.org/html/2606.27326#bib.bib21 "TD-mpc2: scalable, robust world models for continuous control")), so silent hallucination during rollout translates into silently incorrect decisions during control.

Despite the increasing fidelity of these models, _where_ an autoregressive rollout will hallucinate, and _why_, is poorly understood. A natural reading is that hallucination is an architectural problem to be solved by ever-larger backbones and more training compute (Hoffmann et al., [2022](https://arxiv.org/html/2606.27326#bib.bib53 "Training compute-optimal large language models")). However, we argue that hallucination in world models is primarily a data coverage problem: it concentrates in low-coverage regions of the state-action space (Levine et al., [2020](https://arxiv.org/html/2606.27326#bib.bib58 "Offline reinforcement learning: tutorial, review, and perspectives on open problems"); Gadre et al., [2023](https://arxiv.org/html/2606.27326#bib.bib54 "Datacomp: in search of the next generation of multimodal datasets")) and is therefore both _predictable_ from signals available at runtime (Lakshminarayanan et al., [2017](https://arxiv.org/html/2606.27326#bib.bib47 "Simple and scalable predictive uncertainty estimation using deep ensembles"); Gal and Ghahramani, [2016](https://arxiv.org/html/2606.27326#bib.bib48 "Dropout as a Bayesian approximation: representing model uncertainty in deep learning"); Chua et al., [2018](https://arxiv.org/html/2606.27326#bib.bib6 "Deep reinforcement learning in a handful of trials using probabilistic dynamics models")) and _preventable_ by adjusting which data the model is trained on rather than architectural changes. Our experiments reveal that a single underlying cause – coverage gaps – explains failures at every stage of the model pipeline: tokenizer, action-conditioning, and multi-step rollout, and that it manifests as three distinct failure modes shown in Figure[1](https://arxiv.org/html/2606.27326#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Hallucination in World Models is Predictable and Preventable").

\begin{overpic}[width=49.16362pt]{visualizations/hallucination/4_pygame-reacher-easy-enc.png} \put(-0.1,86.85){\hbox{\pagecolor{white}{\color[rgb]{0,0,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,0}\pgfsys@color@gray@stroke{0}\pgfsys@color@gray@fill{0}enc}}} \end{overpic}\begin{overpic}[width=49.16362pt]{visualizations/hallucination/4_pygame-reacher-easy-dec.png} \put(-0.1,86.85){\hbox{\pagecolor{white}{\color[rgb]{0,0,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,0}\pgfsys@color@gray@stroke{0}\pgfsys@color@gray@fill{0}dec}}} \end{overpic}

Reconstruction of OOD scene poor but structurally faithful ✓ 

\begin{overpic}[width=49.16362pt]{visualizations/hallucination/pygame-point-maze-var4-enc.png} \put(-0.1,86.85){\hbox{\pagecolor{white}{\color[rgb]{0,0,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,0}\pgfsys@color@gray@stroke{0}\pgfsys@color@gray@fill{0}enc}}} \end{overpic}\begin{overpic}[width=49.16362pt]{visualizations/hallucination/pygame-point-maze-var4-dec.png} \put(-0.1,86.85){\hbox{\pagecolor{white}{\color[rgb]{0,0,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,0}\pgfsys@color@gray@stroke{0}\pgfsys@color@gray@fill{0}dec}}} \end{overpic}

Unseen layout is reconstructed as a similar, _seen_ layout✗ 

_(i)_ Perceptual

![Image 1: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/hallucination/hallucination-maniskill-action-down2.png)![Image 2: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/hallucination/hallucination-maniskill-action-down-frame2.png)

Predicted next frame reflects the action taken by the user ✓ 

![Image 3: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/hallucination/hallucination-maniskill-action-up2.png)![Image 4: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/hallucination/hallucination-maniskill-action-up-frame2.png)

Visually plausible but _ignores_ the action actually taken✗ 

_(ii)_ Action marginalization

![Image 5: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/hallucination/pygame-pong_ep001_t0034_wm.png)![Image 6: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/hallucination/pygame-pong_ep001_t0037_wm.png)

Plausible dynamics despite slight error accumulation ✓ 

\begin{overpic}[width=49.16362pt]{visualizations/hallucination/hallucination-pong-frame1-thick.png} \put(56.0,86.85){\hbox{\pagecolor{white}{\color[rgb]{0,0,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,0}\pgfsys@color@gray@stroke{0}\pgfsys@color@gray@fill{0}\small$t{=}34$}}} \end{overpic}\begin{overpic}[width=49.16362pt]{visualizations/hallucination/hallucination-pong-frame2-thick.png} \put(56.0,86.85){\hbox{\pagecolor{white}{\color[rgb]{0,0,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,0}\pgfsys@color@gray@stroke{0}\pgfsys@color@gray@fill{0}\small$t{=}36$}}} \end{overpic}

_Implausible_ dynamics (ball teleports back into play)✗ 

_(iii)_ Scene divergence

Figure 1: Hallucination. We categorize hallucinations as three distinct failure modes: _perceptual_, _action marginalization_, and _scene divergence_, and develop methods for detection and mitigation.

In this work, we set out to characterize different types of hallucinations, predict when they will occur using signals internal to the model, and use that information to close the underlying coverage gap. We explore two concrete ways in which the gap can be closed: _(i)_ using coverage-aware sampling techniques during large-scale pretraining, and _(ii)_ by targeted online data collection using our derived predictors of hallucination. Understanding when and why hallucinations happen requires three resources that no single benchmark currently provides: full control over the training corpus, behaviorally diverse data spanning many tasks and domains, as well as live environments to probe coverage gaps via online interaction. To address this gap we develop MMBench2, a massively multitask dataset for visual world modeling that extends MMBench(Hansen et al., [2026](https://arxiv.org/html/2606.27326#bib.bib29 "Learning massively multitask world models for continuous control")) with 65.6 k mixed-quality trajectories equivalent to 427 hours of 224{\times}224 video at 15 fps, complete with ground-truth action and reward labels and live simulators across 210 tasks spanning 10 domains. Of the 210 tasks, 200 form the pretraining corpus and 10 are held out as entirely unseen transfer tasks, allowing us to probe coverage both within and beyond the training distribution.

Using MMBench2, we train a 350 M parameter Dreamer 4 (Hafner et al., [2025](https://arxiv.org/html/2606.27326#bib.bib30 "Training agents inside of scalable world models")) world model and dissect its hallucination behavior. We identify three distinct hallucination modes, each anchored to a different stage of the pipeline: _perceptual_ hallucination in the encoder-decoder pair, _action marginalization_ in the dynamics model, and _scene divergence_ during multi-step rollouts. We develop three predictors of hallucination: tokenizer round-trip residual, flow instability, and inter-seed denoising variance, and use them to detect hallucination at runtime, as well as curiosity reward for online data collection (Sekar et al., [2020](https://arxiv.org/html/2606.27326#bib.bib24 "Planning to explore via self-supervised world models")) in seen and unseen tasks. Empirically, our hallucination-driven data collection enables adaptation to entirely unseen environments with just 50 real trajectories, approaching the effectiveness of human data collection. Together, these results show that hallucination in modern world models is, to a large extent, a coverage problem, and that the same signals that detect it can also be used to mitigate it. We summarize our contributions as follows:

*   •
MMBench2: a large 427-hour dataset for visual world modeling with ground-truth actions and rewards, live environments, and behaviorally diverse data including human play.

*   •
A stage-by-stage characterization of hallucination in generative world models tied to tokenizer, action marginalization, and rollout failures.

*   •
Three hallucination predictors that detect hallucination without labels or additional training.

*   •
A coverage-aware training recipe that reduces hallucination across all three predictors and improves rollout fidelity at no additional cost.

*   •
A framework for targeted data collection that adapts a pretrained 350 M world model to unseen environments with as few as 50 trajectories.

To support further research on generative world modeling, _we release our full dataset, code for training and evaluation, model checkpoints, and a browser interface for open-ended interaction with the world model._ See [https://nicklashansen.com/mmbench2](https://nicklashansen.com/mmbench2) for videos and resources.

![Image 7: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/tasks/pygame-bird-attack-2.png)

![Image 8: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/tasks/rd-open-drawer-1.png)

![Image 9: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/tasks/pygame-whirlpool-1.png)

![Image 10: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/domains/maniskill/ms-anymal-reach.png)

![Image 11: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/domains/pygame/pygame-point-maze-var1.png)

![Image 12: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/domains/box2d/bipedal-walker-rugged.png)

![Image 13: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/domains/pygame/pygame-air-hockey.png)

![Image 14: [Uncaptioned image]](https://arxiv.org/html/2606.27326v1/visualizations/domains/pygame/pygame-coconut-dodge.png)

![Image 15: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/tasks/atari-double-dunk-5.png)

![Image 16: [Uncaptioned image]](https://arxiv.org/html/2606.27326v1/visualizations/domains/mujoco/mujoco-walker.png)

![Image 17: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/domains/dmcontrol-extended/jumper-jump.png)

![Image 18: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/domains/pygame/pygame-chase-evade.png)

![Image 19: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/domains/dmcontrol/finger-turn-hard.png)

![Image 20: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/domains/pygame/pygame-rocket-collect.png)

![Image 21: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/tasks/atari-seaquest-2.png)

![Image 22: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/tasks/pygame-highway-1.png)

![Image 23: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/domains/pygame/pygame-pong.png)

![Image 24: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/tasks/pygame-dungeon-explorer1-0.png)

![Image 25: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/domains/pygame/pygame-cartpole-tremor.png)

![Image 26: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/tasks/atari-road-runner-3.png)

![Image 27: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/domains/maniskill/ms-pick-banana.png)

![Image 28: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/tasks/atari-upndown-3.png)

![Image 29: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/domains/ogbench/og-point-maze.png)

![Image 30: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/domains/pygame/pygame-coinrun.png)

![Image 31: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/domains/dmcontrol-extended/giraffe-run.png)

![Image 32: [Uncaptioned image]](https://arxiv.org/html/2606.27326v1/visualizations/domains/ogbench/og-antball.png)

![Image 33: [Uncaptioned image]](https://arxiv.org/html/2606.27326v1/visualizations/domains/box2d/bipedal-walker-obstacles.png)

![Image 34: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/domains/metaworld/mw-hammer.png)

![Image 35: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/domains/open-loop/atari-assault-48.png)

![Image 36: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/tasks/lunarlander-takeoff-5.png)

![Image 37: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/tasks/cup-catch-var1-1.png)

![Image 38: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/tasks/quadruped-run-4.png)

![Image 39: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/domains/robodesk/rd-open-slide.png)

![Image 40: [Uncaptioned image]](https://arxiv.org/html/2606.27326v1/visualizations/domains/metaworld/mw-stick-pull.png)

![Image 41: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/tasks/pygame-foraging-2.png)

![Image 42: [Uncaptioned image]](https://arxiv.org/html/2606.27326v1/visualizations/domains/maniskill/ms-poke-cube.png)

Figure 2: MMBench2 tasks. A sample of 36 of the 210 tasks in MMBench2, illustrating the visual and morphological diversity of the corpus. All observations are 224{\times}224 RGB frames.

## 2 MMBench2: A Dataset for Visual World Modeling

To fully understand hallucination in large generative world models, it is necessary to have control over the training process as well as access to its training data, but (to our knowledge) no such resource exists today at a satisfactory scale. To address this gap, we propose MMBench2: a massively multitask dataset for visual world modeling. Our dataset consists of \mathbf{65{,}600} trajectories (427 hours of 224{\times}224 video at 15 fps for a total of 23 M frames) with diverse behaviors (including human play data) across \mathbf{210} different continuous control tasks, complete with ground-truth action and reward labels as well as live environments and language instructions for every task included in MMBench2. As the name implies, MMBench2 directly builds on top of MMBench which provides a collection of 200 live environments unified under a common interface. Figure[2](https://arxiv.org/html/2606.27326#S1.F2 "Figure 2 ‣ 1 Introduction ‣ Hallucination in World Models is Predictable and Preventable") shows the breadth of our tasks.

Task overview. MMBench2 spans \mathbf{10} task domains: DMControl(Tassa et al., [2018](https://arxiv.org/html/2606.27326#bib.bib7 "DeepMind control suite")), DMControl Extended(Hansen et al., [2026](https://arxiv.org/html/2606.27326#bib.bib29 "Learning massively multitask world models for continuous control")), Meta-World(Yu et al., [2019](https://arxiv.org/html/2606.27326#bib.bib8 "Meta-world: a benchmark and evaluation for multi-task and meta reinforcement learning")), ManiSkill3(Tao et al., [2025](https://arxiv.org/html/2606.27326#bib.bib15 "ManiSkill3: gpu parallelized robotics simulation and rendering for generalizable embodied ai")), MuJoCo(Todorov et al., [2012](https://arxiv.org/html/2606.27326#bib.bib19 "MuJoCo: a physics engine for model-based control")), MiniArcade(Hansen et al., [2026](https://arxiv.org/html/2606.27326#bib.bib29 "Learning massively multitask world models for continuous control")), Box2D(Brockman et al., [2016](https://arxiv.org/html/2606.27326#bib.bib20 "OpenAI gym")), RoboDesk(Kannan et al., [2021](https://arxiv.org/html/2606.27326#bib.bib16 "RoboDesk: a multi-task reinforcement learning benchmark")), OGBench(Park et al., [2025](https://arxiv.org/html/2606.27326#bib.bib17 "OGBench: benchmarking offline goal-conditioned rl")), and Continuous Atari(Farebrother and Castro, [2024](https://arxiv.org/html/2606.27326#bib.bib18 "CALE: continuous arcade learning environment")). Together they cover locomotion, dexterous and tabletop manipulation, goal-conditioned navigation, and arcade-style environments, with continuous action spaces of 1 to 16 dimensions; we zero-pad every action vector to the maximum dimension (d_{a}{=}16) and supply a per-dimension validity mask so models can ignore inactive dimensions per task. Of the \mathbf{210} tasks, 200 form the pretraining set and 10 (proposed in this work) are held out as _unseen_ transfer tasks.

The 200-task pretraining corpus contains 260 episodes per task. However, episode lengths vary by task, ranging from 25 for certain ManiSkill3 manipulation tasks up to 1{,}000 for Atari games. As a result, the corpus is highly non-uniform across tasks as shown in Figure[3](https://arxiv.org/html/2606.27326#S2.F3 "Figure 3 ‣ 2 MMBench2: A Dataset for Visual World Modeling ‣ Hallucination in World Models is Predictable and Preventable"). Additionally, the visual diversity _within_ a task also differs wildly across the corpus with, _e.g._, Inverted Pendulum (MuJoCo) varies little whereas tasks like Dungeon Explorer (proposed in this work), Bipedal Walker (Box2D), and Krull (Atari) vary greatly between frames and episodes. Please refer to Appendix[C](https://arxiv.org/html/2606.27326#A3 "Appendix C Task Domains ‣ Hallucination in World Models is Predictable and Preventable") for an overview of tasks and Appendix[D](https://arxiv.org/html/2606.27326#A4 "Appendix D Data Collection ‣ Hallucination in World Models is Predictable and Preventable") for a detailed description of the data collection process.

![Image 43: Refer to caption](https://arxiv.org/html/2606.27326v1/x1.png)

Figure 3: Dataset composition. Per-task frame counts across the 210 tasks in our training corpus (20M frames), sorted in descending order and colored by domain. The distribution is heavy-tailed: the top 20 tasks account for 26\% of all frames (dominated by Atari) while the bottom 20 contribute only 0.7\% (short-horizon manipulation tasks). The dashed line marks the per-task median (65 k).

## 3 Training a Large Visual World Model

We seek to answer the following question: _why_ do world models hallucinate, and can we predict _when_ they are likely to hallucinate? To answer this question, we choose to develop and train a 350M parameter visual world model that largely follows the overall architecture and training recipe introduced in Dreamer 4(Hafner et al., [2025](https://arxiv.org/html/2606.27326#bib.bib30 "Training agents inside of scalable world models")). As such, our world model is an action-conditioned generative model that follows an increasingly common two-stage training recipe: _(1)_ first, a video _tokenizer_ is trained via masked auto-encoding of visual observations, and _(2)_ a dynamics model implemented as a block-causal Transformer (Vaswani et al., [2017](https://arxiv.org/html/2606.27326#bib.bib12 "Attention is all you need")) is subsequently trained via flow-matching over spatial latent tokens produced by the frozen tokenizer with additional conditioning on action tokens. We first pretrain our world model on the MMBench2 training corpus, and then address our specific research questions by conducting a series of targeted finetuning experiments. This section aims to provide an overview of our architecture and training recipe.

Tokenizer. Our video tokenizer is instantiated as a symmetric encoder-decoder Transformer architecture. The encoder ingests 224{\times}224 RGB frames “patchified” at stride 14 (256 patch tokens per frame), prepends 64 learnable latent queries, and projects the latent stream to a 64-dimensional bottleneck bounded by a \tanh activation, producing a per-frame code z\in[-1,1]^{64\times 64}. The decoder then reconstructs images conditioned only on latent codes. Training uses a masked reconstruction objective (He et al., [2022](https://arxiv.org/html/2606.27326#bib.bib13 "Masked autoencoders are scalable vision learners")) where a fraction of input patches drawn per-frame from \mathcal{U}(0,0.9) are replaced by a learned mask token, and the loss (pixel MSE + LPIPS (Zhang et al., [2018](https://arxiv.org/html/2606.27326#bib.bib31 "The unreasonable effectiveness of deep features as a perceptual metric"))) is computed only on masked positions. We normalize losses by their running RMS to reduce sensitivity to hyperparameters. Encoder and decoder are 50M learnable parameters each.

Dynamics model. Our dynamics model is a 250M parameter block-causal Transformer over packed tokenizer latents, trained on top of the _frozen_ tokenizer introduced above. Per timestep, the input sequence consists of an action token (a 2-layer MLP over the 16-dimensional padded action), a shortcut-conditioning token (encoding noise level \sigma and step size d), 32 packed spatial latent tokens, 4 register tokens, and optional agent tokens used as the readout for the reward and behavior-cloning heads. The model is trained with the _shortcut flow-matching_ objective of Frans et al. ([2025](https://arxiv.org/html/2606.27326#bib.bib32 "One step diffusion via shortcut models")), which interleaves an empirical one-step regression term with a self-consistency bootstrap that distills two coarser-step predictions into one finer-step target. At inference this enables next-frame sampling in as few as 4 Euler substeps.

Training recipe. The training recipe of Dreamer 4 was developed for a setting where only a small percentage of the training data has action and reward labels. However, MMBench2 provides ground-truth actions and rewards for every trajectory in the dataset which we choose to take advantage of. We first pretrain the tokenizer on the full training corpus, and then proceed to also pretrain the dynamics model, conditioning on actions. After this initial pretraining phase, we experiment with additional mid-training and finetuning recipes introduced in the following sections. As part of our experiments, we initialize additional reward prediction and behavior cloning (BC) policy heads after pretraining. The reward predictor is trained via L{=}8 multi-step discrete regression (using \operatorname{symlog} two-hot encoding) with gradients backpropagated through the dynamics model. The BC policy is a deterministic Gaussian policy trained to predict ground-truth actions via an MSE loss – this is a departure from Dreamer 4 which only considered discrete actions.

## 4 Hallucination in World Models

A generative world model imagines a future by chaining three distinct operations: _(1)_ an encoder maps each observation into a latent code, _(2)_ an action-conditioned dynamics head predicts the next latent, and _(3)_ a decoder renders a reconstruction back to pixel space. Each of these stages is a learned function trained on a finite slice of the state-action space, and each can therefore fail _independently_ when asked to extrapolate beyond what it has seen. We use the term _hallucination_ to refer to any such failure: an output that is fluent and visually plausible, yet decoupled from reality. Crucially, because the three stages compose sequentially, a hallucination introduced early (_e.g._ a corrupted encoding) is propagated and amplified by the stages that follow, so naming _which_ stage produces a given failure is a prerequisite for diagnosing and fixing it. This section explores how hallucinations can be characterized, detected, and finally mitigated.

### 4.1 Characterizing Hallucination

We identify three distinct ways in which a generative world model may hallucinate, each tied to a different stage of the imagination pipeline. We illustrate each type of hallucination in Figure[1](https://arxiv.org/html/2606.27326#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Hallucination in World Models is Predictable and Preventable") and define them as follows:

1.   _(i)_
Perceptual hallucination. The tokenizer’s reconstruction of an observation already differs from the observation itself, before any dynamics rollout has been executed. Concretely, the encoder–decoder pair projects out-of-distribution scene structure onto the closest in-distribution exemplar in its learned latent “vocabulary”. For example, an unseen maze layout might be reconstructed with the agent and goal in the correct positions but with the walls of an entirely _different_ layout seen during training. The dynamics head then rolls out against this corrupted scene as if it were ground-truth. This failure mode is a property of the frozen encoder–decoder pair alone and persists even at horizon H{=}0.

2.   _(ii)_
Action-marginalized (ignored) hallucination. Conditional on a context, the predicted next latent is largely insensitive to the input action. The rollout is visually plausible but collapses onto an action-marginalized future, so the model behaves more like a video generator than a controllable world model. Operationally, we expose this mode by intervening on the action stream at evaluation time, _e.g._ by randomly shuffling actions within a batch and measuring the resulting change in flow MSE; a model that hallucinates in this way is one whose flow MSE barely moves under the intervention.

3.   _(iii)_
Scene-diverging hallucination. It is well understood that autoregressive rollouts accumulate compounding error as the prediction horizon increases. However, _scene-diverging_ hallucination is a very specific failure mode where physically implausible events (such as a ball teleporting back into play when scoring in Pong) are predicted. This type of hallucination is most frequent in states with poor data coverage.

These three types of failure modes probe disjoint pieces of the model: the tokenizer, the action-conditioning of the dynamics model, and the multi-step accumulation of dynamics error.

### 4.2 Detecting Hallucination

Next, we develop _three_ distinct predictors designed to track model hallucination. Empirically we find that these three metrics, although mechanistically distinct, are all strong predictors of hallucination events. We view this convergence as a feature and not a redundancy, since three mechanistically distinct predictors agreeing with respect to hallucination events is in itself evidence that the underlying signal is real. None of them require any labels nor additional training which makes them especially suitable for runtime detection.

Proposed predictor 1: tokenizer round-trip residual.u_{r}=\|\hat{z}-\mathrm{Encode}(\mathrm{Decode}(\hat{z}))\|, the latent-space residual of a single decode–encode round-trip of the dynamics-predicted next latent \hat{z}, targets perceptual hallucination by measuring the very symptom that defines it: a predicted latent whose decoded frame falls off the tokenizer’s manifold, _e.g._ a corrupted scene layout or a fabricated object, does not survive re-encoding and produces a large u_{r}.

Proposed predictor 2: flow instability.u_{f}, the _flow instability_ of the dynamics head at a given (\text{context},\text{action}) pair, measures how much the denoiser’s clean-target prediction \hat{x}_{1} moves between successive Euler integration substeps, averaged over the second half of substeps. A sharp, well-conditioned dynamics head converges quickly to a stable \hat{x}_{1} (low u_{f}); a head whose conditioning provides little signal keeps oscillating across substeps (high u_{f}).

Proposed predictor 3: inter-seed variance.u_{s}, the _inter-seed variance_ of the next-latent prediction across N independent denoising trajectories at fixed past and action, targets scene-diverging hallucination by measuring epistemic uncertainty across noise seeds: regions where seeds disagree are precisely those where multi-step rollouts will fan out.

![Image 44: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/panels_old/og-point-maze_initial_3.png)

![Image 45: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/panels/og-point-maze_density.png)

![Image 46: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/panels/og-point-maze_hallu_u_r.png)

![Image 47: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/panels/pygame-rocket-collect_initial_4.png)

![Image 48: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/panels/pygame-rocket-collect_density.png)

![Image 49: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/panels/pygame-rocket-collect_hallu_u_r.png)

![Image 50: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/panels_old/cup-catch_initial_1.png)

![Image 51: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/panels/cup-catch_density.png)

![Image 52: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/panels/cup-catch_hallu_u_r.png)

![Image 53: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/domains/box2d/lunarlander-takeoff.png)

![Image 54: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/panels/lunarlander-takeoff_hallu_coverage.png)

![Image 55: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/panels/lunarlander-takeoff_hallu_u_r.png)

Figure 4: Data coverage and hallucinations. From left to right: sample frame, state density of key agent/object position, and tokenizer round-trip residual u_{r} of the world model. We use dark blue and dark purple to denote high state density and high{\color[rgb]{0.45703125,0.25,0.61328125}\definecolor[named]{pgfstrokecolor}{rgb}{0.45703125,0.25,0.61328125}\mathbf{u_{r}}}, respectively. Hallucinations concentrate in low-coverage regions, especially near the periphery of the visited state distribution.

Controlling for scene motion. In practice, we find that a naive use of the three signals above is confounded by scene activity: high-motion transitions inflate tokenizer residuals, flow instability, and seed-to-seed denoising dispersion alike. To remove this confound we instead report dynamism-normalized variants u^{\text{norm}}\doteq u/m where m is the per-step RMS change of the latent representation, computed as a per-task average over the dataset, or as a running estimate when used online for data collection. Normalizing by scene motion means that each predictor tracks model uncertainty _relative to how much is happening in the scene_. We treat u_{r}^{\text{norm}},u_{f}^{\text{norm}},u_{s}^{\text{norm}} as our primary hallucination predictors in all subsequent analysis.

### 4.3 Mitigating Hallucination

The taxonomy and predictors above suggest a single data-centric lens on hallucination: each of the three failure modes are, mechanistically, a consequence of the model having seen too little of some region of the state-action space. A perceptual hallucination is a coverage gap in the tokenizer’s reconstruction distribution, an action-marginalized hallucination is a coverage gap in action-conditional transitions, and a scene-diverging hallucination is a coverage gap along the trajectory the model is asked to imagine. Figure[4](https://arxiv.org/html/2606.27326#S4.F4 "Figure 4 ‣ 4.2 Detecting Hallucination ‣ 4 Hallucination in World Models ‣ Hallucination in World Models is Predictable and Preventable") shows the relationship between data coverage and hallucination on four of our tasks. Two interventions follow naturally:

Coverage-aware training. We propose to resample the existing dataset to upweight under-represented regions of the state-action space, then ask whether closing those gaps at training time reduces all three failure modes simultaneously. Because the lens above identifies coverage as the underlying lever for every type of hallucination, a single reweighting recipe is expected to move all three signals in the right direction at once, rather than requiring a separate intervention per type. In practice, we rebalance training data by adjusting sampling to be uniform across _tasks_ rather than _frames_. We also experiment with loss re-weighting but find interventions in sampling to be superior.

Targeted data collection. When the existing dataset does not have enough coverage of a region for re-weighting alone to help, the hallucination predictors can themselves act as an objective for targeted, curiosity-driven data collection. Specifically, during interaction with a live environment, candidate trajectories are rolled out in the world model, scored by predicted hallucination, and the highest-ranked trajectory is executed in the environment, producing data that, by construction, covers transitions that previously caused the model to hallucinate. In practice, we collect data in a closed-loop manner with a prediction horizon of H=32 and replanning every K=16 steps.

## 5 Experiments

We train a 350M parameter action-conditioned world model on the training corpus of MMBench2. Our training data consists of approximately 20M frames at 224\times 224 resolution across 200 distinct continuous control tasks, and we reserve the remaining 3 M frames for testing. Although low-dimensional state information is readily available in MMBench2, we choose to strictly consider RGB observations in this work. We describe our experimental setup below.

Evaluation. We evaluate our world model and its variants across four successive training phases: _(1)_ action-conditioned pretraining _without_ reward labels, _(2)_ coverage-aware "mid-training" that extends the previous phase with our proposed reweighted sampling, _(3)_ action-conditioned world modeling _with_ reward labels, and _(4)_ finetuning on seen and unseen tasks via targeted data collection. Our key evaluation metrics include:

1.   _(i)_
Reconstruction PSNR\uparrow evaluates encoder/decoder quality without considering dynamics.

2.   _(ii)_
Rollout PSNR gain (dB)\uparrow evaluates quality of generated rollouts relative to a baseline that repeats the last frame across the entire rollout horizon. Although naive, this baseline can be surprisingly strong depending on the task. We consider a scene _divergent_ when \Delta PSNR \leq 0.

3.   _(iii)_
Action shuffle ratio\uparrow evaluates action sensitivity in the dynamics model by measuring one-step teacher-forced flow MSE relative to batch-shuffled actions. We consider actions to be _ignored_ (marginalized) when this ratio \leq 1.1.

4.   _(iv)_
Downstream task performance (normalized score)\uparrow evaluates how useful the world model is for downstream tasks via closed-loop MPC using the Cross-Entropy Method (CEM) with horizon H=32 and replanning every 16 steps. We report normalized score s\in[0,1] rather than raw reward since reward scales differ significantly (by several orders of magnitude) across tasks.

Implementation details. The encoder and decoder each have 50M learnable parameters, and the dynamics model (including prediction heads) has 250M parameters. We use 8\times NVIDIA H100 GPUs for training. The tokenizer is pretrained for 300 k steps (14 GPU days), and the dynamics model is pretrained for 180 k steps (24 GPU days). Our final checkpoints (post-finetuning) are trained for 380 k and 210 k steps, respectively, for a total of 58 GPU days using a context length of T=24. The reward head and BC policy are conditioned on CLIP-ViT/B embeddings of per-task language instructions. See Appendix[F](https://arxiv.org/html/2606.27326#A6 "Appendix F Implementation Details ‣ Hallucination in World Models is Predictable and Preventable") for more implementation details.

We seek to answer the following questions:

1.   _(Q1)_
Hallucination detection. Do our proposed metrics accurately predict model hallucination?

2.   _(Q2)_
Pretraining. Does coverage-aware training meaningfully reduce hallucination?

3.   _(Q3)_
Targeted data collection. Can model hallucination be mitigated by collecting additional data? Given a fixed budget, which type of data improves the world model the most?

4.   _(Q4)_
Generalization. How can a pretrained world model be adapted to unseen tasks? Does our world model generalize zero-shot or is finetuning necessary? When does it fail to transfer?

![Image 56: Refer to caption](https://arxiv.org/html/2606.27326v1/x2.png)

Figure 5: Our three hallucination predictors track the same realized rollout error. Each point corresponds to one of 9 k held-out 24-frame sequences from any of 200 training tasks, computed using our pretrained base model; the purple curve shows the median for each of 8 bins.

### 5.1 Main results

Detecting hallucination events. To determine whether our proposed hallucination metrics are predictive of hallucination and by extension model performance, we compute Spearman correlations between open-loop rollout \Delta PSNR and each of our predictors; results are shown in Figure[5](https://arxiv.org/html/2606.27326#S5.F5 "Figure 5 ‣ 5 Experiments ‣ Hallucination in World Models is Predictable and Preventable"). We find a strong negative correlation (\rho\approx 0.80) across all three predictors, which indicates that they all track the same underlying error. We additionally measure their per-task aggregated AUROC against two binary hallucination labels, _action ignored_ (action shuffle ratio \leq 1.1) and _scene divergent_ (rollout \Delta PSNR vs. frame-repeating baseline \leq 0); results are shown in Appendix[E](https://arxiv.org/html/2606.27326#A5 "Appendix E Additional Results ‣ Hallucination in World Models is Predictable and Preventable"). We find that our predictors consistently outperform their unnormalized counterparts (_i.e._, u_{f}^{\text{norm}} is a better predictor than u_{f}) as well as baselines that rely on latent scene motion m or per-task frame count.

Table 1: Coverage-aware training, by stage. Mean change with coverage-aware training vs. the base model on held-out trajectories across 200 tasks. _Tok ft_ finetunes the tokenizer for 30 k steps using coverage-aware training and then finetunes the dynamics model for 30 k steps using default sampling, _Dyn ft_ reverses this, and _Both_ finetunes both for 30 k steps each using coverage-aware training. Best is in bold.

Metric Tok ft Dyn ft Both
Recon PSNR (dB) \uparrow\mathbf{+0.46}-0.01+0.44
Action-shuffle ratio \uparrow+0.02\mathbf{+0.27}+0.29
Rollout \Delta PSNR (dB) \uparrow+0.42+0.68\mathbf{+0.88}
u_{r}^{\text{norm}}\downarrow-0.07-0.16\mathbf{-0.20}
u_{f}^{\text{norm}}\downarrow-0.03-0.06\mathbf{-0.07}
u_{s}^{\text{norm}}\downarrow-0.06-0.13\mathbf{-0.14}

Coverage-aware training. We evaluate the efficacy of our proposed coverage-aware training by extending the pretraining of our base model by an additional 30 k steps for the tokenizer and dynamics model each, while varying the sampling method: uniform sampling across _frames_ (default), and uniform sampling across _tasks_ (ours). Results are shown in Table[1](https://arxiv.org/html/2606.27326#S5.T1 "Table 1 ‣ 5.1 Main results ‣ 5 Experiments ‣ Hallucination in World Models is Predictable and Preventable"). We find that tokenizer and dynamics model both benefit from coverage-aware training, and that adopting it for both achieves best overall performance.

Expert \pi

![Image 57: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/data-collection-panels/cup-catch-var1_expert.png)

Curiosity

![Image 58: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/data-collection-panels/cup-catch-var1_curiosity_u_r_norm.png)

Human

![Image 59: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/data-collection-panels/cup-catch-var1_human.png)

Cup Catch (DMControl)

Expert \pi

![Image 60: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/data-collection-panels/pygame-reacher-easy_expert.png)

Curiosity

![Image 61: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/data-collection-panels/pygame-reacher-easy_curiosity_u_r_norm.png)

Human

![Image 62: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/data-collection-panels/pygame-reacher-easy_human.png)

Reacher Easy (MiniArcade)

Figure 6: Data coverage by collection method. We show state densities for three online data collection policies (expert, curiosity, and human) across two tasks in the unseen task set.

Table 2: Targeted data collection for finetuning on 10 unseen tasks. We finetune our world model on a set of 10 seen + 10 unseen tasks, varying data source and finetuning strategy. Each data source contains 50 trajectories per task. Offline metrics were computed using a test set of expert trajectories and human play data in equal amount. Task performance is measured via closed-loop MPC using CEM; we report mean performance normalized to [0,1] across 3 episodes per task.

Method Tok FT Dyn FT Recon PSNR \uparrow Rollout\Delta PSNR \uparrow Action shuf. \uparrow u_{r}^{\mathrm{norm}}\downarrow Task perf.(MPC) \uparrow
Random policy————0.118
Base✗✗17.37-12.44 1.12 3.860—
Coverage-aware✗✗17.21-12.52 1.29 3.769 0.276
No-op actions✗✓17.21-11.66 1.41 4.175—
No-op actions✓✓34.74+0.66 1.55 1.486 0.163
Random policy✗✓17.21-11.29 1.73 4.167—
Random policy✓✓35.81+2.66 2.00 1.201 0.228
Expert policy✓✓35.86+2.84 2.04 1.131 0.362
Human play✓✓37.11+3.89\mathbf{2.42}1.002 0.362
Curiosity (u_{r}^{\mathrm{norm}})✓✓36.05+3.00 2.00 1.144 0.325
All (combined)✓✓\mathbf{37.91}\mathbf{+4.02}2.34\mathbf{0.975}\mathbf{0.390}

Mitigating hallucination via targeted data collection. Since we observe a strong relationship between data coverage and hallucination, a question that naturally follows is _can hallucination be mitigated by filling the gaps in coverage via targeted data collection?_ To answer this question, we construct a task set that consists of 10 tasks seen during pretraining and 10 completely unseen tasks, and then collect 50 episodes for each of these tasks varying only the behavior policy used. Specifically, we evaluate the following approaches: no-op (all-zero) actions, random actions, expert policies, human play (via keyboard), and curiosity-based exploration (Sekar et al., [2020](https://arxiv.org/html/2606.27326#bib.bib24 "Planning to explore via self-supervised world models")) using our proposed hallucination predictor u_{r}^{\mathrm{norm}} as objective. State densities for different collection policies are shown in Figure[6](https://arxiv.org/html/2606.27326#S5.F6 "Figure 6 ‣ 5.1 Main results ‣ 5 Experiments ‣ Hallucination in World Models is Predictable and Preventable"). We finetune tokenizer and dynamics model for 50 k steps and 30 k steps, respectively, and report offline performance metrics and downstream task performance via closed-loop MPC (planning with CEM). Results for the 10 unseen tasks are shown in Table[2](https://arxiv.org/html/2606.27326#S5.T2 "Table 2 ‣ 5.1 Main results ‣ 5 Experiments ‣ Hallucination in World Models is Predictable and Preventable"), and additional results can be found in Appendix[E](https://arxiv.org/html/2606.27326#A5 "Appendix E Additional Results ‣ Hallucination in World Models is Predictable and Preventable"). Pretraining transfers, to some extent, zero-shot (0.276, 2.3{\times} the 0.118 random policy baseline), and collecting just 50 trajectories per task using u_{r}^{\mathrm{norm}}-based curiosity lifts performance to 0.325, within \sim 90\% of the expert/human oracles (0.362) despite using no privileged behavior.

### 5.2 Discussion

Empirically, we find that hallucination can be mitigated, to a great extent, by targeted data collection, but that not all data sources are equally informative. We believe that there are two concrete reasons for this: _(i)_ world modeling requires broad state-action coverage which, _e.g._, a random policy does not guarantee, and _(ii)_ downstream tasks are goal-directed and have a much narrower state-action distribution than the general space, so _e.g._ our closed-loop MPC evaluations mainly measure model accuracy around that narrower subspace. That is, arguably, by design: there are certain behaviors we are more interested in modeling than others, but it raises a broader question about how we can best evaluate world models moving forward.

Do off-the-shelf tokenizers improve perception? One concrete way in which perceptual hallucination can be mitigated on unseen tasks is by leveraging off-the-shelf tokenizers trained on datasets several orders of magnitude larger than the one considered in this work. To investigate the viability of this approach, we compare our trained tokenizers (before and after finetuning on the unseen task set) against four off-the-shelf tokenizers; results are shown in Table[3](https://arxiv.org/html/2606.27326#S5.T3 "Table 3 ‣ 5.2 Discussion ‣ 5 Experiments ‣ Hallucination in World Models is Predictable and Preventable"). We find that off-the-shelf tokenizers such as Wan 2.1 VAE (the strongest) underperform compared to our in-domain tokenizer when evaluated on tasks from the 200-task training corpus, but perform significantly better on the unseen task set unless we finetune our model in which case our in-domain tokenizer wins again. These results suggest that off-the-shelf tokenizers indeed are promising for world modeling but that there are still tangible benefits to in-domain finetuning when possible.

Related work. We include a detailed discussion of related work in Appendix[A](https://arxiv.org/html/2606.27326#A1 "Appendix A Related Work ‣ Hallucination in World Models is Predictable and Preventable"), and position MMBench2 vs. existing datasets for world modeling in Appendix[B](https://arxiv.org/html/2606.27326#A2 "Appendix B Comparison to Existing Datasets ‣ Hallucination in World Models is Predictable and Preventable").

In conclusion, we argue that hallucination in generative world models is, first and foremost, a data-coverage problem, and identify three failure modes: perceptual, action marginalization, and scene divergence. Our predictors track them with \rho\approx 0.80 against rollout \Delta PSNR and yield two complementary recipes: coverage-aware sampling reduces all three failure modes simultaneously, and using the predictors as curiosity rewards adapts our pretrained model to entirely unseen tasks.

Table 3: Comparison to off-the-shelf tokenizers. Reconstruction PSNR and LPIPS for our tokenizer as well as four off-the-shelf tokenizers, evaluated on the 10 “seen” and “unseen” task sets used in our finetuning experiments. Our tokenizers outperform the best off-the-shelf tokenizer, Wan 2.1 VAE, by a large margin on tasks in the training set but generalizes poorly to new tasks without additional finetuning. Best result in bold; arrows indicate whether higher (\uparrow) or lower (\downarrow) is better.

PSNR (dB) \uparrow LPIPS \downarrow
Tokenizer Params Latent/frame Seen Unseen\Delta_{S-U}Seen Unseen
_Ours_
Base 102 M 4096 38.29 17.34+20.95 0.011 0.389
Coverage-aware 102 M 4096 38.93 17.12+21.81 0.008 0.348
Post-FT 102 M 4096\mathbf{39.66}\mathbf{38.04}\mathbf{+1.62}\mathbf{0.007}\mathbf{0.010}
_Off-the-shelf_
SD-VAE-MSE 84 M 3136 33.32 32.39+0.93 0.031 0.030
Cosmos-CV8x8x8 (1.0)106 M 2048 32.80 32.72+0.08 0.050 0.042
Wan 2.1 VAE 127 M 4096 36.45 36.62-0.17 0.010 0.010
DC-AE-f32c32 323 M 2048 31.49 32.15-0.66 0.035 0.031

Limitations. Our study operates at the 350 M parameter scale across 210 simulated control tasks. Whether our findings translate to billion-parameter models or to real robot data with sensor noise and partial observability remains an open empirical question. We also acknowledge the significant computational cost of training large world models, as well as the need for more diverse datasets.

## References

*   R. Agarwal, D. Schuurmans, and M. Norouzi (2020)An optimistic perspective on offline reinforcement learning. In International Conference on Machine Learning,  pp.104–114. Cited by: [Appendix B](https://arxiv.org/html/2606.27326#A2.p1.1 "Appendix B Comparison to Existing Datasets ‣ Hallucination in World Models is Predictable and Preventable"). 
*   E. Alonso, A. Jelley, V. Micheli, A. Kanervisto, A. Storkey, T. Pearce, and F. Fleuret (2024)Diffusion for world modeling: visual details matter in atari. In Thirty-eighth Conference on Neural Information Processing Systems, Cited by: [Appendix A](https://arxiv.org/html/2606.27326#A1.p1.1 "Appendix A Related Work ‣ Hallucination in World Models is Predictable and Preventable"), [§1](https://arxiv.org/html/2606.27326#S1.p1.1 "1 Introduction ‣ Hallucination in World Models is Predictable and Preventable"). 
*   B. Baker, I. Akkaya, P. Zhokov, J. Huizinga, J. Tang, A. Ecoffet, B. Houghton, R. Sampedro, and J. Clune (2022)Video pretraining (vpt): learning to act by watching unlabeled online videos. Advances in Neural Information Processing Systems 35,  pp.24639–24654. Cited by: [Appendix B](https://arxiv.org/html/2606.27326#A2.p1.1 "Appendix B Comparison to Existing Datasets ‣ Hallucination in World Models is Predictable and Preventable"). 
*   G. Brockman, V. Cheung, L. Pettersson, J. Schneider, J. Schulman, J. Tang, and W. Zaremba (2016)OpenAI gym. External Links: arXiv:1606.01540 Cited by: [§2](https://arxiv.org/html/2606.27326#S2.p2.7 "2 MMBench2: A Dataset for Visual World Modeling ‣ Hallucination in World Models is Predictable and Preventable"). 
*   J. Bruce, M. D. Dennis, A. Edwards, J. Parker-Holder, Y. Shi, E. Hughes, M. Lai, A. Mavalankar, R. Steigerwald, C. Apps, et al. (2024)Genie: generative interactive environments. In Forty-first International Conference on Machine Learning, Cited by: [Appendix A](https://arxiv.org/html/2606.27326#A1.p1.1 "Appendix A Related Work ‣ Hallucination in World Models is Predictable and Preventable"). 
*   Y. Burda, H. Edwards, A. Storkey, and O. Klimov (2019)Exploration by random network distillation. In International Conference on Learning Representations (ICLR), Cited by: [Appendix A](https://arxiv.org/html/2606.27326#A1.p3.1 "Appendix A Related Work ‣ Hallucination in World Models is Predictable and Preventable"). 
*   K. Chua, R. Calandra, R. McAllister, and S. Levine (2018)Deep reinforcement learning in a handful of trials using probabilistic dynamics models. In NeurIPS, Cited by: [§1](https://arxiv.org/html/2606.27326#S1.p2.1 "1 Introduction ‣ Hallucination in World Models is Predictable and Preventable"). 
*   O. X. Collaboration, A. O’Neill, A. Rehman, A. Gupta, A. Maddukuri, A. Gupta, A. Padalkar, A. Lee, A. Pooley, A. Gupta, A. Mandlekar, A. Jain, A. Tung, A. Bewley, A. Herzog, A. Irpan, A. Khazatsky, A. Rai, A. Gupta, A. Wang, A. Kolobov, A. Singh, A. Garg, A. Kembhavi, A. Xie, A. Brohan, A. Raffin, A. Sharma, A. Yavary, A. Jain, A. Balakrishna, A. Wahid, B. Burgess-Limerick, B. Kim, B. Schölkopf, B. Wulfe, B. Ichter, C. Lu, C. Xu, C. Le, C. Finn, C. Wang, C. Xu, C. Chi, C. Huang, C. Chan, C. Agia, C. Pan, C. Fu, C. Devin, D. Xu, D. Morton, D. Driess, D. Chen, D. Pathak, D. Shah, D. Büchler, D. Jayaraman, D. Kalashnikov, D. Sadigh, E. Johns, E. Foster, F. Liu, F. Ceola, F. Xia, F. Zhao, F. V. Frujeri, F. Stulp, G. Zhou, G. S. Sukhatme, G. Salhotra, G. Yan, G. Feng, G. Schiavi, G. Berseth, G. Kahn, G. Yang, G. Wang, H. Su, H. Fang, H. Shi, H. Bao, H. B. Amor, H. I. Christensen, H. Furuta, H. Bharadhwaj, H. Walke, H. Fang, H. Ha, I. Mordatch, I. Radosavovic, I. Leal, J. Liang, J. Abou-Chakra, J. Kim, J. Drake, J. Peters, J. Schneider, J. Hsu, J. Vakil, J. Bohg, J. Bingham, J. Wu, J. Gao, J. Hu, J. Wu, J. Wu, J. Sun, J. Luo, J. Gu, J. Tan, J. Oh, J. Wu, J. Lu, J. Yang, J. Malik, J. Silvério, J. Hejna, J. Booher, J. Tompson, J. Yang, J. Salvador, J. J. Lim, J. Han, K. Wang, K. Rao, K. Pertsch, K. Hausman, K. Go, K. Gopalakrishnan, K. Goldberg, K. Byrne, K. Oslund, K. Kawaharazuka, K. Black, K. Lin, K. Zhang, K. Ehsani, K. Lekkala, K. Ellis, K. Rana, K. Srinivasan, K. Fang, K. P. Singh, K. Zeng, K. Hatch, K. Hsu, L. Itti, L. Y. Chen, L. Pinto, L. Fei-Fei, L. Tan, L. ". Fan, L. Ott, L. Lee, L. Weihs, M. Chen, M. Lepert, M. Memmel, M. Tomizuka, M. Itkina, M. G. Castro, M. Spero, M. Du, M. Ahn, M. C. Yip, M. Zhang, M. Ding, M. Heo, M. K. Srirama, M. Sharma, M. J. Kim, M. Z. Irshad, N. Kanazawa, N. Hansen, N. Heess, N. J. Joshi, N. Suenderhauf, N. Liu, N. D. Palo, N. M. M. Shafiullah, O. Mees, O. Kroemer, O. Bastani, P. R. Sanketi, P. ". Miller, P. Yin, P. Wohlhart, P. Xu, P. D. Fagan, P. Mitrano, P. Sermanet, P. Abbeel, P. Sundaresan, Q. Chen, Q. Vuong, R. Rafailov, R. Tian, R. Doshi, R. Mart’in-Mart’in, R. Baijal, R. Scalise, R. Hendrix, R. Lin, R. Qian, R. Zhang, R. Mendonca, R. Shah, R. Hoque, R. Julian, S. Bustamante, S. Kirmani, S. Levine, S. Lin, S. Moore, S. Bahl, S. Dass, S. Sonawani, S. Tulsiani, S. Song, S. Xu, S. Haldar, S. Karamcheti, S. Adebola, S. Guist, S. Nasiriany, S. Schaal, S. Welker, S. Tian, S. Ramamoorthy, S. Dasari, S. Belkhale, S. Park, S. Nair, S. Mirchandani, T. Osa, T. Gupta, T. Harada, T. Matsushima, T. Xiao, T. Kollar, T. Yu, T. Ding, T. Davchev, T. Z. Zhao, T. Armstrong, T. Darrell, T. Chung, V. Jain, V. Kumar, V. Vanhoucke, V. Guizilini, W. Zhan, W. Zhou, W. Burgard, X. Chen, X. Chen, X. Wang, X. Zhu, X. Geng, X. Liu, X. Liangwei, X. Li, Y. Pang, Y. Lu, Y. J. Ma, Y. Kim, Y. Chebotar, Y. Zhou, Y. Zhu, Y. Wu, Y. Xu, Y. Wang, Y. Bisk, Y. Dou, Y. Cho, Y. Lee, Y. Cui, Y. Cao, Y. Wu, Y. Tang, Y. Zhu, Y. Zhang, Y. Jiang, Y. Li, Y. Li, Y. Iwasawa, Y. Matsuo, Z. Ma, Z. Xu, Z. J. Cui, Z. Zhang, Z. Fu, and Z. Lin (2023)Open X-Embodiment: robotic learning datasets and RT-X models. Note: [https://arxiv.org/abs/2310.08864](https://arxiv.org/abs/2310.08864)Cited by: [Appendix B](https://arxiv.org/html/2606.27326#A2.p1.1 "Appendix B Comparison to Existing Datasets ‣ Hallucination in World Models is Predictable and Preventable"). 
*   T. Darcet, M. Oquab, J. Mairal, and P. Bojanowski (2024)Vision transformers need registers. In International Conference on Learning Representations (ICLR), Cited by: [Appendix F](https://arxiv.org/html/2606.27326#A6.p5.7 "Appendix F Implementation Details ‣ Hallucination in World Models is Predictable and Preventable"). 
*   S. Dasari, F. Ebert, S. Tian, S. Nair, B. Bucher, K. Schmeckpeper, S. Singh, S. Levine, and C. Finn (2019)RoboNet: large-scale multi-robot learning. In Conference on Robot Learning,  pp.885–897. Cited by: [Appendix B](https://arxiv.org/html/2606.27326#A2.p1.1 "Appendix B Comparison to Existing Datasets ‣ Hallucination in World Models is Predictable and Preventable"). 
*   J. Farebrother and P. S. Castro (2024)CALE: continuous arcade learning environment. Advances in Neural Information Processing Systems 37,  pp.134927–134946. Cited by: [§2](https://arxiv.org/html/2606.27326#S2.p2.7 "2 MMBench2: A Dataset for Visual World Modeling ‣ Hallucination in World Models is Predictable and Preventable"). 
*   K. Frans, D. Hafner, S. Levine, and P. Abbeel (2025)One step diffusion via shortcut models. In The Thirteenth International Conference on Learning Representations, Cited by: [Appendix F](https://arxiv.org/html/2606.27326#A6.p7.11 "Appendix F Implementation Details ‣ Hallucination in World Models is Predictable and Preventable"), [§3](https://arxiv.org/html/2606.27326#S3.p3.7 "3 Training a Large Visual World Model ‣ Hallucination in World Models is Predictable and Preventable"). 
*   S. Y. Gadre, G. Ilharco, A. Fang, J. Hayase, G. Smyrnis, T. Nguyen, R. Marten, M. Wortsman, D. Ghosh, J. Zhang, et al. (2023)Datacomp: in search of the next generation of multimodal datasets. Advances in Neural Information Processing Systems 36,  pp.27092–27112. Cited by: [Appendix A](https://arxiv.org/html/2606.27326#A1.p3.1 "Appendix A Related Work ‣ Hallucination in World Models is Predictable and Preventable"), [§1](https://arxiv.org/html/2606.27326#S1.p2.1 "1 Introduction ‣ Hallucination in World Models is Predictable and Preventable"). 
*   Y. Gal and Z. Ghahramani (2016)Dropout as a Bayesian approximation: representing model uncertainty in deep learning. In Proceedings of the 33rd International Conference on Machine Learning (ICML), Proceedings of Machine Learning Research, Vol. 48,  pp.1050–1059. Cited by: [Appendix A](https://arxiv.org/html/2606.27326#A1.p2.1 "Appendix A Related Work ‣ Hallucination in World Models is Predictable and Preventable"), [§1](https://arxiv.org/html/2606.27326#S1.p2.1 "1 Introduction ‣ Hallucination in World Models is Predictable and Preventable"). 
*   C. Gulcehre, Z. Wang, A. Novikov, T. Paine, S. Gómez, K. Zolna, R. Agarwal, J. S. Merel, D. J. Mankowitz, C. Paduraru, G. Dulac-Arnold, J. Li, M. Norouzi, M. Hoffman, N. Heess, and N. de Freitas (2020)RL unplugged: a suite of benchmarks for offline reinforcement learning. In Advances in Neural Information Processing Systems (NeurIPS), Cited by: [Appendix A](https://arxiv.org/html/2606.27326#A1.p3.1 "Appendix A Related Work ‣ Hallucination in World Models is Predictable and Preventable"), [Appendix B](https://arxiv.org/html/2606.27326#A2.p1.1 "Appendix B Comparison to Existing Datasets ‣ Hallucination in World Models is Predictable and Preventable"). 
*   D. Ha and J. Schmidhuber (2018)Recurrent world models facilitate policy evolution. In Advances in Neural Information Processing Systems 31,  pp.2451–2463. Cited by: [Appendix A](https://arxiv.org/html/2606.27326#A1.p1.1 "Appendix A Related Work ‣ Hallucination in World Models is Predictable and Preventable"). 
*   D. Hafner, T. P. Lillicrap, J. Ba, and M. Norouzi (2020)Dream to control: learning behaviors by latent imagination. ArXiv abs/1912.01603. Cited by: [Appendix A](https://arxiv.org/html/2606.27326#A1.p1.1 "Appendix A Related Work ‣ Hallucination in World Models is Predictable and Preventable"). 
*   D. Hafner, J. Pasukonis, J. Ba, and T. Lillicrap (2023)Mastering diverse domains through world models. arXiv preprint arXiv:2301.04104. Cited by: [§1](https://arxiv.org/html/2606.27326#S1.p1.1 "1 Introduction ‣ Hallucination in World Models is Predictable and Preventable"). 
*   D. Hafner, W. Yan, and T. Lillicrap (2025)Training agents inside of scalable world models. arXiv preprint arXiv:2509.24527. Cited by: [Appendix A](https://arxiv.org/html/2606.27326#A1.p1.1 "Appendix A Related Work ‣ Hallucination in World Models is Predictable and Preventable"), [Appendix F](https://arxiv.org/html/2606.27326#A6.p1.1 "Appendix F Implementation Details ‣ Hallucination in World Models is Predictable and Preventable"), [§1](https://arxiv.org/html/2606.27326#S1.p1.1 "1 Introduction ‣ Hallucination in World Models is Predictable and Preventable"), [§1](https://arxiv.org/html/2606.27326#S1.p4.2 "1 Introduction ‣ Hallucination in World Models is Predictable and Preventable"), [§3](https://arxiv.org/html/2606.27326#S3.p1.1 "3 Training a Large Visual World Model ‣ Hallucination in World Models is Predictable and Preventable"). 
*   N. Hansen, H. Su, and X. Wang (2024)TD-mpc2: scalable, robust world models for continuous control. In International Conference on Learning Representations (ICLR), Cited by: [Appendix A](https://arxiv.org/html/2606.27326#A1.p1.1 "Appendix A Related Work ‣ Hallucination in World Models is Predictable and Preventable"), [Appendix B](https://arxiv.org/html/2606.27326#A2.p1.1 "Appendix B Comparison to Existing Datasets ‣ Hallucination in World Models is Predictable and Preventable"), [§1](https://arxiv.org/html/2606.27326#S1.p1.1 "1 Introduction ‣ Hallucination in World Models is Predictable and Preventable"). 
*   N. Hansen, H. Su, and X. Wang (2026)Learning massively multitask world models for continuous control. In International Conference on Learning Representations (ICLR), Cited by: [Appendix B](https://arxiv.org/html/2606.27326#A2.p1.1 "Appendix B Comparison to Existing Datasets ‣ Hallucination in World Models is Predictable and Preventable"), [Table 5](https://arxiv.org/html/2606.27326#A3.T5.70 "In Appendix C Task Domains ‣ Hallucination in World Models is Predictable and Preventable"), [Table 5](https://arxiv.org/html/2606.27326#A3.T5.70.74.2.1 "In Appendix C Task Domains ‣ Hallucination in World Models is Predictable and Preventable"), [§1](https://arxiv.org/html/2606.27326#S1.p3.9 "1 Introduction ‣ Hallucination in World Models is Predictable and Preventable"), [§2](https://arxiv.org/html/2606.27326#S2.p2.7 "2 MMBench2: A Dataset for Visual World Modeling ‣ Hallucination in World Models is Predictable and Preventable"). 
*   N. Hansen, X. Wang, and H. Su (2022)Temporal difference learning for model predictive control. In International Conference on Machine Learning, Cited by: [Appendix A](https://arxiv.org/html/2606.27326#A1.p1.1 "Appendix A Related Work ‣ Hallucination in World Models is Predictable and Preventable"). 
*   K. He, X. Chen, S. Xie, Y. Li, P. Dollár, and R. Girshick (2022)Masked autoencoders are scalable vision learners. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition,  pp.16000–16009. Cited by: [§3](https://arxiv.org/html/2606.27326#S3.p2.8 "3 Training a Large Visual World Model ‣ Hallucination in World Models is Predictable and Preventable"). 
*   A. Henry, P. R. Dachapally, S. S. Pawar, and Y. Chen (2020)Query-key normalization for transformers. In Findings of the Association for Computational Linguistics: EMNLP 2020,  pp.4246–4253. Cited by: [Appendix F](https://arxiv.org/html/2606.27326#A6.p3.1 "Appendix F Implementation Details ‣ Hallucination in World Models is Predictable and Preventable"). 
*   J. Hoffmann, S. Borgeaud, A. Mensch, E. Buchatskaya, T. Cai, E. Rutherford, D. Casas, L. A. Hendricks, J. Welbl, A. Clark, et al. (2022)Training compute-optimal large language models. arXiv preprint arXiv:2203.15556 10. Cited by: [Appendix A](https://arxiv.org/html/2606.27326#A1.p3.1 "Appendix A Related Work ‣ Hallucination in World Models is Predictable and Preventable"), [§1](https://arxiv.org/html/2606.27326#S1.p2.1 "1 Introduction ‣ Hallucination in World Models is Predictable and Preventable"). 
*   L. Huang, W. Yu, W. Ma, W. Zhong, Z. Feng, H. Wang, Q. Chen, W. Peng, X. Feng, B. Qin, and T. Liu (2025)A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions. ACM Transactions on Information Systems 43 (2),  pp.1–55. External Links: [Document](https://dx.doi.org/10.1145/3703155)Cited by: [Appendix A](https://arxiv.org/html/2606.27326#A1.p2.1 "Appendix A Related Work ‣ Hallucination in World Models is Predictable and Preventable"), [§1](https://arxiv.org/html/2606.27326#S1.p1.1 "1 Introduction ‣ Hallucination in World Models is Predictable and Preventable"). 
*   Z. Huang, Y. He, J. Yu, F. Zhang, C. Si, Y. Jiang, Y. Zhang, T. Wu, Q. Jin, N. Chanpaisit, Y. Wang, X. Chen, L. Wang, D. Lin, Y. Qiao, and Z. Liu (2024)VBench: comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Cited by: [Appendix A](https://arxiv.org/html/2606.27326#A1.p2.1 "Appendix A Related Work ‣ Hallucination in World Models is Predictable and Preventable"), [§1](https://arxiv.org/html/2606.27326#S1.p1.1 "1 Introduction ‣ Hallucination in World Models is Predictable and Preventable"). 
*   M. Janner, J. Fu, M. Zhang, and S. Levine (2019)When to trust your model: model-based policy optimization. ArXiv abs/1906.08253. Cited by: [§1](https://arxiv.org/html/2606.27326#S1.p1.1 "1 Introduction ‣ Hallucination in World Models is Predictable and Preventable"). 
*   Z. Ji, N. Lee, R. Frieske, T. Yu, D. Su, Y. Xu, E. Ishii, Y. Bang, A. Madotto, and P. Fung (2023)Survey of hallucination in natural language generation. ACM Computing Surveys 55 (12),  pp.1–38. External Links: [Document](https://dx.doi.org/10.1145/3571730)Cited by: [Appendix A](https://arxiv.org/html/2606.27326#A1.p2.1 "Appendix A Related Work ‣ Hallucination in World Models is Predictable and Preventable"), [§1](https://arxiv.org/html/2606.27326#S1.p1.1 "1 Introduction ‣ Hallucination in World Models is Predictable and Preventable"). 
*   H. Kannan, D. Hafner, C. Finn, and D. Erhan (2021)RoboDesk: a multi-task reinforcement learning benchmark. Note: [https://github.com/google-research/robodesk](https://github.com/google-research/robodesk)Cited by: [§2](https://arxiv.org/html/2606.27326#S2.p2.7 "2 MMBench2: A Dataset for Visual World Modeling ‣ Hallucination in World Models is Predictable and Preventable"). 
*   A. Khazatsky, K. Pertsch, S. Nair, A. Balakrishna, S. Dasari, S. Karamcheti, S. Nasiriany, M. K. Srirama, L. Y. Chen, K. Ellis, P. D. Fagan, J. Hejna, M. Itkina, M. Lepert, Y. J. Ma, P. T. Miller, J. Wu, S. Belkhale, S. Dass, H. Ha, A. Jain, A. Lee, Y. Lee, M. Memmel, S. Park, I. Radosavovic, K. Wang, A. Zhan, K. Black, C. Chi, K. B. Hatch, S. Lin, J. Lu, J. Mercat, A. Rehman, P. R. Sanketi, A. Sharma, C. Simpson, Q. Vuong, H. R. Walke, B. Wulfe, T. Xiao, J. H. Yang, A. Yavary, T. Z. Zhao, C. Agia, R. Baijal, M. G. Castro, D. Chen, Q. Chen, T. Chung, J. Drake, E. P. Foster, J. Gao, V. Guizilini, D. A. Herrera, M. Heo, K. Hsu, J. Hu, M. Z. Irshad, D. Jackson, C. Le, Y. Li, K. Lin, R. Lin, Z. Ma, A. Maddukuri, S. Mirchandani, D. Morton, T. Nguyen, A. O’Neill, R. Scalise, D. Seale, V. Son, S. Tian, E. Tran, A. E. Wang, Y. Wu, A. Xie, J. Yang, P. Yin, Y. Zhang, O. Bastani, G. Berseth, J. Bohg, K. Goldberg, A. Gupta, A. Gupta, D. Jayaraman, J. J. Lim, J. Malik, R. Martín-Martín, S. Ramamoorthy, D. Sadigh, S. Song, J. Wu, M. C. Yip, Y. Zhu, T. Kollar, S. Levine, and C. Finn (2024)DROID: a large-scale in-the-wild robot manipulation dataset. Cited by: [Appendix B](https://arxiv.org/html/2606.27326#A2.p1.1 "Appendix B Comparison to Existing Datasets ‣ Hallucination in World Models is Predictable and Preventable"). 
*   R. Kidambi, A. Rajeswaran, P. Netrapalli, and T. Joachims (2020)Morel: model-based offline reinforcement learning. Advances in neural information processing systems 33,  pp.21810–21823. Cited by: [Appendix A](https://arxiv.org/html/2606.27326#A1.p2.1 "Appendix A Related Work ‣ Hallucination in World Models is Predictable and Preventable"). 
*   A. Kumar, A. Zhou, G. Tucker, and S. Levine (2020)Conservative Q-learning for offline reinforcement learning. In Advances in Neural Information Processing Systems (NeurIPS), Cited by: [Appendix A](https://arxiv.org/html/2606.27326#A1.p3.1 "Appendix A Related Work ‣ Hallucination in World Models is Predictable and Preventable"). 
*   B. Lakshminarayanan, A. Pritzel, and C. Blundell (2017)Simple and scalable predictive uncertainty estimation using deep ensembles. In Advances in Neural Information Processing Systems (NeurIPS), Cited by: [Appendix A](https://arxiv.org/html/2606.27326#A1.p2.1 "Appendix A Related Work ‣ Hallucination in World Models is Predictable and Preventable"), [§1](https://arxiv.org/html/2606.27326#S1.p2.1 "1 Introduction ‣ Hallucination in World Models is Predictable and Preventable"). 
*   K. Lee, K. Lee, H. Lee, and J. Shin (2018)A simple unified framework for detecting out-of-distribution samples and adversarial attacks. In Advances in Neural Information Processing Systems (NeurIPS), Cited by: [Appendix A](https://arxiv.org/html/2606.27326#A1.p2.1 "Appendix A Related Work ‣ Hallucination in World Models is Predictable and Preventable"). 
*   S. Levine, A. Kumar, G. Tucker, and J. Fu (2020)Offline reinforcement learning: tutorial, review, and perspectives on open problems. arXiv preprint arXiv:2005.01643. Cited by: [Appendix A](https://arxiv.org/html/2606.27326#A1.p3.1 "Appendix A Related Work ‣ Hallucination in World Models is Predictable and Preventable"), [§1](https://arxiv.org/html/2606.27326#S1.p2.1 "1 Introduction ‣ Hallucination in World Models is Predictable and Preventable"). 
*   Y. Li, Y. Du, K. Zhou, J. Wang, W. X. Zhao, and J. Wen (2023)Evaluating object hallucination in large vision-language models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP), Cited by: [Appendix A](https://arxiv.org/html/2606.27326#A1.p2.1 "Appendix A Related Work ‣ Hallucination in World Models is Predictable and Preventable"), [§1](https://arxiv.org/html/2606.27326#S1.p1.1 "1 Introduction ‣ Hallucination in World Models is Predictable and Preventable"). 
*   W. Liu, X. Wang, J. D. Owens, and Y. Li (2020)Energy-based out-of-distribution detection. In Advances in Neural Information Processing Systems (NeurIPS), Cited by: [Appendix A](https://arxiv.org/html/2606.27326#A1.p2.1 "Appendix A Related Work ‣ Hallucination in World Models is Predictable and Preventable"). 
*   C. Lu, P. J. Ball, T. G. J. Rudner, J. Parker-Holder, M. A. Osborne, and Y. W. Teh (2023)Challenges and opportunities in offline reinforcement learning from visual observations. Transactions on Machine Learning Research. External Links: [Link](https://openreview.net/forum?id=1QqIfGZOWu)Cited by: [Appendix A](https://arxiv.org/html/2606.27326#A1.p3.1 "Appendix A Related Work ‣ Hallucination in World Models is Predictable and Preventable"), [Appendix B](https://arxiv.org/html/2606.27326#A2.p1.1 "Appendix B Comparison to Existing Datasets ‣ Hallucination in World Models is Predictable and Preventable"). 
*   L. Magne, A. Awadalla, G. Wang, Y. Xu, J. Belofsky, F. Hu, J. Kim, L. Schmidt, G. Gkioxari, J. Kautz, Y. Yue, Y. Choi, Y. Zhu, and L. ". Fan (2026)NitroGen: an open foundation model for generalist gaming agents. External Links: 2601.02427, [Link](https://arxiv.org/abs/2601.02427)Cited by: [Appendix B](https://arxiv.org/html/2606.27326#A2.p1.1 "Appendix B Comparison to Existing Datasets ‣ Hallucination in World Models is Predictable and Preventable"). 
*   V. Micheli, E. Alonso, and F. Fleuret (2023)Transformers are sample-efficient world models. In The Eleventh International Conference on Learning Representations, External Links: [Link](https://openreview.net/forum?id=vhFu1Acb0xb)Cited by: [Appendix A](https://arxiv.org/html/2606.27326#A1.p1.1 "Appendix A Related Work ‣ Hallucination in World Models is Predictable and Preventable"). 
*   NVIDIA (2025)Cosmos world foundation model platform for physical ai. arXiv preprint arXiv:2501.03575. Cited by: [Appendix A](https://arxiv.org/html/2606.27326#A1.p1.1 "Appendix A Related Work ‣ Hallucination in World Models is Predictable and Preventable"). 
*   S. Park, K. Frans, B. Eysenbach, and S. Levine (2025)OGBench: benchmarking offline goal-conditioned rl. In International Conference on Learning Representations (ICLR), Cited by: [§2](https://arxiv.org/html/2606.27326#S2.p2.7 "2 MMBench2: A Dataset for Visual World Modeling ‣ Hallucination in World Models is Predictable and Preventable"). 
*   J. Parker-Holder, S. Fruchter, and Google DeepMind (2025)Genie 3: a new frontier for world models. Note: Google DeepMind Blog External Links: [Link](https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/)Cited by: [Appendix A](https://arxiv.org/html/2606.27326#A1.p1.1 "Appendix A Related Work ‣ Hallucination in World Models is Predictable and Preventable"), [§1](https://arxiv.org/html/2606.27326#S1.p1.1 "1 Introduction ‣ Hallucination in World Models is Predictable and Preventable"). 
*   D. Pathak, P. Agrawal, A. A. Efros, and T. Darrell (2017)Curiosity-driven exploration by self-supervised prediction. In International conference on machine learning,  pp.2778–2787. Cited by: [Appendix A](https://arxiv.org/html/2606.27326#A1.p3.1 "Appendix A Related Work ‣ Hallucination in World Models is Predictable and Preventable"). 
*   D. Pathak, D. Gandhi, and A. Gupta (2019)Self-supervised exploration via disagreement. In Proceedings of the 36th International Conference on Machine Learning (ICML), Cited by: [Appendix A](https://arxiv.org/html/2606.27326#A1.p2.1 "Appendix A Related Work ‣ Hallucination in World Models is Predictable and Preventable"), [Appendix A](https://arxiv.org/html/2606.27326#A1.p3.1 "Appendix A Related Work ‣ Hallucination in World Models is Predictable and Preventable"). 
*   J. Quevedo, Q. McIntyre, S. Campbell, X. Chen, R. Wachen, Decart, and Etched (2024)Oasis: a universe in a transformer. Note: Decart External Links: [Link](https://oasis-model.github.io/)Cited by: [Appendix A](https://arxiv.org/html/2606.27326#A1.p1.1 "Appendix A Related Work ‣ Hallucination in World Models is Predictable and Preventable"). 
*   A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, et al. (2021)Learning transferable visual models from natural language supervision. In International conference on machine learning,  pp.8748–8763. Cited by: [Appendix F](https://arxiv.org/html/2606.27326#A6.p2.1 "Appendix F Implementation Details ‣ Hallucination in World Models is Predictable and Preventable"). 
*   Rami (2024)Lucid v1: a world model that does go brrr on consumer hardware. Note: Substack External Links: [Link](https://ramimo.substack.com/p/lucid-v1-a-world-model-that-does)Cited by: [Appendix A](https://arxiv.org/html/2606.27326#A1.p1.1 "Appendix A Related Work ‣ Hallucination in World Models is Predictable and Preventable"). 
*   J. Schrittwieser, I. Antonoglou, T. Hubert, K. Simonyan, L. Sifre, S. Schmitt, A. Guez, E. Lockhart, D. Hassabis, T. Graepel, et al. (2020)Mastering atari, go, chess and shogi by planning with a learned model. Nature 588 (7839),  pp.604–609. Cited by: [§1](https://arxiv.org/html/2606.27326#S1.p1.1 "1 Introduction ‣ Hallucination in World Models is Predictable and Preventable"). 
*   R. Sekar, O. Rybkin, K. Daniilidis, P. Abbeel, D. Hafner, and D. Pathak (2020)Planning to explore via self-supervised world models. In Proceedings of the 37th International Conference on Machine Learning, H. D. III and A. Singh (Eds.), Proceedings of Machine Learning Research, Vol. 119,  pp.8583–8592. Cited by: [Appendix A](https://arxiv.org/html/2606.27326#A1.p2.1 "Appendix A Related Work ‣ Hallucination in World Models is Predictable and Preventable"), [Appendix A](https://arxiv.org/html/2606.27326#A1.p3.1 "Appendix A Related Work ‣ Hallucination in World Models is Predictable and Preventable"), [§1](https://arxiv.org/html/2606.27326#S1.p4.2 "1 Introduction ‣ Hallucination in World Models is Predictable and Preventable"), [§5.1](https://arxiv.org/html/2606.27326#S5.SS1.p3.11 "5.1 Main results ‣ 5 Experiments ‣ Hallucination in World Models is Predictable and Preventable"). 
*   J. Su, Y. Lu, S. Pan, A. Murtadha, B. Wen, and Y. Liu (2024)RoFormer: enhanced transformer with rotary position embedding. Neurocomputing 568,  pp.127063. Cited by: [Appendix F](https://arxiv.org/html/2606.27326#A6.p3.1 "Appendix F Implementation Details ‣ Hallucination in World Models is Predictable and Preventable"). 
*   S. Tao, F. Xiang, A. Shukla, Y. Qin, X. Hinrichsen, X. Yuan, C. Bao, X. Lin, Y. Liu, T. Chan, Y. Gao, X. Li, T. Mu, N. Xiao, A. Gurha, V. N. Rajesh, Y. W. Choi, Y. Chen, Z. Huang, R. Calandra, R. Chen, S. Luo, and H. Su (2025)ManiSkill3: gpu parallelized robotics simulation and rendering for generalizable embodied ai. Robotics: Science and Systems. Cited by: [§2](https://arxiv.org/html/2606.27326#S2.p2.7 "2 MMBench2: A Dataset for Visual World Modeling ‣ Hallucination in World Models is Predictable and Preventable"). 
*   Y. Tassa, Y. Doron, A. Muldal, T. Erez, Y. Li, D. de Las Casas, D. Budden, A. Abdolmaleki, et al. (2018)DeepMind control suite. Technical report DeepMind. External Links: 1504.04804 Cited by: [§2](https://arxiv.org/html/2606.27326#S2.p2.7 "2 MMBench2: A Dataset for Visual World Modeling ‣ Hallucination in World Models is Predictable and Preventable"). 
*   E. Todorov, T. Erez, and Y. Tassa (2012)MuJoCo: a physics engine for model-based control. In 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems,  pp.5026–5033. External Links: [Document](https://dx.doi.org/10.1109/IROS.2012.6386109)Cited by: [§2](https://arxiv.org/html/2606.27326#S2.p2.7 "2 MMBench2: A Dataset for Visual World Modeling ‣ Hallucination in World Models is Predictable and Preventable"). 
*   D. Valevski, Y. Leviathan, M. Arar, and S. Fruchter (2024)Diffusion models are real-time game engines. External Links: 2408.14837, [Link](https://arxiv.org/abs/2408.14837)Cited by: [Appendix A](https://arxiv.org/html/2606.27326#A1.p1.1 "Appendix A Related Work ‣ Hallucination in World Models is Predictable and Preventable"), [§1](https://arxiv.org/html/2606.27326#S1.p1.1 "1 Introduction ‣ Hallucination in World Models is Predictable and Preventable"). 
*   A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin (2017)Attention is all you need. Advances in neural information processing systems 30. Cited by: [§3](https://arxiv.org/html/2606.27326#S3.p1.1 "3 Training a Large Visual World Model ‣ Hallucination in World Models is Predictable and Preventable"). 
*   H. Walke, K. Black, A. Lee, M. J. Kim, M. Du, C. Zheng, T. Zhao, P. Hansen-Estruch, Q. Vuong, A. He, V. Myers, K. Fang, C. Finn, and S. Levine (2023)BridgeData v2: a dataset for robot learning at scale. In Conference on Robot Learning,  pp.1723–1736. Cited by: [Appendix B](https://arxiv.org/html/2606.27326#A2.p1.1 "Appendix B Comparison to Existing Datasets ‣ Hallucination in World Models is Predictable and Preventable"). 
*   T. Wan, A. Wang, B. Ai, B. Wen, C. Mao, C. Xie, D. Chen, F. Yu, H. Zhao, J. Yang, J. Zeng, J. Wang, J. Zhang, J. Zhou, J. Wang, J. Chen, K. Zhu, K. Zhao, K. Yan, L. Huang, M. Feng, N. Zhang, P. Li, P. Wu, R. Chu, R. Feng, S. Zhang, S. Sun, T. Fang, T. Wang, T. Gui, T. Weng, T. Shen, W. Lin, W. Wang, W. Wang, W. Zhou, W. Wang, W. Shen, W. Yu, X. Shi, X. Huang, X. Xu, Y. Kou, Y. Lv, Y. Li, Y. Liu, Y. Wang, Y. Zhang, Y. Huang, Y. Li, Y. Wu, Y. Liu, Y. Pan, Y. Zheng, Y. Hong, Y. Shi, Y. Feng, Z. Jiang, Z. Han, Z. Wu, and Z. Liu (2025)Wan: open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314. Cited by: [Appendix A](https://arxiv.org/html/2606.27326#A1.p1.1 "Appendix A Related Work ‣ Hallucination in World Models is Predictable and Preventable"). 
*   D. Yarats, D. Brandfonbrener, H. Liu, M. Laskin, P. Abbeel, A. Lazaric, and L. Pinto (2022)Don’t change the algorithm, change the data: exploratory data for offline reinforcement learning. In International Conference on Learning Representations (ICLR), Cited by: [Appendix A](https://arxiv.org/html/2606.27326#A1.p3.1 "Appendix A Related Work ‣ Hallucination in World Models is Predictable and Preventable"), [Appendix B](https://arxiv.org/html/2606.27326#A2.p1.1 "Appendix B Comparison to Existing Datasets ‣ Hallucination in World Models is Predictable and Preventable"). 
*   T. Yu, D. Quillen, Z. He, R. Julian, K. Hausman, C. Finn, and S. Levine (2019)Meta-world: a benchmark and evaluation for multi-task and meta reinforcement learning. In Conference on Robot Learning, External Links: 1910.10897 Cited by: [§2](https://arxiv.org/html/2606.27326#S2.p2.7 "2 MMBench2: A Dataset for Visual World Modeling ‣ Hallucination in World Models is Predictable and Preventable"). 
*   T. Yu, G. Thomas, L. Yu, S. Ermon, J. Zou, S. Levine, C. Finn, and T. Ma (2020)MOPO: model-based offline policy optimization. In Advances in Neural Information Processing Systems (NeurIPS), Cited by: [Appendix A](https://arxiv.org/html/2606.27326#A1.p2.1 "Appendix A Related Work ‣ Hallucination in World Models is Predictable and Preventable"). 
*   B. Zhang and R. Sennrich (2019)Root mean square layer normalization. In Advances in Neural Information Processing Systems (NeurIPS), Cited by: [Appendix F](https://arxiv.org/html/2606.27326#A6.p3.1 "Appendix F Implementation Details ‣ Hallucination in World Models is Predictable and Preventable"). 
*   R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang (2018)The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition,  pp.586–595. Cited by: [§3](https://arxiv.org/html/2606.27326#S3.p2.8 "3 Training a Large Visual World Model ‣ Hallucination in World Models is Predictable and Preventable"). 

## Appendix A Related Work

World models for control. World models that learn environment dynamics from data have a long history in model-based RL, ranging from abstract latent dynamics models [Ha and Schmidhuber, [2018](https://arxiv.org/html/2606.27326#bib.bib10 "Recurrent world models facilitate policy evolution"), Hafner et al., [2020](https://arxiv.org/html/2606.27326#bib.bib3 "Dream to control: learning behaviors by latent imagination"), Hansen et al., [2022](https://arxiv.org/html/2606.27326#bib.bib2 "Temporal difference learning for model predictive control"), [2024](https://arxiv.org/html/2606.27326#bib.bib21 "TD-mpc2: scalable, robust world models for continuous control")] to high-capacity generative models that render full pixel observations [Micheli et al., [2023](https://arxiv.org/html/2606.27326#bib.bib33 "Transformers are sample-efficient world models"), Alonso et al., [2024](https://arxiv.org/html/2606.27326#bib.bib34 "Diffusion for world modeling: visual details matter in atari"), Valevski et al., [2024](https://arxiv.org/html/2606.27326#bib.bib36 "Diffusion models are real-time game engines")]. Recent work scales these models to heterogeneous video corpora [Bruce et al., [2024](https://arxiv.org/html/2606.27326#bib.bib35 "Genie: generative interactive environments"), NVIDIA, [2025](https://arxiv.org/html/2606.27326#bib.bib37 "Cosmos world foundation model platform for physical ai"), Wan et al., [2025](https://arxiv.org/html/2606.27326#bib.bib38 "Wan: open and advanced large-scale video generative models")] and to real-time, playable neural environments [Quevedo et al., [2024](https://arxiv.org/html/2606.27326#bib.bib40 "Oasis: a universe in a transformer"), Rami, [2024](https://arxiv.org/html/2606.27326#bib.bib41 "Lucid v1: a world model that does go brrr on consumer hardware"), Parker-Holder et al., [2025](https://arxiv.org/html/2606.27326#bib.bib39 "Genie 3: a new frontier for world models")]. We build on Dreamer 4 [Hafner et al., [2025](https://arxiv.org/html/2606.27326#bib.bib30 "Training agents inside of scalable world models")], an integrated tokenizer-plus-dynamics architecture with strong action conditioning, but the signals and interventions we propose are largely model-agnostic and apply, in principle, to any modern generative world model. Despite striking visual fidelity, these models continue to hallucinate under distribution shift — a failure mode that we set out to characterize, predict, and mitigate.

Hallucination and uncertainty in generative models. Hallucination has been studied extensively in language models [Ji et al., [2023](https://arxiv.org/html/2606.27326#bib.bib43 "Survey of hallucination in natural language generation"), Huang et al., [2025](https://arxiv.org/html/2606.27326#bib.bib42 "A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions")] and, increasingly, in image [Li et al., [2023](https://arxiv.org/html/2606.27326#bib.bib45 "Evaluating object hallucination in large vision-language models")] and video generation [Huang et al., [2024](https://arxiv.org/html/2606.27326#bib.bib44 "VBench: comprehensive benchmark suite for video generative models")], but the question of _where_ an autoregressive world model will fail along a rollout has received comparatively little attention. A separate but related literature studies uncertainty estimation in deep networks through deep ensembles [Lakshminarayanan et al., [2017](https://arxiv.org/html/2606.27326#bib.bib47 "Simple and scalable predictive uncertainty estimation using deep ensembles")], MC dropout [Gal and Ghahramani, [2016](https://arxiv.org/html/2606.27326#bib.bib48 "Dropout as a Bayesian approximation: representing model uncertainty in deep learning")], and post-hoc out-of-distribution detectors [Lee et al., [2018](https://arxiv.org/html/2606.27326#bib.bib49 "A simple unified framework for detecting out-of-distribution samples and adversarial attacks"), Liu et al., [2020](https://arxiv.org/html/2606.27326#bib.bib50 "Energy-based out-of-distribution detection")], with applications in offline model-based RL via uncertainty-penalized policies [Yu et al., [2020](https://arxiv.org/html/2606.27326#bib.bib51 "MOPO: model-based offline policy optimization"), Kidambi et al., [2020](https://arxiv.org/html/2606.27326#bib.bib46 "Morel: model-based offline reinforcement learning")]. Closest in spirit, prior work uses ensemble disagreement as an exploration signal in single-task RL [Pathak et al., [2019](https://arxiv.org/html/2606.27326#bib.bib52 "Self-supervised exploration via disagreement"), Sekar et al., [2020](https://arxiv.org/html/2606.27326#bib.bib24 "Planning to explore via self-supervised world models")]. In contrast, our proposed predictors are derived directly from the existing world model and target the three distinct stages at which a generative world model can hallucinate (encoder, dynamics, decoder), without auxiliary networks or labels.

Coverage-aware training and data collection. A growing body of work argues that data scale and composition are first-order levers for generative model performance [Hoffmann et al., [2022](https://arxiv.org/html/2606.27326#bib.bib53 "Training compute-optimal large language models"), Gadre et al., [2023](https://arxiv.org/html/2606.27326#bib.bib54 "Datacomp: in search of the next generation of multimodal datasets")]. In offline RL specifically, data coverage is known to bound policy improvement [Levine et al., [2020](https://arxiv.org/html/2606.27326#bib.bib58 "Offline reinforcement learning: tutorial, review, and perspectives on open problems"), Kumar et al., [2020](https://arxiv.org/html/2606.27326#bib.bib59 "Conservative Q-learning for offline reinforcement learning")], motivating curated datasets such as ExoRL [Yarats et al., [2022](https://arxiv.org/html/2606.27326#bib.bib55 "Don’t change the algorithm, change the data: exploratory data for offline reinforcement learning")], RL Unplugged [Gulcehre et al., [2020](https://arxiv.org/html/2606.27326#bib.bib56 "RL unplugged: a suite of benchmarks for offline reinforcement learning")], and V-D4RL [Lu et al., [2023](https://arxiv.org/html/2606.27326#bib.bib57 "Challenges and opportunities in offline reinforcement learning from visual observations")]. Curiosity-driven exploration is the natural online counterpart: agents are incentivized to visit states with high prediction error [Pathak et al., [2017](https://arxiv.org/html/2606.27326#bib.bib23 "Curiosity-driven exploration by self-supervised prediction")], high feature-network novelty [Burda et al., [2019](https://arxiv.org/html/2606.27326#bib.bib60 "Exploration by random network distillation")], or high model disagreement [Pathak et al., [2019](https://arxiv.org/html/2606.27326#bib.bib52 "Self-supervised exploration via disagreement")], scaled to imagined trajectories in Plan2Explore [Sekar et al., [2020](https://arxiv.org/html/2606.27326#bib.bib24 "Planning to explore via self-supervised world models")]. Whereas prior curiosity work uses these signals to drive _single-task policy exploration_, we adapt them to _data-collection for generative world modeling_ in two complementary ways: _(i)_ uniform-task resampling closes a substantial fraction of the hallucination gap at no additional data cost, and _(ii)_ using our proposed predictors as curiosity rewards yields a data-efficient finetuning recipe that generalizes a 350M-parameter world model to entirely unseen environments with as few as 50 trajectories from the target task.

## Appendix B Comparison to Existing Datasets

Table[4](https://arxiv.org/html/2606.27326#A2.T4 "Table 4 ‣ Appendix B Comparison to Existing Datasets ‣ Hallucination in World Models is Predictable and Preventable") compares MMBench2 to a representative set of prior datasets used for offline reinforcement learning, robot imitation learning, and large-scale generative video and world modeling. The offline RL datasets we compare against include RL Unplugged[Gulcehre et al., [2020](https://arxiv.org/html/2606.27326#bib.bib56 "RL unplugged: a suite of benchmarks for offline reinforcement learning")], V-D4RL[Lu et al., [2023](https://arxiv.org/html/2606.27326#bib.bib57 "Challenges and opportunities in offline reinforcement learning from visual observations")], ExoRL[Yarats et al., [2022](https://arxiv.org/html/2606.27326#bib.bib55 "Don’t change the algorithm, change the data: exploratory data for offline reinforcement learning")], the TD-MPC2 multi-task dataset[Hansen et al., [2024](https://arxiv.org/html/2606.27326#bib.bib21 "TD-mpc2: scalable, robust world models for continuous control")], and the Atari DQN Replay dataset[Agarwal et al., [2020](https://arxiv.org/html/2606.27326#bib.bib63 "An optimistic perspective on offline reinforcement learning")]. The robot imitation learning datasets we compare against are RoboNet[Dasari et al., [2019](https://arxiv.org/html/2606.27326#bib.bib61 "RoboNet: large-scale multi-robot learning")], BridgeData V2[Walke et al., [2023](https://arxiv.org/html/2606.27326#bib.bib62 "BridgeData v2: a dataset for robot learning at scale")], DROID[Khazatsky et al., [2024](https://arxiv.org/html/2606.27326#bib.bib27 "DROID: a large-scale in-the-wild robot manipulation dataset")], and Open X-Embodiment[Collaboration et al., [2023](https://arxiv.org/html/2606.27326#bib.bib26 "Open X-Embodiment: robotic learning datasets and RT-X models")]. For large-scale video pretraining corpora with pseudo-labeled actions, we compare against VPT[Baker et al., [2022](https://arxiv.org/html/2606.27326#bib.bib11 "Video pretraining (vpt): learning to act by watching unlabeled online videos")] and NitroGen[Magne et al., [2026](https://arxiv.org/html/2606.27326#bib.bib64 "NitroGen: an open foundation model for generalist gaming agents")]. Finally, we include MMBench[Hansen et al., [2026](https://arxiv.org/html/2606.27326#bib.bib29 "Learning massively multitask world models for continuous control")], the multi-task benchmark on which MMBench2 builds. Across these datasets, MMBench2 is the only corpus that simultaneously offers ground-truth action _and_ reward labels, live simulators for every task, mixed-quality behavior, and broad coverage across both task domains and embodiments.

Table 4: MMBench2 vs. existing datasets. Our dataset consists of mixed-quality data spanning a larger number of tasks and domains, complete with ground-truth action and reward labels as well as live environments. We summarize key characteristics of MMBench2 and existing datasets below. 

a Per-trajectory binary success flag only. b Actions are pseudo-labels. c Estimated from reported 40k hours.

Dataset Tasks Domains Trajs Frames Resolution Action Reward Live env.Behavior
RL Unplugged 66 4—12B Varies✓✓✓Replay
V-D4RL 3 1—6M 64{\times}64✓✓✓Mixed
ExoRL 11 1—45M State✓✓✓Exploratory
TD-MPC2 80 2—545M State✓✓✓Replay
Atari DQN Replay 60 1—15B 84{\times}84✓✓✓Replay
RoboNet—1 162k 15M 128{\times}128✓✗✗Scripted
BridgeData V2 13 1 60k 2.3M 640{\times}480✓✗✗Demos
DROID 86 1 76k 19M 1280{\times}720✓\mathbf{\sim}a✗Demos
Open X-Embod.527 22 1M+—Varies✓✗✗Demos
VPT—1—5B 128{\times}128\mathbf{\sim}b✗✓Human
NitroGen 1,000+—39k 4B c 256{\times}256\mathbf{\sim}b✗✓Human
MMBench 200 10 4k 1.8M 224{\times}224✓✓✓Demos
MMBench2 210 10 65.6k 23M\mathbf{224{\times}224}✓✓✓Mixed

## Appendix C Task Domains

We consider 210 tasks across 10 domains. Our task set is comprised of diverse continuous control tasks spanning robot manipulation, locomotion, navigation, arcade games, and classic control problems, each varying in task complexity, time horizon, observation and action space dimensionality, and reward formulation. Table[5](https://arxiv.org/html/2606.27326#A3.T5.70 "Table 5 ‣ Appendix C Task Domains ‣ Hallucination in World Models is Predictable and Preventable") provides an overview of our task domains.

Table 5: Overview of task domains. Our dataset covers a wide range of task types, state and action dimensionalities, time horizons, and reward formulations. Table courtesy of Hansen et al. [[2026](https://arxiv.org/html/2606.27326#bib.bib29 "Learning massively multitask world models for continuous control")].

Task domain Tasks Observation Action dim Ep. length Reward
min max min max
![Image 63: [Uncaptioned image]](https://arxiv.org/html/2606.27326v1/visualizations/domains/dmcontrol/quadruped-run.png)DMControl 23 224\times 224 1 12 500 500 dense/sparse
![Image 64: [Uncaptioned image]](https://arxiv.org/html/2606.27326v1/visualizations/domains/dmcontrol-extended/spinner-spin.png)DMControl Ext.16 224\times 224 1 7 500 500 dense/sparse
![Image 65: [Uncaptioned image]](https://arxiv.org/html/2606.27326v1/visualizations/domains/metaworld/mw-stick-pull.png)Meta-World 49 224\times 224 4 4 100 100 dense
![Image 66: [Uncaptioned image]](https://arxiv.org/html/2606.27326v1/visualizations/domains/maniskill/ms-poke-cube.png)ManiSkill3 37 224\times 224 1 12 25 500 dense/sparse
![Image 67: [Uncaptioned image]](https://arxiv.org/html/2606.27326v1/visualizations/domains/mujoco/mujoco-walker.png)MuJoCo 6 224\times 224 1 8 50 1000 dense/sparse
![Image 68: [Uncaptioned image]](https://arxiv.org/html/2606.27326v1/visualizations/domains/pygame/pygame-coconut-dodge.png)MiniArcade 24 224\times 224 1 2 200 500 dense/sparse
![Image 69: [Uncaptioned image]](https://arxiv.org/html/2606.27326v1/visualizations/domains/box2d/bipedal-walker-obstacles.png)Box2D 8 224\times 224 2 4 500 500 dense
![Image 70: [Uncaptioned image]](https://arxiv.org/html/2606.27326v1/visualizations/domains/robodesk/rd-open-drawer.png)RoboDesk 6 224\times 224 5 5 100 100 dense
![Image 71: [Uncaptioned image]](https://arxiv.org/html/2606.27326v1/visualizations/domains/ogbench/og-antball.png)OGBench 14 224\times 224 2 8 100 1000 dense
![Image 72: [Uncaptioned image]](https://arxiv.org/html/2606.27326v1/visualizations/domains/atari/atari-chopper-command.png)Atari 27 224\times 224 3 3 1000 1000 sparse

DMControl

![Image 73: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/tasks/quadruped-run-1.png)![Image 74: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/tasks/walker-stand-2.png)![Image 75: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/tasks/cup-catch-1.png)![Image 76: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/tasks/finger-turn-easy-3.png)![Image 77: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/tasks/acrobot-swingup-5.png)![Image 78: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/tasks/hopper-stand-2.png)

DMControl Extended

![Image 79: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/tasks/giraffe-run-0.png)![Image 80: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/tasks/cup-catch-var1-1.png)![Image 81: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/tasks/finger-turn-easy-var1-1.png)![Image 82: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/tasks/jumper-jump-0.png)![Image 83: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/tasks/spinner-spin-5.png)![Image 84: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/tasks/reacher-three-easy-5.png)

Meta-World

![Image 85: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/tasks/mw-hammer-2.png)![Image 86: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/tasks/mw-basketball-4.png)![Image 87: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/tasks/mw-stick-push-0.png)![Image 88: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/tasks/mw-door-open-1.png)![Image 89: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/tasks/mw-box-close-0.png)![Image 90: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/tasks/mw-bin-picking-2.png)

ManiSkill3

![Image 91: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/tasks/ms-ant-run-0.png)![Image 92: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/tasks/ms-anymal-reach-1.png)![Image 93: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/tasks/ms-place-sphere-1.png)![Image 94: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/tasks/ms-pick-banana-2.png)![Image 95: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/tasks/ms-lift-peg-2.png)![Image 96: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/tasks/ms-pick-cube-so-0.png)

MuJoCo

![Image 97: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/tasks/mujoco-ant-5.png)![Image 98: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/tasks/mujoco-hopper-4.png)![Image 99: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/tasks/mujoco-reacher-3.png)![Image 100: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/tasks/mujoco-walker-3.png)![Image 101: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/tasks/mujoco-halfcheetah-0.png)![Image 102: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/tasks/mujoco-inverted-pendulum-0.png)

Box2D

![Image 103: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/tasks/bipedal-walker-hills-5.png)![Image 104: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/tasks/bipedal-walker-obstacles-4.png)![Image 105: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/tasks/bipedal-walker-rugged-4.png)![Image 106: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/tasks/lunarlander-takeoff-5.png)![Image 107: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/tasks/ms-lift-peg-2.png)![Image 108: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/tasks/ms-pick-cube-so-0.png)

Figure 7: Visualization of task domains (1 of 2). We show sample tasks from each of the 10 task domains that we consider.

RoboDesk

![Image 109: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/tasks/rd-push-red-1.png)![Image 110: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/tasks/rd-push-green-0.png)![Image 111: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/tasks/rd-push-blue-2.png)![Image 112: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/tasks/rd-open-slide-4.png)![Image 113: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/tasks/rd-open-drawer-1.png)![Image 114: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/tasks/rd-flat-block-in-bin-0.png)

OGBench

![Image 115: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/tasks/og-ant-maze-1.png)![Image 116: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/tasks/og-ant-bottleneck-1.png)![Image 117: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/tasks/og-antball-2.png)![Image 118: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/tasks/og-point-maze-0.png)![Image 119: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/tasks/og-point-var2-0.png)![Image 120: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/tasks/og-point-circle-1.png)

MiniArcade

![Image 121: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/tasks/pygame-bird-attack-2.png)![Image 122: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/tasks/pygame-foraging-2.png)![Image 123: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/tasks/pygame-rocket-collect-5.png)![Image 124: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/tasks/pygame-air-hockey-4.png)![Image 125: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/tasks/pygame-dungeon-explorer1-0.png)![Image 126: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/tasks/pygame-whirlpool-1.png)

Atari

![Image 127: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/tasks/atari-alien-4.png)![Image 128: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/tasks/atari-battle-zone-3.png)![Image 129: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/tasks/atari-chopper-command-3.png)![Image 130: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/tasks/atari-upndown-0.png)![Image 131: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/tasks/atari-road-runner-3.png)![Image 132: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/tasks/atari-seaquest-4.png)

Figure 8: Visualization of task domains (2 of 2). We show sample tasks from each of the 10 task domains that we consider.

Table 6: Task list for finetuning experiments. We finetune on a set of 10 seen (also used in pretraining) + 10 unseen (held out) tasks. The two task sets are listed below. Note that unseen tasks are developed specifically for MMBench2.

Task Domain
Seen task set
cup-catch DMControl
finger-turn-easy DMControl
mw-push Meta-World
ms-push-cube ManiSkill
lunarlander-hover Box2D
og-point-maze OGBench
og-point-bottleneck OGBench
pygame-point-maze-var1 MiniArcade
pygame-pong MiniArcade
pygame-bird-attack MiniArcade
Unseen task set
cup-catch-var1 DMControl
finger-turn-easy-var1 DMControl
ms-push-banana ManiSkill
og-point-var1 OGBench
og-point-var2 OGBench
pygame-point-maze-var4 MiniArcade
pygame-reacher-easy MiniArcade
pygame-dungeon-explorer1 MiniArcade
pygame-foraging MiniArcade
pygame-whirlpool MiniArcade

## Appendix D Data Collection

We base our data collection on that of MMBench, a 200-task benchmark designed primarily for online RL; it provides a total of 4k expert demonstrations collected via single-task expert policies, as well as live environments with ground-truth action and reward labels for all tasks. While MMBench is well suited for the multi-task online RL setting it was originally developed for, a dataset that consists solely of expert demonstrations lacks diversity in terms of behavior which, as our experiments show, is a significant source of hallucination in world models.

The overarching goal of MMBench2 is to produce a large, diverse dataset for visual world modeling and research in common failure modes such as hallucination. To do so, we use the single-task expert policies \pi^{\star} of MMBench as a basis for our data collection, but crucially augment policies to generate diverse behaviors, and also collect data via non-expert policies, including humans. Our methods of data collection can be summarized as follows:

*   •
Random policy. Sample actions uniformly in [-1,1]. Diverse actions; poor task performance.

*   •
No-op actions. Set actions \mathbf{a}=\mathbf{0}. Models dynamics without agent interference.

*   •
Expert actions. Sample from the expert policy \pi^{\star} without added noise. High task performance; low data diversity despite it being a stochastic policy.

*   •
Transformed expert actions. Sample from \pi^{\star}, then apply any of five transforms with some probability: scale by \varepsilon\in[0,1], action dropout, flip sign, all-zero action, repeat previous action. The same transform may be applied for multiple steps. Provides counterfactual transitions.

*   •
Structured noise. Sample from \pi^{\star}, then apply any of three noise types with parameters sampled on a per-episode basis: Gaussian noise, Ornstein-Uhlenbeck noise (temporally correlated), convex mixture with random policy. Mixed performance; improves data diversity.

*   •
Curiosity-driven. Sample trajectories that maximize our developed hallucination predictor u_{r}^{\mathrm{norm}}, selected by CEM-based planning with the pretrained world model.

*   •
Human play data. Actions are selected by a human interacting with the environment via a keyboard interface. High data diversity; not task-driven.

With the exception of human play data, we frequently switch between any of the above behaviors within a single episode to maximize diversity of our training data. For example, we may apply structured noise at random for a number of steps to visit a less frequently visited part of the state-action space and then switch back to the expert policy to generate recovery behavior. Data is collected across all 210 live environments, and ground-truth actions (the action selected by the behavior policy) and reward labels (task reward obtained from the environment) are logged. We collect pretraining data for 200 tasks, and additional data for targeted finetuning across 10 seen (included in pretraining) as well as 10 unseen tasks, with data partitions separated by behavior policy: _random_, _noop_, _expert_, _mixed_, _curiosity_, and _human_. Figure[9](https://arxiv.org/html/2606.27326#A4.F9 "Figure 9 ‣ Appendix D Data Collection ‣ Hallucination in World Models is Predictable and Preventable") shows a screenshot of the interface used to collect human play data.

![Image 133: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/data-collection-interface.png)

Figure 9: Web interface for human play data collection. We develop a simple web interface for data collection. A human user interacts with the environment using the keyboard, and interaction data is saved and later used for world model training. We collect a total of 1{,}400 trajectories using this interface.

## Appendix E Additional Results

Table 7: Detecting hallucination events. Per-task AUROC against two hallucination labels (action-ignored, scene-diverging) on held-out test data from all 200 pretraining tasks. Our three proposed predictors u_{r}^{\text{norm}},u_{f}^{\text{norm}},u_{s}^{\text{norm}} reliably predict hallucinations. Higher is better (\uparrow).

Predictor Action ignored Scene divergent
Tokenizer residual u_{r}^{\text{norm}}\mathbf{0.887}0.919
Flow instability u_{f}^{\text{norm}}0.868\mathbf{0.939}
Inter-seed variance u_{s}^{\text{norm}}0.873 0.934
Scene motion (latent) m 0.803 0.927
kNN distance (global)0.814 0.731
Flow instability u_{f} (raw)0.752 0.854
n_{\text{frames}} (baseline)0.596 0.534

Task

![Image 134: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/tasks/og-point-bottleneck-3.png)

No-op

![Image 135: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/data-collection-panels/og-point-bottleneck_zero.png)

Random

![Image 136: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/data-collection-panels/og-point-bottleneck_random.png)

Expert \pi

![Image 137: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/data-collection-panels/og-point-bottleneck_expert.png)

Curiosity

![Image 138: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/data-collection-panels/og-point-bottleneck_curiosity.png)

Human

![Image 139: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/data-collection-panels/og-point-bottleneck_human.png)

Point Maze Bottleneck (OGBench)

Task

![Image 140: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/tasks/pygame-point-maze-var4-2.png)

No-op

![Image 141: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/data-collection-panels/pygame-point-maze-var4_zero.png)

Random

![Image 142: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/data-collection-panels/pygame-point-maze-var4_random.png)

Expert \pi

![Image 143: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/data-collection-panels/pygame-point-maze-var4_expert.png)

Curiosity

![Image 144: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/data-collection-panels/pygame-point-maze-var4_curiosity.png)

Human

![Image 145: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/data-collection-panels/pygame-point-maze-var4_human.png)

Point Maze Variation 4 (MiniArcade)

Task

![Image 146: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/tasks/cup-catch-1.png)

No-op

![Image 147: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/data-collection-panels/cup-catch_zero.png)

Random

![Image 148: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/data-collection-panels/cup-catch_random.png)

Expert \pi

![Image 149: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/data-collection-panels/cup-catch_expert.png)

Curiosity

![Image 150: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/data-collection-panels/cup-catch_curiosity.png)

Human

![Image 151: Refer to caption](https://arxiv.org/html/2606.27326v1/visualizations/data-collection-panels/cup-catch_human.png)

Cup Catch (DMControl)

Figure 10: Data coverage by collection method. We show state densities for five online data collection policies (no-op, random, expert, curiosity, and human) across three additional tasks besides those in Figure[6](https://arxiv.org/html/2606.27326#S5.F6 "Figure 6 ‣ 5.1 Main results ‣ 5 Experiments ‣ Hallucination in World Models is Predictable and Preventable").

Table 8: Targeted data collection for finetuning on 10 unseen tasks (_expert_ test set). We finetune our world model on a set of 10 seen + 10 unseen tasks, varying data source and finetuning strategy. Each data source contains 50 trajectories per task. The offline results in Table[2](https://arxiv.org/html/2606.27326#S5.T2 "Table 2 ‣ 5.1 Main results ‣ 5 Experiments ‣ Hallucination in World Models is Predictable and Preventable") are computed over a test set that consists of trajectories from both expert policies and human play; this table complements those results by evaluating only on the _expert_ trajectories. We find that _the policy that aligns best with the test set performs the best in evaluations_. Task performance is measured via closed-loop MPC using CEM; we report mean performance normalized to [0,1] across 3 episodes per task.

Method Tok FT Dyn FT Recon PSNR \uparrow Rollout\Delta PSNR \uparrow Action shuf. \uparrow u_{r}^{\mathrm{norm}}\downarrow Task perf.(MPC) \uparrow
Random policy————0.118
Base✗✗17.29-14.19 1.08 4.884—
Ours, pre-FT✗✗17.07-14.22 1.20 4.591 0.276
No-op actions✗✓17.09-13.40 1.27 5.052—
No-op actions✓✓34.73-0.09 1.38 1.843 0.163
Random policy✗✓17.07-13.03 1.50 5.058—
Random policy✓✓35.59+1.74 1.72 1.423 0.228
Expert policy✓✓37.05+2.64 1.95 1.281 0.362
Human play✓✓36.37+2.20 1.90 1.185 0.362
Curiosity (u_{r}^{\mathrm{norm}})✓✓36.63+2.66 1.75 1.342 0.325
All (combined)✓✓\mathbf{38.20}\mathbf{+3.41}\mathbf{2.11}\mathbf{1.090}\mathbf{0.390}

Table 9: Targeted data collection for finetuning on 10 unseen tasks (_human play_ test set). We finetune our world model on a set of 10 seen + 10 unseen tasks, varying data source and finetuning strategy. Each data source contains 50 trajectories per task. The offline results in Table[2](https://arxiv.org/html/2606.27326#S5.T2 "Table 2 ‣ 5.1 Main results ‣ 5 Experiments ‣ Hallucination in World Models is Predictable and Preventable") are computed over a test set that consists of trajectories from both expert policies and human play; this table complements those results by evaluating only on the _human play_ trajectories. We find that _the policy that aligns best with the test set performs the best in evaluations_. Task performance is measured via closed-loop MPC using CEM; we report mean performance normalized to [0,1] across 3 episodes per task.

Method Tok FT Dyn FT Recon PSNR \uparrow Rollout\Delta PSNR \uparrow Action shuf. \uparrow u_{r}^{\mathrm{norm}}\downarrow Task perf.(MPC) \uparrow
Random policy————0.118
Base✗✗17.44-10.70 1.16 2.835—
Ours, pre-FT✗✗17.35-10.82 1.39 2.947 0.276
No-op actions✗✓17.32-9.92 1.56 3.298—
No-op actions✓✓34.75+1.41 1.71 1.128 0.163
Random policy✗✓17.35-9.55 1.96 3.276—
Random policy✓✓36.04+3.59 2.29 0.980 0.228
Expert policy✓✓34.68+3.04 2.14 0.982 0.362
Human play✓✓\mathbf{37.84}\mathbf{+5.58}\mathbf{2.93}\mathbf{0.820}0.362
Curiosity (u_{r}^{\mathrm{norm}})✓✓35.47+3.35 2.24 0.946 0.325
All (combined)✓✓37.61+4.63 2.57 0.861\mathbf{0.390}

Table 10: Targeted data collection for finetuning on 10 _seen_ tasks. We finetune our world model on a set of 10 seen + 10 unseen tasks, varying data source and finetuning strategy. Each data source contains 50 trajectories per task. Offline metrics were computed using a test set of expert trajectories and human play data in equal amount. Task performance is measured via closed-loop MPC using CEM; we report mean performance normalized to [0,1] across 3 episodes per task.

Method Tok FT Dyn FT Recon PSNR \uparrow Rollout\Delta PSNR \uparrow Action shuf. \uparrow u_{r}^{\mathrm{norm}}\downarrow Task perf.(MPC) \uparrow
Random policy————0.083
Base✗✗37.70+3.00 1.59 1.325—
Ours, pre-FT✗✗38.18+4.24 1.84 1.133 0.256
No-op actions✗✓38.18+4.87 2.11 1.113—
No-op actions✓✓39.04\mathbf{+5.30}2.04 1.039 0.316
Random policy✗✓38.14+4.46 2.07 1.063—
Random policy✓✓39.09+5.17 2.07 1.057 0.286
Expert policy✓✓38.71+5.21\mathbf{2.16}\mathbf{0.982}0.355
Human play✓✓39.00+5.05\mathbf{2.16}1.036\mathbf{0.384}
Curiosity (u_{r}^{\mathrm{norm}})✓✓38.96+5.24 2.13 1.015 0.332
All (combined)✓✓\mathbf{39.21}+5.09 2.08 1.019 0.308

Table 11: Effect of reward finetuning. Results for two variants that both extend the pretrained base model with 30 k additional dynamics steps; one jointly trains the dynamics model and a reward head (backpropagating gradients from the reward back into dynamics), and the other is a reward-free control. Mean over all 200 pretraining tasks on a held-out test set. We do not observe a significant difference in results as a result of finetuning with rewards.

Metric w/o reward w/ reward\Delta_{\text{rew}}
Recon PSNR (dB) \uparrow 35.67 35.68+0.01
Action-shuffle ratio \uparrow 1.67 1.62-0.05
Rollout \Delta PSNR (dB) \uparrow 3.01 3.14+0.13
u_{r}^{\text{norm}}\downarrow 1.306 1.318+0.012
u_{f}^{\text{norm}}\downarrow 0.304 0.286-0.018
u_{s}^{\text{norm}}\downarrow 0.515 0.510-0.005

## Appendix F Implementation Details

Our Dreamer 4 world model is a reproduction of the original method as described in Hafner et al. [[2025](https://arxiv.org/html/2606.27326#bib.bib30 "Training agents inside of scalable world models")]. This section provides an overview of our implementation and design choices.

Language embeddings. For task conditioning we use frozen text embeddings from openai/clip-vit-base-patch32 (CLIP; Radford et al. [[2021](https://arxiv.org/html/2606.27326#bib.bib22 "Learning transferable visual models from natural language supervision")]), which produces 512-dimensional continuous embeddings of per-task language instructions. Only the reward prediction and BC policy heads are conditioned on language embeddings.

Block-causal Transformer backbone. Each Transformer is a stack of block-causal layers consisting of _(i)_ space self-attention over the per-frame token sequence, _(ii)_ causal time self-attention along the temporal axis, and _(iii)_ a SiLU-gated MLP with ratio 4. Attention uses RoPE [Su et al., [2024](https://arxiv.org/html/2606.27326#bib.bib65 "RoFormer: enhanced transformer with rotary position embedding")] on Q/K with a KV-cache-aware offset, QK-normalization [Henry et al., [2020](https://arxiv.org/html/2606.27326#bib.bib66 "Query-key normalization for transformers")], and RMSNorm [Zhang and Sennrich, [2019](https://arxiv.org/html/2606.27326#bib.bib67 "Root mean square layer normalization")] pre-norm with no biases on the normalization layers.

Modality-aware mask. Within space self-attention, we apply a modality-aware mask that depends on the role of each token. In the tokenizer encoder, latent queries attend to all tokens while patch queries only attend within the image modality, preventing patch tokens from mixing across modalities before being bottlenecked through the latents; in the decoder, the directions are reversed so patch queries can read from the latent bottleneck but not from each other directly. In the dynamics model, action, shortcut, spatial, and register tokens are mutually visible while agent tokens (reward head and BC policy) are asymmetrically isolated: agent queries attend to everything, but non-agent queries do not see agent keys.

Spatial packing. The tokenizer produces per-frame latents of shape (n_{L},d_{b})=(64,64). Before being fed to the dynamics, these are spatially packed at factor k{=}2 to shape (n_{\mathrm{spatial}},d_{\mathrm{spatial}})=(32,128), halving attention cost along the spatial axis at the cost of doubling channel dimension; the inverse unpack is applied at decode time. Concretely, the dynamics model sees the following token layout per timestep: [ACTION x 1, SHORTCUT x 1, SPATIAL x 32, REGISTER x 4, AGENT x 4] where the action token is produced by a 2-layer MLP, the shortcut-conditioning token concatenates two embeddings of the discretized noise level \sigma and step size d{=}1/2^{\mathrm{step}}, register tokens [Darcet et al., [2024](https://arxiv.org/html/2606.27326#bib.bib68 "Vision transformers need registers")] are 4 learnable “sink” tokens, and agent tokens are initialized from the per-task CLIP embedding broadcast over time.

Loss normalization. Pixel MSE and LPIPS terms in the tokenizer, the empirical and self-consistency branches of the dynamics objective, and reward two-hot cross-entropy are each separately normalized by their own running RMS before weighting. This decouples loss weights from absolute scale and removes the need to tune them when the dataset, resolution, or backbone changes.

Shortcut flow matching. The dynamics model is trained with the shortcut flow-matching objective of Frans et al. [[2025](https://arxiv.org/html/2606.27326#bib.bib32 "One step diffusion via shortcut models")]. The discretized noise level \sigma is indexed by an integer in \{0,\dots,k_{\mathrm{max}}\} (with k_{\mathrm{max}}{=}64 in our experiments; 0 is pure noise, k_{\mathrm{max}} is clean) and the step is indexed by an integer in \{0,\dots,\log_{2}(k_{\mathrm{max}})\} corresponding to step size d{=}1/2^{\mathrm{step}}. For a fraction \rho_{\mathrm{self}}{=}0.25 of each batch we apply a self-consistency bootstrap: at (\sigma,\mathrm{step}), two coarser half-steps at \mathrm{step}{+}1 are run under no_grad and their averaged velocity is used as a stop-gradient target for the current step’s predicted velocity. The remaining 0.75 of the batch uses the empirical one-step regression term at the finest step.

Reward and BC heads. Both heads read from the agent tokens via attention pooling against a learnable query. The reward head predicts L{=}8 multi-step symlog two-hot distributions over 255 bins on the range [-10,10]. The BC head is a deterministic Gaussian policy with diagonal covariance trained via an MSE loss on the ground-truth 16-dimensional padded action.

Sampling. At inference we run a shortcut Euler integrator with step size d{=}0.125 (_i.e._, K{=}8 substeps) for each new frame: starting from z\sim\mathcal{N}(0,I), we solve b=(\hat{x}_{1}-z)/(1-\sigma) followed by z\leftarrow z+b\cdot d. We apply context corruption to past tokens.

Hyperparameters. We summarize our hyperparameters in Table[12](https://arxiv.org/html/2606.27326#A6.T12.43 "Table 12 ‣ Appendix F Implementation Details ‣ Hallucination in World Models is Predictable and Preventable").

Table 12: Hyperparameters. Key hyperparameters used to train our world model.

Hyperparameter Value
Data
Image resolution 224\times 224\times 3
Action dim. (zero-padded)16
Tokenizer (~100M)
Patch size 14
Embedding dim. (d_{\mathrm{model}})512
Heads 8
Depth 12
MLP ratio 4
Number of latents (n_{L})64
Bottleneck dim. (d_{b})64
MAE keep range[0.1,\,1.0]
LPIPS weight 0.2
Dynamics (~230M)
Embedding dim. (d_{\mathrm{model}})1024
Heads 8
Depth 16
MLP ratio 4
Spatial packing factor 2
Register tokens 4
Agent tokens 4
Self-consistency fraction 0.25
Context corruption (\tau_{\mathrm{ctx}})0.1
Reward + BC heads (~20M)
Multi-step horizon (L)8
Reward bins 255
Reward symlog range[-10,\,10]
Optimization
Optimizer AdamW
Learning rate 1\times 10^{-4}
Weight decay 1\times 10^{-2}
Sequence length 24
Effective batch size Tok: 96 / Dyn: 512
Sampling (inference)
Integrator schedule Shortcut
Step size (d)0.125
Planning (CEM)
Plan horizon (H)32
Replan every (K)16
CEM Iterations 3
Population size 32
Rollouts per candidate 2
Warm start (mean)BC prior

