# 2608.16885 (from arXiv HTML; MinerU fallback)



# $\tau_{0}$-VLA: a Hierarchical Robot Foundation Model with World-Model-Guided Test-Time Computation Authors are listed alphabetically. See the Author Contributions section for individual contributions.

Xiaowei Cai2,
Yunuo Cai1,2,
Bingao Chen2,3,
Jingxiao Chen2,
Zhi Chen2,
Siyuan Feng2,
Tengyu Hou2,
Jingshun Huang1,2,
Han Jiang2,
Runkun Ju2,
Dong Li2,
Mingxiang Li2,
Shaowei Li2,
Xinchen Li2,
Yifan Li1,2,
Yi Liu1,2,
Zhongyuan Liu2,
Jianlan Luo1,2,
Junwen Miao2,
Ruiqi Ni2,
Buqing Nie2,
Mingjie Pan1,2,
Xinlin Ren2,
Jianheng Song2,
Jiaxu Wang2,3,
Peiqi Wang2,
Sen Wang2,
Xiaoyan Wang2,
Dafeng Wei2,
Dongming Wu2,3,
Pengwei Xie2,
Pu Yang2,
Hangjian Ye1,2,
Xiangyu Yue2,3,
Jinyu Zhang1,2,
Qinglin Zhang2,
Xueyong Zhao2,
Pengfei Zhou2,
Yue Zhou2

Affiliation: 1Shanghai Innovation Institute   2Agibot Finch   3The Chinese University of Hong Kong
[https://tau0-vla.github.io/](https://tau0-vla.github.io/)

###### Abstract

Long-horizon robot manipulation requires a robot to both execute individual skills reliably and sequence them coherently over extended tasks. Most hierarchical vision-language-action (VLA) models make each such decision with a single forward pass, leaving no mechanism to allocate additional computation to difficult or consequential choices.
We introduce $\tau_{0}$-VLA, a hierarchical robot foundation model that formulates high-level subtask generation as a compute-scalable inference problem through world-model-guided test-time computation.
At each inference step, the high-level policy uses execution memory to generate a subtask and, when needed, searches over alternatives before committing to its output.
A low-level policy then executes the generated subtask across multiple robot embodiments. The policy is trained on 40,115 hours of heterogeneous real-world data with multimodal co-training. Across in-domain and distribution-shifted settings, allocating additional test-time computation substantially improves next-subtask prediction accuracy, and these gains translate into higher closed-loop success on long-horizon robot manipulation tasks.

††aftertitle:

![[Uncaptioned image]](drafts/images/tau0vla-2608.16885/teaser_v8.png)

Fig. 1: Overview of $\tau_{0}$-VLA.
The high-level policy uses world-model-guided test-time computation to search over subtask sequences. At each expansion step, a VLM proposes candidate subtasks, a world model predicts their visual outcomes, and a value model evaluates the resulting task progress. Beam search retains promising branches, after which a reflective model commits to the next subtask based on the retained candidates and their predicted consequences. The figure illustrates a single candidate expansion with $N=3$; recursive beam expansion to depth $D$ is omitted for brevity. The selected subtask then conditions a low-level VLA policy trained on $40{,}115$ hours of heterogeneous real-world data, enabling long-horizon mobile manipulation, precise execution, and deployment across multiple robot embodiments.

## I Introduction

Long-horizon manipulation encompasses a broad class of real-world robotic
tasks, such as cleaning a room, cooking a meal, or preparing a drink. These
tasks require coherent sequences of subtasks carried out over minutes or hours.
The robot must locate and manipulate objects, interact with articulated
structures, verify intermediate outcomes, and recover from local failures.
Such tasks are not simply long sequences of motor commands; they are
sequences of consequential decisions.
Throughout execution, the robot must infer what has been completed, determine
what remains, and maintain an appropriate subtask command.
An incorrect subtask choice cannot generally be corrected by more precise
motor control: the robot may execute the wrong subtask perfectly.

Language-conditioned robot policies and vision-language-action (VLA) models provide a natural foundation for long-horizon robotic control by mapping visual observations and language instructions to robot actions [6, 46, 21, 30, 3, 2]. Hierarchical robot policies further connect semantic reasoning to visuomotor control through language commands, explicit subtask interfaces, or latent representations [20, 1, 36, 26, 15, 16, 2, 34]. Many hierarchical systems still make each high-level decision with a fixed inference budget, directly mapping the current observation and execution history to the next subtask. In this setting, the model neither compares alternative subtasks nor evaluates them through the physical states they are expected to produce. Recent systems instead use high-level tree search or recursive subgoal refinement [32, 42, 44], motivating the question we study here: how to couple open-ended language-subtask search with visual outcome prediction and execution memory in a generalist real-robot hierarchy.
Without pre-commitment evaluation, an incorrect decision is typically detected only after execution has already altered the environment, at which point replanning can respond but cannot recover the cost of the failed commitment. Thus, the central bottleneck in long-horizon execution is no longer the availability of a hierarchical action interface, but the inference procedure used to generate the next subtask.

We instead formulate generating the next subtask as an inference-time reasoning problem, allowing computation to scale with the difficulty of each decision, as test-time computation does in language models [31, 10]. The subtask is a natural unit for this reasoning. Action-level search can resolve local control ambiguities, but often exposes only local physical consequences [29, 22, 9, 41, 17, 43]. Language-only reasoning operates over longer horizons, but compares alternatives without grounding them in the physical states they would produce. Subtasks lie between these extremes: they are sparse enough to justify additional computation, yet temporally extended enough to induce meaningful and distinguishable changes in the environment. Comparing candidate subtasks before execution therefore requires predicting the state that each candidate would produce. Generative models of future observations provide precisely this capability [13, 5, 12, 14, 27], enabling candidate subtasks to be evaluated through their anticipated physical consequences before they are issued for execution.

We present $\tau_{0}$-VLA, a hierarchical VLA system that instantiates this approach. At each inference step, a memory-augmented high-level policy uses the current observation and execution memory to generate the appropriate subtask. The proposal is used directly when the policy is confident. Otherwise, token-confidence statistics trigger an additional reasoning procedure. This procedure follows a propose–predict–evaluate loop. The proposal model generates open-ended candidate subtasks, a world model predicts the terminal observation induced by each candidate, and a value model scores candidate quality from the predicted outcomes. Beam search recursively expands promising candidates, allowing the inference budget to scale through the branching factor, beam width, and search depth. A reflective model then conditions on the retained branches and generates the final subtask. Its output may coincide with a proposal but is not restricted to the candidate set. The low-level policy then executes the generated subtask. The resulting real observation is incorporated into execution memory at the next inference step, closing the loop between predicted and observed consequences. To support reliable reasoning over extended executions, we train the memory mechanism to correct its own state. Specifically, we perturb memories derived from existing demonstrations and train the high-level policy to repair records that lag behind, run ahead of, or otherwise misrepresent the robot’s actual progress, requiring no additional annotation.

Once generated, each subtask is executed by a generalist low-level policy that combines a pretrained vision-language backbone with a Mixture-of-Transformers action expert. The policy operates over a unified $40$-dimensional state and action space covering end effectors, arm joints, grippers, the waist, and the mobile base. This unified representation allows a single model to support fixed-base manipulation, bimanual coordination, and mobile whole-body control across multiple robot embodiments. We train the policy on approximately $40{,}000$ hours of robot data collected from heterogeneous sources, together with multimodal co-training data, yielding a language-conditioned execution interface without the need for subtask-specific controllers.

We evaluate the complete system across multiple robot embodiments on real-world,
long-horizon manipulation tasks. The evaluation covers room cleaning, meal
preparation, tea making, and laundry collection, with episodes lasting up to
12 minutes.
Hierarchical test-time computation substantially improves task success over whole-task inference using the same low-level policy and increases next-subtask prediction accuracy, with further gains obtained from larger inference budgets.

Our central contribution is a complete VLA system, $\tau_{0}$-VLA, that formulates high-level subtask generation as a compute-scalable inference problem while retaining a unified low-level control interface across
multiple robot embodiments.
Our experiments show that allocating additional computation to high-level
decisions substantially improves next-subtask prediction, and yields higher end-to-end task success on
long-horizon real-robot tasks.
We further provide a detailed empirical analysis of execution memory, consequence-aware search, and the compute-accuracy trade-off.

## II Related Work

$\tau_{0}$-VLA brings together three lines of work: generalist VLA models
for low-level control, hierarchical robot policies for subtask-level
decision making, and world models with test-time computation for evaluating
candidate decisions. We thus survey works in these areas.

### II-A Vision-Language-Action Models

Vision-language-action (VLA) models map visual observations and language
instructions to robot actions, often by adapting pretrained vision-language
backbones for continuous control. Early work established scalable
language-conditioned robot learning and transfer from vision-language
pretraining [6, 46]. Subsequent work broadened this paradigm across
tasks and embodiments [21, 30], while recent generalist models
target continuous control, open-world generalization, and cross-embodiment
transfer [3, 2, 26, 45, 40, 16].
Complementary work studies compact policies, efficient action tokenization, knowledge-preserving training, and asynchronous action-chunk execution [37, 33, 11, 4].
Cross-embodiment policies must also reconcile heterogeneous control
interfaces. A physically interpretable unified action space is introduced
by 24, whereas heterogeneous controls are mapped into semantically
aligned slots by 28.

This line of work primarily improves control representations, training scale,
and transfer across tasks and embodiments. Our low-level policy follows the
same generalist VLA paradigm. Our primary focus is complementary: making
high-level subtask generation an explicit, compute-scalable inference
problem.

### II-B Hierarchical Robot Policies

Hierarchical policies separate temporally extended decisions from continuous
control. Prior work grounds language-model decisions in learned robot
affordances [20], represents action hierarchies through
language [1], and connects slower semantic reasoning to faster
visuomotor control through explicit commands, latent representations, or
staged inference [36, 15, 26, 16, 2].
Recent systems address long-horizon state tracking and adaptation more
explicitly.
39 combine short-term visual memory with long-term textual memory.
38 retrieve task-relevant historical keyframes
before generating instructions for a low-level VLA, while 25
combine structured memory, outcome verification, and reflection in a
high-level VLM and low-level VLA hierarchy. 44
adaptively generate recursive subgoals, whereas 27 use a world
model as the high-level policy to generate textual and visual subgoal
sequences.

Among generalist hierarchical systems, one recent approach couples a semantic
high-level policy, a world model, and a low-level policy [34]. Its
world model generates subgoal images for a subtask
already produced by the high-level policy. In $\tau_{0}$-VLA, visual prediction
instead occurs before commitment and supports comparison among multiple
candidate subtasks through value-guided search and reflection. Thus, visual
prediction conditions how a generated subtask is executed in that framework,
whereas it informs which subtask to execute in $\tau_{0}$-VLA. Our high-level
policy also maintains a correctable execution memory and scales
decision-time computation through search width and depth.

### II-C World Models and Test-Time Computation

World models predict future states under candidate behavior and support
planning and policy learning [18, 19]. In robot
manipulation, generated images or videos have been used as goals for inverse
dynamics and low-level control [13, 5], modeled jointly with
actions [8, 7], and incorporated into search or iterative plan
revision [12].
14 iteratively revise a VLM plan using imagined
future states. Our search instead maintains multiple language-subtask
branches and prunes them with a dedicated value model before reflective
generation.
At the action level, additional inference is allocated through sampling,
verification, or value-guided selection [29, 22, 9].
41 predict latent outcomes for low-level action plans and use a
VLM to select among them, while 17 perform
world-model-guided Monte Carlo tree search over action trajectories.
43 jointly predict textual subtasks, future images, and actions,
and their test-time scaling mode selects among action chunks using imagined
future frames and a value model.

Search has also been applied to high-level robot decisions. 32
search structured scene-graph subgoal transitions, predict their success
probabilities from successful and failed demonstrations, and prune
low-feasibility branches before execution. 42 instead
search discrete manipulation plans with a learned critic, combining predicted
visual dynamics, value-guided beam search, confidence-based routing, and
multi-path reflection. Both are among the closest search-based mechanisms to
our work. By contrast, $\tau_{0}$-VLA searches open-ended language subtasks and
evaluates each candidate through its predicted terminal image. It conditions
subsequent decisions on a correctable execution memory, then uses a reflective
model to generate the final subtask from the retained branches.

## III Preliminaries

We consider a robot task specified by a language instruction $\ell$. At
inference step $t$, a vision-language-action (VLA) policy maps the current
multi-view observation $o_{t}$, proprioceptive state $\mathbf{s}_{t}$, and language
command $c_{t}$ to an $H$-step action chunk
$\mathbf{a}_{t:t+H-1}$. It also receives textual control metadata $\eta$
specifying the embodiment, control mode, and whole-body configuration. Its
concrete text serialization is provided in
Appendix C-B:

|  | $$ \mathbf{a}_{t:t+H-1}=\pi_{\theta}(o_{t},\mathbf{s}_{t},c_{t},\eta), $$ |  | (1) |
|---|---|---|---|

where $\theta$ denotes the policy parameters and $H$ is the action-chunk
horizon. In direct execution, the full task instruction is used throughout the
episode, so $c_{t}=\ell$. The policy must therefore infer the current stage of
the task while generating the corresponding motor actions.

A hierarchical VLA separates high-level subtask generation from low-level
action generation. At each inference step, the high-level policy first
generates a subtask, and the low-level policy then generates actions
conditioned on it. Let $\mu$ denote the high-level policy, let
$\mathcal{M}_{0}$ denote the initial empty execution memory, and let
$z_{0}^{\star}=\varnothing$. At inference step $t$, $\mu$ forms the high-level
decision context

|  | $$ h_{t}=\left(\ell,\,\mathcal{M}_{t-1},\,z_{t-1}^{\star},\,o_{t}\right), $$ |  | (2) |
|---|---|---|---|

where $\mathcal{M}_{t-1}$ is the carried execution memory and
$z_{t-1}^{\star}$ is the subtask generated at the preceding inference step. The
policy brings the execution memory up to date and generates the subtask for the
current observation:

|  | $$ \left(\mathcal{M}_{t},\,z_{t}^{\star}\right)=\mu(h_{t}). $$ |  | (3) |
|---|---|---|---|

Here $\mathcal{M}_{t}$ summarizes the execution history observed up to $o_{t}$ and
therefore does not yet contain the outcome of executing $z_{t}^{\star}$, while
$z_{t}^{\star}$ is the subtask generated for the current observation. The generated
subtask is used as the language command, $c_{t}=z_{t}^{\star}$, yielding

|  | $$ \mathbf{a}_{t:t+H-1}=\pi_{\theta}(o_{t},\mathbf{s}_{t},z_{t}^{\star},\eta). $$ |  | (4) |
|---|---|---|---|

After executing the action chunk, the resulting real observation is used at
the next inference step. At every inference step, the low-level policy
conditions on the generated subtask $z_{t}^{\star}$. If it remains the same as
$z_{t-1}^{\star}$, execution of the current subtask continues. If it changes,
the low-level policy begins executing the newly generated subtask.

The high-level policy can either commit to its direct proposal or invoke
additional test-time computation before producing $z_{t}^{\star}$. Thus,
the high-level policy determines which subtask is
currently appropriate, while the low-level policy determines how to
execute it.

## IV Method

### IV-A System Overview

$\tau_{0}$-VLA consists of two components applied sequentially at each logical inference step. The high-level policy maintains execution memory, decides when to invoke additional test-time computation, and generates the current subtask. The low-level policy then maps the current observation and generated subtask to robot actions. This hierarchy supports long-horizon progress tracking and consequence-aware planning without modifying the low-level control interface. Figure 2 provides an overview. The following subsections define both policies and then describe their joint inference procedure.

![Refer to caption](drafts/images/tau0vla-2608.16885/framework_0726.png)

*Fig. 2: The hierarchical $\tau_{0}$-VLA architecture.
(a) At a high-level inference step $t$, the proposal model $P$ conditions
on the latest multi-view observation $o_{t}$, task instruction $\ell$,
carried execution memory $\mathcal{M}_{t-1}$, and previously generated
subtask $z_{t-1}^{\star}$. It produces observation-aligned memory
$\mathcal{M}_{t}$ and a direct proposal $z_{t}^{\mathrm{dir}}$. (b) The
low-level policy conditions on the generated subtask
$z_{t}^{\star}$, multi-view observation $o_{t}$, proprioceptive state
$\mathbf{s}_{t}$, and textual control metadata $\eta$. A
vision-language backbone and Mixture-of-Transformers (MoT) action expert
generate the action chunk $\mathbf{a}_{t:t+H-1}$ through conditional flow
matching from a noisy action chunk. (c) On the TTC route, the proposal
model is invoked $N$ times for each retained branch to generate $N$
candidates. Given the branch’s head-camera image and a candidate,
the world model predicts the terminal head-camera image, and the value model
assigns a candidate-quality score conditioned on the task instruction,
candidate, and predicted image. Beam search globally retains the top-$B$
branches by cumulative score and recursively expands them to depth $D$.
The figure illustrates the root expansion with $N=3$ and $B=2$, where
local and cumulative scores coincide. Deeper expansion is omitted for
clarity. The reflective model then conditions on $\bar{h}_{t}$ and the final
branch summaries $\mathcal{C}_{t}$ to generate the final subtask
$z_{t}^{\star}$. This output may coincide with a retained proposal but is not
restricted to the retained set and is passed to the low-level policy for
execution.*

### IV-B High-Level Policy

Unlike a conventional fixed-compute high-level policy which commits to a subtask through a single forward pass, our high-level policy $\mu$ generates the next subtask with a budget-adaptive test-time computation (TTC) procedure.
On uncertain predictions, TTC performs world-model-guided beam search over possible subtask sequences. It expands candidate branches, predicts and scores their visual outcomes, and passes the retained beam to a reflective model that generates the final subtask.
This procedure allows the policy to improve next-subtask prediction by allocating additional test-time computation.

Accordingly, the high-level policy $\mu$ consists of a proposal model $P$, a world model $\mathcal{W}$, a value model $V$, and a reflective model $F$.
Their inference interfaces are defined below.

Proposal model.  At the beginning of inference step $t$, the
proposal model receives the context $h_{t}$ defined in
Section III, updates the execution memory, and generates a
direct subtask proposal $z_{t}^{\mathrm{dir}}$:

|  | $$ \left(z_{t}^{\mathrm{dir}},\mathcal{M}_{t}\right)=P(h_{t}). $$ |  | (5) |
|---|---|---|---|

The proposal model updates the memory from $\mathcal{M}_{t-1}$, the previously
generated subtask $z_{t-1}^{\star}$, and the current observation $o_{t}$. From
token confidences produced by the same forward pass, the adaptive router
computes $g_{t}\in\{0,1\}$. Here $g_{t}=0$
selects the fast route and $g_{t}=1$ invokes TTC. The routing rule is
defined in Appendix C-J.

World model and value model.  For a generic candidate within beam
search, let $\tilde{o}$ denote a head-camera RGB image and $z$ the candidate
subtask. The world model predicts the terminal head-camera image, and the value
model scores that outcome:

|  | $$ \hat{o}=\mathcal{W}(\tilde{o},z),\qquad v=V(\ell,z,\hat{o}). $$ |  | (6) |
|---|---|---|---|

Thus, the world model always operates on a single head-camera image. The value
model receives the global task instruction, candidate subtask, and predicted
terminal image, and returns a scalar candidate-quality score $v$.

Test-time search.  When $g_{t}=1$, the high-level policy invokes the
Search operation in Algorithm 1:

|  | $$ \mathcal{C}_{t}=\operatorname{Search}\left(h_{t},P,\mathcal{W},V,N,B,D\right). $$ |  | (7) |
|---|---|---|---|

The operation performs beam search over candidate subtask sequences. The
positive integers $N$, $B$, and $D$ denote the branching factor, beam width,
and search depth. A branch $b$ stores a proposal context $h(b)$, an ordered
sequence of imagined subtasks $\rho(b)$, a cumulative score $S(b)$, and, for a
non-root branch, a terminal predicted image $\hat{o}(b)$. We initialize
$\mathcal{Q}_{t,0}$ with a single root branch
$b_{\mathrm{root}}$ whose context is $h_{t}$, whose path is the empty sequence
$\rho(b_{\mathrm{root}})=()$, and whose score is
$S(b_{\mathrm{root}})=0$.

At each depth $d\in\{1,\ldots,D\}$, we independently invoke the proposal model
$N$ times for every retained branch $b\in\mathcal{Q}_{t,d-1}$:

|  | $$ \left(z^{b,i}_{t,d},\mathcal{M}^{b,i}_{t,d}\right)\sim P(\,\cdot\mid h(b)),\qquad i=1,\ldots,N. $$ |  | (8) |
|---|---|---|---|

Here $P(\,\cdot\mid h(b))$ denotes the proposal model’s decoding distribution.
The sample index $i$ keeps repeated text proposals distinct and associates
each proposal with its branch-local memory. For the root branch,
$h(b_{\mathrm{root}})=h_{t}$, so the proposal model receives the current
multi-view observation $o_{t}$. For every non-root branch, the visual component
of $h(b)$ is $\hat{o}(b)$, the terminal head-camera image imagined for that
branch. Routing decisions from proposal calls inside search are ignored.

We write $b\oplus z$ for the child branch obtained by appending subtask $z$ to
branch $b$. Its path $\rho(b\oplus z)$ is the ordered sequence $\rho(b)$
followed by $z$. Let $\tilde{o}(b)$ be the head-camera image in $o_{t}$ when
$b=b_{\mathrm{root}}$ and $\hat{o}(b)$ otherwise. For each indexed proposal,
the world and value models compute

|  | $\displaystyle\hat{o}\!\left(b\oplus z^{b,i}_{t,d}\right)$ | $\displaystyle=\mathcal{W}\!\left(\tilde{o}(b),z^{b,i}_{t,d}\right),$ |  | (9) |
|---|---|---|---|---|
|  | $\displaystyle v\!\left(b\oplus z^{b,i}_{t,d}\right)$ | $\displaystyle=V\!\left(\ell,z^{b,i}_{t,d},\hat{o}\!\left(b\oplus z^{b,i}_{t,d}\right)\right).$ |  |

The child retains the corresponding memory
$\mathcal{M}^{b,i}_{t,d}$ and predicted image. Its cumulative score and context
for the next expansion are

|  | $\displaystyle S\!\left(b\oplus z^{b,i}_{t,d}\right)$ | $\displaystyle=S(b)+v\!\left(b\oplus z^{b,i}_{t,d}\right),$ |  | (10) |
|---|---|---|---|---|
|  | $\displaystyle h\!\left(b\oplus z^{b,i}_{t,d}\right)$ | $\displaystyle=\left(\ell,\mathcal{M}^{b,i}_{t,d},z^{b,i}_{t,d},\hat{o}\!\left(b\oplus z^{b,i}_{t,d}\right)\right).$ |  |

All memories produced inside search are branch-local and never overwrite the
persistent execution memory $\mathcal{M}_{t}$.

At depth $d$, the indexed collection of all children is

|  | $$ \mathcal{A}_{t,d}=\left(b\oplus z^{b,i}_{t,d}\right)_{\begin{subarray}{c}b\in\mathcal{Q}_{t,d-1}\\ i=1,\ldots,N\end{subarray}},\qquad\mathcal{Q}_{t,d}=\operatorname{Top}_{B}\!\left(\mathcal{A}_{t,d},S\right). $$ |  | (11) |
|---|---|---|---|

Before pruning, $\mathcal{A}_{t,1}$ contains $N$ children, while each
subsequent expansion produces at most $BN$ children. The top-$B$ operation
ranks all children globally by cumulative score. Each retained child’s
predicted image and branch-local memory define the context for its next
expansion. This process repeats until depth $D$.

The final beam is summarized by the indexed collection

|  | $$ \mathcal{C}_{t}=\left(\bigl(\rho(b),\hat{o}(b),S(b)\bigr)\right)_{b\in\mathcal{Q}_{t,D}}, $$ |  | (12) |
|---|---|---|---|

where $\rho(b)$ is the imagined subtask path and $\hat{o}(b)$ is its terminal
predicted head-camera image.

Reflective model.  At the end of the TTC route at inference step
$t$, the reflective model conditions on the retained branch summaries
$\mathcal{C}_{t}$ and the observation-aligned real context
$\bar{h}_{t}=(\ell,\mathcal{M}_{t},z_{t-1}^{\star},o_{t})$. It generates the final
subtask passed to the low-level policy:

|  | $$ z_{t}^{\star}=F(\bar{h}_{t},\mathcal{C}_{t}). $$ |  | (13) |
|---|---|---|---|

Its output may reproduce a retained proposal but is not constrained to the
candidate set. The reflective model does not update persistent execution
memory.

### IV-C Low-Level Policy

The low-level policy couples a vision-language backbone with a
Mixture-of-Transformers (MoT) action expert. At each full-attention layer, the
action tokens and backbone tokens interact through joint attention while being
processed by separately parameterized Transformer streams. The policy
conditions on multi-view observations, proprioceptive state, and a language
command. The action expert is trained with conditional flow matching to learn
a velocity field from noise to the distribution of action chunks. At
inference, this field is integrated to generate executable actions.

Unified state and action space. 
We represent heterogeneous embodiments in a shared $40$-dimensional state and
action space. Per-sample state and action masks select the valid channels,
while the control metadata specifies the control parameterization. This
interface allows the same policy to support fixed-base and whole-body control
without embodiment-specific output heads.
Appendix C-B provides the complete layout and action
encoding.

Masked flow matching. 
Because different embodiments occupy different channels in the unified action
space, we mask both the flow path and its training objective. Let $d_{a}$ denote
the action dimension and let
$\mathbf{M}\in\{0,1\}^{d_{a}\times d_{a}}$ be a diagonal action-mask matrix, where
$[\mathbf{M}]_{ii}=1$ if channel $i$ is active and $[\mathbf{M}]_{ii}=0$
otherwise. The same matrix is applied to every action vector in the chunk.
For each $j\in\{0,\ldots,H-1\}$, let
$\mathbf{a}_{t+j},\boldsymbol{\epsilon}_{t+j}\in\mathbb{R}^{d_{a}}$, where
$\boldsymbol{\epsilon}_{t+j}\sim\mathcal{N}(\mathbf{0},\mathbf{I}_{d_{a}})$ and $\mathbf{I}_{d_{a}}$ is the
$d_{a}$-dimensional identity matrix. Following the linear Gaussian flow path
of 3, we use its algebraically equivalent reverse-time
reparameterization to match the task-specific checkpoints evaluated in this
work. A flow time
$\tau\in[0,1]$ is shared across the chunk, with $\tau=1$ denoting noise and
$\tau=0$ denoting a clean action. The interpolated action and target velocity
are

|  | $\displaystyle\mathbf{a}_{t+j}^{\tau}$ | $\displaystyle=\tau\mathbf{M}\boldsymbol{\epsilon}_{t+j}+(1-\tau)\mathbf{M}\mathbf{a}_{t+j},$ |  | (14) |
|---|---|---|---|---|
|  | $\displaystyle\mathbf{u}_{t+j}$ | $\displaystyle=\mathbf{M}(\boldsymbol{\epsilon}_{t+j}-\mathbf{a}_{t+j}).$ |  |

Given the complete noisy chunk, the action expert jointly predicts the
velocities of all $H$ action tokens. We supervise only active channels and
project inactive channels before each velocity-field evaluation and at the
final output. Appendix C-B provides the full loss and
sampling details.

### IV-D System Inference

At inference, the high-level policy and low-level policy form a closed loop.
At each logical inference step, the proposal model updates the execution
memory and generates $z_{t}^{\mathrm{dir}}$. When $g_{t}=0$, this direct proposal
becomes the final subtask $z_{t}^{\star}$. When $g_{t}=1$, test-time search produces
$\mathcal{C}_{t}$, and the reflective model generates $z_{t}^{\star}$ from the
observation-aligned context and retained branch summaries. The low-level policy
then generates and executes an action chunk conditioned on $z_{t}^{\star}$. The
resulting real observation is incorporated into memory at the next inference
step.

Algorithm 1 presents these logical dependencies as a
sequential procedure. In deployment, the high-level and low-level policies are
pipelined asynchronously, as detailed in Appendix C-F. The
same routing statistics are used across tasks, with task-specific thresholds
calibrated on held-out data. In direct-execution mode, the system bypasses the
high-level policy and conditions the low-level policy directly on $\ell$.

*Algorithm 1  Closed-loop system inference*

1:


            Task instruction $\ell$, models $P,\mathcal{W},V,F,\pi_{\theta}$

2:


            Control metadata $\eta$, action horizon $H$, branching factor $N$, beam width $B$, depth $D$

3:


            Initialize $\mathcal{M}_{0}\leftarrow\varnothing$ and
$z_{0}^{\star}\leftarrow\varnothing$

4:
for $t=1,2,\ldots$ until task termination do

5:


              Observe $o_{t}$ and $\mathbf{s}_{t}$

6:


              $h_{t}\leftarrow(\ell,\mathcal{M}_{t-1},z_{t-1}^{\star},o_{t})$

7:


              $(z_{t}^{\mathrm{dir}},\mathcal{M}_{t})\leftarrow P(h_{t})$

8:


              Compute $g_{t}$ from the proposal token confidences

9:


              $\bar{h}_{t}\leftarrow(\ell,\mathcal{M}_{t},z_{t-1}^{\star},o_{t})$

10:


              if $g_{t}=0$ then

11:


               $z_{t}^{\star}\leftarrow z_{t}^{\mathrm{dir}}$ $\triangleright$ fast route

12:


              else

13:


               $\mathcal{C}_{t}\leftarrow\textsc{Search}(h_{t},P,\mathcal{W},V,N,B,D)$

14:


               $z_{t}^{\star}\leftarrow F(\bar{h}_{t},\mathcal{C}_{t})$

15:


              end if

16:


              $\mathbf{a}_{t:t+H-1}\leftarrow\pi_{\theta}(o_{t},\mathbf{s}_{t},z_{t}^{\star},\eta)$

17:


              Execute action chunk $\mathbf{a}_{t:t+H-1}$

18:
end for

## V Training

### V-A Data Sources

Low-level policy data. The low-level policy is trained on $40{,}115$ hours of heterogeneous real-world
data spanning fixed-base, mobile, and bimanual embodiments. The corpus
combines human demonstrations, autonomous policy rollouts, and UMI-style
recordings, covering manipulation, navigation, and whole-body coordination
across diverse robot morphologies and control interfaces. We interleave these
trajectories with multimodal vision-language data for instruction following,
visual grounding, spatial and depth reasoning, and robot-centric perception.
This co-training mixture preserves the semantic and visual capabilities of the
VLM backbone during action learning. Appendix C-D provides
further details on the data composition and processing.

High-level policy data. Supervision for the high-level policy is derived automatically from existing
task, stage, and executable-subtask annotations together with segmented
multi-view demonstrations. These sources supervise reasoning over task
progress, execution-memory updates, and subtask generation. We additionally
construct memory-perturbed examples that teach the policy to recover when its
execution history lags behind, runs ahead of, or otherwise conflicts with the
observed state. Subtask-aligned start–end frames supervise the world model.
This construction requires no additional per-sample human labeling.
Appendix C-E describes the annotation, filtering, and quality
control pipeline.

### V-B Low-Level Policy Training

We initialize the low-level vision-language backbone from
Qwen3.5-2B [35] and attach the MoT action expert described in
Section IV-C. The resulting policy maps the current
observation and language command to robot actions in the unified action space,
as defined in Eq. (1). We train it in three successive
stages.

Stage 1: Knowledge-isolated co-training. 
Following 11, we jointly train on multimodal and
robot-action data with knowledge isolation (KI). The MoT action expert attends
to the vision-language backbone, while action-loss gradients are blocked at
the backbone interface. This allows the action expert to learn
control-relevant representations without prematurely perturbing the
pretrained backbone, with multimodal supervision stabilizing training.

Stage 2: End-to-end co-training. We remove KI and optimize the full model end to end, retaining multimodal data as auxiliary supervision. This allows action gradients to adapt the backbone toward representations better suited for robot control and strengthens the coupling between perception, language, and action prediction.

Stage 3: Task-specific adaptation. For each deployment task, we fine-tune the policy on a small set of task-specific demonstrations, adapting it to the target embodiment, camera viewpoints, object configurations, and success criteria.

### V-C High-Level Policy Training

The proposal, value, and reflective models are independently fine-tuned from
the same robot-pretrained VLM checkpoint initialized from
Qwen3.5-9B [35]. The world model is initialized from
Step1X-Edit [23] and trained separately. We describe the specific supervision for each model below.

Proposal model. The proposal model generates candidate subtasks while maintaining and updating
execution memory. We train it on both aligned execution histories and automatically
perturbed histories that lag behind, run ahead of, or misrepresent progress
after a failure. The corrected targets teach the model to reconcile memory with
visual evidence before generating the next subtask. The perturbation types,
recovery targets, and sampling mixture are detailed in
Appendix C-E.

World model. The world model is trained to predict the visual
outcome of a candidate subtask at completion. Each training example pairs the
head-camera RGB observations at the beginning and end of an annotated subtask
segment, conditioned on the corresponding subtask instruction. These
subtask-aligned transitions are drawn from real trajectories.

Starting from real observations, we construct multi-step simulated rollouts by recursively alternating subtask proposal with $P$ and visual outcome prediction with $\mathcal{W}$, using each predicted observation to condition the next proposal step.
These simulated rollouts form an offline dataset used to train the value and reflective models described below.

Value model. We formulate value prediction as a multiple-choice
VQA task. Conditioned on the global instruction, candidate subtask, and its
imagined terminal image, the model predicts one of five ordinal quality levels,
from clearly wrong to clearly correct, mapped to scalar values in
$[0.05,0.95]$.
Training pairs each sampled state with a candidate subtask from offline
rollouts generated by repeatedly applying the proposal and world models. The
candidates are graded against the ground-truth next step, aligning training
with the proposals that the value model scores during search.

Reflective model. We train the reflective model with the same
observation-aligned context and retained-branch summaries used at inference.
The candidate plans come from the same offline rollouts together with their
predicted future states and scores. Each set is paired with the ground-truth
next subtask as an autoregressive target. This supervision teaches the model
to ground its generation in visual evidence and repair flawed candidate plans
rather than simply copy the first proposal.

## VI Experiments

Our evaluation separates high-level decision quality, long-horizon system
performance, and short-horizon execution across embodiments. At the high level,
we evaluate next-subtask prediction, test-time computation, and the
contribution of execution memory. We then evaluate the hierarchical system on
four long-horizon tasks, study test-time computation on Book Organization task, and
evaluate the VLA through direct execution on the shorter Collect Laundry and
Tidy Makeup Table tasks.

![Refer to caption](drafts/images/tau0vla-2608.16885/demo_new.png)

*Fig. 3: Representative physical-robot evaluation tasks.
(a) Clean Room requires collecting two dirty garments, placing them in a
laundry basket, hanging a handbag, handing a blanket to a person, and
disposing of table trash. (b) Prepare Ingredients requires retrieving a
tomato and an egg, then cracking and stirring the egg while returning the
tools. (c) Tomato and Egg Stir Fry requires chopping and transferring the
tomatoes, cooking and seasoning the ingredients, plating the dish, and
returning the cookware and utensils.
(d) Make Milk Tea requires adding toppings, pouring milk and tea, sealing the
cup, and inserting a straw. (e) Collect Laundry requires transferring a
T-shirt from the bedside table to a laundry basket. (f) Tidy Makeup Table
comprises three separately scored instruction-following groups that require
different object selections and action sequences from matched visual states.*

### VI-A Evaluation Setup

Robot platforms. 
We evaluate on AGIBOT G1 for the four primary long-horizon tasks, ARX AC One
for Book Organization and Collect Laundry, and a bimanual Franka Research 3
setup for Tidy Makeup Table.
All platforms provide multi-view RGB observations and proprioceptive state and
map their controls into the unified action space. Hardware and sensing
details are deferred to Appendix C-A.

Task suite. 
The evaluation covers six household tasks and one benchmark comprising three
separately evaluated instruction-following task groups. Clean Room, Prepare
Ingredients, Tomato and Egg Stir Fry, and Collect Laundry require mobile
manipulation. Make Milk Tea, Book Organization, and the three Tidy Makeup Table
groups require manipulation without base motion. We use Tidy Makeup Table for
instruction-following evaluation. Prepare Ingredients and Tomato and Egg Stir
Fry form a collaborative cooking workflow for the same tomato and egg dish. One
robot prepares the ingredients, and another completes the cooking stage. We
therefore evaluate and report the two stages as separate tasks.

- •

Clean Room (25 steps). The robot enters the bedroom, retrieves two dirty garments from the nightstand and bed, and places them in a laundry basket. It then hangs a handbag on a clothes rack, retrieves a blanket and hands it to a person, leaves the bedroom, and disposes of snack-bag trash from the coffee table. A typical successful rollout lasts approximately $8$ min.
- •

Prepare Ingredients (14 steps). The robot moves between the refrigerator and preparation table, opens the refrigerator, retrieves a tomato and an egg, and closes the refrigerator. It then places the ingredients in bowls, cracks the egg with an egg cracker, stirs the egg mixture, and returns both tools to a porcelain plate. A typical successful rollout lasts approximately $4$ min.
- •

Tomato and Egg Stir Fry (22 steps). The robot chops the tomatoes and transfers the tomatoes and prepared egg mixture to the induction cooker. It turns on the cooker, adds oil and the egg mixture, stir-fries the eggs, adds the tomatoes and salt, finishes stir-frying, plates the dish, returns the cookware and utensils, turns off the cooker, places the finished dish on the preparation table, and returns both arms to the home position. A typical successful rollout lasts approximately $10$ min.
- •

Make Milk Tea (13 steps). The robot places a cup at the preparation station, adds two toppings, pours milk and tea in sequence, seals the cup with a lid, and inserts a straw. A typical successful rollout lasts approximately $3$ min.
- •

Collect Laundry (5 steps). The robot searches for and moves to the drawer cabinet, picks up the dirty T-shirt from the bedside table with its left arm, closes the cabinet door with its right arm, searches for and moves to the laundry basket, and places the T-shirt in the basket with its left arm. A typical successful rollout lasts approximately $1$ min.
- •

Tidy Makeup Table (three task groups). This benchmark pairs matched visual states with instructions that specify different target objects, active arms, action sequences, and destinations. We score three groups independently. In Cotton Pad (2 steps), the left arm picks up the cotton pad and places it in its designated compartment. In Eyelash Curler (2 steps), the left arm picks up the eyelash curler and places it in its designated compartment. In Makeup Puff (4 steps), both arms open the drawer, the left arm picks up the makeup puff from the tabletop and places it inside, and both arms close the drawer. Executing all eight steps across the three task groups takes approximately $30$ s in total.
- •

Book Organization (3 steps). The robot uses its two arms to reorder four books of different heights across four fixed slots, swapping any pair of books at each step until they are arranged from tallest to shortest. A typical successful rollout lasts approximately $1$ min.
We evaluate two initial-state regimes while keeping the task objective and action space fixed: in-domain initial arrangements appear in the training data, whereas out-of-domain (OOD) initial arrangements are not observed during training.
The OOD setting therefore evaluates the high-level policy’s ability to predict appropriate swaps from observations induced by unseen initial configurations.

Evaluation protocols. 
We use complementary protocols for high-level decision making and
physical-robot execution.

High-level policy evaluation. We isolate high-level decision making on annotated subtask-boundary samples and
report next-subtask prediction accuracy and inference time. All methods within
each comparison receive the same inputs and output format. Evaluation-set
construction, variant-specific inputs, and judging details are provided in
Appendix C-H.

Physical-robot evaluation. We report success rate (SR) and progress using the same
task-specific terminal conditions for all methods. SR measures the fraction
of successful trials. A trial is successful only when every required milestone
is completed. Skipping a required milestone makes the trial unsuccessful.
Progress measures how much of the task is completed using annotated
milestones. Each milestone receives full, half, or zero credit based on
completion quality, and dependent milestones count only after their required
earlier steps are completed. Exact scoring rules, trial durations, reset
procedures, and termination conditions are provided in
Appendix C-C.

### VI-B Overall System Evaluation

Table I reports the four long-horizon tasks: Clean Room,
Prepare Ingredients, Tomato and Egg Stir Fry, and Make Milk Tea. These tasks require
navigation, object search, manipulation, state tracking, and recovery across
multiple stages. For the controlled comparison of our two variants, the
observation and action interfaces and the low-level policy are fixed.
Standalone $\tau_{0}$-VLA conditions the low-level policy on the full task
instruction. The hierarchical system instead supplies bounded subtasks from
the high-level policy. Both variants in this comparison run without beam
search, which is evaluated separately in the test-time computation experiments.

*TABLE I: Long-horizon task performance. Each
method–task setting uses 10 independently collected physical-robot trials. SR
reports successful trials as $x/10$, and Progress is the normalized
milestone-completion score. Avg. is the unweighted mean of the four task-level rates.
The first four rows use direct execution. The final row uses the Hierarchical
System with Plan Once and no beam search.*

| Method | Clean Room | Prepare Ingredients | Tomato and Egg Stir Fry | Make Milk Tea | Avg. |
|---|---|---|---|---|---|
|  | SR $\uparrow$ | Progress $\uparrow$ | SR $\uparrow$ | Progress $\uparrow$ | SR $\uparrow$ | Progress $\uparrow$ | SR $\uparrow$ | Progress $\uparrow$ | SR $\uparrow$ | Progress $\uparrow$ |
| $\mathrm{GR00T}$ N1.7 [26] | 0/10 | 59.80% | 1/10 | 68.57% | 0/10 | 24.32% | 0/10 | 28.46% | 2.50% | 45.29% |
| LingBot-VLA [40] | 0/10 | 66.60% | 0/10 | 35.00% | 0/10 | 12.27% | 0/10 | 63.85% | 0.00% | 44.43% |
| $\pi_{0.5}$ [2] | 4/10 | 86.20% | 2/10 | 73.93% | 0/10 | 49.77% | 3/10 | 82.31% | 22.50% | 73.05% |
| $\tau_{0}$-VLA | 4/10 | 92.80% | 2/10 | 66.43% | 0/10 | 65.00% | 5/10 | 96.15% | 27.50% | 80.10% |
| $\tau_{0}$-VLA (Hierarchical System, Plan Once) | 5/10 | 94.80% | 4/10 | 82.86% | 4/10 | 81.82% | 5/10 | 91.92% | 45.00% | 87.85% |

Clean Room. 
Performance on Clean Room benefits clearly from explicit execution memory. The
hierarchical system retains progress across room transitions, whereas failures
without this memory concentrate on handbag hanging and the later room-tidying
stages.

Prepare Ingredients. 
Most failures in Prepare Ingredients trials arise during egg pickup, cracking,
or stirring. Because these actions are prerequisites for later preparation
stages, an early error blocks substantial downstream progress. Tracking
completed stages is therefore particularly valuable on this task.

Tomato and Egg Stir Fry. 
The decisive bottleneck in Tomato and Egg Stir Fry is seasoning. Adding salt
causes little visible change, so the current observation alone cannot reliably
reveal whether this step has already been completed. Direct-execution policies
therefore tend to add salt repeatedly or skip it entirely, either of which
violates the success criterion. The hierarchical system resolves this ambiguity
by recording seasoning progress explicitly.

Make Milk Tea. 
Make Milk Tea presents a complementary regime in which both $\tau_{0}$-VLA
variants already complete most of the sequence reliably. Both achieve an SR of
5/10 and more than 91% progress, outperforming the external baselines.
Their remaining failures occur during lid attachment or straw insertion, after
the preceding preparation stages have been completed. This pattern identifies
final contact-rich manipulation as the main remaining bottleneck. When TTC is
enabled, the hierarchical policy further improves to an SR of 7/10 and 95.38%
progress, as reported in Table III.

### VI-C Evaluation Across Embodiments

We evaluate the low-level policy on two additional embodiments using Collect
Laundry and Tidy Makeup Table. Because these
tasks contain only two to five steps, each method executes the full task
instruction without task decomposition, execution memory, or test-time search.
This setting evaluates low-level control and language-conditioned manipulation
across embodiments, separately from the high-level policy used in the
long-horizon tasks.
Table II reports the results.

*TABLE II: Direct-execution performance across embodiments. Tidy Makeup
Table comprises three independently scored instruction-following groups. All
methods execute the full task instruction without a high-level policy. Each
task or task group is evaluated over 10 trials. SR reports successful trials
as $x/10$, and Progress is the normalized milestone-completion score.*

| Method | Collect Laundry | Tidy Makeup Table |
|---|---|---|
|  | T-shirt | Cotton Pad | Eyelash Curler | Makeup Puff |
|  | SR $\uparrow$ | Progress $\uparrow$ | SR $\uparrow$ | Progress $\uparrow$ | SR $\uparrow$ | Progress $\uparrow$ | SR $\uparrow$ | Progress $\uparrow$ |
| $\mathrm{GR00T}$ N1.7 [26] | 4/10 | 76.00% | 10/10 | 87.50% | 8/10 | 77.50% | 7/10 | 52.50% |
| LingBot-VLA [40] | 2/10 | 35.00% | 9/10 | 67.50% | 3/10 | 22.50% | 3/10 | 33.75% |
| $\pi_{0.5}$ [2] | 9/10 | 88.00% | 9/10 | 85.00% | 8/10 | 85.00% | 7/10 | 73.75% |
| $\tau_{0}$-VLA | 10/10 | 97.00% | 10/10 | 95.00% | 9/10 | 92.50% | 10/10 | 95.00% |

Collect Laundry. 
Collect Laundry evaluates mobile manipulation by the low-level policy. The
observed failures concentrate on T-shirt grasping and navigation
around the bed: LingBot-VLA sometimes fails to lift the T-shirt,
while GR00T collides with the foot of the bed and terminates the rollout.

Tidy Makeup Table. 
The Tidy Makeup Table benchmark evaluates instruction following under matched
visual states. The language instruction changes the target object,
active arm, action order, and destination, so the observation alone does not
determine the correct action sequence. Cotton Pad and Eyelash Curler emphasize
instruction-conditioned object selection and placement, whereas Makeup Puff
adds bimanual drawer manipulation. GR00T occasionally pauses during drawer
motion or selects the makeup puff instead of the eyelash curler. LingBot-VLA
generally follows the specified arm but tends to select visually nearby objects
rather than the instructed target. $\pi_{0.5}$ sometimes releases and re-grasps
the makeup puff, closes the drawer less smoothly, or omits the cotton pad.
These failures distinguish language-grounding errors from low-level execution
inefficiencies: selecting or omitting the wrong object changes task completion,
whereas re-grasping and hesitant drawer motion reduce execution efficiency even
when the intended final state is reached.

### VI-D Test-Time Computation Experiments

We conduct several experiments to quantify the gains in subtask-prediction accuracy achieved by test-time computation (TTC) and assess whether these gains lead to higher success rates in closed-loop real-robot evaluation.
Our TTC experiments cover the following tasks: Make Milk Tea, Book Organization, and Clean Room.
We evaluate Book Organization both in-domain and out-of-domain (OOD) initial book arrangements: the former appears in the training data, whereas the latter does not.
The OOD setting evaluates the high-level policy’s performance when planning from unseen initial observations.
We first measure next-subtask prediction accuracy in an open-loop setting, then evaluate TTC in closed-loop real-robot execution, and finally examine the relationship between computational cost and prediction accuracy.

*Fig. 4: Next-subtask prediction accuracy under different high-level inference methods.
We compare Plan Once, Best-of-$N$, and TTC (Ours) across four evaluation settings:
Make Milk Tea, Book Organization (In-Domain), Book Organization (OOD), and
Clean Room.*

*TABLE III: Closed-loop physical-robot performance with test-time computation.
Each entry uses 10 independently collected trials. Book Organization uses
shuffled initial arrangements and is reported without the in-domain and OOD
split used in the open-loop evaluation. SR denotes task success rate, and
Progress is the normalized milestone-completion score.*

| Method | Make Milk Tea | Book Organization | Clean Room |
|---|---|---|---|
| SR $\uparrow$ | Progress $\uparrow$ | SR $\uparrow$ | Progress $\uparrow$ | SR $\uparrow$ | Progress $\uparrow$ |
| Plan Once | 5/10 | 91.92% | 6/10 | 66.67% | 5/10 | 94.80% |
| TTC | 7/10 | 95.38% | 9/10 | 93.33% | 7/10 | 97.60% |

*Fig. 5: Relationship between computational cost and subtask-prediction accuracy.
The accuracy increases with additional computation for the Make Milk Tea and Book Organization tasks, shown in panels (a) and (b), respectively.
Each point represents an experimental result obtained at a different computational cost. The orange dashed curves show saturation fits to these results, and the gray dashed lines indicate the Plan Once baselines.*

Open-loop Subtask Prediction. 
Following the high-level policy evaluation protocol above, we evaluate next-subtask
prediction at different stages of each task. We compare TTC against two
baselines. Plan Once makes one high-level prediction at each decision
point without test-time search. It remains a variant of the high-level policy
and is distinct from direct execution, which bypasses the high-level policy.
Best-of-$N$ samples $N$ candidates, predicts the next
observation for each candidate using the same world model as TTC, and selects
the candidate assigned the highest score by the same value model.

As shown in Figure 4, TTC achieves the highest accuracy across all four evaluation settings, demonstrating its effectiveness in improving next-subtask prediction.
Take the OOD Book Organization setting as an example, where TTC achieves $74.0\%$ accuracy, compared with $50.0\%$ for Plan Once and $57.5\%$ for Best-of-$N$.
In the OOD setting, the initial book arrangements are absent from the training data, placing the corresponding observations outside the high-level policy’s training distribution.
Under this distribution shift, directly predicting the next subtask with the fine-tuned VLM is more prone to error.
Instead, TTC recursively expands candidate branches by using the world model to
predict future observations and the value model to score the resulting imagined
branches. The reflective model then uses the retained branches as context to
generate the current subtask.
Unlike direct VLM prediction based solely on patterns learned during fine-tuning, TTC evaluates candidate consequences at decision time, resulting in improved next-subtask prediction accuracy.
Although Best-of-$N$ evaluates sampled candidates using the same world and value models, it performs only one-step selection without TTC’s multi-step expansion and reflective commitment.
Consequently, its accuracy gains relative to Plan Once are consistently smaller than those achieved by TTC across the evaluated settings.

Closed-loop Real-robot Evaluation. 
We further assess whether TTC leads to higher task success rates in closed-loop real-robot evaluation.
We deploy the high-level policy with and without TTC while keeping the low-level policy fixed.
As reported in Table III, TTC improves both task progress and success rate across all evaluated tasks.
This benefit is particularly important for tasks such as Book Organization, which do not admit a fixed execution plan.
These results show that the gains in next-subtask prediction accuracy achieved by TTC lead to more effective and efficient closed-loop task execution.

Computational Cost vs. Subtask-Prediction Accuracy. 
Finally, we conduct an analysis of the relationship between test-time computational cost and subtask-prediction accuracy.
Figure 5 reports this relationship for the Make Milk Tea and Book Organization tasks.
Each point represents an experimental result with a different computational cost, and the orange dashed curves provide approximate saturation fits to the observations.
Specifically, we fit a saturating exponential function.
The parameters are estimated by least squares using all experimental observations.
Accuracy rises rapidly when computation is increased from a small budget, showing that additional search and reflection substantially improve subtask prediction in the low-compute regime.
The marginal gain then gradually decreases and the fitted curves approach a plateau.
This trend indicates that TTC provides a favorable compute–accuracy trade-off at moderate budgets, while its benefit eventually saturates as additional computation is allocated.

## VII Conclusion

We present $\tau_{0}$-VLA, a hierarchical VLA system for long-horizon manipulation that combines a memory-augmented high-level policy with a low-level policy. The high-level policy handles routine decisions efficiently and allocates world-model-guided computation when a decision benefits from explicit consequence evaluation. The low-level policy is trained from heterogeneous robot and vision-language data for deployable whole-body mobile manipulation. Together, the two levels provide a practical way to allocate reasoning over long task horizons while retaining stable real-robot execution.

## References

- [1]
S. Belkhale, T. Ding, T. Xiao, P. Sermanet, Q. Vuong, J. Tompson, Y. Chebotar, D. Dwibedi, and D. Sadigh (2024)

RT-H: action hierarchies using language.

In Proceedings of Robotics: Science and Systems,

External Links: [Document](https://dx.doi.org/10.15607/RSS.2024.XX.049),
[Link](https://www.roboticsproceedings.org/rss20/p049.html)

Cited by: §I,
§II-B.
- [2]
K. Black, N. Brown, J. Darpinian, K. Dhabalia, D. Driess, A. Esmail, M. R. Equi, C. Finn, N. Fusai, M. Y. Galliker, et al. (2025)

$\pi_{0.5}$: A vision-language-action model with open-world generalization.

In Proceedings of The 9th Conference on Robot Learning,

Proceedings of Machine Learning Research, Vol. 305, pp. 17–40.

External Links: [Link](https://proceedings.mlr.press/v305/black25a.html)

Cited by: §I,
§II-A,
§II-B,
TABLE I,
TABLE II.
- [3]
K. Black, N. Brown, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, L. Groom, K. Hausman, B. Ichter, et al. (2024)

$\pi_{0}$: A vision-language-action flow model for general robot control.

arXiv preprint arXiv:2410.24164.

Cited by: §I,
§II-A,
§IV-C.
- [4]
K. Black, M. Y. Galliker, and S. Levine (2025)

Real-time execution of action chunking flow policies.

In Advances in Neural Information Processing Systems,

Vol. 38.

Cited by: §II-A.
- [5]
K. Black, M. Nakamoto, P. Atreya, H. R. Walke, C. Finn, A. Kumar, and S. Levine (2024)

Zero-shot robotic manipulation with pre-trained image-editing diffusion models.

In The Twelfth International Conference on Learning Representations,

External Links: [Link](https://openreview.net/forum?id=c0chJTSbci)

Cited by: §I,
§II-C.
- [6]
A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, J. Dabis, C. Finn, K. Gopalakrishnan, K. Hausman, A. Herzog, J. Hsu, et al. (2023)

RT-1: robotics transformer for real-world control at scale.

In Proceedings of Robotics: Science and Systems,

External Links: [Document](https://dx.doi.org/10.15607/RSS.2023.XIX.025),
[Link](https://roboticsproceedings.org/rss19/p025.html)

Cited by: §I,
§II-A.
- [7]
J. Cen, C. Yu, H. Yuan, Y. Jiang, S. Huang, J. Guo, X. Li, Y. Song, H. Luo, F. Wang, et al. (2025)

WorldVLA: towards autoregressive action world model.

arXiv preprint arXiv:2506.21539.

Cited by: §II-C.
- [8]
C. Cheang, G. Chen, Y. Jing, T. Kong, H. Li, Y. Li, Y. Liu, H. Wu, J. Xu, Y. Yang, et al. (2024)

GR-2: a generative video-language-action model with web-scale knowledge for robot manipulation.

arXiv preprint arXiv:2410.06158.

Cited by: §II-C.
- [9]
M. Dai, L. Liu, Y. Bai, Y. Liu, Z. Wang, R. Su, C. Chen, L. Lin, and X. Wu (2025)

RoVer: robot reward model as test-time verifier for vision-language-action model.

arXiv preprint arXiv:2510.10975.

Cited by: §I,
§II-C.
- [10]
DeepSeek-AI (2025)

DeepSeek-r1: incentivizing reasoning capability in LLMs via reinforcement learning.

arXiv preprint arXiv:2501.12948.

Cited by: §I.
- [11]
D. Driess, J. T. Springenberg, B. Ichter, L. Yu, A. Li-Bell, K. Pertsch, A. Z. Ren, H. Walke, Q. Vuong, L. X. Shi, and S. Levine (2025)

Knowledge insulating vision-language-action models: train fast, run fast, generalize better.

In Advances in Neural Information Processing Systems,

Vol. 38.

External Links: [Link](https://openreview.net/forum?id=cb0xbZ3APM)

Cited by: §II-A,
§V-B.
- [12]
Y. Du, M. Yang, P. Florence, F. Xia, A. Wahid, B. Ichter, P. Sermanet, T. Yu, P. Abbeel, J. B. Tenenbaum, L. P. Kaelbling, A. Zeng, and J. Tompson (2024)

Video language planning.

In The Twelfth International Conference on Learning Representations,

External Links: [Link](https://openreview.net/forum?id=9pKtcJcMP3)

Cited by: §I,
§II-C.
- [13]
Y. Du, S. Yang, B. Dai, H. Dai, O. Nachum, J. Tenenbaum, D. Schuurmans, and P. Abbeel (2023)

Learning universal policies via text-guided video generation.

Advances in Neural Information Processing Systems 36, pp. 9156–9172.

Cited by: §I,
§II-C.
- [14]
Y. Feng, J. Han, Z. Yang, X. Yue, S. Levine, and J. Luo (2025)

Reflective planning: vision-language models for multi-stage long-horizon robotic manipulation.

In Proceedings of The 9th Conference on Robot Learning,  J. Lim, S. Song, and H. Park (Eds.),

Proceedings of Machine Learning Research, Vol. 305, pp. 2038–2062.

External Links: [Link](https://proceedings.mlr.press/v305/feng25b.html)

Cited by: §I,
§II-C.
- [15]
Figure AI (2025)

Helix: a vision-language-action model for generalist humanoid control.

Note: [https://www.figure.ai/news/helix](https://www.figure.ai/news/helix)

Cited by: §I,
§II-B.
- [16]
Gemini Robotics Team, A. Abdolmaleki, S. Abeyruwan, J. Ainslie, et al. (2025)

Gemini Robotics 1.5: pushing the frontier of generalist robots with advanced embodied reasoning, thinking, and motion transfer.

External Links: 2510.03342

Cited by: §I,
§II-A,
§II-B.
- [17]
W. Guo, G. Lu, H. Deng, Z. Wu, Y. Tang, and Z. Wang (2026)

VLA-Reasoner: empowering vision-language-action models with reasoning via online Monte Carlo tree search.

In 2026 IEEE International Conference on Robotics and Automation (ICRA),

Cited by: §I,
§II-C.
- [18]
D. Ha and J. Schmidhuber (2018)

World models.

arXiv preprint arXiv:1803.10122.

Cited by: §II-C.
- [19]
D. Hafner, J. Pasukonis, J. Ba, and T. P. Lillicrap (2025)

Mastering diverse control tasks through world models.

Nature 640 (8059), pp. 647–653.

External Links: [Document](https://dx.doi.org/10.1038/s41586-025-08744-2),
[Link](https://doi.org/10.1038/s41586-025-08744-2)

Cited by: §II-C.
- [20]
B. Ichter, A. Brohan, Y. Chebotar, C. Finn, K. Hausman, A. Herzog, D. Ho, J. Ibarz, A. Irpan, E. Jang, et al. (2023)

Do as i can, not as i say: grounding language in robotic affordances.

In Conference on Robot Learning,

pp. 287–318.

Cited by: §I,
§II-B.
- [21]
M. J. Kim, K. Pertsch, S. Karamcheti, T. Xiao, A. Balakrishna, S. Nair, R. Rafailov, E. P. Foster, P. R. Sanketi, Q. Vuong, T. Kollar, B. Burchfiel, R. Tedrake, D. Sadigh, S. Levine, P. Liang, and C. Finn (2025)

OpenVLA: an open-source vision-language-action model.

In Proceedings of The 8th Conference on Robot Learning,

Proceedings of Machine Learning Research, Vol. 270, pp. 2679–2713.

External Links: [Link](https://proceedings.mlr.press/v270/kim25c.html)

Cited by: §I,
§II-A.
- [22]
J. Kwok, C. Agia, R. Sinha, M. Foutter, S. Li, I. Stoica, A. Mirhoseini, and M. Pavone (2025)

RoboMonkey: scaling test-time sampling and verification for vision-language-action models.

In Proceedings of The 9th Conference on Robot Learning,  J. Lim, S. Song, and H. Park (Eds.),

Proceedings of Machine Learning Research, Vol. 305, pp. 3200–3217.

External Links: [Link](https://proceedings.mlr.press/v305/kwok25a.html)

Cited by: §I,
§II-C.
- [23]
S. Liu, Y. Han, P. Xing, F. Yin, R. Wang, W. Cheng, J. Liao, Y. Wang, H. Fu, C. Han, et al. (2025)

Step1X-Edit: a practical framework for general image editing.

arXiv preprint arXiv:2504.17761.

Cited by: §V-C.
- [24]
S. Liu, L. Wu, B. Li, H. Tan, H. Chen, Z. Wang, K. Xu, H. Su, and J. Zhu (2025)

RDT-1B: a diffusion foundation model for bimanual manipulation.

In The Thirteenth International Conference on Learning Representations,

External Links: [Link](https://openreview.net/forum?id=yAzN4tz7oI)

Cited by: §II-A.
- [25]
Z. Liu, X. Ning, Z. Hu, X. Xie, W. Li, Z. Tang, C. Wang, Z. Yang, H. Wang, Y. Liu, and Z. Pu (2026)

Goal2Skill: long-horizon manipulation with adaptive planning and reflection.

arXiv preprint arXiv:2604.13942.

Cited by: §II-B.
- [26]
E. Llontop and K. Vadrevu (2026)

NVIDIA Isaac GR00T N1.7: open reasoning vla model for humanoid robots.

Note: [https://huggingface.co/blog/nvidia/gr00t-n1-7](https://huggingface.co/blog/nvidia/gr00t-n1-7)NVIDIA community article, published April 17, 2026

Cited by: §I,
§II-A,
§II-B,
TABLE I,
TABLE II.
- [27]
Q. Long, Y. Wang, J. Song, J. Zhang, P. Li, W. Wang, Y. Wang, H. Li, S. Xie, G. Yao, H. Zhang, X. Wang, Z. Wang, X. Lan, H. Liu, and X. Li (2026)

Scaling world model for hierarchical manipulation policies.

arXiv preprint arXiv:2602.10983.

Cited by: §I,
§II-B.
- [28]
H. Luo, Y. Wang, W. Zhang, S. Zheng, Z. Xi, C. Xu, H. Xu, H. Yuan, C. Zhang, Y. Wang, Y. Feng, and Z. Lu (2026)

Being-H0.5: scaling human-centric robot learning for cross-embodiment generalization.

arXiv preprint arXiv:2601.12993.

Cited by: §II-A.
- [29]
M. Nakamoto, O. Mees, A. Kumar, and S. Levine (2025)

Steering your generalists: improving robotic foundation models via value guidance.

In Proceedings of The 8th Conference on Robot Learning,  P. Agrawal, O. Kroemer, and W. Burgard (Eds.),

Proceedings of Machine Learning Research, Vol. 270, pp. 4996–5013.

External Links: [Link](https://proceedings.mlr.press/v270/nakamoto25a.html)

Cited by: §I,
§II-C.
- [30]
Octo Model Team, D. Ghosh, H. R. Walke, K. Pertsch, K. Black, O. Mees, S. Dasari, J. Hejna, T. Kreiman, C. Xu, J. Luo, Y. L. Tan, L. Y. Chen, Q. Vuong, T. Xiao, P. R. Sanketi, D. Sadigh, C. Finn, and S. Levine (2024)

Octo: an open-source generalist robot policy.

In Proceedings of Robotics: Science and Systems,

External Links: [Document](https://dx.doi.org/10.15607/RSS.2024.XX.090),
[Link](https://www.roboticsproceedings.org/rss20/p090.html)

Cited by: §I,
§II-A.
- [31]
OpenAI (2024)

OpenAI o1 system card.

arXiv preprint arXiv:2412.16720.

Cited by: §I.
- [32]
J. Park, J. Yoon, B. Jeon, J. Park, J. Shin, N. Cho, K. Lee, S. Yun, and S. Choi (2025)

Hierarchical vision language action model using success and failure demonstrations.

arXiv preprint arXiv:2512.03913.

Cited by: §I,
§II-C.
- [33]
K. Pertsch, K. Stachowicz, B. Ichter, D. Driess, S. Nair, Q. Vuong, O. Mees, C. Finn, and S. Levine (2025)

FAST: efficient action tokenization for vision-language-action models.

In Proceedings of Robotics: Science and Systems,

External Links: [Document](https://dx.doi.org/10.15607/RSS.2025.XXI.012),
[Link](https://www.roboticsproceedings.org/rss21/p012.html)

Cited by: §II-A.
- [34]
Physical Intelligence, B. Ai, A. Amin, R. Aniceto, A. Balakrishna, G. Balke, K. Black, et al. (2026)

$\pi_{0.7}$: A steerable generalist robotic foundation model with emergent capabilities.

External Links: 2604.15483

Cited by: §I,
§II-B.
- [35]
Qwen Team (2026)

Qwen3.5: towards native multimodal agents.

Note: [https://qwen.ai/blog?id=qwen3.5](https://qwen.ai/blog?id=qwen3.5)Qwen3.5 model family

Cited by: §V-B,
§V-C.
- [36]
L. X. Shi, B. Ichter, M. R. Equi, L. Ke, K. Pertsch, Q. Vuong, J. Tanner, A. Walling, H. Wang, N. Fusai, A. Li-Bell, D. Driess, L. Groom, S. Levine, and C. Finn (2025)

Hi Robot: open-ended instruction following with hierarchical vision-language-action models.

In Proceedings of the 42nd International Conference on Machine Learning,

Proceedings of Machine Learning Research, Vol. 267, pp. 54919–54933.

External Links: [Link](https://proceedings.mlr.press/v267/shi25d.html)

Cited by: §I,
§II-B.
- [37]
M. Shukor, D. Aubakirova, F. Capuano, P. Kooijmans, S. Palma, A. Zouitine, M. Aractingi, C. Pascal, M. Russi, A. Marafioti, et al. (2025)

SmolVLA: a vision-language-action model for affordable and efficient robotics.

arXiv preprint arXiv:2506.01844.

Cited by: §II-A.
- [38]
A. Sridhar, J. Pan, S. Sharma, and C. Finn (2026)

Scaling up memory for robot control via experience retrieval.

In The Fourteenth International Conference on Learning Representations,

External Links: [Link](https://openreview.net/forum?id=1dH4ARGdwD)

Cited by: §II-B.
- [39]
M. Torne, K. Pertsch, H. Walke, K. Vedder, S. Nair, B. Ichter, A. Z. Ren, H. Wang, J. Tang, K. Stachowicz, K. Dhabalia, M. Equi, Q. Vuong, J. T. Springenberg, S. Levine, C. Finn, and D. Driess (2026)

MEM: multi-scale embodied memory for vision language action models.

arXiv preprint arXiv:2603.03596.

External Links: [Link](https://arxiv.org/abs/2603.03596)

Cited by: §II-B.
- [40]
W. Wu, F. Lu, Y. Wang, S. Yang, S. Liu, F. Wang, Q. Zhu, H. Sun, Y. Wang, S. Ma, Y. Ren, K. Zhang, H. Yu, J. Zhao, S. Zhou, Z. Qiu, H. Xiong, Z. Wang, Z. Wang, R. Cheng, Y. Li, Y. Huang, X. Zhu, Y. Shen, and K. Zheng (2026)

A pragmatic vla foundation model.

External Links: 2601.18692

Cited by: §II-A,
TABLE I,
TABLE II.
- [41]
Y. Wu, T. Tian, G. Swamy, and A. Bajcsy (2025)

From foresight to forethought: VLM-in-the-loop policy steering via latent alignment.

In Proceedings of Robotics: Science and Systems,

External Links: [Document](https://dx.doi.org/10.15607/RSS.2025.XXI.076)

Cited by: §I,
§II-C.
- [42]
Y. Yang, S. Gao, Q. Bu, L. Chen, and D. N. Metaxas (2026)

Seeing farther and smarter: value-guided multi-path reflection for VLM policy optimization.

In 2026 IEEE International Conference on Robotics and Automation (ICRA),

Cited by: §I,
§II-C.
- [43]
Y. Yang, Z. Liu, S. Kou, Y. Chen, Y. Hu, J. Zhou, B. Zhao, Z. Wei, X. Xia, X. Li, P. Liu, and Z. Deng (2026)

World-language-action model for unified world modeling, language reasoning, and action synthesis.

arXiv preprint arXiv:2606.05979.

Cited by: §I,
§II-C.
- [44]
Z. Zhang, W. Luo, H. Wang, Y. Sheng, Y. Wang, H. Guo, H. Ren, X. Du, Y. Che, T. Cao, L. Yuan, and Y. Yu (2026)

Anticipation-VLA: solving long-horizon embodied tasks via anticipation-based subgoal generation.

arXiv preprint arXiv:2605.01772.

Cited by: §I,
§II-B.
- [45]
J. Zheng, J. Li, Z. Wang, D. Liu, X. Kang, Y. Feng, Y. Zheng, J. Zou, Y. Chen, J. Zeng, Y. Zhang, J. Pang, J. Liu, T. Wang, and X. Zhan (2025)

X-VLA: soft-prompted transformer as scalable cross-embodiment vision-language-action model.

External Links: 2510.10274

Cited by: §II-A.
- [46]
B. Zitkovich, T. Yu, S. Xu, P. Xu, T. Xiao, F. Xia, J. Wu, P. Wohlhart, S. Welker, A. Wahid, et al. (2023)

RT-2: vision-language-action models transfer web knowledge to robotic control.

In Conference on Robot Learning,

pp. 2165–2183.

Cited by: §I,
§II-A.

## Appendix A Author Contributions

High-Level Policy Training and Research:
Xiaowei Cai, Jingxiao Chen, Xinchen Li, Yifan Li, Yi Liu, Jianlan Luo,
Junwen Miao, Ruiqi Ni, Buqing Nie, Jiaxu Wang, Dafeng Wei, Pengwei Xie,
Pu Yang, Hangjian Ye, Xiangyu Yue, and Jinyu Zhang.

Low-Level Policy Training and Research:
Xiaowei Cai, Bingao Chen, Jingxiao Chen, Jingshun Huang, Yi Liu,
Jianlan Luo, Junwen Miao, Dafeng Wei, Dongming Wu, Hangjian Ye,
Jinyu Zhang, and Pengfei Zhou.

Training Infra:
Peiqi Wang, Sen Wang, and Qinglin Zhang.

Robot Infra:
Tengyu Hou, Dong Li, Zhongyuan Liu, and Xiaoyan Wang.

Writing and Illustration:
Xiaowei Cai, Yunuo Cai, Jingxiao Chen, Zhi Chen, Yi Liu, Jianlan Luo, Junwen Miao,
Buqing Nie, Dafeng Wei, Dongming Wu, and Jinyu Zhang.

Data Collection and Deployment:
Zhi Chen, Siyuan Feng, Han Jiang, Runkun Ju, Shaowei Li, Mingjie Pan, Xinlin Ren,
Jianheng Song, and Yue Zhou.

Data Infra:
Mingxiang Li, Xueyong Zhao and Yue Zhou.

First Authors:
Jinyu Zhang and Yi Liu.

Corresponding Author:
Jianlan Luo.

## Appendix B Acknowledgments

We gratefully acknowledge Hongwen Cai, Haoran Chen, Yebin Chen, Zhibo Cui, Zihao Fan, Tao Gao, Yanan Gong, Minghao Gui, Bin He,
Xinyan Hou, Lei Hua, Shuaishuai Li, Tong Li, Qingxiang Liu,
Xiang Lü, Zongbao Song, Yuna Sun, Zhe Sun, Zhifeng Tang,
Fuming Tian, Jiahao Wan, Hanying Wang, Hao Wang, Sijing Wang,
Tao Wang, Le Yuan, Guozhu Zhang, Hao Zhang, Tianbao Zhang,
Xinbo Zhi, Mingyan Zhou, and Rongxing Zhu for their valuable
contributions to data creation, annotation, and related data
operations. We also thank Yifei Chen for assistance with video
recording and editing.

## Appendix C Additional Details

This appendix provides additional details on the robot platforms,
physical-robot evaluation protocol, low-level training data, and high-level
policy data construction.

### C-A Robot Platform Details

AGIBOT G1.
AGIBOT G1 is a wheeled humanoid with an omnidirectional
four-wheel-steering base, dual 7-DoF arms, and modular end-effectors. Our setup
uses parallel-jaw grippers. Its sensing suite includes head-mounted RGB-D and
fisheye cameras and a wrist-mounted camera on each arm.

ARX AC One.
ARX AC One is a bimanual platform with dual 6-DoF X5
arms and parallel-jaw grippers. We use wrist-mounted cameras together with a
fixed central camera.

Franka Research 3.
Our bimanual Franka Research 3 setup comprises two
7-DoF torque-controlled arms with custom 3D-printed grippers and three RGB
cameras. Across all three embodiments, the policy receives multi-view RGB
images and proprioceptive state, and the native robot commands are represented
in the unified 40-dimensional action layout defined in
Section IV-C.

### C-B Unified State and Action Representation

The canonical state vector $\mathbf{s}_{t}$ and each action vector
$\mathbf{a}_{t+j}$, for $j\in\{0,\ldots,H-1\}$, lie in
$\mathbb{R}^{40}$. The stacked action chunk therefore lies in
$\mathbb{R}^{H\times 40}$. State and action vectors use the slot ordering shown
in Table IV. State channels store the current values, and
the end-effector and joint action channels use the relative parameterization
defined below. The left and right arm blocks follow each embodiment’s native
kinematic joint order. A robot with fewer than eight joints per arm fills the
leading entries and masks the remainder.

*TABLE IV: Canonical 40-D state and action layout. Dimensions are
one-indexed. For a rotation matrix
$\mathbf{R}=[\mathbf{r}_{1},\mathbf{r}_{2},\mathbf{r}_{3}]$, we use
$\operatorname{Rot6D}(\mathbf{R})=[\mathbf{r}_{1}^{\top},\mathbf{r}_{2}^{\top}]^{\top}$.*

| Coordinates | Dimensions | State representation |
|---|---|---|
| Left EEF position | 1–3 | Cartesian position in meters |
| Left EEF orientation | 4–9 | $\operatorname{Rot6D}(\mathbf{R}^{L})$ |
| Right EEF position | 10–12 | Cartesian position in meters |
| Right EEF orientation | 13–18 | $\operatorname{Rot6D}(\mathbf{R}^{R})$ |
| Left gripper | 19 | native opening coordinate |
| Right gripper | 20 | native opening coordinate |
| Waist | 21–22 | two native coordinates |
| Planar base velocity | 23–24 | two native coordinates |
| Left arm joints | 25–32 | $q^{L}_{1},\ldots,q^{L}_{8}$ in radians |
| Right arm joints | 33–40 | $q^{R}_{1},\ldots,q^{R}_{8}$ in radians |

Relative action encoding.
Both end-effector and joint actions are expressed relative to the current
state. End-effector actions use position and rotation deltas from the current
pose, with the position delta expressed in the current end-effector frame.
Joint actions use angular offsets from the current joint configuration.

Masks.
Separate state and action masks identify the valid dimensions for each
embodiment. They activate the applicable end-effector or joint slots, set
unavailable dimensions to zero, and exclude inactive action dimensions from
the training loss.

Control metadata.
We serialize $\eta$ together with the language command $c_{t}$ using the
following prompt. The robot type, control mode, and whole-body fields comprise
$\eta$, while the final field contains $c_{t}$:

You are controlling a robot.
Robot type: <embodiment>
Control mode: <eef/eef_wbc/joint>
Whole-body control: <enabled/disabled>
Task: <language command>

Masked flow-matching details.
Using the notation in Eq. (14), the stacked action chunk
$\mathbf{a}_{t:t+H-1}$ lies in $\mathbb{R}^{H\times d_{a}}$. Given the complete
noisy chunk and the policy conditioning inputs
$(o_{t},\mathbf{s}_{t},c_{t},\eta,\tau)$, the action expert jointly predicts
$\mathbf{v}_{\theta,t+j}^{\tau}\in\mathbb{R}^{d_{a}}$ for each action token. The
masked flow-matching objective is

|  | $$ \mathcal{L}_{\mathrm{FM}}=\mathbb{E}\!\left[\frac{1}{H\,\operatorname{tr}(\mathbf{M})}\sum_{j=0}^{H-1}\left\|\mathbf{M}\mathbf{v}_{\theta,t+j}^{\tau}-\mathbf{u}_{t+j}\right\|_{2}^{2}\right]. $$ |  | (15) |
|---|---|---|---|---|---|

The expectation is over training examples, Gaussian noise, and
$\tau=0.001+0.999x$ with $x\sim\operatorname{Beta}(1.5,1)$.
The implementation caps each normalized per-sample loss at $100$ before batch
averaging. At inference, we initialize the chunk from masked Gaussian noise and
integrate the learned field from $\tau=1$ to $\tau=0$. We apply
the action mask to the current noisy chunk before every velocity-field
evaluation and to the final chunk after integration. The reported policies use
an action horizon of $H=30$ and ten uniform Euler updates.

### C-C Physical-Robot Evaluation Protocol

Success rate.
We apply the same task-specific terminal conditions and milestone annotations
to every method. SR requires every required annotated subtask and the terminal
condition to be completed. A required subtask completed after one or more failed
autonomous attempts still satisfies the SR criterion, provided that no required
subtask is skipped and no task-specific prohibited action occurs.

Progress.
For each task, we construct a directed acyclic prerequisite graph
$G=(\mathcal{V},\mathcal{E})$ from its original fine-grained annotations. Each
index $i\in\mathcal{V}$ identifies one required milestone. A directed edge
$(i,j)\in\mathcal{E}$ means that milestone $i$ must reach its required task
state before milestone $j$ is eligible for credit. Let
$\operatorname{Anc}(i)$ denote all direct and indirect prerequisites of
milestone $i$.

For trial $r$, let $e_{r,i}=1$ when every milestone in
$\operatorname{Anc}(i)$ has reached its required task state before milestone
$i$, and let $e_{r,i}=0$ otherwise. The local credit
$\alpha_{r,i}\in\{0,0.5,1\}$ is $1$ for first-attempt completion, $0.5$ for
completion after one or more failed autonomous attempts or for a predefined
task-specific partial-completion state, and $0$ otherwise. The
prerequisite-aware credit is

|  | $$ s_{r,i}=e_{r,i}\alpha_{r,i}. $$ |  | (16) |
|---|---|---|---|

A completed retry can therefore receive half credit while still enabling later
milestones. A partial-completion state enables descendants only when it meets
the task-specific prerequisite condition. A skipped required milestone receives
no credit and blocks only its descendants. Milestones on independent branches
remain eligible. For example, failure to add the first topping in Make Milk Tea
does not prevent credit for a correctly added second topping. When an execution
record groups multiple subtasks, we resolve the outcomes at the original
subtask granularity using the accompanying action-level annotations.

For a task with $|\mathcal{V}|$ annotated milestones and $T$ evaluated trials,
progress is

|  | $$ \mathrm{Progress}=\frac{100}{T|\mathcal{V}|}\sum_{r=1}^{T}\sum_{i\in\mathcal{V}}s_{r,i}. $$ |  | (17) |
|---|---|---|---|---|---|

The graph sizes used in Table I are
$|\mathcal{V}|=25$, $14$, $22$, and $13$ for Clean Room, Prepare Ingredients,
Tomato and Egg Stir Fry, and Make Milk Tea, respectively. Book Organization
uses $|\mathcal{V}|=3$, and Collect Laundry uses $|\mathcal{V}|=5$. The three
Tidy Makeup Table groups use $|\mathcal{V}|=2$, $2$, and $4$ and are evaluated
separately.

Task-specific rules.
Repeated salt addition in Tomato and Egg Stir Fry is a prohibited action rather
than an autonomous retry. It invalidates SR and assigns $0.5$ progress credit
to the salt-addition subtask. Other autonomously completed subtasks are scored
according to the prerequisite graph.

Trial accounting.
Every method–task entry in Table I uses ten independently
collected physical-robot trials.

Each physical-robot trial uses a fixed wall-clock limit measured from the start
of autonomous execution. The task-specific limits are summarized in
Table V.

*TABLE V: Maximum duration of each physical-robot trial.*

| Task | Maximum duration |
|---|---|
| Clean Room | 20 min |
| Prepare Ingredients | 20 min |
| Tomato and Egg Stir Fry | 20 min |
| Make Milk Tea | 10 min |
| Book Organization | 5 min |
| Collect Laundry | 5 min |
| Tidy Makeup Table (each group) | 5 min |

Reset policy.
Before every trial, the robot is returned to the task-specific standardized
initial pose, and all task objects, tools, articulated furniture, and
consumables are restored to their designated initial states. Object positions
are either restored to fixed designated locations or, when a task specifies
randomized placement, resampled within the same predefined range for every
method. All state carried over from the preceding trial is cleared.

Termination conditions.
A trial terminates when the task success conditions are satisfied, the
task-specific time limit is reached, or the scene enters an irrecoverable state.
Only the first case is counted as success. All other termination cases are
counted as failures. For progress, only valid milestones completed before
termination receive credit, following the scoring rule in the main text.

### C-D Low-Level Data Details

The low-level training corpus combines approximately $23.4$K hours of
internally collected demonstrations with $16.7$K hours from public robot
datasets. The internal data comprise $21.9$K hours on AGIBOT G1, $585$ hours
on AGIBOT G2, $578$ hours on ARX AC One, and $347$ hours on Franka. The public
data include $9.25$K hours of open-source UMI data together with additional
selected datasets, supplying further embodiments, tasks, and contact-rich
manipulation skills.

We additionally interleave multimodal instruction-following and robot-centric
perception examples with the action data. All sources follow a consistent
sampling and preprocessing pipeline throughout low-level training; the
multimodal samples serve as auxiliary supervision for preserving the
backbone’s vision-language capabilities.

### C-E High-Level Policy Data Construction

Supervision for the high-level policy is generated automatically from task instructions, stage descriptions, executable subtask annotations, segmented demonstrations, and videos. In the data pipeline, we denote these three instruction levels as L1, L2, and L3, respectively. Construction proceeds in three steps. (1) Pre-labeling. Using google/Gemma4-31B-it as the annotator, we generate a <think> field that summarizes scene state, progress, constraints, and failures, together with a <memory> field that compresses completed stages while retaining detail for the active subtask. The input includes the carried memory and previously committed subtask. The target <memory> is aligned with the current observation, while the target <subtask> reuses the executable-subtask annotation for what should be executed next. (2) Keyframe extraction. Following each subtask’s temporal range, we extract three synchronized views (top_head, hand_left, hand_right) with ffmpeg. (3) VQA assembly. We combine the synchronized views with the generated fields and executable-subtask annotations to form structured training examples for the high-level policy.

To make the high-level policy robust to deployment-time memory misalignment, we perturb only the input memory while reading the corrected target from the demonstration. This procedure derives the five instance families in Table VI without additional annotation. Here $\mathcal{M}_{n}$ denotes the memory upon entering segment $n$. The within-subtask family is the aligned case, and the remaining families target distinct failure modes.

*TABLE VI: High-level instance families, synthesized by perturbing only the input memory while reading the corrected target from the demonstration. $\mathcal{M}_{n}$ is the memory upon entering segment $n$. A single unified <think>/<memory>/<subtask> format instantiates all of them at zero extra annotation.*

| Family | Sampling position | Input $\rightarrow$ target memory | Target subtask | Deployment failure countered | Mix |
|---|---|---|---|---|---|
| within-subtask | anywhere in seg. $n$ | $\mathcal{M}_{n}\rightarrow\mathcal{M}_{n}$ | seg. $n$ | — (aligned, normal progression) | 58% |
| transition | tail of seg. $n$ | $\mathcal{M}_{n}\rightarrow\mathcal{M}_{n+1}$ | seg. $n{+}1$ | starting a new subtask after completion | 15% |
| catch-up | head of seg. $n$ | $\mathcal{M}_{n-1}\rightarrow\mathcal{M}_{n}$ | seg. $n$ | memory lag (behind the visual state) | 10% |
| rollback | late in seg. $n$ | $\mathcal{M}_{n+1\ldots n+3}\rightarrow\mathcal{M}_{n}$ | retry seg. $n$ | memory run-ahead (over-optimistic) | 12% |
| error-think | annotated failure frame | $\mathcal{M}_{n}\rightarrow$ type-dependent | recovery step | unnoticed execution failure | 5% |

For error-think, the <think> field first flags the failure and
then repairs memory according to the failure type. Recoverable failures, such
as an empty grasp, are retried with memory unchanged. Failures that undo prior
progress, such as a dropped object, trigger rollback to the preceding subtask.
Samples marked restorable$=$false are skipped, and rollback
instances are capped at $10$–$15\%$ to avoid teaching the high-level policy to
distrust correct memory. We additionally apply six-dimensional visually grounded
augmentation to the L1/L2 task and stage instructions to improve instruction
diversity and zero-shot steerability.

Quality control. 
Quality control operated in two phases. During prompt development, fixed-seed
stratified human spot-checks on small batches surfaced failure modes, chiefly
cross-task contamination, where the annotator hallucinated action constraints
from an unrelated task (e.g. milk-tea constraints appearing under a
fruit-shelving task). All visual prompts embed a verification rule that requires
omitting any attribute that cannot be confirmed from the images (“when in
doubt, leave it out”), and each identified failure mode was hardened into a
programmatic filter applied at scale: a cross-task contamination filter, an
empty-<subtask> detector, and a memory-contamination filter. Once the
prompts and filters were fixed, the full corpus was pre-labeled and filtered
programmatically, without further per-sample human inspection.

The contamination filter builds a per-episode noun vocabulary from the L1 and L3
annotations and rejects a constraint whose operated object lies entirely outside
this vocabulary, or whose slot nouns overlap it by less than $20\%$. Tightening
the criterion from a naive disjointness test (discard rate $0.33\%$, with
residual contamination) through an over-aggressive ratio threshold (discard rate
$2.32\%$, which removed many valid but verbose constraints) to the final
two-rule filter with an expanded stop-word list (discard rate $0.42\%$) left
zero residual contamination on a $3.26$M-sample validation set. Structural
dirty-data filtering additionally removes empty-<subtask> episodes and
the downstream memory corruption they induce, using a three-tier policy by
per-task dirty ratio (tasks above $90\%$ dirty are excluded outright, tasks
between $10\%$ and $90\%$ are filtered at the episode level, and tasks below
$10\%$ have only their empty-<subtask> samples removed). This discards
$11.74\%$ of episodes and retains $40.4$M ($88.26\%$) clean samples. The
held-out evaluation set for the high-level policy is drawn from frames unseen in training.

### C-F Asynchronous Dual-System Serving

The main text describes the logical dependency between high-level subtask
generation and low-level action generation as a sequential inference step. In
deployment, we pipeline these stages asynchronously because a high-level
decision is far slower than one low-level control period. A background worker
continuously recomputes the next subtask for each
active episode and publishes the generated subtask to a per-episode cache
(refreshed about every $1$ s). The control loop only reads this cache: each tick
takes the cached subtask as the current language command and returns an
action chunk immediately, without waiting for the high-level policy. A slow or failed high-level decision thus
only delays the next command refresh, while $\pi_{\theta}$ keeps executing the
current cached subtask at the control rate ($\sim\!30$ Hz).

### C-G Book Organization Task Settings

The Book Organization task contains four books of different heights and four fixed slots.
At the beginning of each trial, the four books are placed in a shuffled order.
At each step, the robot can use its two arms to exchange the positions of any two books.
The task is completed when all four books are arranged from tallest to shortest.
The high-level policy must therefore predict which pair of books should be exchanged next to make progress toward the target ordering.
In the in-domain setting, the initial arrangement of the four books has appeared in the training data.
In the out-of-domain (OOD) setting, the initial arrangement of the four books has not appeared in the training data.
The OOD setting is designed to evaluate the robustness of the high-level policy to observations arising from initial book arrangements not encountered during training.

### C-H High-Level Evaluation Protocol

The high-level evaluation isolates decision making from physical execution
using annotated samples collected at subtask boundaries. Each sample contains
the task instruction, current robot observation, execution memory when
applicable, the previously committed subtask, and the ground-truth next
subtask. Within each comparison, all methods receive the same samples and
produce the same next-subtask output format. The memory ablation changes only
whether execution memory is provided.
The comparison among Plan Once, Best-of-$N$, and TTC instead holds the
evaluation inputs and output interface fixed while varying the decision-time
inference procedure.

### C-I LLM-as-a-Judge Protocol

For the open-loop TTC evaluation, each sample contains a robot observation from a particular stage of a task and the corresponding ground-truth next subtask.
Each method predicts the next subtask from the same input, and an LLM judge evaluates whether the prediction is correct with respect to the ground truth.

Exact string matching can substantially underestimate performance because the same executable subtask may be expressed with different wording. We therefore use GPT-5.4 as a semantic judge. For each sample, the judge is given the task goal, the ground-truth next subtask, and the predicted next subtask, and assigns one of three labels: equivalent, adjacent, or wrong.

A prediction is equivalent only if it describes the same immediate physical state transition as the ground truth. The required arm, action, object, destination, and relevant material state must agree, while synonyms and harmless differences in wording or granularity are allowed. A prediction is adjacent if it is a physically valid preceding, following, or reorderable step toward the same task goal, but does not match the immediate transition specified by the ground truth. A prediction is wrong if it uses an incompatible arm, action, object, destination, or state, hallucinates an action, or incorrectly terminates or resets the task. We do not allow the judge to infer omitted objects, destinations, or second-arm actions.

Only equivalent predictions are counted as successful; adjacent is retained as a diagnostic category. Each unique (goal, ground truth, prediction) tuple is evaluated by two separate temperature-zero calls. Disagreements receive a third judgment; if all three labels differ, a fourth judgment is obtained. The final label is determined by majority vote.

prompt
You are a strict evaluator of the immediate next executable subtask in a robot plan. Given GOAL, GROUND-TRUTH next subtask, and MODEL-PREDICTED subtask, output one label.

equivalent: the same immediate physical state transition. Required arm(s), action, object(s), destination, and material state must match. Synonyms, capitalization, and harmless wording or granularity differences are allowed.

adjacent: a physically valid preceding, following, or reorderable step for the same goal, but it is a different immediate state transition from the ground truth.

wrong: incompatible arm, action, object, destination, or state; hallucinated/nonsensical action; or done/reset when the ground truth is another action.

Do not infer missing material objects, destinations, or a missing second-arm action. Respond only as compact JSON:”label”:”equivalent—adjacent—wrong”,”reason”:”¡=16 words”.

GOAL: {task_goal}
GROUND-TRUTH next subtask: {ground_truth}
MODEL-PREDICTED subtask: {prediction}
Classify.

### C-J Adaptive Routing Details

At deployment, the high-level policy and low-level policy operate in the closed
loop summarized in Algorithm 1. At inference step
$t$, the proposal model produces $z_{t}^{\mathrm{dir}}$ and $\mathcal{M}_{t}$ from
$h_{t}$. We compute the routing decision from token logits already produced by
this forward pass, so routing requires no additional model invocation.

Let $p_{i}$ be the probability assigned to the generated token at position $i$.
Let $\lambda_{i}^{(1)}$ and $\lambda_{i}^{(2)}$ be the largest and second-largest
logits at that position, and define their margin as

|  | $$ m_{i}=\lambda_{i}^{(1)}-\lambda_{i}^{(2)}. $$ |  | (18) |
|---|---|---|---|

The router uses the mean generated-token probability and the mean logit margin
within the <memory> field:

|  | $\displaystyle u_{t}^{\mathrm{all}}$ | $\displaystyle=\frac{1}{|\mathcal{I}_{\mathrm{all}}|}\sum_{i\in\mathcal{I}_{\mathrm{all}}}p_{i},$ |  | (19) |
|---|---|---|---|---|---|---|
|  | $\displaystyle u_{t}^{\mathrm{mem}}$ | $\displaystyle=\frac{1}{|\mathcal{I}_{\mathrm{mem}}|}\sum_{i\in\mathcal{I}_{\mathrm{mem}}}m_{i},$ |  |

where $\mathcal{I}_{\mathrm{all}}$ indexes all generated tokens and
$\mathcal{I}_{\mathrm{mem}}$ indexes the tokens in the
<memory> field. The binary routing decision is

|  | $$ g_{t}=\mathbf{1}\!\left[u_{t}^{\mathrm{all}}\leq\delta_{\mathrm{all}}\lor u_{t}^{\mathrm{mem}}\leq\delta_{\mathrm{mem}}\right], $$ |  | (20) |
|---|---|---|---|

where $\mathbf{1}[\cdot]$ is the indicator function and
$\delta_{\mathrm{all}}$ and $\delta_{\mathrm{mem}}$ are routing thresholds.
The statistics and routing rule are shared across tasks. The thresholds are
calibrated separately for each task on held-out validation data.

When $g_{t}=0$, the system sends $z_{t}^{\mathrm{dir}}$ directly to the low-level
policy. When $g_{t}=1$, it searches over candidate subtasks, predicts and scores
their visual outcomes, and conditions the reflective model on the retained
branches to generate $z_{t}^{\star}$.

