# 2608.18746 (from arXiv HTML; MinerU fallback)



# Decision-Metric Alignment in Latent World Models:
Diagnostics and Action-Conditioned Objectives for MPC Planning

Jiawei Wang

Thanks: Equal contribution.

Thanks: Corresponding author
(jarvisustc@gmail.com).

  
Yushen Zuo11footnotemark:
1

  
Ke Rui11footnotemark:
1

  
Yichun Feng

  
Minglei Li22footnotemark:
2

###### Abstract

JEPA-style latent world models can use Euclidean distance to a goal
latent as the cost for model-predictive control (MPC). Strong decoding
of task variables, however, does not guarantee that this particular
cost ranks candidate action sequences by real task progress. We call
the latter property decision-metric alignment. We introduce
Plan-Real Spearman, which measures latent–real rank agreement on random
plans, and CEM-stage Spearman, which measures the same agreement as
cross-entropy-method (CEM) search concentrates its proposal. We analyze
sufficient conditions under which latent distance preserves real-cost
rankings, identifying encoder distortion, terminal rollout error, and
candidate margins as the controlling quantities. Guided by the observed
empirical alignment gap, DA-LeWM augments LeWM
with inverse-dynamics and demonstration-conditioned goal-action heads.
Across all our experiments, DA-LeWM accelerates convergence and achieves
higher online success than LeWM, while probe scores remain similar. These
results show that action-conditioned objectives improve the geometry used
by Euclidean-cost, CEM-based latent MPC.

## 1 Introduction

Model predictive control on a learned latent world model is a
principled recipe for robotic manipulation: roll out the world model
along candidate action sequences, select the lowest-cost sequence,
execute, and replan (23; 13; 15; 14; 19). When the
world model operates in a JEPA-style learned latent
space (22; 3), the planning cost is
typically the squared Euclidean distance between the predicted latent
trajectory and an encoded goal observation, dispensing with the need
for an explicit reward function.

This recipe exhibits a striking pattern: small changes to the training
objective induce large and often counterintuitive changes in online
planning success, while linear probes for state, action, reward, and
value remain nearly unchanged across the same variants. The dominant
question asked of a latent world model, “what does the latent
encode?”, is therefore incomplete for planning.

The missing property is geometric rather than informational. The
planner’s cost depends on Euclidean distances between latents, so for
the ranking of candidate plans to track real-world outcomes, the
latent’s metric structure must be compatible with the structure of
action consequences in the environment. We formalize this as the
distinction between information sufficiency (whether
task-relevant quantities can be decoded from the latent) and
decision-metric alignment (whether the latent cost ranks
candidate plans consistently with their environmental outcomes). The
two are independent: as Figure 1 illustrates, a latent
can be information-sufficient yet geometrically misaligned.

*Figure 1: Information sufficiency does not imply
decision-metric alignment. In each panel, the left schematic places
candidate plans around the goal latent $z_{g}$, while the right plot
compares their real costs (horizontal) with their latent Euclidean
costs (vertical). In (a), the two costs induce the same ordering. In
(b), candidate 1 is truly low-cost but lies far from $z_{g}$, whereas
candidate 3 has higher real cost but lies close. Latent distance
therefore reverses their preference even though candidate identity
and rank remain decodable.*

To make decision-metric alignment testable, we introduce two
diagnostics. Plan-Real Spearman measures the rank correlation
between latent costs and real costs over random candidate plans.
CEM-stage Spearman extends it to the converged elite candidates
the agent actually executes. We analyze sufficient conditions under which
latent distance preserves real-cost rankings
(Section 4). The analysis identifies encoder distortion,
terminal rollout error, and candidate margins as the controlling quantities,
motivating stage-wise evaluation as CEM concentrates its proposal. We treat
SIGReg measurements as empirical collapse diagnostics, not as certificates
of the analytical encoder constants. Guided by the observed empirical
alignment gap, we then study action-conditioned auxiliary supervision through
DA-LeWM (Decision-Aligned LeWM), which augments LeWM with
two lightweight auxiliary heads, an inverse-dynamics head and a
goal-conditioned action head. Across all our experiments, DA-LeWM
accelerates convergence and achieves higher online success than LeWM,
while linear-probe scores remain similar.

Our key contributions are as follows:

- •

We distinguish information sufficiency from decision-metric
alignment and introduce Plan-Real Spearman and CEM-stage Spearman
to measure the latter.
- •

We analyze sufficient conditions for rank preservation, identifying
encoder distortion, terminal rollout error, and candidate margins as
the controlling quantities.
- •

Guided by the observed alignment gap, we introduce DA-LeWM with
action-conditioned auxiliary supervision. It improves latent–real rank
agreement in our geometry diagnostics and, across all four environments,
accelerates convergence and achieves higher online success than LeWM
despite similar linear-probe scores.

## 2 Background and Problem Setup

### JEPA Latent World Models

We build on LeWM (23), a JEPA-style world model for robotic
manipulation. An encoder $f_{\theta}$ maps an observation $o_{t}$ to a latent
embedding $z_{t}=f_{\theta}(o_{t})$ via a Vision Transformer backbone followed
by a linear projector. An autoregressive predictor $g_{\phi}$ estimates
future latents conditioned on actions:

|  | $$ \hat{z}_{t+1}=g_{\phi}(z_{t-H:t},\;a_{t-H:t}). $$ |  | (1) |
|---|---|---|---|

Training minimizes a prediction loss together with Sketched Isotropic
Gaussian Regularization (SIGReg), introduced by LeJEPA
(4) and adopted by LeWM:

|  | $$ \mathcal{L}_{\text{WM}}=\mathcal{L}_{\text{pred}}+\lambda_{\text{sig}}\mathcal{L}_{\text{SIGReg}}. $$ |  | (2) |
|---|---|---|---|

SIGReg projects embeddings onto random unit directions and
minimizes the one-dimensional Epps–Pulley statistic against a standard
Gaussian, encouraging the aggregate embedding distribution to match an
isotropic Gaussian and discouraging trivial collapse. This is a
distribution-level target: it does not bound encoder-Jacobian singular
values or imply local metric isotropy, injectivity, or bi-Lipschitz
behavior.

### MPC with Latent Goal Distance

At test time, MPC is realized by the cross-entropy method (CEM)
(26). Given an initial observation $o_{0}$ and a goal
observation $o_{g}$, the planner solves

|  | $$ a^{*}=\arg\min_{a_{0:H}}\|\hat{z}_{H}-z_{g}\|^{2},\quad z_{g}=f_{\theta}(o_{g}), $$ |  | (3) |
|---|---|---|---|---|---|

where $\hat{z}_{H}$ is obtained by rolling out $g_{\phi}$ under candidate
action sequences. CEM iteratively refits its proposal distribution to the
top-$E$ elite candidates with lowest latent cost.

#### Goal acquisition at evaluation time.

For evaluation, $o_{g}$ is drawn from a held-out demonstration trajectory
at a fixed offset $\Delta$ ahead of the initial observation. This
guarantees that the goal latent lies in the data distribution and is
reachable by some action sequence. In deployment, the goal image would
typically come from a single human demonstration or sub-goal generation.
Our analysis is independent of the goal-acquisition mechanism.

### Auxiliary Objectives

A standard avenue for improving latent representations is to add
auxiliary prediction heads jointly trained with the world model
(28; 21; 11). We
consider inverse-dynamics, goal-conditioned action, reward-proxy, and
value-proxy heads (full losses are in Supplementary Section A).
Such objectives are usually judged by probes or downstream control. We
instead ask how they affect the planner’s fixed Euclidean cost.

## 3 Information Sufficiency vs. Decision-Metric Alignment

### Definitions

A latent space $z=f_{\theta}(o)$ is $\epsilon$-information-sufficient
for task quantities $\{q_{k}\}$ if probe-specific regressors
$\hat{q}_{k}=h_{k}(x_{k})$ attain held-out error at most $\epsilon$.
We test state, inverse-action, reward, value, and goal-action with linear
probes. Supplementary Section A specifies each input $x_{k}$.

A latent space is decision-metric aligned with respect to a planning
cost $c_{\text{lat}}$ if, for any pair of candidate action sequences
$\mathbf{a}^{(i)},\mathbf{a}^{(j)}$ with corresponding real-environment
costs $c_{\text{real}}^{(i)},c_{\text{real}}^{(j)}$,

|  | $$ c_{\text{lat}}(\mathbf{a}^{(i)})<c_{\text{lat}}(\mathbf{a}^{(j)})\quad\Longleftrightarrow\quad c_{\text{real}}(\mathbf{a}^{(i)})<c_{\text{real}}(\mathbf{a}^{(j)}). $$ |  | (4) |
|---|---|---|---|

Decision-metric alignment is an ordinal property, not a numerical one. It
requires the latent cost to correctly rank candidates rather than to be
numerically equal to any reward function. The two properties are
logically independent. As Figure 1 illustrates, a
latent can encode all task-relevant information while arranging it in
a geometry where Euclidean distance ranks candidates poorly.

### Plan-Real and CEM-stage Spearman

For each (start, goal) pair sampled from a held-out dataset with
offset $\Delta$, we draw $N=64$ candidate action sequences from the
CEM proposal distribution, score each candidate by the world model’s
latent cost $c_{\text{lat}}^{(i)}=\|\hat{z}_{H}^{(i)}-z_{g}\|^{2}$ and by
environment-rollout cost $c_{\text{real}}^{(i)}=d_{\text{task}}(s_{H}^{(i)},s_{g})$, and compute the Spearman rank correlation between the two
across the $N$ candidates. On PushT, $d_{\text{task}}=\|s_{H}-s_{g}\|_{2}$
over all seven simulator coordinates (agent and block pose, and agent
velocity), rather than thresholded online success. Plan-Real
Spearman is the mean over $n=30$ pairs. Spearman correlation lies in $[-1,1]$: $+1$ denotes
identical rankings, $0$ no monotone association, and $-1$ reversed
rankings. Exact ties receive their average rank. If either cost vector
is constant, the pair-level correlation is undefined. We exclude that
pair from the mean and record the number of defined pairs. Ranking is
the relevant quantity because CEM updates its
proposal from relative candidate order rather than calibrated costs.

CEM does not act on random plans: it iteratively concentrates
candidates toward low-cost regions, and a latent cost that ranks
random plans well may still fail when candidates are near-optimal.
CEM-stage Spearman measures the same correlation at three CEM
stages: random (iteration $0$), mid (iteration $15$), and elite (the
final top-$E$ set). Random-stage agreement tests global coarse ordering.
Mid-stage agreement tests the ordering that shapes proposal updates.
Elite-stage agreement tests local discrimination among near-optimal
candidates.

![Refer to caption](drafts/images/dalewm-2608.18746/figure2_new3-compatible.png)

*Figure 2: Plan-Real Spearman measurement procedure. This operationalizes the
candidate-order comparison in Figure 1. For each
held-out pair $k$, the same $N=64$ action sequences are evaluated by
the world model and the environment, yielding paired latent- and
real-cost vectors. Spearman gives a pair-level rank correlation $\rho_{k}$;
Plan–Real averages the defined correlations among $n=30$ sampled pairs.
CEM-stage Spearman reuses the paired
scoring procedure with candidates from the selected CEM population.*

## 4 DA-LeWM: A Decision-Aligned Latent World Model

### Method

We propose DA-LeWM, which augments the base LeWM training with
two lightweight auxiliary heads. The full training objective is

|  | $$ \mathcal{L}_{\text{DA-LeWM}}=\mathcal{L}_{\text{WM}}+\alpha\,\mathcal{L}_{\text{inv}}+\beta\,\mathcal{L}_{\text{goal}}, $$ |  | (5) |
|---|---|---|---|

where
$\mathcal{L}_{\text{inv}}=\|h_{\text{inv}}(z_{t},z_{t+1})-a_{t}\|^{2}$
is the inverse-dynamics loss and
$\mathcal{L}_{\text{goal}}=\|h_{\text{goal}}(z_{t},z_{g})-a_{t}\|^{2}$
is the goal-conditioned action loss. Both heads are small MLPs with
two hidden layers of width $256$. We set $\alpha=0.1$ and
$\beta=0.1$ as the cross-environment default. Sensitivity to $\beta$
is reported in Supplementary Section C.

#### Goal-action supervision.

Training uses four-frame demonstration clips at raw offsets
$0,5,10,15$. For $k\in\{0,1,2\}$, $h_{\text{goal}}(z_{t+5k},z_{t+15})$ is
supervised by the concatenated, normalized five-raw-action block
$a_{t+5k:t+5k+4}$. The goal is $15,10,$ or $5$ raw steps ahead.
The target reflects demonstrated progress, not an optimizer-computed
globally optimal action. With multimodal demonstrations, squared-error
training may average incompatible actions. The head shapes the
representation and is not used as a policy.

#### Rationale.

The inverse-dynamics loss encourages the latent transition to retain
information predictive of $a_{t}$.
The goal-conditioned action head provides analogous structure relative
to the goal latent. An exploratory all-heads checkpoint using the state-derived proxies
in Supplementary Section A is retained as an implementation-specific
comparison. Because those proxies are not the environment’s reward and
value targets, this checkpoint neither isolates an R/V effect nor tests
correctly specified reward- or value-aware training in general.

### Sufficient-Condition Analysis

We complement the empirical diagnostics with an analytical account of
when latent-cost ranking can be expected to track real-cost ranking. The
goal is to identify the principal error sources, not to prove a tight
rank-correlation bound.

#### Setup and assumptions.

We use the notation of Section 2. For compactness,
$f_{\theta}(s)$ denotes encoding the rendered observation $o=R(s)$. The
encoder does not consume simulator state directly. Let $T$ denote the
environment dynamics, and write $s_{H}(\mathbf{a})$ and $\hat{z}_{H}(\mathbf{a})$
for the real and predicted terminal states under plan $\mathbf{a}$. Real
and latent costs are $c_{\text{real}}(\mathbf{a})=\|s_{H}(\mathbf{a})-s_{g}\|_{\Sigma}$ and $c_{\text{lat}}(\mathbf{a})=\|\hat{z}_{H}(\mathbf{a})-z_{g}\|_{2}$, respectively, for some PSD metric $\Sigma$. The planner
uses the squared latent norm, but positive squaring preserves every
candidate ranking, so the analysis uses the unsquared norm. The PushT
diagnostics instantiate $\Sigma=I$ on the full seven-dimensional state.

###### Assumption 1 (Pointwise terminal-rollout consistency).

For every horizon-$H$ candidate plan in the analysis, terminal rollout
error satisfies
$\|\hat{z}_{H}(\mathbf{a})-f_{\theta}(s_{H}(\mathbf{a}))\|_{2}\leq\epsilon_{H}$.

###### Assumption 2 (Encoder bi-Lipschitz).

There exist constants $0<\mu_{f}\leq L_{f}$ such that, for the
terminal-state/goal pairs induced by candidate plans in the analysis,

|  | $$ \mu_{f}\|s-s^{\prime}\|_{\Sigma}\;\leq\;\|f_{\theta}(s)-f_{\theta}(s^{\prime})\|_{2}\;\leq\;L_{f}\|s-s^{\prime}\|_{\Sigma}. $$ |  |
|---|---|---|---|---|---|---|---|---|

SIGReg empirically discourages representational collapse, but by itself
does not establish Assumption 2 or a lower bound on
$\mu_{f}$. We therefore state bi-Lipschitz behavior as an assumption and
measure collapse-related surrogates separately. Training losses are
empirical diagnostics, not proofs that these assumptions hold off
distribution.

#### Pointwise cost-approximation bound.

###### Proposition 1 (Two-sided cost-approximation bound).

Under Assumptions 1 and 2, for any
horizon-$H$ plan $\mathbf{a}$,

|  | $$ \mu_{f}\,c_{\text{real}}(\mathbf{a})-\epsilon_{H}\;\leq\;c_{\text{lat}}(\mathbf{a})\;\leq\;L_{f}\,c_{\text{real}}(\mathbf{a})+\epsilon_{H}. $$ |  | (6) |
|---|---|---|---|

The bracket width $(L_{f}-\mu_{f})c_{\text{real}}+2\epsilon_{H}$
decomposes into an encoder distortion term and a
predictor error term. Our experiments measure collapse-related
and rollout-error diagnostics, but do not identify these constants.

###### Proof sketch.

The reverse triangle inequality and Assumption 1 give
$\big|c_{\text{lat}}-\|f_{\theta}(s_{H})-z_{g}\|_{2}\big|\leq\epsilon_{H}$.
Assumption 2 (and $z_{g}\!=\!f_{\theta}(s_{g})$) gives
$\|f_{\theta}(s_{H})-z_{g}\|_{2}\in[\mu_{f}c_{\text{real}},L_{f}c_{\text{real}}]$.
∎

###### Corollary 1 (Margin-implied rank preservation).

Under the assumptions of Prop. 1, for any pair
with $c_{\text{real}}^{(i)}<c_{\text{real}}^{(j)}$,
$\mu_{f}c_{\text{real}}^{(j)}>L_{f}c_{\text{real}}^{(i)}+2\epsilon_{H}$
implies $c_{\text{lat}}^{(i)}<c_{\text{lat}}^{(j)}$. If every pair of
$N$ distinct candidates satisfies this, Plan-Real Spearman is $+1$.
(Proof: subtract the upper bound at $i$ from the lower bound at $j$.)

The condition requires (i) a low-distortion encoder
($L_{f}/\mu_{f}\!\to\!1$) and (ii) non-vanishing
candidate margins
$\Delta_{ij}\!=\!c_{\text{real}}^{(j)}\!-\!c_{\text{real}}^{(i)}$. The
latter can be small on contact-rich tasks under random sampling and can
shrink further as CEM concentrates its proposal. This motivates
measuring rank agreement at random, intermediate, and elite stages
rather than assuming that one sampling regime is representative.

#### Empirical consistency check.

Fit $c_{\text{lat}}\!\approx\!\kappa c_{\text{real}}$ and let
$\eta_{i}\!=\!|c_{\text{lat}}^{(i)}-\kappa c_{\text{real}}^{(i)}|$.
Kendall’s $\tau_{a}$ averages pairwise concordance signs (ties count as
zero). The soft-margin rate $p$ is the fraction of distinct-real-cost
pairs satisfying $\kappa\Delta_{ij}>\eta_{i}+\eta_{j}$. Thus
$\tau_{a}\!\geq\!2p\!-\!1$ without real-cost ties (tie-aware form in
Supplementary Section F). On PushT, $p$ tracks Plan-Real Spearman
with pooled Pearson $+0.895$ over $100$ pairs (Supplementary Section F).

#### The role of inverse-dynamics training.

###### Claim 1 (Inverse-action mechanism).

Inverse-action prediction encourages action-relevant latent transitions:
the latent displacement $z_{t+1}-z_{t}$ must contain enough information
to recover $a_{t}$. If the latent transition geometry is locally
well-conditioned, neither degenerate along action-relevant directions
nor anisotropically dilated relative to the data distribution, this
constraint manifests as an increased rank correlation between latent
displacement magnitude and action magnitude. Empirically, we observe
that inverse-action training increases this global rank correlation
from $-0.03$ for the LeWM baseline to $+0.38$ for inverse-only and
$+0.43$ for DA-LeWM. Inverse-only produces the largest single increment
in Plan-Real Spearman among the measured ablations.

The claim is mechanistic, not a theorem. Low held-out inverse-prediction
error is an empirical diagnostic rather than an assumption of
Proposition 1, and it does not force norm alignment:
action information could in principle be encoded directionally. We report
latent-displacement-vs-action-norm rank correlation in
Supplementary Section B as a falsifiable signature of the mechanism.

#### Implications.

Encoder distortion ($L_{f}/\mu_{f}$) and terminal rollout error
($\epsilon_{H}$) are the two analytical quantities. SIGReg and predictor
training are empirical interventions associated with their diagnostics,
not certificates of the constants. The inverse-action loss is instead
mediated by Claim 1. Reward/value decodability is
absent from the bound. We retain the exploratory all-heads checkpoint as
an implementation-specific comparison using the proxy R/V targets
described in Supplementary Section A. It is not a general test of
reward- or value-aware world-model training.

## 5 Experiments

*Figure 3: Probe accuracy vs. online success on PushT
($3$ evaluation seeds). Probe scores are nearly identical across the
four non-collapsed variants, while online success differs sharply.*

### Setup

We evaluate on four environments: PushT, Reacher, Cube, and TwoRoom,
spanning planar object pushing, continuous-control reaching, contact-rich
manipulation, and visual navigation. Dataset and task details are provided
in Supplementary Section A.

#### Evaluation protocol.

All methods use matched training and planning budgets. Online success is
measured under closed-loop, receding-horizon MPC over $50$ episodes for
each of $3$ evaluation seeds. Full training, CEM, and replanning details
are provided in Supplementary Section A.

#### Variants compared.

We compare five variants. LeWM is the SIGReg-regularized baseline with no
auxiliary heads. No-SIGReg removes the regularizer and serves as the
collapse control. Inverse-only adds only the inverse-action head.
All-heads adds inverse, goal-action, reward-proxy, and value-proxy heads,
whereas DA-LeWM uses only the inverse and goal-action heads. We treat
all-heads as exploratory because its reward/value targets are
implementation-specific.
Supplementary Section C reports sensitivity to the goal-action weight.

#### Evaluation sequence.

We first use No-SIGReg to establish the collapse sanity check. We then
compare non-collapsed PushT variants across information probes, Plan-Real
Spearman, held-out action geometry, and online success. Next, CEM-stage
Spearman tracks rank agreement throughout planning. Finally, short-budget
and ten-epoch evaluations test control across tasks before
Table 5 places DA-LeWM alongside published
baselines.

### SIGReg Preserves Cost-Surface Variation

We audit cost-surface variation by the ratio of maximum to minimum latent
cost over a random candidate population. Removing SIGReg contracts this
ratio from $3\!-\!30\times$ to $\approx\!1.005\times$. A ratio near one
means that the planner receives almost the same objective for every plan.
Table 1 shows that this flattening coincides with near-zero
Plan-Real Spearman and large online-success drops on PushT, TwoRoom, and
Reacher.

|  | Success (%) | Plan-Real Sp |
|---|---|---|
| Task | LeWM | No-SIG | LeWM | No-SIG |
| PushT | $49.3\pm 12.2$ | $\phantom{0}2.0\pm 2.0$ | $+0.280$ | $+0.031$ |
| TwoRoom | $98.0\pm\phantom{0}2.0$ | $41.3\pm 6.1$ | $+0.549$ | $+0.012$ |
| Reacher | $82.0\pm\phantom{0}2.0$ | $10.7\pm 8.1$ | $+0.504$ | $+0.001$ |

*Table 1: Effect of removing SIGReg across three environments
($3$ evaluation seeds, mean $\pm$ std).*

Cube is handled separately in Section 5, because random
no-contact candidates provide too few distinct real costs for a supported
Spearman estimate.

*Figure 4: Online success throughout ten training epochs
($3$ evaluation seeds per task, mean $\pm$ std). Shading shows one
standard deviation. Annotations report DA-LeWM minus LeWM averaged over
epochs (Avg. $\Delta$) and at epoch 10 ($\Delta_{10}$).*

### Non-Collapsed Variants Separate Information from Planning Geometry

We next compare LeWM, inverse-only, all-heads, and DA-LeWM after excluding
the collapsed control. Their state, action, and goal-action probe scores vary
by at most $0.03$ in $R^{2}$ (Figure 3). In contrast,
Plan-Real Spearman changes from $+0.280$ for LeWM to $+0.410$–$+0.420$
for action-supervised variants (Table 2), while online
success spans $43$ percentage points ($49.3\%\!\to\!92.7\%$). This comparison
isolates decision-metric alignment from information sufficiency rather than
conflating either property with representation collapse.
Table 2 retains No-SIGReg only as the collapse
reference.
For each of the $30$ held-out start–goal pairs, Spearman ranks $64$
candidates. The table reports the mean, and “Positive pairs” counts how
many pair-level correlations exceed zero.

#### Inverse-action mechanism.

On PushT, inverse-only raises Plan-Real Spearman from $+0.280$ to $+0.420$
and online success from $49.3\%$ to $64.0\%$. DA-LeWM retains a similar
$+0.412$ Spearman and reaches $92.7\%$ success. Thus inverse-only accounts
for most of the measured ranking lift, while the combined objective is
associated with the additional online lift.

The checkpoint-level change is accompanied by the held-out transition
signature in Claim 1: correlation between latent- and
action-displacement magnitudes rises from $-0.03$ for LeWM to $+0.38$ for
inverse-only and $+0.43$ for DA-LeWM, versus $-0.13$ without SIGReg.
Because MPC continues to use Euclidean goal distance rather than an
auxiliary-head output, this evidence concerns the geometry exposed to the
planner, not an added test-time cost.

| Model | Plan-Real Sp | Positive pairs (out of $30$) |
|---|---|---|
| LeWM | $+0.280$ | $24$ |
| No-SIGReg | $+0.031$ | $14$ |
| Inverse-only | $\mathbf{+0.420}$ | $30$ |
| All-heads | $+0.410$ | $30$ |
| DA-LeWM | $+0.412$ | $29$ |

*Table 2: Full-state Plan-Real Spearman on PushT ($n=30$).*

Supplementary Section B gives the corresponding values for all variants.

### CEM-stage Spearman: Global Ranking Lift and Elite Saturation on PushT

Table 3 follows CEM iterations $0/15/29$, corresponding
to the random, mid, and elite stages. The random stage measures global
candidate ordering. The later snapshots test ordering after the proposal
has adapted to the model’s own cost. This stage-wise audit operationalizes
the candidate-margin term in the analysis instead of relying on a single
global correlation.

| Model | Random | Mid | Elite |
|---|---|---|---|
| LeWM | $+0.403$ | $+0.227$ | $+0.036$ |
| No-SIGReg | $+0.029$ | $-0.017$ | $-0.102$ |
| Inverse-only | $+0.523$ | $\mathbf{+0.261}$ | $-0.089$ |
| All-heads | $+0.515$ | $+0.249$ | $-0.010$ |
| DA-LeWM | $\mathbf{+0.536}$ | $+0.253$ | $-0.011$ |

*Table 3: Full-state CEM-stage Spearman on PushT ($n\!=\!15$
held-out pairs, CEM budget $300\!\times\!30$, top-$30$ elites).*

Relative to LeWM, the three action-supervised variants produce consistent
random-stage gains of $+0.112$ to $+0.133$ (paired
$p\!\leq\!0.012$), with DA-LeWM highest at $+0.536$. These variants remain at
$+0.249$–$+0.261$ mid-stage versus $+0.227$ for LeWM. Elite-stage
correlations are near zero for every variant, indicating shared local
saturation. Supplementary Section E reports paired significance and
local-geometry checks.
Tables 2 and 3 therefore answer
complementary questions. The former tests independently sampled plans,
while the latter follows the distributions visited by the optimizer.

### Short-Budget Control Across Environments

Having established the PushT ranking effect, we test whether it translates
to control under the same one-epoch budget. Table 4
shows that DA-LeWM improves PushT by $43.4$ pp and Cube by $10.6$ pp,
while Reacher remains within its strong baseline band.

| Model | PushT | Reacher | Cube |
|---|---|---|---|
| LeWM | $49.3\pm 12.2$ | $82.0\pm 2.0$ | $62.7\pm 4.2$ |
| No-SIGReg | $\phantom{0}2.0\pm\phantom{0}2.0$ | $10.7\pm 8.1$ | $52.7\pm 1.9$ |
| Inverse-only | $64.0\pm\phantom{0}7.2$ | $82.7\pm 3.1$ | $68.0\pm 4.0$ |
| DA-LeWM | $\mathbf{92.7\pm\phantom{0}1.2}$ | $\mathbf{84.0\pm\phantom{0}3.5}$ | $\mathbf{73.3\pm 1.2}$ |

*Table 4: Cross-environment online success (matched training
budget, $3$ evaluation seeds for all variants.)*

#### Diagnostic scope.

Reacher variants stay within $[+0.49,+0.53]$, consistent with strong
baseline rank agreement. For Cube, exact no-contact ties leave too few
distinct real costs for a supported Plan-Real estimate. Its No-SIGReg
latent-cost range still contracts from $4\times$ to $1.07\times$, while
online success drops from $62.7\%$ to $52.7\%$
(Table 4). We treat these as collapse and online-control
evidence, not as a Cube Spearman or global bi-Lipschitz claim. Sensitivity
to the goal-action weight $\beta$ is in Supplementary Section C.

### Learning Dynamics under a Larger Budget

We next increase the training budget to ten epochs and compare LeWM with
DA-LeWM on all four tasks. Figure 4 reports the
resulting learning curves, with complete per-epoch statistics in
Supplementary Section H.

The learning curves test whether the large one-epoch gains are merely an
early-training effect. DA-LeWM enters the high-success regime earlier and
maintains higher success throughout training on all four tasks. Its
advantage therefore persists as LeWM receives more optimization, supporting
accelerated convergence rather than a favorable single checkpoint.

Table 5 then asks whether faster learning also
yields strong absolute performance against LeWM (23),
PLDM (29), and DINO-WM (33).
Relative to LeWM, DA-LeWM improves success by $2.7$ pp on PushT, $1.3$ pp
on Reacher, $6.7$ pp on Cube, and $9.0$ pp on TwoRoom. It ranks first on
PushT ($98.7\%$) and Reacher ($87.3\%$). On Cube, it surpasses PLDM and
remains within $5.3$ pp of DINO-WM. On TwoRoom, it reaches $96.0\%$,
within $1.0$ pp of PLDM and $4.0$ pp of DINO-WM while remaining $9.0$ pp
above LeWM. Thus the faster learning is accompanied by stronger control
performance rather than only an earlier optimization advantage.

The auxiliary heads are used only during training and are discarded at
evaluation. LeWM and DA-LeWM therefore use the same inference-time MPC
procedure and incur the same planning-time computation. Their separation
along the learning curves isolates a representation-learning benefit:
action-conditioned supervision makes the latent geometry useful to the
unchanged planner earlier in training. Together,
Figure 4 and Table 5
show improvements in both learning efficiency and control performance across
task regimes.

| Method | PushT | Reacher | Cube | TwoRoom |
|---|---|---|---|---|
| Random | $2$ | $10$ | $48$ | $0$ |
| PLDM (29) | $78$ | $78$ | $65$ | $97$ |
| DINO-WM (33) | $74$ | $79$ | $\mathbf{86}$ | $\mathbf{100}$ |
| LeWM (23) | $96$ | $86$ | $74$ | $87$ |
| DA-LeWM (ours) | $\mathbf{98.7}$ | $\mathbf{87.3}$ | $80.7$ | $96.0$ |

*Table 5: Online success comparison (%). DA-LeWM values are
means over $3$ evaluation seeds.*

## 6 Related Work

#### JEPA and latent world models.

Joint-embedding predictive architectures learn visual representations by
predicting in embedding space rather than reconstructing pixels
(22; 3; 5). Masked autoencoding
and self-distillation provide complementary self-supervised routes
(20; 6). LeWM adapts joint-embedding prediction to
action-conditioned visual dynamics for manipulation, PLDM learns latent
dynamics for offline reward-free control, and DINO-WM plans over frozen
pretrained features (23; 29; 33). These
works establish capable representations and predictors for latent planning.
Linear probes assess what is decodable (1). Our question
is whether Euclidean distance to a goal latent induces the correct ordering
over candidate action sequences.

#### Auxiliary objectives in RL and world models.

Inverse dynamics has been used for curiosity and action-relevant
representation learning
(25; 32; 11), while
contrastive and self-predictive objectives improve pixel-based RL
(21; 28). Demonstration-conditioned
goal-action prediction relates to goal-conditioned imitation and hindsight
relabeling (9; 2). Here these established
objectives provide action-conditioned supervision. We test how that
supervision reshapes the metric structure consumed by fixed-cost latent MPC.

#### MPC and planning with learned models.

PETS plans under learned rewards, while Dreamer learns behavior through
reward-driven latent imagination (8; 14; 16; 17). MuZero and TD-MPC learn
value-equivalent dynamics, and DayDreamer extends world-model control to
physical robots (27; 18; 19; 31). Our setting isolates a complementary
requirement: when Euclidean goal distance is itself the planning cost,
accurate prediction or decoding need not imply correct relative ordering.
We measure that ordering globally and as CEM concentrates its proposal.

#### Decision-aware and bisimulation-based representations.

Bisimulation metrics ground state similarity in reward and transition
consequences (10; 7; 32). Value equivalence provides an analogous criterion for
model predictions (12). Plan-Real and CEM-stage Spearman
are planner-aligned empirical counterparts in this spirit: they audit the
fixed metric actually consumed by the planner rather than learning a new
task-specific metric.

## 7 Conclusion

We study when Euclidean goal distance provides a reliable objective for latent
MPC, separating information sufficiency from decision-metric alignment.
Plan-Real and CEM-stage Spearman operationalize this distinction, while the
sufficient-condition analysis identifies encoder distortion, terminal rollout
error, and candidate margins as the quantities controlling rank preservation.
Across the non-collapsed variants, similar linear-probe scores coexist with
substantial differences in Plan-Real agreement and online success, showing
that this geometric distinction matters for control. Guided by this empirical
gap, DA-LeWM uses action-conditioned supervision to improve the cost geometry
exposed to the planner. Across all four environments, DA-LeWM accelerates
convergence and achieves higher online success than LeWM.
Latent world models used for control should therefore be evaluated both for
what they encode and for the ordering induced by the planning cost they
expose.

#### Limitations.

Our evidence is limited to four short-horizon simulated tasks, LeWM-family
checkpoints with ViT-Tiny, Euclidean goal costs, and CEM. Each configuration
has one training run, so uncertainty covers evaluation variation rather than
training-initialization variation. Plan-Real requires simulator rollouts and
loses statistical power under exact real-cost ties. We fix $\alpha=0.1$ and
test neither its sensitivity nor gradient-based planning. Generalization to
robots, partial observability, larger backbones, other planners, and learned
rewards, goal costs, or positive-semidefinite latent metrics remains open.

#### Generative-AI disclosure.

Generative AI assisted language editing and code review. It was not a component
of the method or evaluation, and the authors verified all resulting text and
code.

## References

- Alain and Bengio (2017)
G. Alain and Y. Bengio

Understanding intermediate layers using linear classifier probes.

In International Conference on Learning Representations (ICLR) Workshop Track,

External Links: [Link](https://openreview.net/forum?id=HJ4-rAVtl)

Cited by: §6.
- Andrychowicz et al. (2017)
M. Andrychowicz, F. Wolski, A. Ray, J. Schneider, R. Fong, P. Welinder, B. McGrew, J. Tobin, P. Abbeel, and W. Zaremba

Hindsight experience replay.

In Advances in Neural Information Processing Systems 30 (NeurIPS),

External Links: [Link](https://proceedings.neurips.cc/paper/2017/hash/453fadbd8a1a3af50a9df4df899537b5-Abstract.html)

Cited by: §6.
- Assran et al. (2023)
M. Assran, Q. Duval, I. Misra, P. Bojanowski, P. Vincent, M. Rabbat, Y. LeCun, and N. Ballas

Self-supervised learning from images with a joint-embedding predictive architecture.

In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),

pp. 15619–15629.

Cited by: §1,
§6.
- Balestriero and LeCun (2025)
R. Balestriero and Y. LeCun

LeJEPA: provable and scalable self-supervised learning without the heuristics.

arXiv preprint arXiv:2511.08544.

External Links: [Link](https://arxiv.org/abs/2511.08544)

Cited by: §2.
- Bardes et al. (2024)
A. Bardes, Q. Garrido, J. Ponce, X. Chen, M. Rabbat, Y. LeCun, M. Assran, and N. Ballas

Revisiting feature prediction for learning visual representations from video.

arXiv preprint arXiv:2404.08471.

External Links: [Link](https://arxiv.org/abs/2404.08471)

Cited by: §6.
- Caron et al. (2021)
M. Caron, H. Touvron, I. Misra, H. Jégou, J. Mairal, P. Bojanowski, and A. Joulin

Emerging properties in self-supervised vision transformers.

In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV),

pp. 9650–9660.

Cited by: §6.
- Castro (2020)
P. S. Castro

Scalable methods for computing state similarity in deterministic Markov decision processes.

In Proceedings of the 34th AAAI Conference on Artificial Intelligence,

pp. 10069–10076.

Cited by: §6.
- Chua et al. (2018)
K. Chua, R. Calandra, R. McAllister, and S. Levine

Deep reinforcement learning in a handful of trials using probabilistic dynamics models.

In Advances in Neural Information Processing Systems 31 (NeurIPS),

External Links: [Link](https://proceedings.neurips.cc/paper_files/paper/2018/hash/3de568f8597b94bda53149c7d7f5958c-Abstract.html)

Cited by: §6.
- Ding et al. (2019)
Y. Ding, C. Florensa, P. Abbeel, and M. Phielipp

Goal-conditioned imitation learning.

In Advances in Neural Information Processing Systems 32 (NeurIPS),

External Links: [Link](https://proceedings.neurips.cc/paper/2019/hash/c8d3a760ebab631565f8509d84b3b3f1-Abstract.html)

Cited by: §6.
- Ferns et al. (2004)
N. Ferns, P. Panangaden, and D. Precup

Metrics for finite Markov decision processes.

In Proceedings of the 20th Conference on Uncertainty in Artificial Intelligence (UAI),

pp. 162–169.

Cited by: §6.
- Gelada et al. (2019)
C. Gelada, S. Kumar, J. Buckman, O. Nachum, and M. G. Bellemare

DeepMDP: learning continuous latent space models for representation learning.

In Proceedings of the 36th International Conference on Machine Learning (ICML),

Proceedings of Machine Learning Research, Vol. 97, pp. 2170–2179.

Cited by: §2,
§6.
- Grimm et al. (2020)
C. Grimm, A. Barreto, S. Singh, and D. Silver

The value equivalence principle for model-based reinforcement learning.

In Advances in Neural Information Processing Systems 33 (NeurIPS),

Cited by: §6.
- Ha and Schmidhuber (2018)
D. Ha and J. Schmidhuber

Recurrent world models facilitate policy evolution.

In Advances in Neural Information Processing Systems 31 (NeurIPS),

External Links: [Link](https://proceedings.neurips.cc/paper/2018/hash/2de5d16682c3c35007e4e92982f1a2ba-Abstract.html)

Cited by: §1.
- Hafner et al. (2020)
D. Hafner, T. Lillicrap, J. Ba, and M. Norouzi

Dream to control: learning behaviors by latent imagination.

In Proceedings of the 8th International Conference on Learning Representations (ICLR),

External Links: [Link](https://openreview.net/forum?id=S1lOTC4tDS)

Cited by: §1,
§6.
- Hafner et al. (2019)
D. Hafner, T. Lillicrap, I. Fischer, R. Villegas, D. Ha, H. Lee, and J. Davidson

Learning latent dynamics for planning from pixels.

In Proceedings of the 36th International Conference on Machine Learning (ICML),

Proceedings of Machine Learning Research, Vol. 97, pp. 2555–2565.

Cited by: §1.
- Hafner et al. (2021)
D. Hafner, T. Lillicrap, M. Norouzi, and J. Ba

Mastering Atari with discrete world models.

In Proceedings of the 9th International Conference on Learning Representations (ICLR),

External Links: [Link](https://openreview.net/forum?id=0oabwyZbOu)

Cited by: §6.
- Hafner et al. (2023)
D. Hafner, J. Pasukonis, J. Ba, and T. Lillicrap

Mastering diverse domains through world models.

arXiv preprint arXiv:2301.04104.

External Links: [Link](https://arxiv.org/abs/2301.04104)

Cited by: §6.
- Hansen et al. (2022)
N. Hansen, H. Su, and X. Wang

Temporal difference learning for model predictive control.

In Proceedings of the 39th International Conference on Machine Learning (ICML),

Proceedings of Machine Learning Research, Vol. 162, pp. 8387–8406.

Cited by: §6.
- Hansen et al. (2024)
N. Hansen, H. Su, and X. Wang

TD-MPC2: scalable, robust world models for continuous control.

In The Twelfth International Conference on Learning Representations (ICLR),

External Links: [Link](https://openreview.net/forum?id=Oxh5CstDJU)

Cited by: §1,
§6.
- He et al. (2022)
K. He, X. Chen, S. Xie, Y. Li, P. Dollár, and R. Girshick

Masked autoencoders are scalable vision learners.

In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),

pp. 16000–16009.

Cited by: §6.
- Laskin et al. (2020)
M. Laskin, A. Srinivas, and P. Abbeel

CURL: contrastive unsupervised representations for reinforcement learning.

In Proceedings of the 37th International Conference on Machine Learning (ICML),

Proceedings of Machine Learning Research, Vol. 119, pp. 5639–5650.

Cited by: §2,
§6.
- LeCun (2022)
Y. LeCun

A path towards autonomous machine intelligence.

Note: Position paper, version 0.9.2, OpenReview

External Links: [Link](https://openreview.net/forum?id=BZ5a1r-kVsf)

Cited by: §1,
§6.
- Maes et al. (2026)
L. Maes, Q. Le Lidec, D. Scieur, Y. LeCun, and R. Balestriero

LeWorldModel: stable end-to-end joint-embedding predictive architecture from pixels.

arXiv preprint arXiv:2603.19312.

External Links: [Link](https://arxiv.org/abs/2603.19312)

Cited by: Appendix A,
§1,
§2,
§5,
§5,
§6.
- Park et al. (2025)
S. Park, K. Frans, B. Eysenbach, and S. Levine

OGBench: benchmarking offline goal-conditioned RL.

In International Conference on Learning Representations (ICLR),

Cited by: Appendix A.
- Pathak et al. (2017)
D. Pathak, P. Agrawal, A. A. Efros, and T. Darrell

Curiosity-driven exploration by self-supervised prediction.

In Proceedings of the 34th International Conference on Machine Learning (ICML),

Proceedings of Machine Learning Research, Vol. 70, pp. 2778–2787.

Cited by: §6.
- Rubinstein (1999)
R. Y. Rubinstein

The cross-entropy method for combinatorial and continuous optimization.

Methodology and Computing in Applied Probability 1 (2), pp. 127–190.

External Links: [Document](https://dx.doi.org/10.1023/A%3A1010091220143)

Cited by: §2.
- Schrittwieser et al. (2020)
J. Schrittwieser, I. Antonoglou, T. Hubert, K. Simonyan, L. Sifre, S. Schmitt, A. Guez, E. Lockhart, D. Hassabis, T. Graepel, T. Lillicrap, and D. Silver

Mastering Atari, Go, chess and Shogi by planning with a learned model.

Nature 588 (7839), pp. 604–609.

Cited by: §6.
- Schwarzer et al. (2021)
M. Schwarzer, A. Anand, R. Goel, R. D. Hjelm, A. Courville, and P. Bachman

Data-efficient reinforcement learning with self-predictive representations.

In Proceedings of the 9th International Conference on Learning Representations (ICLR),

External Links: [Link](https://openreview.net/forum?id=uCQfPZwRaUu)

Cited by: §2,
§6.
- Sobal et al. (2025)
V. Sobal, W. Zhang, K. Cho, R. Balestriero, T. G. J. Rudner, and Y. LeCun

Stress-testing offline reward-free reinforcement learning: a case for planning with latent dynamics models.

In 7th Robot Learning Workshop: Towards Robots with Human-Level Abilities,

External Links: [Link](https://openreview.net/forum?id=jON7H6A9UU)

Cited by: §5,
§5,
§6.
- Tassa et al. (2018)
Y. Tassa, Y. Doron, A. Muldal, T. Erez, Y. Li, D. d. L. Casas, D. Budden, A. Abdolmaleki, J. Merel, A. Lefrancq, T. Lillicrap, and M. Riedmiller

DeepMind control suite.

arXiv preprint arXiv:1801.00690.

Cited by: Appendix A.
- Wu et al. (2023)
P. Wu, A. Escontrela, D. Hafner, P. Abbeel, and K. Goldberg

DayDreamer: world models for physical robot learning.

In Proceedings of the 6th Conference on Robot Learning (CoRL),

Proceedings of Machine Learning Research, Vol. 205, pp. 2226–2240.

Cited by: §6.
- Zhang et al. (2021)
A. Zhang, R. McAllister, R. Calandra, Y. Gal, and S. Levine

Learning invariant representations for reinforcement learning without reconstruction.

In Proceedings of the 9th International Conference on Learning Representations (ICLR),

External Links: [Link](https://openreview.net/forum?id=-2FCwDKRREu)

Cited by: §6,
§6.
- Zhou et al. (2025)
G. Zhou, H. Pan, Y. LeCun, and L. Pinto

DINO-WM: world models on pre-trained visual features enable zero-shot planning.

In Proceedings of the 42nd International Conference on Machine Learning (ICML),

Proceedings of Machine Learning Research, Vol. 267, pp. 79115–79135.

External Links: [Link](https://proceedings.mlr.press/v267/zhou25t.html)

Cited by: §5,
§5,
§6.

## Appendix

## Appendix A Experimental Details

We provide here the full details necessary to reproduce all experiments.
The implementation is built on top of the open-source LeWM codebase
(23), with auxiliary heads and decision losses added in a
modular fashion that does not modify the base world model architecture.

### Backbone Architecture

The encoder is a Vision Transformer (ViT-Tiny) backbone with patch size
$14$ and image resolution $224\times 224$. The CLS token is projected
to a latent embedding of dimension $192$. The predictor is a $6$-layer
Transformer with $16$ attention heads, head dimension $64$, MLP
dimension $2048$, dropout $0.1$, and embedding dropout $0$. History
length is $3$ frames and the predictor outputs one future latent per
forward pass. All variants share this exact backbone configuration. The
only differences are in the auxiliary head set and the loss weights.

### Auxiliary Head Architecture

Each auxiliary head is a $2$-layer MLP with hidden dimension $256$,
GELU activations, and LayerNorm, followed by a final linear layer to
the prediction target. Concretely:

- •

Inverse-action head: input $[z_{t},z_{t+1}]\in\mathbb{R}^{2\cdot 192}$, output $\hat{a}_{t}\in\mathbb{R}^{d_{a}}$.
- •

Goal-conditioned action head: input $[z_{t},z_{g}]\in\mathbb{R}^{2\cdot 192}$, output $\hat{a}_{t}\in\mathbb{R}^{d_{a}}$.
- •

Reward proxy head: input $[z_{t},a_{t}]$, scalar output.
- •

Value proxy head: input $z_{t}$, scalar output.

Here $d_{a}=5d_{\rm raw}$, where $d_{\rm raw}$ is the per-step action dimension.
For the all-heads configuration we instantiate all four heads. The
canonical no-R/V implementation used for the PushT DA-LeWM checkpoint
instantiates only the inverse-action and goal-action heads
(DecisionHeadsNoRV). For compatibility with the original
cross-environment training jobs, the Reacher, Cube, and TwoRoom
learning-curve checkpoints retain unused reward/value modules but set
both corresponding loss weights to zero. Thus these checkpoints share
the DA-LeWM training objective, although their parameter-initialization
stream is not byte-identical to the no-R/V implementation.

#### Reward and value proxy targets.

For the stored all-heads checkpoint, the implementation used

|  | $\displaystyle r_{t}$ | $\displaystyle=-\|s_{t}^{[:2]}-s_{g}^{[:2]}\|_{2}$ |  | (7) |
|---|---|---|---|---|---|---|
|  |  | $\displaystyle-0.1\,d_{\angle}(s_{t}^{[2]},s_{g}^{[2]}),$ |  |
|  | $\displaystyle V_{t}$ | $\displaystyle=\sum_{k=t}^{T-1}\gamma^{k-t}r_{k},\qquad\gamma=0.99.$ |  |

Here $d_{\angle}$ is wrapped angular distance. Neither target is an environment reward nor optimal-policy value.
Under the stored PushT state layout, these coordinates do not match the
intended block-pose fields, and no corrected all-heads checkpoint was trained.
We therefore retain this result only as an exploratory, implementation-specific ablation. It
cannot determine whether correctly specified R/V supervision helps or hurts, or explain the gap causally.

### Training Configuration

The matched-budget ablations train every variant for one epoch. The
learning-dynamics comparison uses matched 100-epoch training schedules
for LeWM and DA-LeWM and analyzes the first ten saved checkpoints.
Optimizer, data, and model settings are otherwise identical within each
comparison.

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | $5\times 10^{-5}$ |
| Weight decay | $10^{-3}$ |
| Batch size | $128$ |
| Gradient clipping | $1.0$ |
| Mixed precision | bf16 |
| Training schedule | $1$ epoch (ablations), $100$ epochs (first $10$ analyzed) |
| Train/val split | $0.9/0.1$ |
| Training randomness | Matched across variants |
| SIGReg weight $\lambda_{\text{sig}}$ | $0.09$ |
| SIGReg knots | $17$ |
| SIGReg projections | $1024$ |
| Inverse weight $\alpha$ | $0.1$ |
| Goal-action weight $\beta$ | $0.1$ |
| Reward weight (all-heads only) | $0.05$ |
| Value weight (all-heads only) | $0.01$ |
| Reward/value discount $\gamma$ | $0.99$ |
| Goal strategy | clip-final-state |

The objective-level difference between all-heads and DA-LeWM is the
presence or absence of reward/value losses. The implementation detail
above records whether zero-weight modules are retained. The base world
model loss
($\mathcal{L}_{\text{pred}}+\lambda_{\text{sig}}\mathcal{L}_{\text{SIGReg}}$)
is identical across all variants except No-SIGReg, where the SIGReg
term is dropped.

### Computing Infrastructure

Training and evaluation use one NVIDIA A100-SXM4-80GB or
A800-SXM4-80GB GPU per process (80 GB memory, driver 535.129.03) on
x86-64 Linux 5.4 cluster nodes with Intel Xeon Platinum 8362 host CPUs.
The software environment is Python 3.10.20, PyTorch 2.7.1
(cu118), Lightning 2.6.1, Hydra 1.3.2,
stable-worldmodel 0.0.6, stable-pretraining 0.1.6,
dm-control 1.0.40, MuJoCo 3.8.0, OGBench 1.2.1, NumPy 2.2.6,
SciPy 1.15.3, and scikit-learn 1.7.2. Training uses bfloat16 mixed
precision. Diagnostics and summary statistics use float32 model
inference and double-precision host-side aggregation where required.

### Datasets

All data are simulated trajectories from established public
environments. This work introduces no new dataset and uses no human
subjects or personally identifiable information. PushT and TwoRoom
exercise contact-rich manipulation and obstacle-constrained
navigation, Reacher supplies a standard continuous-control test from
the DeepMind Control Suite (30), and Cube supplies
an offline goal-conditioned manipulation task from
OGBench (24). The exact dataset identifiers are:

| Task | Dataset identifier | Provenance |
|---|---|---|
| PushT | pusht_expert_train | LeWM release |
| Reacher | dmc/reacher_random | Control Suite / LeWM |
| Cube | ogbench/cube_single_expert | OGBench / LeWM |
| TwoRoom | tworoom | LeWM release |

Training follows the four-frame, frame-skip-5 construction in
Section 4. The 25-row goal clip below is evaluation-only.
All tasks use the same 0.9/0.1 trajectory-level train/validation split.
The code package records dataset identifiers and preprocessing.

#### Task definitions.

The diagnostic state, raw action dimension, online success rule, and
terminal diagnostic cost are:

| Task | Diagnostic state / cost | $d_{\rm raw}$ | Online success |
|---|---|---|---|
| PushT | agent $xy$, block $xy$/angle, agent velocity (7-D), L2 | 2 | $\|\Delta(\text{agent }xy,\text{block }xy)\|_{2}<20$ and wrapped $|\Delta\text{ block angle}|<\pi/9$ |
| TwoRoom | agent $xy$, L2 | 2 | $\|\Delta xy\|_{2}<16$ pixels |
| Reacher | 2-D joint position, L2 | 2 | $\|\Delta q\|_{\infty}<0.05$ radians |
| Cube | cube $xyz$, L2 | 5 | $\|\Delta xyz\|_{2}\leq 0.04$ m |

Here $d_{\rm raw}$ is the per-raw-step action dimension. Each model action block
concatenates five raw actions. All listed L2 costs are unweighted,
simulator-only terminal distances and are not exposed to the planner.

### Evaluation Configuration

Online evaluation is performed by initializing $50$ vectorized
environments per evaluation seed and running CEM-based planning. Each
reported mean aggregates $3\times 50=150$ episodes. Short-budget
ablations and ten-checkpoint learning curves each use $3$ evaluation
seeds per task. Task-specific sweeps are evaluated separately and are
not pooled.

| Parameter | Value |
|---|---|
| Episodes per evaluation seed | $50$ |
| Short-budget evaluation repeats | $3$ seeds per task |
| Learning-curve evaluation repeats | $3$ seeds per task, task sweeps separate |
| Goal clip (goal_offset_steps) | $25$ rows (endpoint $+24$) |
| Evaluation horizon (eval_budget) | $50$ raw steps |
| Planning horizon $H$ | $5$ action blocks |
| Receding-horizon stride | $5$ action blocks |
| Action block size (frame-skip) | $5$ raw actions |
| CEM samples $N$ (online) | $300$ |
| CEM iterations $K$ (online) | $30$ |
| CEM elite size $E$ (online) | $30$ |
| CEM proposal variance scale | $1.0$ |

The dataset loader uses the end-exclusive slice
$[\text{start},\text{start}+25)$, so the goal is the row at index
$\text{start}+24$. The five-block planning and receding horizons each
execute $25$ raw actions. These parameters are identical across
environments.
Environment-specific callables (state setters, goal setters) are
implemented as configured in the codebase and are not changed across
variants.

### Plan-Real and CEM-stage Spearman

Diagnostic Spearman correlations use a separate evaluation procedure
that reuses the same trained checkpoints. We sample $n=30$ (start,
goal) pairs per checkpoint from the held-out validation split. For each
pair we draw $N=64$ candidate action sequences from $\mathcal{N}(0,I)$ in normalized action-block space, compute latent costs from the
world model, and roll out each candidate in the environment to compute
real costs. For PushT, both Plan-Real and CEM-stage diagnostics use
$c_{\text{real}}=\|s_{H}-s_{g}\|_{2}$ on the unweighted seven-dimensional
simulator state (agent position, block position and angle, and agent
velocity), recorded as pusht_full_state_l2_v1.

CEM-stage Spearman uses the same CEM trajectory as the online MPC
controller ($N\!=\!300$ samples $\times$ $30$ iterations with the
top $30$ as the elite set, $n_{\text{pairs}}\!=\!15$) and reports
Spearman at iteration $0$ (random), iteration $15$ (mid), and the
last iteration (elite). At the elite stage each pair’s Spearman is
computed over the top-$30$ candidates of the converged CEM
distribution. At the random/mid stages it is computed over $\min(N,\max(2K,64))\!=\!64$
candidates uniformly subsampled from the $N\!=\!300$ population.
This budget is identical for every checkpoint reported in
Table 3 and Appendix C.
All Spearman calculations use average ranks for exact ties. Diagnostic
artifacts also record defined and undefined pair counts.

### Information Probes

Linear probes are fit on frozen latents from the trained world model.
Each probe is an affine ridge regressor (closed-form least squares with
$\ell_{2}$ coefficient $10^{-4}$ and an intercept). No hidden layer or
nonlinear activation is used. The exact input–target pairs are:
$z_{t}\!\mapsto\!s_{t}$ (state),
$[z_{t},z_{t+1}]\!\mapsto\!a_{t}$ (inverse action),
$[z_{t},z_{g}]\!\mapsto\!a_{t}$ (goal action),
$[z_{t},a_{t}]\!\mapsto\!r_{t}$ (reward proxy), and
$z_{t}\!\mapsto\!V_{t}$ (value proxy). The R/V targets are the legacy
proxies defined above. Probe $R^{2}$ is reported on a fixed held-out
$20\%$ split. The probe definition and split are identical across model
variants, so differences in $R^{2}$ reflect properties of the frozen
latent rather than probe capacity.

## Appendix B Geometric Signature of Inverse-Action Training

We measure the global rank correlation between latent displacement
magnitude $\|\Delta z\|$ and action magnitude $\|a\|$ over $38{,}400$
held-out transitions. Inverse-action training changes a near-zero
baseline correlation into a moderate positive correlation, a
falsifiable empirical signature of Claim 1. We do
not assert this is a logical consequence of the inverse loss alone:
the inverse head can in principle recover actions through directional
not norm structure. The figure reports what we observe under the actual
training procedure.

*Figure 5: Global Spearman correlation between $\|\Delta z\|$
and $\|a\|$ on $38{,}400$ held-out PushT transitions. Values are
read directly from the saved diagnostic output. No synthetic points
are shown.*

## Appendix C Additional Ablations

### Per-Method Probes and Online Success on PushT

Figure 3 (main text) plots the four
non-degenerate methods. Table 6 below restates
the same numbers and additionally lists the No-SIGReg degenerate
baseline as a sanity check. Probes detect the outright information
loss of No-SIGReg ($R^{2}$ negative or near zero) but do not separate
the four methods that train without representation collapse.

| Model | Success | State $R^{2}$ | Action $R^{2}$ | Goal $R^{2}$ |
|---|---|---|---|---|
| LeWM | $49.3\pm 12.2$ | $0.90$ | $0.86$ | $0.78$ |
| No-SIGReg | $\phantom{0}2.0\pm\phantom{0}2.0$ | $-6.14$ | $0.14$ | $-0.67$ |
| Inverse-only | $64.0\pm\phantom{0}7.2$ | $0.89$ | $0.88$ | $0.77$ |
| All-heads | $71.3\pm\phantom{0}4.2$ | $0.89$ | $0.88$ | $0.78$ |
| DA-LeWM | $\mathbf{92.7\pm\phantom{0}1.2}$ | $\mathbf{0.90}$ | $\mathbf{0.89}$ | $\mathbf{0.80}$ |

*Table 6: Information probes and online success on PushT
($3$ evaluation seeds). Across the four non-collapsed methods, probe scores
change by less than $0.03$ in $R^{2}$ while online success spans
$43$ percentage points. No-SIGReg is shown as a degenerate sanity
check.*

### Goal-Action Weight Sensitivity

We sweep the goal-action weight $\beta$ on PushT online success and
report Plan-Real / CEM-stage Spearman for $\beta=0.3$ alongside
$\beta=0.1$.

| $\beta$ | PushT online (%) |
|---|---|
| $0.03$ | $63.3\pm 12.2$ |
| $\mathbf{0.1}$ | $\mathbf{92.7\pm\phantom{0}1.2}$ |
| $0.3$ | $89.3\pm\phantom{0}3.1$ |

#### Spearman at $\beta=0.3$ on PushT.

Plan-Real Spearman is
$+0.448\pm 0.207$ ($29/30$ positive pairs), and CEM-stage Spearman
is $(\text{random},\text{mid},\text{elite})=(+0.549,+0.232,-0.038)$. Relative to LeWM, the Plan-Real and random-stage gains are
significant (paired $p\!=\!0.002$ for both), whereas mid and elite are
not ($p\!=\!0.90/0.33$).

#### Additional same-budget initialization.

A separately initialized $\beta=0.1$ checkpoint gives Plan-Real
Spearman $+0.436\pm 0.224$ ($29/30$ positive pairs) and CEM-stage
Spearman $(+0.542,+0.237,+0.028)$ at random/mid/elite. Its Plan-Real
and random-stage gains over LeWM are significant (paired
$p\!<\!0.001$ and $p\!=\!0.002$), whereas mid and elite are not
($p\!=\!0.85/0.91$). Together, the two additional checks reproduce
global alignment improvement and shared local saturation.

#### Cross-environment behavior.

On Reacher, $\beta=0.3$
regresses to $78.0\%$ (vs. $84.0\%$ at $\beta=0.1$ and $82.0\%$
LeWM baseline). Reacher per-variant Spearman numbers including
$\beta=0.3$ are reported in Appendix G. The
Reacher regression motivates the conservative cross-environment
default $\beta=0.1$.

## Appendix D Scope of the Sufficient-Condition Diagnostics

Proposition 1 is a sufficient-condition analysis,
not a claim that its constants are identified by the training losses.
In particular, empirical average prediction loss does not establish
the pointwise terminal-rollout constant $\epsilon_{H}$, and finite held-out pairs
cannot certify global bi-Lipschitz constants $\mu_{f}$ and $L_{f}$.
We therefore do not attach numerical estimates to those constants.
Instead, the paper tests observable implications at three levels:
SIGReg collapse via latent-cost dynamic range and tie-aware Plan-Real
Spearman on PushT, TwoRoom, and Reacher (main text), action-conditioned local geometry via
Figure 5, and the rank-preservation implication via the
soft-margin check in Supplementary Section F. These measurements are
diagnostics consistent with the analysis, not proofs that its
assumptions hold out of distribution.

## Appendix E Elite-Neighborhood Local Latent Geometry

This appendix details the elite-neighborhood diagnostic referenced in
Section 5. The diagnostic was originally designed to
discriminate between two candidate mechanisms for an apparent elite-stage
Spearman gap:
(i) gradients from the R/V losses distort the local latent metric in
the elite neighborhood (gradient interference), or
(ii) R/V heads encode global task-progress scalars that are
uninformative once candidates are near-optimal but do not distort
local geometry (global-progress encoding).
After the same-budget re-evaluation in Table 3
and Supplementary Section C showed that elite-stage Spearman is
uniformly weak across the five main variants and two additional DA-LeWM
checks (with paired tests vs LeWM all non-significant), the elite-stage
gap no longer requires a mechanistic explanation. The
diagnostic results below remain useful: they provide direct evidence
that all four head configurations share the same weakly-informative
local latent metric in the elite neighborhood
($\mathrm{Pearson}(\log d^{z},\log d^{s})\approx 0.08$ across variants),
which independently corroborates the elite-stage uniformity reported
in Table 3.

#### Procedure.

For each of $n\!=\!15$ held-out
$(\text{start},\text{goal})$ pairs from the PushT validation set we
run CEM with $300$ candidates, $30$ iterations and elite size
$E\!=\!K\!=\!30$ (matching the planner setting) and retain the
final-iteration elite set. For every elite candidate $i$ we record

- •

$z_{i}\in\mathbb{R}^{192}$, the predicted final-step embedding
produced by the JEPA rollout (the latent the planner scores against
$z_{g}$), and
- •

$s_{i}\in\mathbb{R}^{7}$, the real PushT final-step state after
rolling out the action sequence in the simulator.

We then compute pairwise distances within the elite set,
$d^{z}_{ij}=\|z_{i}-z_{j}\|_{2}$ and
$d^{s}_{ij}=\|s_{i}-s_{j}\|_{\Sigma}$, with
$\Sigma=\mathrm{diag}(\sigma^{-2})$ and $\sigma$ the per-coordinate
state standard deviation on the training set (diagonal Mahalanobis). The local geometric ratio is
$\rho_{ij}=d^{z}_{ij}/d^{s}_{ij}$. A locally isotropic latent has
roughly constant $\rho_{ij}$. Gradient interference would produce
direction-dependent contraction with elevated $\rho_{ij}$ variance.

#### Results.

Table 7 reports per-pair
statistics aggregated over the $n\!=\!15$ pairs.

| Variant | $\rho$ mean | $\rho$ CV | Log spread |
|---|---|---|---|
| LeWM (no DS) | $2.13\pm 0.86$ | $0.710\pm 0.200$ | $1.87\pm 0.26$ |
| Inverse-only | $2.90\pm 1.08$ | $0.714\pm 0.172$ | $2.03\pm 0.36$ |
| All-heads | $2.92\pm 1.20$ | $0.685\pm 0.116$ | $1.98\pm 0.41$ |
| DA-LeWM ($\beta\!=\!0.1$) | $2.90\pm 1.04$ | $0.767\pm 0.118$ | $2.12\pm 0.20$ |

| Variant | $\mathrm{Pearson}(\log d^{z},\log d^{s})$ |
|---|---|
| LeWM (no DS) | $+0.092\pm 0.067$ |
| Inverse-only | $+0.080\pm 0.084$ |
| All-heads | $+0.084\pm 0.065$ |
| DA-LeWM ($\beta\!=\!0.1$) | $+0.050\pm 0.066$ |

*Table 7: Elite-neighborhood local latent geometry on PushT
($K\!=\!30$ elites/pair, $n\!=\!15$ pairs). Mean $\pm$ std across
pairs. CV is $\sigma(\rho)/\mu(\rho)$. Log spread is
$\log(p_{95}/p_{5})$.*

Across the four variants, the coefficient of variation spans
$0.685$–$0.767$ and log spread spans $1.87$–$2.12$ in the elite
neighborhood. In particular, all-heads does not differ significantly
from inverse-only, DA-LeWM, or LeWM on either statistic. This provides
no evidence for the R/V-specific gradient-interference hypothesis:
if the proxy R/V losses produced a detectable increase in local
anisotropy, all-heads should differ from the R/V-free configurations.
Table 8 reports the paired contrasts.

| Contrast | $\Delta\mathrm{CV}\ (t)$ | $\Delta$ log spread $(t)$ |
|---|---|---|
| All-heads $-$ Inverse-only | $-0.029\ (-0.73)$ | $-0.049\ (-0.87)$ |
| All-heads $-$ DA-LeWM | $-0.081\ (-1.85)$ | $-0.135\ (-1.09)$ |
| All-heads $-$ LeWM | $-0.024\ (-0.47)$ | $+0.119\ (+0.85)$ |
| DA-LeWM $-$ Inverse-only | $+0.053\ (+0.99)$ | $+0.086\ (+0.87)$ |
| DA-LeWM $-$ LeWM | $+0.057\ (+0.93)$ | $+0.255\ (+2.50)$ |
| Inverse-only $-$ LeWM | $+0.004\ (+0.06)$ | $+0.169\ (+1.34)$ |

*Table 8: Paired comparisons of elite-neighborhood local
geometry on PushT ($n\!=\!15$ pairs, same pair indices across
variants). Positive numbers indicate the first variant has higher
anisotropy. Parentheses report paired $t$. All all-heads contrasts
are non-significant. Among the remaining contrasts, only DA-LeWM versus
LeWM log spread is significant ($p=0.026$).*

The Pearson correlation
$\mathrm{Pearson}(\log d^{z}_{ij},\log d^{s}_{ij})$
is uniformly low ($+0.050$–$+0.092$) across all four variants,
showing weak within-elite association between latent and real-state
distance regardless of which auxiliary heads are present. This is
consistent with the full-state elite-stage results in
Table 3: elite correlations remain near zero and the
paired test against LeWM is non-significant for every main
configuration. The isolated DA-LeWM–LeWM log-spread contrast does not
support an R/V-specific effect because all three contrasts involving
the all-heads checkpoint are non-significant.

#### Implementation.

The diagnostic is implemented in
decision/eval_elite_local_geometry.py and re-uses the CEM
inner loop and dataset / scaler / env wiring of the behavioral
Plan-Real Spearman pipeline. Total wall-clock for the four variants
on a single 80 GB A100/A800 GPU is $\approx\!9$ minutes
($n\!=\!15$ pairs,
$30$ CEM iterations $\times$ $300$ samples per iteration), so the
diagnostic is essentially free relative to the CEM-stage Spearman
evaluation it complements.

## Appendix F Soft-Margin Consistency Check for Corollary 1

#### Procedure.

Implementation:
decision/eval_corollary_margin_check.py. For each PushT
validation pair we sample $N\!=\!64$ random plans (matching the
Plan-Real Spearman protocol), compute $c_{\text{lat}}^{(i)}$ via JEPA
and the full-state PushT cost $c_{\text{real}}^{(i)}=\|s_{H}^{(i)}-s_{g}\|_{2}$
via environment rollout, fit
$\kappa\!=\!\sum_{i}c_{\text{lat}}^{(i)}c_{\text{real}}^{(i)}/\sum_{i}(c_{\text{real}}^{(i)})^{2}$ (OLS through origin), and define
residuals $\eta_{i}\!=\!|c_{\text{lat}}^{(i)}\!-\!\kappa c_{\text{real}}^{(i)}|$.
The soft margin condition is
$\kappa\,\Delta_{ij}\!>\!\eta_{i}\!+\!\eta_{j}$ for the ordered pair with
$c_{\text{real}}^{(i)}\!<\!c_{\text{real}}^{(j)}$.
We define $p$ as the fraction of such distinct-real-cost pairs satisfying
the condition. Kendall’s $\tau_{a}$ is the mean concordance sign over all
unordered candidate pairs, with tied pairs contributing zero. If $q$ is
the fraction of pairs with distinct real costs, the tie-aware bound is
$\tau_{a}\geq q(2p-1)$. Here $q=1$ for the reported full-state PushT pairs.

#### Results.

Table 9 reports per-variant
means over $n\!=\!20$ pairs.

| Variant | $p$ | $\rho_{s}$ | $\tau_{a}$ |
|---|---|---|---|
| LeWM (no DS) | $0.288\pm 0.073$ | $+0.338\pm 0.206$ | $+0.239\pm 0.148$ |
| No-SIGReg | $0.221\pm 0.052$ | $+0.053\pm 0.209$ | $+0.037\pm 0.142$ |
| Inverse-only | $0.314\pm 0.088$ | $+0.448\pm 0.207$ | $+0.324\pm 0.154$ |
| All-heads | $0.317\pm 0.093$ | $+0.457\pm 0.236$ | $+0.331\pm 0.178$ |
| DA ($\beta\!=\!0.1$) | $0.325\pm 0.091$ | $+0.479\pm 0.228$ | $+0.347\pm 0.174$ |

| Variant | $2p\!-\!1$ | $\kappa$ |
|---|---|---|
| LeWM (no DS) | $-0.424\pm 0.146$ | $+1.93$ |
| No-SIGReg | $-0.557\pm 0.103$ | $+0.003$ |
| Inverse-only | $-0.371\pm 0.176$ | $+2.28$ |
| All-heads | $-0.367\pm 0.187$ | $+2.32$ |
| DA ($\beta\!=\!0.1$) | $-0.351\pm 0.182$ | $+2.34$ |

*Table 9: Corollary 1 soft-margin consistency check on
PushT ($N\!=\!64$ random plans/pair, $n\!=\!20$ pairs, mean
$\pm$ std across pairs). $\tau_{a}\geq 2p-1$ holds for every pair.*

Across the five variant means, $p$ tracks $\rho_{s}$ with Pearson
correlation $+0.9996$ (Spearman rank $+1.0$). Pooling all
$5\!\times\!20\!=\!100$ (variant, pair) points gives
Pearson$(p,\rho_{s})=+0.895$ ($p=3.54\!\times\!10^{-36}$). The Kendall
lower bound $\tau_{a}\!\geq\!2p\!-\!1$ holds for all $100$ pairs. The
empirical gap $\tau_{a}-(2p-1)$ is $0.670\pm 0.083$. Thus the soft condition
tracks ranking consistently while remaining a conservative, loose
lower bound.

## Appendix G Cross-Environment Spearman Scope

This appendix collects the behavioral Plan-Real Spearman and CEM-stage
Spearman measurements on Reacher and explains why the same summary is
not meaningful on contact-sparse Cube, supporting the cross-environment
discussion in Section 5.
Table 2 and Table 3 in the
main body cover PushT in detail.

### Reacher

Reacher exhibits a high baseline alignment (LeWM $\mathrm{Sp}=+0.504$). Auxiliary objectives leave behavioral Spearman essentially
unchanged, consistent with limited empirical headroom in this
diagnostic. These measurements do not estimate the constants or
tightness of Proposition 1.

| Variant | Behavior | Random | Mid | Elite |
|---|---|---|---|---|
| LeWM | $+0.504\pm 0.244$ | $+0.468$ | $+0.446$ | $+0.220$ |
| No-SIGReg | $+0.001\pm 0.135$ | $-0.015$ | $-0.018$ | $-0.045$ |
| Inverse-only | $+0.506\pm 0.254$ | $+0.486$ | $+0.457$ | $+0.111$ |
| DA ($\beta\!=\!0.1$) | $+0.523\pm 0.217$ | $+0.492$ | $+0.460$ | $\mathbf{+0.231}$ |
| DA ($\beta\!=\!0.3$) | $+0.514\pm 0.242$ | $+0.486$ | $+0.467$ | $+0.097$ |

The variation across alignment-improving variants is within sampling
noise ($n=30$ pairs, std $\sim 0.22$), and online success similarly
varies by less than $3$ percentage points across the same variants.

### Cube: Tie-Dominated Real Costs

Random Cube plans often fail to contact the object. Consequently, many
candidates have exactly the same terminal object position and real
cost. Some $64$-candidate populations contain a tie block of size $63$,
and fully constant real-cost vectors also occur in both behavioral and
CEM-stage populations. Average-rank Spearman is undefined for a
constant vector and is low-support when only one or two distinct costs
remain.

We therefore do not report Cube Spearman as a cross-variant metric.
The appropriate conclusion is narrower: random-plan rank agreement is
not informative in this contact-sparse regime. Cube comparisons in the
paper use online success, while the tied-cost frequency is itself
reported as a limitation of Plan-Real and CEM-stage diagnostics.

## Appendix H Training Dynamics and Checkpoint Selection

We evaluate checkpoints from epochs 1 through 10 of matched 100-epoch
schedules for LeWM and DA-LeWM. Training randomness is matched across
the two methods, and online evaluation uses $50$ episodes for each of
$3$ evaluation seeds per task. Task-specific sweeps are evaluated
separately and are not pooled. Intermediate checkpoints are
non-monotonic, especially on Reacher and Cube. We therefore use epoch
10 as a common, pre-specified endpoint and do not select a separate
best epoch for each method. The complete per-epoch means and sample
standard deviations underlying Figure 4 are
included in the accompanying results file.

