Title: SIEVE: Structure-Aware Data Selection for Imitation Learning with VLA Models

URL Source: https://arxiv.org/html/2607.06442

Markdown Content:
Changti Wu 1,2, Bin Yu 3,2 1 1 footnotemark: 1, Zhaolong Shen 4,2 1 1 footnotemark: 1, Shijie Lian 5,2, Xiaopeng Lin 6,8, 

Cong Huang 7, Zhirui Zhang 8, Lei Zhang 1, Kai Chen 7,2,8 2 2 footnotemark: 2

###### Abstract

Vision-Language-Action (VLA) models are typically trained by imitation learning on large-scale robot demonstration datasets, but more data does not necessarily yield better policies due to redundancy, noise, and uneven coverage. Existing data selection methods often assess demonstrations at either the trajectory or state-action level, missing the reusable structures that compose long-horizon behaviors. In this paper, we propose SIEVE, a structure-aware data selection method for VLA imitation learning. SIEVE views demonstrations as compositions of reusable primitives and transition interfaces. It first discovers visuo-motor primitives from segmented trajectories, then allocates selection budgets to composition patterns by maximizing reuse-aware structural exposure under diminishing returns. Finally, it selects medoid trajectories within each composition-pattern bucket to retain central, stable, and imitation-friendly demonstrations. Experiments across multiple datasets, benchmarks, and VLA models show that SIEVE consistently outperforms competitive data selection baselines. Notably, SIEVE can surpass full-data training while using only 50% of demonstrations and 50% of training steps, suggesting that reusable structure, captured through primitives and transitions, is an important signal for efficient VLA imitation learning.

Code — https://github.com/ChangtiWu/SIEVE

## Introduction

Vision-Language-Action (VLA) models have emerged as a scalable paradigm for robotic control, typically acquiring manipulation skills through imitation learning (IL) over large-scale demonstrations (Zitkovich et al.[2023](https://arxiv.org/html/2607.06442#bib.bib1 "Rt-2: vision-language-action models transfer web knowledge to robotic control"); O’Neill et al.[2024](https://arxiv.org/html/2607.06442#bib.bib2 "Open x-embodiment: robotic learning datasets and rt-x models: open x-embodiment collaboration 0"); Kim et al.[2024](https://arxiv.org/html/2607.06442#bib.bib3 "Openvla: an open-source vision-language-action model"); Black et al.[2024](https://arxiv.org/html/2607.06442#bib.bib5 "π0: A vision-language-action flow model for general robot control"); Intelligence et al.[2025](https://arxiv.org/html/2607.06442#bib.bib6 "π0.5: A vision-language-action model with open-world generalization")). However, the rapid growth of robot demonstration datasets does not automatically translate into better policies. In practice, such datasets often contain substantial trajectory redundancy, noisy human demonstrations, suboptimal behaviors, and uneven task coverage (Sathyanarayan et al.[2025](https://arxiv.org/html/2607.06442#bib.bib14 "Quality over quantity: curating contact-based robot datasets improves learning"); Belkhale et al.[2023](https://arxiv.org/html/2607.06442#bib.bib15 "Data quality in imitation learning"); Lin et al.[2025a](https://arxiv.org/html/2607.06442#bib.bib16 "Data scaling laws in imitation learning for robotic manipulation"); Xing et al.[2025](https://arxiv.org/html/2607.06442#bib.bib10 "Shortcut learning in generalist robot policies: the role of dataset diversity and fragmentation")). Training on unfiltered data can repeatedly expose the model to near-duplicate behaviors while also propagating inconsistent or low-quality supervision. These issues make data selection an increasingly important problem for VLA imitation learning: given a large demonstration pool, we aim to retain a compact subset that is more beneficial for policy learning.

![Image 1: Refer to caption](https://arxiv.org/html/2607.06442v1/x1.png)

Figure 1:  Motivation of SIEVE. Training demonstrations contain recurring primitive-transition patterns that can be viewed as reusable behavioral subprograms. Inspired by the MDL principle, SIEVE aims to select demonstrations that expose such reusable structures, enabling the policy to internalize shared subprograms. 

Existing data selection methods for imitation learning typically curate demonstrations by estimating sample utility at different granularities. One line relies on trajectory-level signals, such as trajectory-representation similarity for redundancy removal, demonstration reliability, or downstream task feedback (Dass et al.[2025](https://arxiv.org/html/2607.06442#bib.bib21 "Datamil: selecting data for robot imitation learning with datamodels"); Chen et al.[2025](https://arxiv.org/html/2607.06442#bib.bib22 "Curating demonstrations using online experience"); Zhang et al.[2025](https://arxiv.org/html/2607.06442#bib.bib19 "Scizor: a self-supervised approach to data curation for large-scale imitation learning"); Xu et al.[2026](https://arxiv.org/html/2607.06442#bib.bib24 "ATHENA: accelerated multi-task heterogeneous influence functions for robot data curation")). While these signals provide a global view of demonstration utility, they may collapse a long-horizon trajectory into a single score and obscure which internal stages or behavior compositions are useful; moreover, feedback-based methods often require costly additional model training. Another line estimates utility from state-action-level signals, such as state-action mutual information, task progress, or joint state-action similarity (Zhang et al.[2025](https://arxiv.org/html/2607.06442#bib.bib19 "Scizor: a self-supervised approach to data curation for large-scale imitation learning"); Hejna et al.[2025](https://arxiv.org/html/2607.06442#bib.bib20 "Robot data curation with mutual information estimators")). These fine-grained signals can capture local predictability or redundancy, but they are often either too myopic to characterize coherent long-horizon task semantics or primarily designed for local pruning. This creates a granularity mismatch: effective IL data selection requires evidence that is coarser than individual state-action pairs, yet more structured than holistic trajectory scores.

Inspired by the Minimum Description Length (MDL) principle (Barron et al.[1998](https://arxiv.org/html/2607.06442#bib.bib29 "The minimum description length principle in coding and modeling"); Rissanen [2004](https://arxiv.org/html/2607.06442#bib.bib27 "Minimum description length principle"); Grünwald [2007](https://arxiv.org/html/2607.06442#bib.bib28 "The minimum description length principle")), we view useful demonstrations as those that expose reusable behavioral regularities. In its two-part form, MDL seeks a description that minimizes the cost of encoding both the model and the data:

L(x)=\min_{H\in\mathcal{H}}\left[L(H)-\log P(x\mid H)\right],(1)

where L(H) is the number of bits required to encode the model H, and -\log P(x\mid H) is the number of bits required to encode the data x. This formulation operationalizes Occam’s razor: a model is penalized unless it yields a shorter description of the data. When repeating patterns exist, they can be stored in the model rather than redundantly encoded in each data instance. This view implies that a learner tends to compress data by absorbing repeating patterns into its parameters, so as to encode more data under limited parameters and computation budgets. Useful data, therefore, is not merely abundant or locally predictable, but rich in extractable structure (i.e., non-random regularities that a bounded learner can internalize and reuse as shared behavioral subprograms) (Finzi et al.[2026](https://arxiv.org/html/2607.06442#bib.bib30 "From entropy to epiplexity: rethinking information for computationally bounded intelligence")). For robotic imitation learning, such structure is naturally expressed as primitive composition: a trajectory consists of atomic behavior primitives and transition interfaces. Thus, we estimate trajectory utility by the primitives it exposes, how they are composed, and which transitions support long-horizon execution, yielding a mid-level scoring granularity that captures reusable behavior patterns shared across demonstrations.

In addition, since IL is typically optimized by behavior cloning (BC), selected demonstrations should also provide stable and predictable action supervision. Atypical or noisy realizations of the same behavior pattern may introduce inconsistent actions under similar observations, making the conditional action prediction target harder to fit (Ross and Bagnell [2010](https://arxiv.org/html/2607.06442#bib.bib13 "Efficient reductions for imitation learning"); Hussein et al.[2017](https://arxiv.org/html/2607.06442#bib.bib12 "Imitation learning: a survey of learning methods"); Belkhale et al.[2023](https://arxiv.org/html/2607.06442#bib.bib15 "Data quality in imitation learning")). Therefore, effective data selection should not only decide _which behavioral structures to preserve_, but also _which concrete demonstrations to imitate_. This motivates an IL-friendly selection principle: after identifying useful behavior patterns, one should prefer stable central trajectory realizations that provide more consistent supervision for BC.

Based on these insights, we propose SIEVE, a structure-aware data selection framework for VLA imitation learning. SIEVE first discovers reusable atomic behavior primitives by segmenting trajectories at physically grounded interaction boundaries and clustering segment representations. Each trajectory is then represented as a primitive sequence, which defines a composition pattern and its adjacent transitions. SIEVE allocates the selection budget over composition-pattern buckets to maximize reusable primitive and transition exposure under diminishing returns, and then selects medoid trajectories within each bucket to obtain central, stable, and imitation-friendly realizations. This yields a compact subset that exposes reusable behavioral structure while providing predictable supervision for BC. Our contributions are as follows:

*   •
We propose a primitive-compositional view of trajectory utility, realized by Primitive Discovery and Structural Exposure Allocation, which allocate selection budgets according to reuse-aware primitive and transition exposure under diminishing returns.

*   •
We introduce Learning-Friendly Trajectory Selection, which selects medoid trajectories within each composition-pattern bucket to favor central, stable, and predictable realizations for behavior cloning.

*   •
We present SIEVE, a structure-aware data selection method for VLA imitation learning, and demonstrate its effectiveness across datasets, benchmarks, and models. SIEVE can outperform full-data training using only 50% of demonstrations and 50% of training steps, while consistently improving over competitive baselines under multiple experimental settings.

## Related Work

### Vision-Language-Action Models.

Recent advances in vision-language models (VLMs) (Shao et al.[2025](https://arxiv.org/html/2607.06442#bib.bib38 "Large vlm-based vision-language-action models for robotic manipulation: a survey"); Bai et al.[2025](https://arxiv.org/html/2607.06442#bib.bib36 "Qwen3-vl technical report"); Lin et al.[2025b](https://arxiv.org/html/2607.06442#bib.bib37 "Physbrain: human egocentric data as a bridge from vision language models to physical intelligence")) have accelerated the development of Vision-Language-Action (VLA) models, establishing a scalable paradigm for language-conditioned robotic control (Zitkovich et al.[2023](https://arxiv.org/html/2607.06442#bib.bib1 "Rt-2: vision-language-action models transfer web knowledge to robotic control"); O’Neill et al.[2024](https://arxiv.org/html/2607.06442#bib.bib2 "Open x-embodiment: robotic learning datasets and rt-x models: open x-embodiment collaboration 0"); Kim et al.[2024](https://arxiv.org/html/2607.06442#bib.bib3 "Openvla: an open-source vision-language-action model")). Modern VLAs map visual observations and language instructions to robot actions through generative policy architectures, such as autoregressive action tokenization, diffusion, or flow-matching models (Zitkovich et al.[2023](https://arxiv.org/html/2607.06442#bib.bib1 "Rt-2: vision-language-action models transfer web knowledge to robotic control"); Chi et al.[2025](https://arxiv.org/html/2607.06442#bib.bib4 "Diffusion policy: visuomotor policy learning via action diffusion"); Black et al.[2024](https://arxiv.org/html/2607.06442#bib.bib5 "π0: A vision-language-action flow model for general robot control"); Intelligence et al.[2025](https://arxiv.org/html/2607.06442#bib.bib6 "π0.5: A vision-language-action model with open-world generalization"); Bjorck et al.[2025](https://arxiv.org/html/2607.06442#bib.bib9 "Gr00t n1: an open foundation model for generalist humanoid robots"); Kim et al.[2025](https://arxiv.org/html/2607.06442#bib.bib8 "Fine-tuning vision-language-action models: optimizing speed and success"); Lian et al.[2026](https://arxiv.org/html/2607.06442#bib.bib11 "IntentVLA: short-horizon intent modeling for aliased robot manipulation"); Intelligence et al.[2026](https://arxiv.org/html/2607.06442#bib.bib7 "π0.7: A steerable generalist robotic foundation model with emergent capabilities")). These models are typically trained on large-scale demonstrations via imitation learning (IL), most commonly formulated as behavior cloning (Hussein et al.[2017](https://arxiv.org/html/2607.06442#bib.bib12 "Imitation learning: a survey of learning methods")). However, robotic datasets often contain noisy, redundant, and unevenly distributed demonstrations, so scaling training data indiscriminately can waste computation and yield diminishing returns (Xing et al.[2025](https://arxiv.org/html/2607.06442#bib.bib10 "Shortcut learning in generalist robot policies: the role of dataset diversity and fragmentation"); Sathyanarayan et al.[2025](https://arxiv.org/html/2607.06442#bib.bib14 "Quality over quantity: curating contact-based robot datasets improves learning"); Belkhale et al.[2023](https://arxiv.org/html/2607.06442#bib.bib15 "Data quality in imitation learning"); Lin et al.[2025a](https://arxiv.org/html/2607.06442#bib.bib16 "Data scaling laws in imitation learning for robotic manipulation"); O’Neill et al.[2024](https://arxiv.org/html/2607.06442#bib.bib2 "Open x-embodiment: robotic learning datasets and rt-x models: open x-embodiment collaboration 0")). This motivates selecting compact yet informative subsets for efficient VLA imitation learning.

### Data Selection for Imitation Learning.

Some related studies mainly focus on macroscopic dataset-level evaluation, mixing optimization across data sources and VLM data selection (Xiao et al.[2025](https://arxiv.org/html/2607.06442#bib.bib17 "Data assessment for embodied intelligence"); Belkhale et al.[2023](https://arxiv.org/html/2607.06442#bib.bib15 "Data quality in imitation learning"); Hejna et al.[2024](https://arxiv.org/html/2607.06442#bib.bib18 "Re-mix: optimizing data mixtures for large scale imitation learning"); Wu et al.[2026](https://arxiv.org/html/2607.06442#bib.bib25 "ScalSelect: scalable training-free multimodal data selection for efficient visual instruction tuning"); Zhou et al.[2026](https://arxiv.org/html/2607.06442#bib.bib26 "Synthetic data for multimodal large language models: a lifecycle-oriented survey")). Beyond dataset-level curation, existing IL data selection methods typically estimate sample utility either at the trajectory or state-action level. Trajectory-level methods use global signals such as representation similarity, reliability, downstream task feedback, or validation-loss-based influence estimation (Zhang et al.[2025](https://arxiv.org/html/2607.06442#bib.bib19 "Scizor: a self-supervised approach to data curation for large-scale imitation learning"); Chen et al.[2025](https://arxiv.org/html/2607.06442#bib.bib22 "Curating demonstrations using online experience"); Dass et al.[2025](https://arxiv.org/html/2607.06442#bib.bib21 "Datamil: selecting data for robot imitation learning with datamodels"); Xu et al.[2026](https://arxiv.org/html/2607.06442#bib.bib24 "ATHENA: accelerated multi-task heterogeneous influence functions for robot data curation")), but they often collapse long-horizon demonstrations into a single score and, for feedback-based criteria, require costly additional model training. State-action-level methods instead rely on local signals such as mutual information, task progress, or joint state-action similarity (Hejna et al.[2025](https://arxiv.org/html/2607.06442#bib.bib20 "Robot data curation with mutual information estimators"); Zhang et al.[2025](https://arxiv.org/html/2607.06442#bib.bib19 "Scizor: a self-supervised approach to data curation for large-scale imitation learning"); Yu et al.[2026](https://arxiv.org/html/2607.06442#bib.bib23 "FrameSkip: learning from fewer but more informative frames in vla training")). However, these criteria are often either too myopic to capture coherent task semantics or primarily designed for local pruning.

![Image 2: Refer to caption](https://arxiv.org/html/2607.06442v1/x2.png)

Figure 2: Overview of SIEVE. 1) Primitive Discovery: trajectories are segmented into primitives through representation learning and clustering. 2) Structural Exposure Allocation: a structural exposure objective estimates primitive composition coverage and allocates the selection budget across composition patterns. 3) Learning-Friendly Trajectory Selection: representative trajectories are selected within each composition-pattern bucket to construct a compact, diverse, and learning-friendly training subset.

## SIEVE

We propose SIEVE, a structure-aware data selection method for imitation learning with VLA models. The key idea is to exploit the structural exposure of demonstrations for data selection. To this end, SIEVE first discovers visuo-motor primitives from trajectory segments and represents each trajectory as a composition of these primitives. It then allocates the selection budget based on structural exposure and selects representative, learning-friendly trajectories within each composition pattern. The resulting subset retains informative behavioral structures while improving the efficiency of imitation learning.

### Primitive Discovery

A demonstration trajectory is typically a composition of several reusable visuo-motor segments. This underlying structural prior motivates us to uncover a vocabulary of primitives from trajectories. We define a primitive as a reusable visuo-motor behavior unit discovered from trajectories, which serves as a proxy for a reusable behavioral subprogram.

#### Trajectory segmentation.

Let \mathcal{D}=\{\mathcal{T}_{1},\dots,\mathcal{T}_{N}\} denote the original dataset, where each sample \mathcal{T}_{i} is a demonstration trajectory. For each trajectory \mathcal{T}_{i}\in\mathcal{D}, we segment it using end effector (gripper/dexterous-hand) state flips (i.e., grasp/release flips) as physically grounded interaction boundaries to obtain L_{i} segments \mathcal{T}_{i}=\{S_{i}^{1},\dots,S_{i}^{L_{i}}\}. To suppress spurious boundaries caused by transient actuation jitter, a state transition is accepted only if it persists for five consecutive frames.

#### Segment representation.

We then extract the representation for each segment using a pretrained video encoder (\operatorname{VE}). Specifically, for each segment S_{i}^{j} (j\in\{1,\dots,L_{i}\}), we uniformly sample 8 frames and encode them with V-JEPA2 (Assran et al.[2025](https://arxiv.org/html/2607.06442#bib.bib31 "V-jepa2: self-supervised video models enable understanding, prediction and planning")). As the start, middle, and end frames provide a compact summary of the segment’s state evolution, we concatenate their representations to form a single feature that captures both semantic content and coarse temporal structure. To suppress noise, improve clustering stability, and reduce computational cost, we further reduce the representation to 256 dimensions with PCA:

\displaystyle z_{i}^{j}\leftarrow\displaystyle\operatorname{PCA}(\operatorname{Concat}(\operatorname{VE}(S_{i}^{j})_{\texttt{start}};(2)
\displaystyle\operatorname{VE}(S_{i}^{j})_{\texttt{mid}};\operatorname{VE}(S_{i}^{j})_{\texttt{end}})).

#### Primitive discovery via clustering.

We discover primitives by clustering the segment representations using MiniBatch K-Means (Sculley [2010](https://arxiv.org/html/2607.06442#bib.bib32 "Web-scale k-means clustering")). Rather than manually specifying the number of clusters, we select K automatically on a randomly sampled subset of trajectories. We seek a primitive vocabulary that is both _reusable_ across trajectories and _discriminative_ at the trajectory level: useful primitives should recur across demonstrations, while not collapsing structurally different trajectories into nearly identical primitive composition patterns.

For each candidate K, we first cluster the segment representations from the randomly sampled subset and summarize each trajectory \mathcal{T}_{i} by the set of clusters it covers:

\mathcal{C}_{i}=\{c_{i}^{1},\dots,c_{i}^{|\mathcal{C}_{i}|}\}\subseteq\{1,\dots,K\},(3)

where each c_{i}^{u}\in\{1,\dots,K\} is a cluster index covered by at least one segment in \mathcal{T}_{i}. Based on these trajectory-level cluster sets, we evaluate each candidate vocabulary using a reuse-aware criterion comprising two components. The first, _trajectory-level discriminability_\mathcal{J}, measures how distinct different trajectories remain after being represented by cluster coverage. We compute \mathcal{J} as the median of the average pairwise Jaccard similarities over the cluster sets \mathcal{C}_{i} across all sampled trajectories:

\mathcal{J}=\operatorname{median}_{i=1}^{n}\,\frac{1}{n-1}\sum_{j\neq i}^{n}\operatorname{Jaccard}(\mathcal{C}_{i},\mathcal{C}_{j}).(4)

A lower \mathcal{J} indicates better preservation of structural distinctions between trajectories.

The second component, _cross-trajectory reuse_\mathcal{R}, quantifies how broadly each discovered cluster recurs across trajectories. \mathcal{R} is defined as the median occurrence count of each cluster across all trajectories:

\mathcal{R}=\operatorname{median}_{k=1}^{K}\,\sum_{i=1}^{n}\mathbf{1}[\,k\in\mathcal{C}_{i}\,].(5)

A higher \mathcal{R} implies that the discovered primitives capture widely applicable behavior units.

We choose the number of clusters by maximizing

K^{*}=\arg\max_{K}\left[\bigl(1-\mathcal{J}\bigr)\log\mathcal{R}\right].(6)

This criterion favors primitive vocabularies that are broadly reused across trajectories while still preserving trajectory-level discriminability. After selecting K^{*}, we rerun MiniBatch K-Means on all segment representations, and each resulting cluster is treated as a discovered primitive.

### Structural Exposure Allocation

After primitive discovery, each trajectory \mathcal{T}_{i} is represented by an ordered primitive sequence, referred to as a _composition pattern_:

P_{i}=[c_{i}^{1},\dots,c_{i}^{|P_{i}|}],(7)

where each c_{i}^{m}\in\mathcal{C}_{i} is a discovered primitive.

We further define a _transition_ as the local compositional interface between two adjacent primitives in a composition pattern. For trajectories with |P_{i}|\geq 2, the transitions are

e_{i}^{j}=(c_{i}^{j}\rightarrow c_{i}^{j+1}),\qquad j=1,\dots,|P_{i}|-1.(8)

For trajectories consisting of a single primitive, we introduce a terminal null state \varnothing and define the terminal transition:

e_{i}=(c_{i}^{1}\rightarrow\varnothing),(9)

so that every trajectory contains at least one transition.

Let \mathcal{P}=\{P^{(1)},\dots,P^{(|\mathcal{P}|)}\} denote the set of unique composition patterns in the dataset, where P^{(\ell)} denotes the \ell-th unique composition pattern, corresponding to a pattern bucket containing trajectories with the same composition pattern. This composition-pattern space provides the structural reference for measuring how broadly reusable structures are shared across demonstrations. In SIEVE, we capture such reusable structures through primitive composition, where trajectories expose both atomic behavior primitives and the transition interfaces connecting them. To effectively learn reusable structures, the model should be repeatedly exposed not only to primitives themselves but also to how they are composed through transitions. In our setting, these transition interfaces often coincide with critical state changes (e.g., gripper grasp/release) in the task, and therefore provide informative cues for local behavior progression. Moreover, primitives and transitions that are reused across more composition patterns support a broader range of executable behaviors and should be preferentially preserved.

To retain diverse and important structural information, we allocate the selection budget over the composition-pattern space. Let

\mathcal{B}=[b_{1},\dots,b_{|\mathcal{P}|}](10)

denote the allocation vector, where b_{\ell} is the number of trajectories retained for composition pattern P^{(\ell)}. We optimize the budget allocation \mathcal{B} by maximizing the following structural exposure objective:

F(\mathcal{B})=\sum_{c\in\mathcal{C}}w_{c}\log\bigl(1+n_{c}(\mathcal{B})\bigr)+\sum_{e\in\mathcal{E}}w_{e}\log\bigl(1+n_{e}(\mathcal{B})\bigr),(11)

where \mathcal{C} and \mathcal{E} denote the sets of all discovered primitives and transitions, respectively. Here, n_{c}(\mathcal{B}) and n_{e}(\mathcal{B}) denote the numbers of occurrences of primitive c and transition e among the selected trajectories under allocation \mathcal{B}. The primitive and transition weights are defined by their reuse frequency over the composition-pattern space:

\displaystyle w_{c}\displaystyle=\frac{q_{c}}{|\mathcal{P}|},\qquad q_{c}=\left|\{\,P\in\mathcal{P}:c\in P\,\}\right|,(12)
\displaystyle w_{e}\displaystyle=\frac{q_{e}}{|\mathcal{P}|},\qquad q_{e}=\left|\{\,P\in\mathcal{P}:e\in P\,\}\right|.

Here, q_{c} and q_{e} denote the numbers of composition patterns containing primitive c and transition e, respectively. Consequently, primitives and transitions that participate in more composition patterns receive larger weights. The logarithmic utility introduces diminishing returns, encouraging the budget to expose the model to a broader set of reusable structures rather than repeatedly reinforcing the same ones.

We optimize F(\mathcal{B}) greedily. Starting from \mathcal{B}^{(0)}=\mathbf{0}, each iteration allocates one additional sample to the composition pattern with the largest marginal gain:

\Delta(P^{(\ell)}\mid\mathcal{B})=F(\mathcal{B}+\mathbf{b}_{\ell})-F(\mathcal{B}),(13)

where \mathbf{b}_{\ell} is the one-hot allocation vector that increases the budget of pattern P^{(\ell)} by one. We select

\ell^{\star}=\mathop{\mathrm{arg\,max}}_{\ell\in\{1,\dots,|\mathcal{P}|\}}\Delta(P^{(\ell)}\mid\mathcal{B}),(14)

and update

\mathcal{B}\leftarrow\mathcal{B}+\mathbf{b}_{\ell^{\star}}.(15)

After the budget is exhausted, \mathcal{B} specifies how many trajectories should be retained from each pattern bucket. The actual trajectories are then selected within each pattern bucket in the next stage.

### Learning-Friendly Trajectory Selection

Given the pattern-level budget allocation, this stage selects representative and learning-friendly trajectories within each pattern bucket. Behavior cloning trains the policy by minimizing

\mathcal{L}_{\mathrm{BC}}(\theta)=\mathbb{E}_{(s,a)\sim\mathcal{D}}[-\log\pi_{\theta}(a|s)],(16)

where (s,a) is a state-action pair sampled from the demonstration dataset \mathcal{D}, and \pi_{\theta}(a|s) is the probability assigned by the policy \pi_{\theta} to action a under state s. Demonstrations with more consistent state-action mappings provide clearer supervision and are easier for imitation learning to optimize. Since directly estimating conditional action entropy is impractical, we use representation-space centrality as a practical proxy: trajectories near the center of a composition pattern are less likely to be outliers or ambiguous demonstrations, and thus tend to provide more stable supervision.

For each trajectory \mathcal{T}_{i} with composition pattern P^{(\ell)}, we construct a trajectory representation x_{i} by concatenating the segment representations along its primitive sequence:

x_{i}=\operatorname{Concat}(z_{i}^{1};\dots;z_{i}^{|P_{i}|}).(17)

Within each composition pattern, we compare trajectories using cosine similarity and identify the medoid trajectory \mathcal{T}_{\mathrm{med}}, which has the largest aggregate similarity S_{i}=\sum_{j\neq i}\cos(x_{i},x_{j}) to other trajectories in the same pattern. We then rank trajectories by their distance to the medoid, defined as d_{i}=1-\cos(x_{i},x_{\mathrm{med}}), and retain the \mathcal{B}[\ell] trajectories with the smallest distances, where \mathcal{B}[\ell] is the budget assigned to composition pattern P^{(\ell)} by Structural Exposure Allocation. This selects trajectories closest to the pattern bucket center, yielding representative and learning-friendly demonstrations.

Method Training Steps Stack Green Cube On Yellow Cube Put Carrot On Plate Put Spoon On Table Cloth Put Eggplant In Basket Avg.
Full-Training 50K 22.9 53.1 68.8 62.5 51.8
Selection Budget: 26.5K (50%)
Random 25K (50%)20.8 41.7 64.6 31.3 39.6
DemInf 11.5 37.5 58.3 65.6 43.2
SCIZOR 13.5 36.5 68.8 90.6 52.2
\rowcolor navyblue!10SIEVE (Ours)25.0 54.2 70.8 75.0 56.3
Random 50K (100%)25.0 36.5 66.7 33.3 40.4
DemInf 12.5 40.6 69.8 63.5 46.6
SCIZOR 16.7 39.6 72.9 92.7 55.5
\rowcolor navyblue!10SIEVE (Ours)29.2 57.3 75.0 76.0 59.4
Selection Budget: 37.1K (70%)
Random 35K (70%)16.7 50.0 74.0 37.5 44.6
DemInf 18.8 51.0 78.1 72.9 55.2
SCIZOR 20.8 41.7 72.9 91.7 56.8
\rowcolor navyblue!10SIEVE (Ours)22.9 57.3 81.3 87.5 62.3
Random 50K (100%)12.5 51.0 77.1 46.9 46.9
DemInf 17.7 57.3 79.2 74.0 57.1
SCIZOR 19.8 45.8 76.0 90.6 58.1
\rowcolor navyblue!10SIEVE (Ours)21.9 58.3 86.5 83.3 62.5

Table 1: Performance comparison with baselines on SimplerEnv-WidowX using Qwen3-VL-4B-GR00T. We compare SIEVE with baselines under 50% and 70% selection budgets, at different training steps. “Avg.” denotes the average success rate.

## Experiments

### Experiment Setup

Datasets and Evaluation. Here we evaluate SIEVE on three representative robot imitation learning datasets to assess both its data selection effectiveness and its applicability across different embodiments and environments. Unless otherwise specified, Bridge-V2 with SimplerEnv-WidowX serves as the default training and evaluation setting.

*   •
_Bridge-V2_: We train models on the Bridge-V2 dataset (Walke et al.[2023](https://arxiv.org/html/2607.06442#bib.bib33 "Bridgedata v2: a dataset for robot learning at scale")), a real-world subset of Open X-Embodiment, containing approximately 53K demonstration trajectories collected with a WidowX robot equipped with a parallel gripper. Policies are evaluated in SimplerEnv-WidowX (Li et al.[2024](https://arxiv.org/html/2607.06442#bib.bib34 "Evaluating real-world robot manipulation policies in simulation")), which includes four manipulation tasks: _Stack Cube_, _Put Carrot_, _Put Spoon_, and _Put Eggplant_. Evaluation is conducted under multiple unseen kitchen backgrounds and randomized object configurations, providing a challenging out-of-distribution (OOD) benchmark.

*   •
_Fractal_: We further evaluate SIEVE on the Fractal subset of Open X-Embodiment (O’Neill et al.[2024](https://arxiv.org/html/2607.06442#bib.bib2 "Open x-embodiment: robotic learning datasets and rt-x models: open x-embodiment collaboration 0")), which contains approximately 87K real-world manipulation trajectories collected with a Google Robot manipulator. Policies are evaluated in SimplerEnv-GoogleRobot (Li et al.[2024](https://arxiv.org/html/2607.06442#bib.bib34 "Evaluating real-world robot manipulation policies in simulation")) on three manipulation tasks: _Grasp Coke Can_, _Move Near_, and _Close/Open Drawer_.

*   •
_GR00T-X-Sim_: We additionally evaluate SIEVE on the downsampled Humanoid Robot Tabletop Manipulation subset of GR00T-X-Embodiment-Sim (Bjorck et al.[2025](https://arxiv.org/html/2607.06442#bib.bib9 "Gr00t n1: an open foundation model for generalist humanoid robots")), which contains 24K simulated demonstration trajectories collected with a humanoid robot equipped with dexterous hands. Policies are evaluated in RoboCasa-GR1 (Nasiriany et al.[2024](https://arxiv.org/html/2607.06442#bib.bib35 "Robocasa: large-scale simulation of everyday tasks for generalist robots")), a benchmark comprising 24 tabletop manipulation tasks across diverse scene layouts and object configurations.

Models. We evaluate SIEVE on two representative VLA models, Qwen3-VL-4B-GR00T and Qwen3-VL-4B-OFT. Unless otherwise specified, all main experiments and ablation studies are conducted using Qwen3-VL-4B-GR00T.

*   •
_Qwen3-VL-4B-GR00T_(Bjorck et al.[2025](https://arxiv.org/html/2607.06442#bib.bib9 "Gr00t n1: an open foundation model for generalist humanoid robots")): This model combines the Qwen3-VL-4B vision-language backbone (Bai et al.[2025](https://arxiv.org/html/2607.06442#bib.bib36 "Qwen3-vl technical report")) with a GR00T-style flow-matching policy head that predicts continuous robot actions through conditional flow matching.

*   •
_Qwen3-VL-4B-OFT_(Kim et al.[2025](https://arxiv.org/html/2607.06442#bib.bib8 "Fine-tuning vision-language-action models: optimizing speed and success")): This model uses the Qwen3-VL-4B backbone and adopts the OpenVLA-OFT action decoding recipe, which performs parallel continuous action prediction with an MLP-based prediction head trained using an L1 regression objective.

Baselines. We compare SIEVE with the following baselines. All experiments use the same training hyperparameters for a fair comparison (see Appendix for details):

*   •
_Full-Training_: trains on the complete dataset without data selection and serves as the full-data reference.

*   •
_Random_: uniformly samples demonstrations from the original training set.

*   •
_DemInf_(Hejna et al.[2025](https://arxiv.org/html/2607.06442#bib.bib20 "Robot data curation with mutual information estimators")): selects demonstrations according to state-action mutual information estimates.

*   •
_SCIZOR_(Zhang et al.[2025](https://arxiv.org/html/2607.06442#bib.bib19 "Scizor: a self-supervised approach to data curation for large-scale imitation learning")): filters low-quality data by identifying redundant trajectories and suboptimal state-action pairs.

### Main Results

Table[1](https://arxiv.org/html/2607.06442#Sx3.T1 "Table 1 ‣ Learning-Friendly Trajectory Selection ‣ SIEVE ‣ SIEVE: Structure-Aware Data Selection for Imitation Learning with VLA Models") summarizes the main results on Bridge-V2 using Qwen3-VL-4B-GR00T. We evaluate two selection budgets, 50% and 70%, corresponding to 26.5K and 37.1K demonstrations, respectively. For each budget, we consider two training schedules. The first scales the number of training steps proportionally to the selected data size (25K and 35K steps for the 50% and 70% budgets, respectively), reducing both training data and computation. The second trains each selected subset for the same 50K training steps as Full-Training. Comparing these two settings allows us to disentangle improvements brought by data selection from those potentially arising from increased optimization per sample under a fixed training budget.

As shown in Table[1](https://arxiv.org/html/2607.06442#Sx3.T1 "Table 1 ‣ Learning-Friendly Trajectory Selection ‣ SIEVE ‣ SIEVE: Structure-Aware Data Selection for Imitation Learning with VLA Models"), SIEVE consistently achieves the highest average success rate across all selection budgets and training schedules. Notably, using only 50% of the demonstrations and 25K training steps, SIEVE attains an average success rate of 56.3%, outperforming Full-Training (51.8%), which uses the complete dataset and twice the training steps. This result indicates that SIEVE is able to identify a compact subset that is more effective for imitation learning while substantially reducing both training data and computation.

Compared with existing data selection baselines, SIEVE consistently delivers the best performance. Under the 50% selection budget, SIEVE achieves average success rates of 56.3% and 59.4% under the proportional and 50K training schedules, outperforming SCIZOR (52.2% and 55.5%, respectively). Under the 70% budget, SIEVE further achieves average success rates of 62.3% and 62.5% with 35K and 50K training steps, respectively, again outperforming all competing methods. SIEVE also exhibits consistently stronger per-task performance. It outperforms Random on every task under all evaluation settings, suggesting that the performance gains arise from selecting structurally informative demonstrations rather than simply reducing the training set size. Compared with SCIZOR, SIEVE produces more balanced performance across tasks. While SCIZOR performs particularly well on _Put Eggplant In Basket_, its performance drops noticeably on _Stack Green Cube On Yellow Cube_ and _Put Carrot On Plate_. In contrast, SIEVE maintains competitive performance across all four tasks, resulting in the best overall average success rate.

Since SimplerEnv evaluates policies under substantial visual and environmental variations relative to Bridge-V2, these results further demonstrate that the subsets selected by SIEVE generalize well under distribution shift.

### Extended Results

Method Bridge-V2 Fractal GR00T-X-Sim
Full-Training 51.8 75.0 52.7
Random 39.6 55.6 53.5
DemInf 43.2 67.4 53.8
SCIZOR 52.2 71.9 54.2
\rowcolor navyblue!10SIEVE (Ours)56.3 76.4 54.8

Table 2: Performance across different datasets using Qwen3-VL-4B-GR00T. Full-Training uses the complete dataset, while all selection methods use 50% of the data and 50% of the training steps. We report average success rates (%).

Performance Across Different Datasets. We further evaluate SIEVE across different training datasets and evaluation benchmarks. For Bridge-V2, Full-Training is trained for 50K steps, whereas for Fractal and GR00T-X-Sim, Full-Training is trained for 100K steps. For all data selection methods, we retain 50% of the training data and train for 50% of the corresponding Full-Training steps (25K for Bridge-V2 and 50K for Fractal and GR00T-X-Sim). As shown in Table[2](https://arxiv.org/html/2607.06442#Sx4.T2 "Table 2 ‣ Extended Results ‣ Experiments ‣ SIEVE: Structure-Aware Data Selection for Imitation Learning with VLA Models") (see the Appendix for detailed results), SIEVE achieves the highest average success rate across all three settings. On Bridge-V2 and Fractal, SIEVE achieves average success rates of 56.3% and 76.4%, outperforming Full-Training (51.8% and 75.0%, respectively), while using only half of the training data and half of the training steps. On GR00T-X-Sim, Random selection already slightly outperforms Full-Training. One possible explanation is that both GR00T-X-Sim and RoboCasa-GR1 are simulated tabletop manipulation environments, making RoboCasa-GR1 relatively in-domain for GR00T-X-Sim. In this setting, training on a smaller subset may provide more effective optimization per demonstration. Nevertheless, SIEVE still achieves the highest average success rate among all methods. These results demonstrate that SIEVE generalizes effectively across different datasets, robot embodiments, and evaluation benchmarks.

Method Stack Green Cube On Yellow Cube Put Carrot On Plate Put Spoon On Table Cloth Put Eggplant In Basket Avg.
Qwen3-VL-4B-GR00T
Full-Training 22.9 53.1 68.8 62.5 51.8
Random 20.8 41.7 64.6 31.3 39.6
\rowcolor navyblue!9SIEVE (Ours)25.0 54.2 70.8 75.0 56.3
Qwen3-VL-4B-OFT
Full-Training 21.9 25.0 45.8 62.5 38.8
Random 9.4 33.3 16.6 45.8 26.3
\rowcolor navyblue!9SIEVE (Ours)21.9 58.3 50.0 95.8 56.5

Table 3: Performance across different models on SimplerEnv-WidowX. Full-Training uses the complete dataset with 50K training steps, while all selection methods use 50% of the data and 50% of the training steps.

Performance Across Different Models. We further evaluate whether SIEVE-selected data remains effective across different VLA models. Since this experiment aims to evaluate model generalization rather than compare all data selection methods, we compare SIEVE with Full-Training and Random under the same 50% selection budget. As shown in Table[3](https://arxiv.org/html/2607.06442#Sx4.T3 "Table 3 ‣ Extended Results ‣ Experiments ‣ SIEVE: Structure-Aware Data Selection for Imitation Learning with VLA Models"), SIEVE consistently achieves the highest average success rate on both Qwen3-VL-4B-GR00T and Qwen3-VL-4B-OFT. On Qwen3-VL-4B-GR00T, SIEVE achieves an average success rate of 56.3%, outperforming both Random (39.6%) and Full-Training (51.8%). On Qwen3-VL-4B-OFT, SIEVE further achieves an average success rate of 56.5%, outperforming Random (26.3%) and Full-Training (38.8%). The gains are particularly evident on tasks such as _Put Carrot On Plate_ and _Put Eggplant In Basket_, where SIEVE-selected data consistently achieves substantially higher success rates than both reference methods. These results indicate that the effectiveness of SIEVE is not tied to a particular VLA model.

Method Stack Green Cube On Yellow Cube Put Carrot On Plate Put Spoon On Table Cloth Put Eggplant In Basket Avg.
Ablation of Structural Exposure Allocation
w/o Trans.22.9 47.9 67.7 64.6 50.8
w/o Prim.25.0 51.0 68.8 61.5 51.6
Ablation of Learning-Friendly Trajectory Selection
Most-Dissim 18.8 45.8 64.6 31.3 40.1
Random 29.2 36.5 77.1 71.9 53.7
\rowcolor navyblue!10SIEVE (Ours)25.0 54.2 70.8 75.0 56.3

Table 4:  Ablation study on SimplerEnv-WidowX using Qwen3-VL-4B-GR00T under the 50% selection budget and 50% of the training steps. “w/o Trans.” and “w/o Prim.” remove the transition and primitive exposure terms, respectively. “Most-Dissim” selects the most dissimilar samples to maximize within-bucket trajectory diversity, while “Random” uses random within-bucket selection. 

### Ablation Study and Further Analysis

Ablation of Structural Exposure Allocation. We ablate the two structural exposure terms in SIEVE. Removing the transition or primitive exposure term decreases the average success rate from 56.3% to 50.8% and 51.6%, respectively, indicating that both primitives and transitions contribute to effective data selection. The larger performance drop without transition exposure further suggests that transition interfaces provide important supervision for modeling how reusable behavior units are connected during task execution.

Ablation of Learning-Friendly Trajectory Selection. We ablate the within-bucket sample selection strategy. Replacing the proposed selection with Most-Dissim, which selects trajectories with the lowest aggregate cosine similarity to other samples to maximize trajectory diversity, reduces the average success rate from 56.3% to 40.1%, indicating that atypical trajectories within the same composition pattern are less suitable for imitation learning. Random within-bucket selection performs better than most-dissimilar selection but still trails SIEVE, improving the average success rate only from 53.7% to 56.3%. These results validate our design choice of selecting representative and stable demonstrations within each composition-pattern bucket.

![Image 3: Refer to caption](https://arxiv.org/html/2607.06442v1/x3.png)

Figure 3:  Distribution of top-50 composition patterns before and after SIEVE selection (Selection budget=50%) on Bridge-V2. Each x-axis label denotes a composition pattern, represented as an ordered list of primitive IDs. 

Composition Pattern Redistribution. Figure[3](https://arxiv.org/html/2607.06442#Sx4.F3 "Figure 3 ‣ Ablation Study and Further Analysis ‣ Experiments ‣ SIEVE: Structure-Aware Data Selection for Imitation Learning with VLA Models") illustrates how SIEVE reshapes the composition-pattern distribution on Bridge-V2. The original dataset is dominated by a small number of high-frequency patterns, many of which consist of a single primitive, indicating limited compositional diversity. After selection, the distribution becomes substantially more balanced, and the most frequent selected patterns are predominantly multi-primitive sequences, covering a wider range of primitives and transitions. This redistribution suggests that SIEVE shifts the selected subset away from frequency-dominated patterns toward structurally richer composition patterns, thereby exposing more reusable primitives and transitions for imitation learning.

## Conclusion

We introduced SIEVE, a structure-aware data selection method for VLA imitation learning. SIEVE selects demonstrations by exposing reusable primitive compositions and transition interfaces, while favoring central and stable realizations within each composition pattern for behavior cloning. Across multiple datasets, benchmarks, and VLA models, SIEVE consistently improves over competitive baselines and can outperform full-data training using fewer demonstrations and training steps. These results highlight the importance of selecting data according to reusable structure, offering a practical route toward more efficient VLA imitation learning.

## References

*   M. Assran, A. Bardes, D. Fan, Q. Garrido, R. Howes, M. Muckley, A. Rizvi, C. Roberts, K. Sinha, A. Zholus, et al. (2025)V-jepa2: self-supervised video models enable understanding, prediction and planning. arXiv preprint arXiv:2506.09985. Cited by: [Segment representation.](https://arxiv.org/html/2607.06442#Sx3.SSx1.SSS0.Px2.p1.3 "Segment representation. ‣ Primitive Discovery ‣ SIEVE ‣ SIEVE: Structure-Aware Data Selection for Imitation Learning with VLA Models"). 
*   S. Bai, Y. Cai, R. Chen, K. Chen, X. Chen, Z. Cheng, L. Deng, W. Ding, C. Gao, C. Ge, et al. (2025)Qwen3-vl technical report. arXiv preprint arXiv:2511.21631. Cited by: [Vision-Language-Action Models.](https://arxiv.org/html/2607.06442#Sx2.SSx1.p1.1 "Vision-Language-Action Models. ‣ Related Work ‣ SIEVE: Structure-Aware Data Selection for Imitation Learning with VLA Models"), [1st item](https://arxiv.org/html/2607.06442#Sx4.I3.i1.p1.1 "In Experiment Setup ‣ Experiments ‣ SIEVE: Structure-Aware Data Selection for Imitation Learning with VLA Models"). 
*   A. Barron, J. Rissanen, and B. Yu (1998)The minimum description length principle in coding and modeling. IEEE transactions on information theory 44 (6),  pp.2743–2760. Cited by: [Introduction](https://arxiv.org/html/2607.06442#Sx1.p3.5 "Introduction ‣ SIEVE: Structure-Aware Data Selection for Imitation Learning with VLA Models"). 
*   S. Belkhale, Y. Cui, and D. Sadigh (2023)Data quality in imitation learning. Advances in neural information processing systems 36,  pp.80375–80395. Cited by: [Introduction](https://arxiv.org/html/2607.06442#Sx1.p1.1 "Introduction ‣ SIEVE: Structure-Aware Data Selection for Imitation Learning with VLA Models"), [Introduction](https://arxiv.org/html/2607.06442#Sx1.p4.1 "Introduction ‣ SIEVE: Structure-Aware Data Selection for Imitation Learning with VLA Models"), [Vision-Language-Action Models.](https://arxiv.org/html/2607.06442#Sx2.SSx1.p1.1 "Vision-Language-Action Models. ‣ Related Work ‣ SIEVE: Structure-Aware Data Selection for Imitation Learning with VLA Models"), [Data Selection for Imitation Learning.](https://arxiv.org/html/2607.06442#Sx2.SSx2.p1.1 "Data Selection for Imitation Learning. ‣ Related Work ‣ SIEVE: Structure-Aware Data Selection for Imitation Learning with VLA Models"). 
*   J. Bjorck, F. Castañeda, N. Cherniadev, X. Da, R. Ding, L. Fan, Y. Fang, D. Fox, F. Hu, S. Huang, et al. (2025)Gr00t n1: an open foundation model for generalist humanoid robots. arXiv preprint arXiv:2503.14734. Cited by: [Vision-Language-Action Models.](https://arxiv.org/html/2607.06442#Sx2.SSx1.p1.1 "Vision-Language-Action Models. ‣ Related Work ‣ SIEVE: Structure-Aware Data Selection for Imitation Learning with VLA Models"), [3rd item](https://arxiv.org/html/2607.06442#Sx4.I2.i3.p1.1 "In Experiment Setup ‣ Experiments ‣ SIEVE: Structure-Aware Data Selection for Imitation Learning with VLA Models"), [1st item](https://arxiv.org/html/2607.06442#Sx4.I3.i1.p1.1 "In Experiment Setup ‣ Experiments ‣ SIEVE: Structure-Aware Data Selection for Imitation Learning with VLA Models"). 
*   K. Black, N. Brown, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, L. Groom, K. Hausman, B. Ichter, et al. (2024)\pi_{0}: A vision-language-action flow model for general robot control. arXiv preprint arXiv:2410.24164. Cited by: [Introduction](https://arxiv.org/html/2607.06442#Sx1.p1.1 "Introduction ‣ SIEVE: Structure-Aware Data Selection for Imitation Learning with VLA Models"), [Vision-Language-Action Models.](https://arxiv.org/html/2607.06442#Sx2.SSx1.p1.1 "Vision-Language-Action Models. ‣ Related Work ‣ SIEVE: Structure-Aware Data Selection for Imitation Learning with VLA Models"). 
*   A. S. Chen, A. M. Lessing, Y. Liu, and C. Finn (2025)Curating demonstrations using online experience. arXiv preprint arXiv:2503.03707. Cited by: [Introduction](https://arxiv.org/html/2607.06442#Sx1.p2.1 "Introduction ‣ SIEVE: Structure-Aware Data Selection for Imitation Learning with VLA Models"), [Data Selection for Imitation Learning.](https://arxiv.org/html/2607.06442#Sx2.SSx2.p1.1 "Data Selection for Imitation Learning. ‣ Related Work ‣ SIEVE: Structure-Aware Data Selection for Imitation Learning with VLA Models"). 
*   C. Chi, Z. Xu, S. Feng, E. Cousineau, Y. Du, B. Burchfiel, R. Tedrake, and S. Song (2025)Diffusion policy: visuomotor policy learning via action diffusion. The International Journal of Robotics Research 44 (10-11),  pp.1684–1704. Cited by: [Vision-Language-Action Models.](https://arxiv.org/html/2607.06442#Sx2.SSx1.p1.1 "Vision-Language-Action Models. ‣ Related Work ‣ SIEVE: Structure-Aware Data Selection for Imitation Learning with VLA Models"). 
*   S. Dass, A. Khaddaj, L. Engstrom, A. Madry, A. Ilyas, and R. Martín-Martín (2025)Datamil: selecting data for robot imitation learning with datamodels. arXiv preprint arXiv:2505.09603. Cited by: [Introduction](https://arxiv.org/html/2607.06442#Sx1.p2.1 "Introduction ‣ SIEVE: Structure-Aware Data Selection for Imitation Learning with VLA Models"), [Data Selection for Imitation Learning.](https://arxiv.org/html/2607.06442#Sx2.SSx2.p1.1 "Data Selection for Imitation Learning. ‣ Related Work ‣ SIEVE: Structure-Aware Data Selection for Imitation Learning with VLA Models"). 
*   M. Finzi, S. Qiu, Y. Jiang, P. Izmailov, J. Z. Kolter, and A. G. Wilson (2026)From entropy to epiplexity: rethinking information for computationally bounded intelligence. arXiv preprint arXiv:2601.03220. Cited by: [Introduction](https://arxiv.org/html/2607.06442#Sx1.p3.4 "Introduction ‣ SIEVE: Structure-Aware Data Selection for Imitation Learning with VLA Models"). 
*   P. D. Grünwald (2007)The minimum description length principle. MIT press. Cited by: [Introduction](https://arxiv.org/html/2607.06442#Sx1.p3.5 "Introduction ‣ SIEVE: Structure-Aware Data Selection for Imitation Learning with VLA Models"). 
*   J. Hejna, C. Bhateja, Y. Jiang, K. Pertsch, and D. Sadigh (2024)Re-mix: optimizing data mixtures for large scale imitation learning. arXiv preprint arXiv:2408.14037. Cited by: [Data Selection for Imitation Learning.](https://arxiv.org/html/2607.06442#Sx2.SSx2.p1.1 "Data Selection for Imitation Learning. ‣ Related Work ‣ SIEVE: Structure-Aware Data Selection for Imitation Learning with VLA Models"). 
*   J. Hejna, S. Mirchandani, A. Balakrishna, A. Xie, A. Wahid, J. Tompson, P. Sanketi, D. Shah, C. Devin, and D. Sadigh (2025)Robot data curation with mutual information estimators. arXiv preprint arXiv:2502.08623. Cited by: [Introduction](https://arxiv.org/html/2607.06442#Sx1.p2.1 "Introduction ‣ SIEVE: Structure-Aware Data Selection for Imitation Learning with VLA Models"), [Data Selection for Imitation Learning.](https://arxiv.org/html/2607.06442#Sx2.SSx2.p1.1 "Data Selection for Imitation Learning. ‣ Related Work ‣ SIEVE: Structure-Aware Data Selection for Imitation Learning with VLA Models"), [3rd item](https://arxiv.org/html/2607.06442#Sx4.I4.i3.p1.1 "In Experiment Setup ‣ Experiments ‣ SIEVE: Structure-Aware Data Selection for Imitation Learning with VLA Models"). 
*   A. Hussein, M. M. Gaber, E. Elyan, and C. Jayne (2017)Imitation learning: a survey of learning methods. ACM Computing Surveys (CSUR)50 (2),  pp.1–35. Cited by: [Introduction](https://arxiv.org/html/2607.06442#Sx1.p4.1 "Introduction ‣ SIEVE: Structure-Aware Data Selection for Imitation Learning with VLA Models"), [Vision-Language-Action Models.](https://arxiv.org/html/2607.06442#Sx2.SSx1.p1.1 "Vision-Language-Action Models. ‣ Related Work ‣ SIEVE: Structure-Aware Data Selection for Imitation Learning with VLA Models"). 
*   P. Intelligence, B. Ai, A. Amin, R. Aniceto, A. Balakrishna, G. Balke, K. Black, G. Bokinsky, S. Cao, T. Charbonnier, et al. (2026)\pi_{0.7}: A steerable generalist robotic foundation model with emergent capabilities. arXiv preprint arXiv:2604.15483. Cited by: [Vision-Language-Action Models.](https://arxiv.org/html/2607.06442#Sx2.SSx1.p1.1 "Vision-Language-Action Models. ‣ Related Work ‣ SIEVE: Structure-Aware Data Selection for Imitation Learning with VLA Models"). 
*   P. Intelligence, K. Black, N. Brown, J. Darpinian, K. Dhabalia, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, et al. (2025)\pi_{0.5}: A vision-language-action model with open-world generalization. arXiv preprint arXiv:2504.16054. Cited by: [Introduction](https://arxiv.org/html/2607.06442#Sx1.p1.1 "Introduction ‣ SIEVE: Structure-Aware Data Selection for Imitation Learning with VLA Models"), [Vision-Language-Action Models.](https://arxiv.org/html/2607.06442#Sx2.SSx1.p1.1 "Vision-Language-Action Models. ‣ Related Work ‣ SIEVE: Structure-Aware Data Selection for Imitation Learning with VLA Models"). 
*   M. J. Kim, C. Finn, and P. Liang (2025)Fine-tuning vision-language-action models: optimizing speed and success. arXiv preprint arXiv:2502.19645. Cited by: [Vision-Language-Action Models.](https://arxiv.org/html/2607.06442#Sx2.SSx1.p1.1 "Vision-Language-Action Models. ‣ Related Work ‣ SIEVE: Structure-Aware Data Selection for Imitation Learning with VLA Models"), [2nd item](https://arxiv.org/html/2607.06442#Sx4.I3.i2.p1.1 "In Experiment Setup ‣ Experiments ‣ SIEVE: Structure-Aware Data Selection for Imitation Learning with VLA Models"). 
*   M. J. Kim, K. Pertsch, S. Karamcheti, T. Xiao, A. Balakrishna, S. Nair, R. Rafailov, E. Foster, G. Lam, P. Sanketi, et al. (2024)Openvla: an open-source vision-language-action model. arXiv preprint arXiv:2406.09246. Cited by: [Introduction](https://arxiv.org/html/2607.06442#Sx1.p1.1 "Introduction ‣ SIEVE: Structure-Aware Data Selection for Imitation Learning with VLA Models"), [Vision-Language-Action Models.](https://arxiv.org/html/2607.06442#Sx2.SSx1.p1.1 "Vision-Language-Action Models. ‣ Related Work ‣ SIEVE: Structure-Aware Data Selection for Imitation Learning with VLA Models"). 
*   X. Li, K. Hsu, J. Gu, K. Pertsch, O. Mees, H. R. Walke, C. Fu, I. Lunawat, I. Sieh, S. Kirmani, et al. (2024)Evaluating real-world robot manipulation policies in simulation. arXiv preprint arXiv:2405.05941. Cited by: [1st item](https://arxiv.org/html/2607.06442#Sx4.I2.i1.p1.1 "In Experiment Setup ‣ Experiments ‣ SIEVE: Structure-Aware Data Selection for Imitation Learning with VLA Models"), [2nd item](https://arxiv.org/html/2607.06442#Sx4.I2.i2.p1.1 "In Experiment Setup ‣ Experiments ‣ SIEVE: Structure-Aware Data Selection for Imitation Learning with VLA Models"). 
*   S. Lian, B. Yu, X. Lin, Z. Shen, L. T. Yang, Y. Jin, H. Liu, C. Wu, H. Yuan, C. Huang, et al. (2026)IntentVLA: short-horizon intent modeling for aliased robot manipulation. arXiv preprint arXiv:2605.14712. Cited by: [Vision-Language-Action Models.](https://arxiv.org/html/2607.06442#Sx2.SSx1.p1.1 "Vision-Language-Action Models. ‣ Related Work ‣ SIEVE: Structure-Aware Data Selection for Imitation Learning with VLA Models"). 
*   F. Lin, Y. Hu, P. Sheng, C. Wen, J. You, and Y. Gao (2025a)Data scaling laws in imitation learning for robotic manipulation. In International Conference on Learning Representations, Vol. 2025,  pp.54877–54910. Cited by: [Introduction](https://arxiv.org/html/2607.06442#Sx1.p1.1 "Introduction ‣ SIEVE: Structure-Aware Data Selection for Imitation Learning with VLA Models"), [Vision-Language-Action Models.](https://arxiv.org/html/2607.06442#Sx2.SSx1.p1.1 "Vision-Language-Action Models. ‣ Related Work ‣ SIEVE: Structure-Aware Data Selection for Imitation Learning with VLA Models"). 
*   X. Lin, S. Lian, B. Yu, R. Yang, Z. Shen, C. Wu, Y. Miao, Y. Jin, Y. Shi, J. He, et al. (2025b)Physbrain: human egocentric data as a bridge from vision language models to physical intelligence. arXiv preprint arXiv:2512.16793. Cited by: [Vision-Language-Action Models.](https://arxiv.org/html/2607.06442#Sx2.SSx1.p1.1 "Vision-Language-Action Models. ‣ Related Work ‣ SIEVE: Structure-Aware Data Selection for Imitation Learning with VLA Models"). 
*   S. Nasiriany, A. Maddukuri, L. Zhang, A. Parikh, A. Lo, A. Joshi, A. Mandlekar, and Y. Zhu (2024)Robocasa: large-scale simulation of everyday tasks for generalist robots. arXiv preprint arXiv:2406.02523. Cited by: [3rd item](https://arxiv.org/html/2607.06442#Sx4.I2.i3.p1.1 "In Experiment Setup ‣ Experiments ‣ SIEVE: Structure-Aware Data Selection for Imitation Learning with VLA Models"). 
*   A. O’Neill, A. Rehman, A. Maddukuri, A. Gupta, A. Padalkar, A. Lee, A. Pooley, A. Gupta, A. Mandlekar, A. Jain, et al. (2024)Open x-embodiment: robotic learning datasets and rt-x models: open x-embodiment collaboration 0. In 2024 IEEE International Conference on Robotics and Automation (ICRA),  pp.6892–6903. Cited by: [Introduction](https://arxiv.org/html/2607.06442#Sx1.p1.1 "Introduction ‣ SIEVE: Structure-Aware Data Selection for Imitation Learning with VLA Models"), [Vision-Language-Action Models.](https://arxiv.org/html/2607.06442#Sx2.SSx1.p1.1 "Vision-Language-Action Models. ‣ Related Work ‣ SIEVE: Structure-Aware Data Selection for Imitation Learning with VLA Models"), [2nd item](https://arxiv.org/html/2607.06442#Sx4.I2.i2.p1.1 "In Experiment Setup ‣ Experiments ‣ SIEVE: Structure-Aware Data Selection for Imitation Learning with VLA Models"). 
*   J. Rissanen (2004)Minimum description length principle. Encyclopedia of statistical sciences 7. Cited by: [Introduction](https://arxiv.org/html/2607.06442#Sx1.p3.5 "Introduction ‣ SIEVE: Structure-Aware Data Selection for Imitation Learning with VLA Models"). 
*   S. Ross and D. Bagnell (2010)Efficient reductions for imitation learning. In Proceedings of the thirteenth international conference on artificial intelligence and statistics,  pp.661–668. Cited by: [Introduction](https://arxiv.org/html/2607.06442#Sx1.p4.1 "Introduction ‣ SIEVE: Structure-Aware Data Selection for Imitation Learning with VLA Models"). 
*   H. Sathyanarayan, V. Vantilborgh, and I. Abraham (2025)Quality over quantity: curating contact-based robot datasets improves learning. arXiv preprint arXiv:2510.18137. Cited by: [Introduction](https://arxiv.org/html/2607.06442#Sx1.p1.1 "Introduction ‣ SIEVE: Structure-Aware Data Selection for Imitation Learning with VLA Models"), [Vision-Language-Action Models.](https://arxiv.org/html/2607.06442#Sx2.SSx1.p1.1 "Vision-Language-Action Models. ‣ Related Work ‣ SIEVE: Structure-Aware Data Selection for Imitation Learning with VLA Models"). 
*   D. Sculley (2010)Web-scale k-means clustering. In Proceedings of the 19th international conference on World wide web,  pp.1177–1178. Cited by: [Primitive discovery via clustering.](https://arxiv.org/html/2607.06442#Sx3.SSx1.SSS0.Px3.p1.1 "Primitive discovery via clustering. ‣ Primitive Discovery ‣ SIEVE ‣ SIEVE: Structure-Aware Data Selection for Imitation Learning with VLA Models"). 
*   R. Shao, W. Li, L. Zhang, R. Zhang, Z. Liu, R. Chen, and L. Nie (2025)Large vlm-based vision-language-action models for robotic manipulation: a survey. arXiv preprint arXiv:2508.13073. Cited by: [Vision-Language-Action Models.](https://arxiv.org/html/2607.06442#Sx2.SSx1.p1.1 "Vision-Language-Action Models. ‣ Related Work ‣ SIEVE: Structure-Aware Data Selection for Imitation Learning with VLA Models"). 
*   H. R. Walke, K. Black, T. Z. Zhao, Q. Vuong, C. Zheng, P. Hansen-Estruch, A. W. He, V. Myers, M. J. Kim, M. Du, et al. (2023)Bridgedata v2: a dataset for robot learning at scale. In Conference on Robot Learning,  pp.1723–1736. Cited by: [1st item](https://arxiv.org/html/2607.06442#Sx4.I2.i1.p1.1 "In Experiment Setup ‣ Experiments ‣ SIEVE: Structure-Aware Data Selection for Imitation Learning with VLA Models"). 
*   C. Wu, J. Mao, Y. Miao, S. Lian, B. Yu, X. Lin, C. Huang, L. Zhang, and K. Chen (2026)ScalSelect: scalable training-free multimodal data selection for efficient visual instruction tuning. arXiv preprint arXiv:2602.11636. Cited by: [Data Selection for Imitation Learning.](https://arxiv.org/html/2607.06442#Sx2.SSx2.p1.1 "Data Selection for Imitation Learning. ‣ Related Work ‣ SIEVE: Structure-Aware Data Selection for Imitation Learning with VLA Models"). 
*   J. Xiao, B. Yan, J. Zhang, J. Wang, C. Li, Z. Cheng, and G. Zhai (2025)Data assessment for embodied intelligence. arXiv preprint arXiv:2511.09119. Cited by: [Data Selection for Imitation Learning.](https://arxiv.org/html/2607.06442#Sx2.SSx2.p1.1 "Data Selection for Imitation Learning. ‣ Related Work ‣ SIEVE: Structure-Aware Data Selection for Imitation Learning with VLA Models"). 
*   Y. Xing, X. Luo, J. Xie, L. Gao, H. Shen, and J. Song (2025)Shortcut learning in generalist robot policies: the role of dataset diversity and fragmentation. arXiv preprint arXiv:2508.06426. Cited by: [Introduction](https://arxiv.org/html/2607.06442#Sx1.p1.1 "Introduction ‣ SIEVE: Structure-Aware Data Selection for Imitation Learning with VLA Models"), [Vision-Language-Action Models.](https://arxiv.org/html/2607.06442#Sx2.SSx1.p1.1 "Vision-Language-Action Models. ‣ Related Work ‣ SIEVE: Structure-Aware Data Selection for Imitation Learning with VLA Models"). 
*   T. Xu, J. Wang, R. Zhang, J. Guan, X. Zeng, W. Song, X. Zhou, Z. Chen, G. Chen, and Y. Li (2026)ATHENA: accelerated multi-task heterogeneous influence functions for robot data curation. arXiv preprint arXiv:2606.16208. Cited by: [Introduction](https://arxiv.org/html/2607.06442#Sx1.p2.1 "Introduction ‣ SIEVE: Structure-Aware Data Selection for Imitation Learning with VLA Models"), [Data Selection for Imitation Learning.](https://arxiv.org/html/2607.06442#Sx2.SSx2.p1.1 "Data Selection for Imitation Learning. ‣ Related Work ‣ SIEVE: Structure-Aware Data Selection for Imitation Learning with VLA Models"). 
*   B. Yu, S. Lian, X. Lin, Z. Shen, Y. Wei, C. Wu, H. Yuan, H. Liu, B. Wang, C. Huang, et al. (2026)FrameSkip: learning from fewer but more informative frames in vla training. arXiv preprint arXiv:2605.13757. Cited by: [Data Selection for Imitation Learning.](https://arxiv.org/html/2607.06442#Sx2.SSx2.p1.1 "Data Selection for Imitation Learning. ‣ Related Work ‣ SIEVE: Structure-Aware Data Selection for Imitation Learning with VLA Models"). 
*   Y. Zhang, Y. Xie, H. Liu, R. Shah, M. Wan, L. Fan, and Y. Zhu (2025)Scizor: a self-supervised approach to data curation for large-scale imitation learning. arXiv preprint arXiv:2505.22626. Cited by: [Introduction](https://arxiv.org/html/2607.06442#Sx1.p2.1 "Introduction ‣ SIEVE: Structure-Aware Data Selection for Imitation Learning with VLA Models"), [Data Selection for Imitation Learning.](https://arxiv.org/html/2607.06442#Sx2.SSx2.p1.1 "Data Selection for Imitation Learning. ‣ Related Work ‣ SIEVE: Structure-Aware Data Selection for Imitation Learning with VLA Models"), [4th item](https://arxiv.org/html/2607.06442#Sx4.I4.i4.p1.1 "In Experiment Setup ‣ Experiments ‣ SIEVE: Structure-Aware Data Selection for Imitation Learning with VLA Models"). 
*   Y. Zhou, Y. Chang, and Y. Wu (2026)Synthetic data for multimodal large language models: a lifecycle-oriented survey. Cited by: [Data Selection for Imitation Learning.](https://arxiv.org/html/2607.06442#Sx2.SSx2.p1.1 "Data Selection for Imitation Learning. ‣ Related Work ‣ SIEVE: Structure-Aware Data Selection for Imitation Learning with VLA Models"). 
*   B. Zitkovich, T. Yu, S. Xu, P. Xu, T. Xiao, F. Xia, J. Wu, P. Wohlhart, S. Welker, A. Wahid, et al. (2023)Rt-2: vision-language-action models transfer web knowledge to robotic control. In Conference on Robot Learning,  pp.2165–2183. Cited by: [Introduction](https://arxiv.org/html/2607.06442#Sx1.p1.1 "Introduction ‣ SIEVE: Structure-Aware Data Selection for Imitation Learning with VLA Models"), [Vision-Language-Action Models.](https://arxiv.org/html/2607.06442#Sx2.SSx1.p1.1 "Vision-Language-Action Models. ‣ Related Work ‣ SIEVE: Structure-Aware Data Selection for Imitation Learning with VLA Models"). 

## Appendix

### Training Hyperparameters

All experiments are conducted using the same training hyperparameters summarized in Table[5](https://arxiv.org/html/2607.06442#Sx6.T5 "Table 5 ‣ Training Hyperparameters ‣ Appendix ‣ SIEVE: Structure-Aware Data Selection for Imitation Learning with VLA Models"), ensuring a fair comparison across different methods. All models are trained on 8 NVIDIA H100 (80GB) GPUs.

Hyperparameter Value
Optimizer AdamW
Learning rate (VLM)1e-5
Learning rate (Action Head)1e-4
LR scheduler Cosine decay
Warmup ratio 10%
AdamW (\beta_{1},\beta_{2})(0.9,\,0.95)
Weight decay 1e-8
Gradient clipping 1.0
Per-device batch size 16
Gradient accumulation 1
Mixed precision BF16
Distributed training DeepSpeed ZeRO-2

Table 5: Training hyperparameters.

### Implementation Details of Primitive Discovery via Clustering.

To determine the number of primitive clusters, we adopt a practical search strategy. Specifically, for each dataset, we randomly sample 10% of the demonstrations and uniformly evaluate 20 candidate values of K within a reasonable search range. For each candidate, we perform primitive clustering and compute the clustering score (1-\mathcal{J})\log\mathcal{R} defined in the _Primitive discovery via clustering_ section. The value of K that maximizes this score is selected for all subsequent experiments. Figure[4](https://arxiv.org/html/2607.06442#Sx6.F4 "Figure 4 ‣ Implementation Details of Primitive Discovery via Clustering. ‣ Appendix ‣ SIEVE: Structure-Aware Data Selection for Imitation Learning with VLA Models") shows the search results on Bridge-V2, Fractal, and GR00T-X-Sim. The clustering score exhibits a clear peak on all three datasets, suggesting that the proposed criterion provides a stable and practical heuristic for determining the number of primitive clusters.

![Image 4: Refer to caption](https://arxiv.org/html/2607.06442v1/x4.png)

(a) Bridge-V2

![Image 5: Refer to caption](https://arxiv.org/html/2607.06442v1/x5.png)

(b) Fractal

![Image 6: Refer to caption](https://arxiv.org/html/2607.06442v1/x6.png)

(c) GR00T-X-Sim

Figure 4: Selection of the number of primitive clusters K on different datasets. Following the practical protocol used in our experiments, we randomly sample 10% of each training dataset and uniformly evaluate 20 candidate K values within a reasonable search range. The optimal K is selected by maximizing the clustering score (1-\mathcal{J})\log\mathcal{R}, and is indicated by the red dashed line.

### Details on Performance Across Different Datasets

Tables[6](https://arxiv.org/html/2607.06442#Sx6.T6 "Table 6 ‣ Details on Performance Across Different Datasets ‣ Appendix ‣ SIEVE: Structure-Aware Data Selection for Imitation Learning with VLA Models") and[7](https://arxiv.org/html/2607.06442#Sx6.T7 "Table 7 ‣ Details on Performance Across Different Datasets ‣ Appendix ‣ SIEVE: Structure-Aware Data Selection for Imitation Learning with VLA Models") present the detailed results corresponding to Table[2](https://arxiv.org/html/2607.06442#Sx4.T2 "Table 2 ‣ Extended Results ‣ Experiments ‣ SIEVE: Structure-Aware Data Selection for Imitation Learning with VLA Models") in the main paper. Consistent with the main results, SIEVE achieves the highest average success rate on both Fractal and GR00T-X-Sim while maintaining competitive performance across individual evaluation categories.

Method Grasp Single Opened Coke Can Move Near Google Baked Tex Close/Open Drawer Custom Avg.
Full-Training 95.8 87.5 41.7 75.0
Random 100.0 66.7 0.0 55.6
DemInf 100.0 85.4 16.7 67.4
SCIZOR 99.0 87.5 29.2 71.9
\rowcolor navyblue!9SIEVE (Ours)100.0 91.7 37.5 76.4

Table 6: Detailed results on Fractal. This table expands the Fractal results reported in Table[2](https://arxiv.org/html/2607.06442#Sx4.T2 "Table 2 ‣ Extended Results ‣ Experiments ‣ SIEVE: Structure-Aware Data Selection for Imitation Learning with VLA Models"), showing the success rate for each evaluation task. Full-Training uses the complete dataset with 100K training steps, while all data selection methods use 50% of the training data and 50% of the training steps.

Method PnP *To * Close PnP Novel From Cuttingboard To *PnP Novel From Placemat To *PnP Novel From Plate To *PnP Novel From Tray To *Avg.
Full-Training 52.8 54.8 46.5 55.5 54.0 52.7
Random 50.5 56.8 49.8 60.3 50.1 53.5
DemInf 53.3 55.1 48.6 58.1 53.9 53.8
SCIZOR 54.0 56.1 48.1 60.5 52.3 54.2
\rowcolor navyblue!9SIEVE (Ours)55.3 56.7 47.9 61.3 52.8 54.8

Table 7: Detailed results on GR00T-X-Sim. This table expands the GR00T-X-Sim results reported in Table[2](https://arxiv.org/html/2607.06442#Sx4.T2 "Table 2 ‣ Extended Results ‣ Experiments ‣ SIEVE: Structure-Aware Data Selection for Imitation Learning with VLA Models"), showing the average success rate for each RoboCasa-GR1 task category. Full-Training uses the complete dataset with 100K training steps, while all data selection methods use 50% of the training data and 50% of the training steps.

