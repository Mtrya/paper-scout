# Not Worth Another Token: Marginal Value Estimation for Efficient Deep Research Agents

Harshitha Kolukuluru<sup>1</sup>, Reshma Ashok<sup>1</sup>, Kirat Arora<sup>1</sup> Evan William Ciccarelli<sup>1</sup>, Nischal Ashok Kumar<sup>1</sup>, Lunyiu Nie<sup>2</sup> Franck Dernoncourt<sup>3</sup>, Samyadeep Basu<sup>3</sup>, Ryan A. Rossi<sup>3</sup>, Nedim Lipka<sup>3</sup>

<sup>1</sup>University of Massachusetts Amherst

<sup>2</sup>The University of Texas at Austin

<sup>3</sup>Adobe Research

hkolukuluru@umass.edu, lipka@adobe.com

## Abstract

Long-horizon research agents solve openended tasks through iterative retrieval, aggregation, and synthesis, but context grows rapidly while the marginal value of additional evidence often declines. This leads to unnecessary token cost, higher latency, and noisier inputs for final report generation. We study marginal value estimation for context management in deep research agents and present the first systematic stage-aware comparison of pruning strategies across the pipeline. We evaluate lightweight heuristic criteria and a learned value model at pre-retrieval, post-retrieval, and pre-synthesis stages. Our results show that pruning effectiveness depends more on where pruning is applied than on the specific scoring rule: early pruning yields the largest end-to-end savings, while later pruning mainly refines the final synthesis context. Lightweight heuristics reduce token usage by up to 73% with little quality degradation, learned pruning remains competitive on selected trade-offs, and no single method dominates across quality, efficiency, and faithfulness. These findings provide practical guidance for designing efficient long-horizon agentic systems.

## 1 Introduction

Long-horizon retrieval agents answer complex, open-ended queries through iterative retrieval, reasoning, and synthesis (Yao et al., 2023b,a; Nie et al., 2026). As these systems operate over multiple steps, accumulated context grows rapidly while the marginal value of each additional retrieval often declines: early steps surface core facts, whereas later ones add redundant or weakly informative content (Liu et al., 2024; Jiang et al., 2024; Xu et al., 2023), increasing token cost, latency, and noise in the final synthesis context.

A natural remedy is to prune low-value candidates by estimating their marginal value—how much new, useful information each contributes beyond what is already collected. Existing deepresearch pipelines often rely on LLM prompting for these decisions, incurring inference overhead and inconsistent behavior (Nie et al., 2026). Yet there is limited empirical understanding of which pipeline stages benefit most from pruning, which strategies work best at each stage, and under which objectives.

We address this through a stage-aware analysis of marginal-value-based pruning in deep research pipelines (Figure 1), where a system decomposes a query into subqueries, accumulates context across retrieval steps, and synthesizes a long-form report. We identify three intervention points: Pre-Retrieval, Post-Retrieval, and Pre-Synthesis, giving rise to one-, two-, and three-stage configurations, and compare heuristic, learned, and LLM-based strategies across all configurations under a shared evaluation setup (Sections 3-5).

Stage placement is often more consequential than the specific scoring rule. Post-Retrieval MMR reduces token cost from 375.4k to 114.6k and explored nodes from 29.0 to 8.84 with modest quality loss; Pre-Synthesis pruning is generally too late to recover upstream costs but can improve report quality. Two-stage pruning yields the strongest qualityefficiency trade-offs; three-stage MMR achieves 73.3% token reduction. No single strategy dominates: relevance–redundancy heuristics excel at early cost control, while richer methods are more competitive when targeting report quality or source grounding.

The main contributions of this work are:

• We formulate marginal value estimation as a stage-aware pruning problem for deep research pipelines, covering three intervention points: Pre-Retrieval, Post-Retrieval, and Pre-Synthesis.

• We provide a controlled empirical comparison of heuristic, learned, and LLM-based pruning strategies across one, two, and three-stage configurations under a shared evaluation setup.

• We identify cross-stage findings about where pruning is most effective, which strategies provide the best quality-cost trade-offs, and where report quality, source grounding, and cost reduction diverge.

## 2 Related Work

Prior work on long-horizon retrieval agents studies how language models can decompose tasks, retrieve iteratively, and reason over multi-step search trajectories. Tree of Thoughts (Yao et al., 2023a) and ReAct (Yao et al., 2023b) introduced branching and tool-augmented reasoning, while ParallelResearch (Nie et al., 2026) extends this direction to deep research through tree-structured query decomposition and parallel exploration. These systems improve multi-step retrieval and coverage, but primarily focus on how to explore rather than how to control the growth of accumulated context.

A separate line of work studies context pruning, compression, and prompt-efficiency methods for long-context language models. Lost in the Middle (Liu et al., 2024) shows that models struggle to use relevant information in long prompts, motivating stronger context management. Selective Context (Li, 2023), DYCP (Choi et al., 2026), LLMLingua and LongLLMLingua (Jiang et al., 2023, 2024), and RECOMP (Xu et al., 2023) reduce prompt length or compress retrieved evidence, typically at or near the final model input. Related work on agent orchestration and adaptive control studies how multi-step pipelines should schedule actions, react to intermediate signals, or terminate search (Mazzolenis and Zhang, 2025; Laju et al., 2026; Pagonas et al., 2025; Shinn et al., 2023; Li et al., 2025). Our focus is complementary: we study marginal-value-based pruning as a stage-aware context management problem, and compare pruning strategies across Pre-Retrieval, Post-Retrieval, and Pre-Synthesis under a shared evaluation setup.

## 3 Problem Formulation

## 3.1 Deep Research Pipeline with Pruning

We consider long-horizon retrieval agents that answer complex queries through iterative decomposition, retrieval, aggregation, and synthesis, instantiated here as a tree-structured deep research workflow (Nie et al., 2026) (Figure 1).

Given query $Q .$ , the system decomposes $Q$ into subqueries ${ \mathcal { S } } .$ At step t, it selects $s _ { t } \in S$ , retrieves context items $C _ { s _ { t } } ,$ , and merges retained evidence into accumulated context $C _ { t }$ . Retrieved findings may spawn additional subqueries, extending S dynamically. The process terminates when $s$ is exhausted or a stopping criterion is met at step $T ,$ after which the accumulated context is passed to synthesis.

Objective. Let $\mathcal { R } ( C _ { T } , Q )$ denote report quality and $\mathrm { C o s t } ( C _ { T } )$ operational cost (token usage, retrieval calls, latency). In the unpruned setting, evidence is added unconditionally:

$$
C _ { t + 1 } = C _ { t } \cup C _ { s _ { t } } .
$$

Since not all retrieved content contributes equally, we seek a policy that maximizes quality subject to a cost budget $B \colon$

$$
\operatorname* { m a x } _ { C _ { T } \subseteq \bigcup _ { t } C _ { s _ { t } } } \mathcal { R } ( C _ { T } , Q ) \quad \mathrm { s . t . } \quad \operatorname { C o s t } ( C _ { T } ) \leq B ,
$$

or equivalently, via Lagrangian relaxation,

$$
\operatorname* { m a x } _ { C _ { T } } \mathcal { R } ( C _ { T } , Q ) - \eta \operatorname { C o s t } ( C _ { T } ) ,
$$

where $\eta \geq 0$ selects an operating point on the quality-efficiency frontier. (We use η to avoid overloading λ, which later denotes the MMR trade-off parameter.)

Joint optimization over all subsets is intractable, so we decompose the problem into local pruning decisions at three intervention points:

• Pre-Retrieval. A candidate subquery $s _ { t } \in S$ is scored by predicted marginal value; low-value subqueries are filtered before incurring retrieval cost.

• Post-Retrieval. Each retrieved item $c \in C _ { s _ { t } }$ is evaluated against $C _ { t } ;$ low-value items are discarded before influencing further branch expansion.

• Pre-Synthesis. The final context $C _ { T }$ is pruned to retain only high-value evidence before report generation.

Stage combinations. A one-stage configuration prunes at exactly one point (Post-Retrieval or Pre-Synthesis); a two-stage configuration combines Post-Retrieval + Pre-Synthesis; a three-stage configuration prunes at all three points. We exclude Pre-Retrieval-only and other partial Pre-Retrieval combinations from the one- and two-stage analyses: Pre-Retrieval decisions are necessarily more predictive and error-sensitive than later-stage decisions conditioned on retrieved context. This decomposition lets us isolate whether gains arise from reducing search expansion, compressing the final synthesis context, or both.

![](images/6c690dad93b53a960a8a6fdde4dcdb015858fb18182a0889e358fc1136010e84.jpg)  
Figure 1: Overview of the deep research pipeline. Given a user query, the system proceeds through four stages: (1) planning via query decomposition, (2) retrieval and branch expansion, (3) context aggregation, and (4) final report synthesis. We study pruning at three intervention points: Pre-Retrieval, which filters candidate subqueries before search; Post-Retrieval, which filters retrieved context during branch expansion; and Pre-Synthesis, which compresses the accumulated context before report generation.

## 3.2 Marginal Value Estimation as a Decision Problem

At each pruning point, the system decides whether candidate x contributes sufficient marginal value to retain, given query Q and accumulated context $C _ { t } \colon$ at Pre-Retrieval, x is a subquery $s \in S ;$ at Post-Retrieval and Pre-Synthesis, x is a context item c. We formalize this through a unified scoring function $\mathcal { V } ( \boldsymbol { x } \mid C _ { t } , Q )$ , retaining x if its score exceeds a stage-specific threshold $( \tau _ { \mathrm { p r e } } , \ \tau _ { \mathrm { p o s t } } .$ , or $\tau _ { \mathrm { s y n } } )$ . Because relevance, novelty, diversity, and coverage may matter differently at different stages, selecting the right instantiation of V motivates the cross-stage comparison in Section 5. Algorithm 1 shows the full pipeline.

## 4 Motivating Analysis

The unpruned pipeline is expensive by design: it explores 29.0 nodes, consumes 375.4k tokens, and requires 3422.6s per report (Table 1). A key source of inefficiency is that low-value context survives until late in the pipeline. The unpruned pipeline’s built-in pre-synthesis trimming reduces accumulated context from 66.10 to 44.08 items (33.3% fewer items; 34.06% fewer tokens), meaning substantial retrieved material is discarded only after most retrieval and processing cost has been paid. Since token cost is dominated by retrieval and result processing rather than planning or query generation, late trimming compresses the synthesis prompt but cannot recover earlier search costs. This motivates our central question: can low-value context be removed earlier, and does intervening at $P r e \mathrm { - }$ Retrieval, Post-Retrieval, or Pre-Synthesis, meaningfully affect quality and cost trade-offs?

Algorithm 1 Deep Research with Context Pruning   
Require: Query Q; thresholds $\tau _ { \mathrm { p r e } } , \tau _ { \mathrm { p o s t } } , \tau _ { \mathrm { s y n } }$   
1: $\mathbf { \bar { \phi } } _ { C _ { 0 } } \gets \bar { \varnothing } , \bar { s _ { \ell - } }$ Decompose $( Q ) , { \dot { t } } \gets 1 ^ { \cdot }$   
2: while $ { \boldsymbol { S } } \neq \emptyset$ do   
3: Select and remove subquery $s _ { t } \in S$   
4: ▷ Pre-retrieval pruning   
5: if $\mathcal { V } ( s _ { t } \mid C _ { t - 1 } , Q ) < \tau _ { \mathrm { p r e } }$ then   
6: $C _ { t } \gets C _ { t - 1 }$   
7: else   
8: Retrieve context items $C _ { s _ { t } }$ using s<sub>t</sub>   
9: ▷ Post-retrieval pruning   
10: $C _ { s _ { t } } \gets \{ c \in \bar { C _ { s _ { t } } } : \mathcal { V } ( \bar { c } | C _ { t - 1 } , Q ) \geq \tau _ { \mathrm { p o s t } } \}$   
11: $C _ { t } \gets \dot { C _ { t - 1 } } \cup C _ { s _ { 1 } }$   
12: Add new subqueries spawned from $C _ { s _ { t } }$ to S   
13: end if   
14: $t \gets t + 1$   
15: end while   
16: ▷ Pre-synthesis pruning   
17: $\widehat { C } _ { T }  \{ c \in C _ { T } : \mathcal { V } ( c \mid C _ { T } , Q ) \geq \tau _ { \mathrm { s y n } } \}$   
18: Generate final report from $\widehat { C } _ { T }$

## 5 Pruning Strategies

We compare heuristic, lexical, LLM-based, and learned pruning strategies, each instantiating a different notion of marginal value under the stageaware framework of Section 3.

Let $\boldsymbol { e } ( \cdot ) \in \mathbb { R } ^ { d }$ denote the embedding function, and let

$$
q = e ( Q )
$$

be the root-query embedding. Candidate items are denoted by x, and retained context by $C .$ When a formula operates in embedding space, we write $e ( x )$ for the embedding of candidate x and $e ( c )$ for the embedding of a retained item $c \in C$ . We use cosine similarity

$$
\mathrm { s i m } ( u , v ) = \frac { u ^ { \top } v } { \| u \| \| v \| }
$$

for vector inputs $u , v$ . For methods that require a nonnegative scalar query-relevance weight, we define

$$
w ( x ) = \operatorname* { m a x } ( \sin ( e ( x ) , q ) , 0 ) .
$$

We use $w ( x )$ exclusively for scalar relevance weights. For projection-based methods, $P _ { C } ( \cdot )$ denotes orthogonal projection onto the span of $\{ e ( c )$ $c \in C \}$ . When used in the DPP kernel, we interpret similarity as the inner product of $\ell _ { 2 } .$ -normalized embeddings, i.e., sim $( e ( i ) , e ( j ) ) \ : = \ : \langle \widetilde { e } ( i ) , \widetilde { e } ( j ) \rangle$ , so the resulting kernel is a weighted Gram matrix.

## 5.1 Compared Strategies

We compare pruning strategies ranging from lightweight heuristics to LLM-based and learned methods, each instantiating a different notion of marginal value. sec:heur-strat

## Heuristic Strategies

Heuristic strategies estimate marginal value through fixed scoring functions requiring no training or inference. We define the following heuristic strategies below.

Maximal Marginal Relevance (MMR)(§ B.1): MMR balances query relevance against redundancy with respect to already retained context (Carbonell and Goldstein, 2017):

$$
\begin{array} { l } { { V _ { \mathrm { M M R } } ( x \mid C , Q ) = \lambda \sin ( e ( x ) , q ) \nonumber \qquad } } \\ { { \qquad - ( 1 - \lambda ) \displaystyle \operatorname* { m a x } _ { c \in C } \sin ( e ( x ) , e ( c ) ) , } } \end{array}
$$

where $\lambda \in [ 0 , 1 ]$ controls the relevance–novelty trade-off.

Geometric Residual Novelty (GRN)(§ B.1): GRN measures whether a candidate introduces a new semantic direction relative to the retained context:

$$
V _ { \mathrm { G R N } } ( x \mid C ) = \| e ( x ) - P _ { C } ( e ( x ) ) \| _ { 2 } ,
$$

where $P _ { C } ( e ( x ) )$ is the projection of $e ( x )$ onto the subspace spanned by the retained context embeddings.

Centroid Drift (CD)(§ B.1): CD measures whether adding a candidate shifts the semantic center of the retained context (Radev et al., 2004). Let $\begin{array} { r } { \mu _ { C } = \frac { 1 } { | C | } \sum _ { c \in C } e ( c ) } \end{array}$ . We define

$$
\mathcal { V } _ { \mathrm { C D } } ( x \mid C ) = 1 - \sin ( \mu _ { C } , \mu _ { C \cup \{ x \} } ) .
$$

Larger scores indicate larger semantic drift. Since cosine similarity lies in $[ - 1 , 1 ]$ , this score lies in $[ 0 , 2 ]$ , although empirical values are typically much smaller.

Determinantal Point Processes (DPP)(§ B.1): DPPs favor subsets that are both relevant and diverse (Kulesza and Taskar, 2012).With kernel entries

$$
L _ { i j } = w ( i ) w ( j ) \langle \tilde { e } ( i ) , \tilde { e } ( j ) \rangle ,
$$

where $w ( i )$ is the nonnegative query-relevance weight defined above and the gain of adding x is

$$
V _ { \mathrm { D P P } } ( x \mid C ) = { \frac { \operatorname * { d e t } ( L _ { C \cup \{ x \} } ) } { \operatorname * { d e t } ( L _ { C } ) } } .
$$

Redundant candidates contribute less additional volume and receive smaller gains.

Submodular Coverage (SC)(§ B.1): SC measures how much a retained set covers the semantic space of the candidate pool while accounting for token cost (Lin and Bilmes, 2011). With coverage function

$$
F ( C ) = \sum _ { j \in P } \operatorname* { m a x } _ { i \in C } \left[ w ( i ) \sin ( e ( i ) , e ( j ) ) \right]
$$

the token-normalized marginal gain is

$$
\mathcal { V } _ { \mathrm { S C } } ( x \mid C , Q ) = \frac { F ( C \cup \{ x \} ) - F ( C ) } { \cos ( x ) } .
$$

Combined and lexical variants (§B.1). We also study secondary variants that either combine relevance, novelty, and coverage into a single score or replace dense semantic similarity with cheaper lexical proxies such as TF–IDF cosine similarity and bigram overlap. These variants test whether richer signal integration or lower-cost similarity measures improve the trade-off frontier.

We further evaluate a small set of targeted mixed variants $( C D + S C , C D + L L M ,$ and $S C + L L M )$ to test whether different pruning objectives are complementary across stages. These are not intended to exhaust all pairings, but to probe whether noveltysensitive early pruning combines effectively with coverage-aware or semantic late-stage refinement.

## LLM-based Strategy (§ B.2)

We evaluate methods in which a language model acts as a pruning judge, predicting whether a candidate should be retained given the current query, context, and candidate set. This method can capture richer semantic judgments than fixed heuristics, but it also introduces additional inference cost.

## Learned Strategy (§ B.3)

We also study a learned pre-retrieval controller trained on execution trajectories as an exploratory test of whether a lightweight data-driven proxy can approximate a downstream utility signal.

## 6 Experimental Setup

## 6.1 Benchmark

We evaluate pruning strategies on DeepResearch-Gym (Coelho et al., 2025), an open-source evaluation sandbox for deep research systems. Its evaluation protocol is built on 1,000 complex, highengagement queries from the Researchy Questions dataset (Rosset et al., 2024). Because evaluating every configuration on the full benchmark is computationally expensive, we report results on a fixed sample of 100 queries.

## 6.2 Pipeline and Baseline

Our experiments are built on the GPT-Researcher framework, which performs multi-step query decomposition, retrieval, aggregation, and report generation. The baseline is the standard GPT-Researcher pipeline without any explicit pruning policy, including the framework’s default late-stage context trimming before synthesis. All pruning variants are implemented on top of the same underlying pipeline, and where applicable, we reuse cached subqueries and retrieved evidence across runs, so observed differences in quality, token usage, node count, and latency are attributable to pruning decisions alone.

## 6.3 Compared Configurations

We compare pruning strategies at three intervention points in the pipeline: Pre-Retrieval, Post-Retrieval, and Pre-Synthesis. We evaluate onestage policies, multi-stage combinations, heuristic criteria, LLM-based pruning, and learned pruning under a shared execution framework. Unless otherwise specified, all methods use the same underlying generator and report-writing configuration, and differ only in how and where pruning is applied.

Unless otherwise noted, the main paper reports one representative operating point per method rather than exhaustively tuning each pruning family to its best possible frontier. Appendix B.7 reports local threshold sweeps for five representative postretrieval methods and shows that the published settings lie within stable sampled operating regions under a 2% quality-degradation criterion.

## 6.4 Evaluation Metrics

We evaluate methods along four dimensions: quality, relevance, faithfulness, and efficiency.

• Quality (§ A.2): Measured with a fixed rubricbased LLM judge applied uniformly across all methods. Because absolute rubric scores can shift across judges, we interpret these values primarily as relative comparisons under a fixed independent judge. Appendix Table 4 reports a judge-sensitivity analysis.

• Relevance (§ A.2): KPR+KPC (Key Point Recall + Key Point Contradiction), measuring coverage of key informational points from ground-truth documents while penalizing conflicting statements.

• Faithfulness (§ A.2): Citation Recall (Cit.), measuring the fraction of report claims grounded in a retrievable source.

• Efficiency (§ A.3): Node count (Nodes), total token cost in thousands (Tokens), and wall-clock runtime (Time).

All metrics are mean values across 100 reports. Formal definitions are in Appendix A.1.

## 7 Results and Discussion

We next compare pruning strategies under the shared evaluation setup above, varying both the pruning stage and the pruning rule while keeping the underlying GPT-Researcher pipeline fixed. This setting isolates the effect of pruning decisions from changes to generation or retrieval. The following subsections examine in detail which pruning stages are most effective, which heuristics work best at each stage, and how quality, efficiency, relevance, and faithfulness trade off across one-, two-, and three-stage settings.

## 7.1 One-Stage Pruning: Post-Retrieval vs. Pre-Synthesis

Best quality vs. best efficiency. Table 1 shows a sharp contrast between the strongest quality point and the strongest efficiency point. Under the fixed rubric-based LLM judge used in the main experiments, the highest one-stage quality is Pre-Synthesis Hybrid at 60.68, improving on the baseline (57.83) by +2.85 points. Its efficiency gains are modest, however: token usage falls only to 332.3k and runtime remains high at 3834.1s, confirming that late pruning can refine the final synthesis context but cannot recover most upstream search cost. By contrast, the strongest efficiency point is Post-Retrieval MMR, which reduces token cost to 114.6k, explored nodes to 8.84, and runtime to 1379.8s. Relative to the baseline, this is roughly 69.5% lower token usage and 59.7% lower runtime while retaining 56.62 overall quality, or about 97.9% of baseline quality. This difference is intuitive: MMR is most effective early, where its relevance–redundancy trade-off prevents lowvalue branch expansion before retrieval and resultprocessing costs are incurred, whereas Hybrid is more useful as a late-stage quality refinement mechanism.

Table 1: One-stage pruning results. Full metric breakdown appears in the appendix. Tokens are reported in thousands (k). Bold denotes the best value among pruned methods within each stage block.
<table><tr><td>Method</td><td>Nodes</td><td>Tokens</td><td>Time</td><td>Ov.Quality</td><td>KPR+KPC</td><td>Cit.</td></tr><tr><td>Baseline</td><td>29.0</td><td>375.4</td><td>3422.6</td><td>57.83</td><td>70.23</td><td>95.54</td></tr><tr><td colspan="7">Post-Retrieval Pruning</td></tr><tr><td>MMR (§B.1)</td><td>8.84</td><td>114.6</td><td>1379.8</td><td>56.62</td><td>63.49</td><td>91.70</td></tr><tr><td>GRN (§B.1)</td><td>13.31</td><td>175.7</td><td>2087.9</td><td>57.02</td><td>42.89</td><td>94.40</td></tr><tr><td>CD (§B.1)</td><td>10.47</td><td>137.5</td><td>1605.7</td><td>56.84</td><td>41.29</td><td>92.48</td></tr><tr><td>SC (§B.1)</td><td>10.82</td><td>141.9</td><td>1737.2</td><td>55.22</td><td>47.94</td><td>91.97</td></tr><tr><td>DPP (§B.1)</td><td>9.88</td><td>129.6</td><td>1520.6</td><td>54.51</td><td>43.33</td><td>95.62</td></tr><tr><td>Combined (§B.1)</td><td>9.00</td><td>117.8</td><td>1399.5</td><td>56.03</td><td>64.13</td><td>92.59</td></tr><tr><td>Hybrid (§B.1)</td><td>9.28</td><td>121.3</td><td>1449.5</td><td>57.63</td><td>65.23</td><td>92.16</td></tr><tr><td>LLM (§B.2)</td><td>14.90</td><td>211.8</td><td>2310.7</td><td>59.65</td><td>49.65</td><td>93.54</td></tr><tr><td colspan="7">Pre-Synthesis Pruning</td></tr><tr><td>MMR (§B.1)</td><td>29.0</td><td>366.0</td><td>4446.9</td><td>57.77</td><td>65.26</td><td>90.41</td></tr><tr><td>GRN (§B.1)</td><td>29.0</td><td>374.1</td><td>4493.3</td><td>55.98</td><td>41.67</td><td>90.38</td></tr><tr><td>CD (§B.1)</td><td>29.0</td><td>374.5</td><td>4562.4</td><td>52.38</td><td>43.75</td><td>90.19</td></tr><tr><td>SC (§B.1)</td><td>29.0</td><td>384.7</td><td>4533.3</td><td>53.22</td><td>42.09</td><td>90.78</td></tr><tr><td>DPP (§B.1)</td><td>29.0</td><td>374.4</td><td>4512.4</td><td>54.70</td><td>54.07</td><td>90.24</td></tr><tr><td>Combined (§B.1)</td><td>29.0</td><td>374.5</td><td>4408.6</td><td>59.38</td><td>66.32</td><td>93.79</td></tr><tr><td>Hybrid (§B.1)</td><td>29.0</td><td>332.3</td><td>3834.1</td><td>60.68</td><td>65.62</td><td>95.07</td></tr><tr><td>LLM (§B.2)</td><td>29.0</td><td>386.7</td><td>4512.0</td><td>57.17</td><td>44.47</td><td>92.43</td></tr></table>

Faithfulness and evidence retention. The best one-stage citation recall among pruned methods is Post-Retrieval DPP at 95.62, whereas Pre-Synthesis Hybrid achieves the highest overall quality. This mismatch illustrates a central trade-off in the paper: stronger final report quality does not necessarily imply stronger evidence retention. More importantly, no pruned one-stage method surpasses the baseline on KPR+KPC, indicating that compression can preserve or improve report quality while still discarding evidence needed for maximal relevance retention. Among the post-retrieval heuristics, DPP remains more competitive on citation recall because its diversity objective favors retaining a broader evidence set, whereas MMR is more aggressive in removing semantically overlapping content and therefore delivers stronger efficiency gains.

Table 2: Two-stage pruning results. The full metric breakdown appears in Appendix Table 5. Tokens are reported in thousands (k). Bold denotes the best value within each subsection among pruned methods. A single method name indicates that the same pruning rule is used at both Post-Retrieval and Pre-Synthesis; “A + B” indicates method A at Post-Retrieval and method B at Pre-Synthesis.
<table><tr><td>Method</td><td>Nodes</td><td>Tokens</td><td>Time</td><td>Ov.Quality</td><td>KPR+KPC</td><td>Cit.</td></tr><tr><td>Baseline</td><td>29.0</td><td>375.4</td><td>3422.6</td><td>57.83</td><td>70.23</td><td>95.54</td></tr><tr><td colspan="7">Main heuristic families</td></tr><tr><td>MMR (§B.1)</td><td>8.84</td><td>114.6</td><td>1381.3</td><td>56.40</td><td>65.16</td><td>92.61</td></tr><tr><td>GRN (§B.1)</td><td>13.2</td><td>175.2</td><td>2080.2</td><td>57.94</td><td>64.47</td><td>93.06</td></tr><tr><td>CD (§B.1)</td><td>9.02</td><td>138.2</td><td>1619.8</td><td>57.44</td><td>43.23</td><td>92.94</td></tr><tr><td>SC (§B.1)</td><td>10.80</td><td>142.1</td><td>1776.5</td><td>57.00</td><td>62.23</td><td>94.74</td></tr><tr><td>DPP (§B.1)</td><td>9.2</td><td>122.8</td><td>1462.3</td><td>56.73</td><td>62.33</td><td>93.62</td></tr><tr><td>LLM (§B.2)</td><td>15.57</td><td>226.6</td><td>2632.3</td><td>58.27</td><td>48.61</td><td>93.01</td></tr><tr><td colspan="7">Mixed variants</td></tr><tr><td>Combined (§B.1)</td><td>8.98</td><td>117.6</td><td>1392.2</td><td>54.87</td><td>64.88</td><td>93.03</td></tr><tr><td>Hybrid (§B.1)</td><td>9.30</td><td>121.7</td><td>1457.4</td><td>56.92</td><td>64.25</td><td>93.51</td></tr><tr><td>CD + SC (§B.1)</td><td>10.45</td><td>137.5</td><td>1599.6</td><td>59.47</td><td>44.29</td><td>89.96</td></tr><tr><td>CD + LLM (§B.1)</td><td>10.24</td><td>136.0</td><td>1589.3</td><td>58.65</td><td>63.40</td><td>94.34</td></tr><tr><td>SC + LLM (§B.1)</td><td>10.54</td><td>139.1</td><td>1700.6</td><td>58.08</td><td>64.54</td><td>93.43</td></tr></table>

## 7.2 Two-Stage Pruning: Post-Retrieval + Pre-Synthesis

Best quality vs. best efficiency. Table 2 shows that combining early and late pruning yields the strongest practical trade-offs in the study. Under the fixed rubric-based LLM judge used in the main experiments, the highest quality point among the evaluated two-stage variants is CD + SC, which reaches 59.47 overall quality, improving on the baseline by +1.64 points while reducing token cost by 63.4%, runtime by 53.3%, and node count from 29.0 to 10.45. This shows that multi-stage pruning can improve report quality while substantially shrinking the search process. The strongest pure efficiency point remains MMR, which reduces token cost further to 114.6k, nodes to 8.84, and runtime to 1381.3s, while reaching 56.40 overall quality. The contrast reflects the underlying objectives: early MMR is highly effective at removing redundant branches before they expand, whereas CD + SC benefits from a later coverage-aware refinement stage that improves final report quality.

Table 3: Three-stage pruning results. The full metric breakdown appears in Appendix Table 5. Tokens are reported in thousands (k). Bold denotes the best value within each subsection among pruned methods. A single method name means the same pruning rule is used at Pre-Retrieval, Post-Retrieval, and Pre-Synthesis; a notation of the form $\mathbf { \ddot { A } } + \mathbf { B } + \mathbf { C } ^ { \prime }$ indicates method A at Pre-Retrieval, method B at Post-Retrieval, and method C at Pre-Synthesis.
<table><tr><td>Method</td><td>Nodes</td><td>Tokens</td><td>Time</td><td>Ov.Quality</td><td>KPR+KPC</td><td>Cit.</td></tr><tr><td>Baseline</td><td>29.0</td><td>375.4</td><td>3422.6</td><td>57.83</td><td>70.23</td><td>95.54</td></tr><tr><td>Main heuristic families</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MMR (§B.1)</td><td>7.82</td><td>100.1</td><td>1157.7</td><td>55.90</td><td>63.43</td><td>91.84</td></tr><tr><td>GRN (§B.1)</td><td>11.57</td><td>150.0</td><td>1937.7</td><td>56.68</td><td>45.13</td><td>91.33</td></tr><tr><td>CD (§B.1)</td><td>9.27</td><td>120.6</td><td>1669.8</td><td>56.00</td><td>43.64</td><td>93.30</td></tr><tr><td>SC (§B.1)</td><td>9.35</td><td>121.8</td><td>1438.9</td><td>55.52</td><td>65.79</td><td>92.03</td></tr><tr><td>DPP (§B.1)</td><td>8.68</td><td>113.3</td><td>1626.2</td><td>55.41</td><td>46.33</td><td>95.40</td></tr><tr><td>LLM (§B.2)</td><td>10.12</td><td>143.6</td><td>1574.7</td><td>59.53</td><td>65.09</td><td>93.12</td></tr><tr><td>Mixed variants</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Lexical + CD + SC (§B.1)</td><td>9.26</td><td>120.1</td><td>1375.0</td><td>57.81</td><td>42.67</td><td>93.31</td></tr><tr><td>Combined (§B.1)</td><td>7.90</td><td>102.8</td><td>1227.6</td><td>55.00</td><td>62.50</td><td>90.83</td></tr><tr><td>Hybrid (§B.1)</td><td>8.16</td><td>106.3</td><td>1282.7</td><td>57.10</td><td>64.80</td><td>93.06</td></tr><tr><td>Learned Query + GRN + GRN (§B.3)</td><td>10.44</td><td>145.7</td><td>1744.4</td><td>58.13</td><td>51.78</td><td>95.48</td></tr></table>

Faithfulness and evidence retention. The twostage setting makes the trade-off frontier most visible. Among the main heuristic families, SC achieves the strongest citation recall (94.74), whereas MMR is strongest on cost and KPR+KPC (65.16) among pruned methods. This suggests that relevance–redundancy control is especially effective for reducing search cost, while coverageoriented objectives are better suited to retaining a broader evidence set for source grounding. LLMaugmented variants remain quality-competitive, but their extra inference cost reduces their end-to-end efficiency advantage. The strongest quality point, CD + SC, also shows that quality and faithfulness can diverge: it improves overall quality, but its citation recall (89.96) is weaker than both the baseline and SC.

## 7.3 Three-Stage Pruning: Pre-Retrieval + Post-Retrieval + Pre-Synthesis

In the three-stage setting, we compare the main heuristic families against a small set of richer or mixed variants that add lexical, learned, or LLMbased pruning components.

Best quality vs. best efficiency. Table 3 shows that three-stage pruning is most useful when the objective is maximal compression. Among the main heuristic families, MMR is the strongest compression point, reducing token cost to 100.1k, explored nodes to 7.82, and runtime to 1157.7s, or roughly 73.3% lower token usage than the baseline. Its overall quality (55.90) remains below both the baseline and the strongest quality-oriented variants, indicating that extending relevance–redundancy pruning across all three stages sharpens compression more than it improves end-task quality.

Faithfulness and evidence retention. Under the fixed rubric-based LLM judge used in the main experiments, the highest overall quality in this family is achieved by LLM at 59.53, while Learned Query + GRN attains the strongest citation recall (95.48) and SC the strongest KPR+KPC (65.79) among pruned methods. Adding a third stage therefore does not remove the trade-off between quality, compression, and evidence retention; it makes it more explicit. Relative to the strongest two-stage configurations, three-stage pruning improves compression more reliably than quality, making the third stage most valuable when aggressive cost reduction is prioritized over maximal end-task quality.

## 7.4 Cross-Stage Findings

How do pruning objectives differ? Across settings, methods based on explicit relevance– redundancy control, especially MMR, are the strongest compression tools. Their advantage is greatest when applied early, where pruning a redundant branch avoids not only final-context growth but also the downstream cost of retrieval, result processing, and recursive expansion. By contrast, coverage-oriented methods such as SC are less aggressive compressors but often stronger on evidence retention because they reward keeping a broader set of complementary support rather than eliminating overlap as aggressively as possible. Diversity-oriented methods such as DPP behave similarly, favoring a wider evidence set and therefore remaining more competitive on citation recall than on pure efficiency. Geometric novelty methods such as GRN preserve novel semantic directions but are less tightly coupled to aggressive cost reduction. Centroid-based methods such as CD lie between these extremes: they capture novelty through shifts in the semantic center of the retained context, helping broader contextual refinement, but with a coarser signal than direct relevance– redundancy control and therefore a weaker alignment with aggressive efficiency gains.

How do methods change across stages? Holding the pruning rule fixed reveals a second consistent pattern. For MMR, moving from one-stage to two-stage leaves the strongest efficiency point essentially unchanged (114.6k tokens and 8.84 nodes in both settings), while three-stage pruning pushes compression further, reducing token cost to 100.1k and nodes to 7.82, but with weaker overall quality. This suggests that MMR captures most of its benefit once applied at Post-Retrieval, and that additional stages mainly sharpen compression rather than improve end-task quality. By contrast, SC benefits less from extra stages on pure efficiency, but remains comparatively stronger on evidence-retention objectives such as citation recall and KPR+KPC. More broadly, additional stages improve cost reduction more reliably than report quality, whereas later-stage refinement is most useful when the objective places greater weight on final synthesis quality than on maximal efficiency.

Cross-benchmark evidence. Although our primary analysis is conducted on DeepResearchGym, supplementary results on DeepResearch Bench (Appendix B.5, Tables 13 and 14) show the same broad efficiency pattern: earlier pruning yields substantially larger token and runtime savings than root-only pruning, and MMR remains one of the strongest compression-oriented heuristics. At the same time, method-level quality rankings are less stable across the two benchmarks (Table 13). We therefore interpret these results as directional support for the stage-ordering conclusion rather than as a full replication of the quality trade-offs observed on DeepResearchGym, and note that the two benchmarks use different evaluation protocols and report-quality metrics.

Which stages and configurations matter most? Comparing stages and configurations head-to-head, Post-Retrieval pruning delivers the largest efficiency gains while remaining relatively close to baseline quality, because it prevents low-value branches from expanding before retrieval and result-processing costs are incurred. By contrast, one-stage Pre-Synthesis pruning gives the strongest quality refinement, but is usually too late to recover most upstream cost. At the configuration level, twostage pruning offers the strongest practical qualityefficiency trade-off by combining early search control with late refinement. Three-stage pruning pushes compression further, but with smaller returns for report quality, making it most useful when the objective is maximal cost reduction rather than maximal end-task quality.

When do Mixed variant methods help? Mixed variant and later-stage pruning rules become more competitive when the target is final report quality rather than absolute efficiency. Pre-Synthesis Hybrid gives the strongest one-stage quality, and LLM-based variants are often quality-competitive because they can make more flexible semantic decisions than fixed heuristics. However, their additional inference cost limits their efficiency advantage, making them most attractive when quality is prioritized over absolute cost.

What the exploratory learned controller suggests. The learned pre-retrieval controller serves as a small-scale data-driven alternative to handdesigned marginal-value criteria. In our results, it produces viable operating points in a limited number of settings, especially when predictive query filtering is useful, but it does not provide strong evidence of consistent superiority over the strongest heuristic baselines. This is informative in itself: most of the attainable gains in our study come from a well-matched pruning objective and stage placement, while the learned component is best viewed as an exploratory proof of concept rather than a mature replacement for simpler heuristics.

## 8 Conclusion

We presented the first systematic stage-aware study of marginal-value-based pruning for long-horizon deep research agents. Across Pre-Retrieval, Post-Retrieval, and Pre-Synthesis, stage placement is often more consequential than the scoring rule: early pruning, especially at Post-Retrieval, yields the largest savings in nodes, tokens, and runtime, whereas later pruning mainly refines the final synthesis context. No single strategy dominates across objectives. Under the fixed rubric-based judge used in the main experiments, Pre-Synthesis Hybrid gives the strongest one-stage quality, CD + SC the strongest observed quality-efficiency tradeoff, and three-stage pruning is most useful for maximal compression. Learned pruning remains exploratory; most gains come from careful stage placement and lightweight heuristics. Because report quality and evidence retention can diverge under pruning, policies should be evaluated jointly in terms of quality, efficiency, relevance, and faithfulness. Overall, our results support a stage-aware view of pruning and offer practical guidance for efficient long-horizon research agents.

## 9 Limitations

This study has several limitations. First, our empirical analysis is conducted within a fixed deep research pipeline and benchmark setup, so the exact trade-offs may change under different agent architectures, retrieval systems, or underlying language models. Our conclusions are therefore strongest as a comparative study of pruning strategies within this setting rather than as universal claims about all long-horizon agents. Although supplementary experiments on DeepResearch Bench (Appendix B.5, Tables 13 and 14) support the broad efficiency advantage of earlier pruning, method-level quality rankings are less stable across benchmarks, suggesting that quality-sensitive conclusions depend more strongly on the evaluation protocol and task formulation than the basic stage-ordering effect. Our quality metric is also judge-dependent. Appendix Table 4 shows that absolute rubric scores can shift substantially across judge choices, even for the same generated reports. We therefore interpret quality-sensitive conclusions more cautiously than efficiency results, and treat the reported quality values primarily as relative comparisons under a fixed independent judge rather than as stable absolute measurements.

Second, several pruning methods require threshold or hyperparameter choices, and the best operating point remains task-dependent. Appendix B.7 reports local threshold sweeps for five representative post-retrieval methods and shows that the published settings lie within stable sampled regions under a 2% quality-degradation criterion. These sweeps suggest that our main conclusions are not driven by isolated brittle thresholds, although finergrained method ordering may still shift under different threshold choices or under exhaustive permethod tuning.

Finally, our evaluation metrics do not capture all aspects of utility or reliability. Automatic measures of report quality, relevance, and citation recall only partially reflect factual correctness, completeness of evidence, and usefulness to end users. Our findings should therefore be interpreted as a systematic empirical study of pruning trade-offs rather than a complete evaluation of deployment readiness.

## 10 Ethics Statement

Improving the efficiency of long-horizon research agents can reduce computational cost, latency, and environmental burden. However, more efficient systems can also make it easier to scale the production of misleading or low-quality synthesized content.

Pruning introduces an additional risk: imperfect pruning signals may discard contradictory evidence, minority viewpoints, or important caveats while preserving fluent but insufficiently supported summaries. Pruning should therefore not be treated as a substitute for verification. We recommend pairing pruning-based research agents with source transparency, citation auditing, and human oversight, especially in high-stakes domains such as medicine, law, public policy, and education.

Learned pruning models may also inherit biases from the trajectories and evaluation signals used to train them. Their outputs should therefore be understood as objective-driven decisions rather than as ground truth about what evidence is important.

## References

Andrei Z. Broder. 1997. On the resemblance and containment of documents. In Compression and Complexity of SEQUENCES 1997, Positano, Amalfitan Coast, Salerno, Italy, June 11-13, 1997, Proceedings, pages 21–29. IEEE.

Jaime G. Carbonell and Jade Goldstein. 2017. The use of mmr, diversity-based reranking for reordering documents and producing summaries. SIGIR Forum, 51(2):209–210.

Nayoung Choi, Jonathan Zhang, and Jinho D. Choi. 2026. DYCP: dynamic context pruning for longform dialogue with llms. CoRR, abs/2601.07994.

João Coelho, Jingjie Ning, Jingyuan He, Kangrui Mao, Abhijay Paladugu, Pranav Setlur, Jiahe Jin, Jamie Callan, João Magalhães, Bruno Martins, and Chenyan Xiong. 2025. Deepresearchgym: A free, transparent, and reproducible evaluation sandbox for deep research. CoRR, abs/2505.19253.

Huiqiang Jiang, Qianhui Wu, Chin-Yew Lin, Yuqing Yang, and Lili Qiu. 2023. Llmlingua: Compressing prompts for accelerated inference of large language models. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023, pages 13358–13376. Association for Computational Linguistics.

Huiqiang Jiang, Qianhui Wu, Xufang Luo, Dongsheng Li, Chin-Yew Lin, Yuqing Yang, and Lili Qiu. 2024. Longllmlingua: Accelerating and enhancing llms in long context scenarios via prompt compression. Preprint, arXiv:2310.06839.

Alex Kulesza and Ben Taskar. 2012. Determinantal point processes for machine learning. Found. Trends Mach. Learn., 5(2-3):123–286.

Marco Laju, Donghyun Son, Saurabh Agarwal, Nitin Kedia, Myungjin Lee, Jayanth Srinivasa, and Aditya

Akella. 2026. Nalar: An agent serving framework. CoRR, abs/2601.05109.

Shiyu Li, Yang Tang, Yifan Wang, Peiming Li, and Xi Chen. 2025. Reseek: A self-correcting framework for search agents with instructive rewards. CoRR, abs/2510.00568.

Yucheng Li. 2023. Unlocking context constraints of llms: Enhancing context efficiency of llms with self-information-based content filtering. CoRR, abs/2304.12102.

Hui Lin and Jeff Bilmes. 2011. A class of submodular functions for document summarization. In Proceedings ofthe 49th Annual Meeting ofthe Associationfor Computational Linguistics: Human Language Technologies, pages 510–520, Portland, Oregon, USA. Association for Computational Linguistics.

Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. 2024. Lost in the middle: How language models use long contexts. Trans. Assoc. Comput. Linguistics, 12:157–173.

Maria Emilia Mazzolenis and Ruirui Zhang. 2025. Agent WARPP: workflow adherence via runtime parallel personalization. CoRR, abs/2507.19543.

Lunyiu Nie, Nedim Lipka, Ryan A. Rossi, and Swarat Chaudhuri. 2026. Efficient tree-structured deep research with adaptive resource allocation. Preprint, arXiv:2510.05145.

Nikos Pagonas, Yeounoh Chung, Kostis Kaffes, and Arvind Krishnamurthy. 2025. Cortex: Workflowaware resource pooling and scheduling for agentic serving. Preprint, arXiv:2510.14126.

Dragomir R. Radev, Hongyan Jing, Magorzata Sty, and Daniel Tam. 2004. Centroid-based summarization of multiple documents. Inf. Process. Manag., 40(6):919–938.

Juan Enrique Ramos. 2003. Using tf-idf to determine word relevance in document queries.

Corby Rosset, Ho-Lam Chung, Guanghui Qin, Ethan C. Chau, Zhuo Feng, Ahmed Awadallah, Jennifer Neville, and Nikhil Rao. 2024. Researchy questions: A dataset of multi-perspective, decompositional questions for LLM web agents. CoRR, abs/2402.17896.

Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. 2023. Reflexion: language agents with verbal reinforcement learning. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023.

Joel A. Tropp and Anna C. Gilbert. 2007. Signal recovery from random measurements via orthogonal matching pursuit. IEEE Trans. Inf. Theory, 53(12):4655–4666.

Fangyuan Xu, Weijia Shi, and Eunsol Choi. 2023. RECOMP: improving retrieval-augmented lms with compression and selective augmentation. CoRR, abs/2310.04408.

Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan. 2023a. Tree of thoughts: Deliberate problem solving with large language models. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R. Narasimhan, and Yuan Cao. 2023b. React: Synergizing reasoning and acting in language models. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net.

## A Appendix

This appendix provides supporting detail for the main paper along three dimensions. First, it defines the reported metrics and derived efficiency quantities, including node counts, token accounting, runtime decomposition, and stage-wise pruning effectiveness. Second, it expands the method descriptions, giving the mathematical form and stagespecific role of each pruning strategy summarized in the main text. Third, it reports the full comparison tables and robustness analyses that underlie the compact main-paper results.

The purpose of these materials is to make the empirical findings more transparent rather than to introduce new claims. In particular, the appendix tables expose the full metric breakdown behind the stage-aware comparisons in the main paper, clarify where efficiency gains arise within the pipeline, and show how different pruning strategies behave under more detailed accounting than can fit in the main text.

## A.1 Metric Computation Details

All reported cost, runtime, and pruning metrics are aggregated over the same set of evaluation reports. For any per-report quantity $x _ { i }$ measured on report $i \in \{ 1 , \ldots , N \}$ , we report the sample mean

$$
\bar { x } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } x _ { i }
$$

and, when available, the standard error

$$
\mathrm { S E } ( x ) = \frac { s _ { x } } { \sqrt { N } } , \qquad s _ { x } ^ { 2 } = \frac { 1 } { N - 1 } \sum _ { i = 1 } ^ { N } ( x _ { i } - \bar { x } ) ^ { 2 } .
$$

Unless otherwise stated, tables report values as mean ± standard error.

## A.2 Quality, relevance, and faithfulness:

The quality dimensions reported in the main comparison tables, namely Overall, Clarity, Depth, Balance, Breadth, Support, and Insight, are obtained by running the generated reports through the Deep-ResearchGym evaluation pipeline and averaging the resulting scores over reports. The same procedure is used for the relevance metric (KPR+KPC) and the faithfulness metric (Citation Recall). These metrics are currently reported as means only.

## A.3 # Nodes:

For the baseline configuration with breadth 4 and depth 3, the nominal full search tree contains

$$
1 + 4 + 8 + 1 6 = 2 9
$$

nodes, consisting of one planning node and 28 research nodes. We report the node count computed as

$$
\mathrm { N } _ { i } ^ { \mathrm { c o r r } } = 1 + R _ { i } ,
$$

where $R _ { i }$ is the number of result-processing calls in report i, and the leading 1 accounts for the planning node. We also report the average number of pruned nodes, denoted ${ \bar { P } } ,$ from the node-level pruning logs. The pruning rate shown in the appendix is then computed from the reported means as

$$
\mathrm { P r u n i n g ~ R a t e } = { \frac { \bar { P } } { \bar { N } ^ { \mathrm { c o r r } } } } \times 1 0 0 .
$$

## A.4 Token cost:

For each report, total token cost is defined as the sum of input and output tokens across all tokentracked pipeline stages:

$$
T _ { i } = T _ { i } ^ { \mathrm { i n } } + T _ { i } ^ { \mathrm { o u t } } = \sum _ { \phi \in \Phi } \left( T _ { i , \phi } ^ { \mathrm { i n } } + T _ { i , \phi } ^ { \mathrm { o u t } } \right) ,
$$

where Φ includes the logged token phases such as planning, query generation, pre-retrieval pruning (when the pruning method itself invokes an LLM), result processing, embedding, and other tracked model calls. The main efficiency table reports $\bar { T }$ as # Tokens. The token-accounting table further decomposes this into average input tokens, average output tokens, and total tokens.

## A.5 Estimated tokens saved by pruning:

For each pruning stage s in report i, let $C _ { i , s } ^ { \mathrm { b e f o r e } }$ and $C _ { i , s } ^ { \mathrm { a f t e r } }$ denote the tokenized context size before and after pruning. The estimated tokens saved at that stage are

$$
\Delta T _ { i , s } = C _ { i , s } ^ { \mathrm { b e f o r e } } - C _ { i , s } ^ { \mathrm { a f t e r } } .
$$

The total estimated tokens saved by pruning in report i are then

$$
T _ { i } ^ { \mathrm { s a v e d } } = \sum _ { s } \Delta T _ { i , s } .
$$

## A.6 Estimated token reduction:

The estimated pre-pruning token budget of a method is defined as the final observed token cost plus the estimated tokens removed by pruning:

$$
T _ { i } ^ { \mathrm { b e f o r e } } = T _ { i } + T _ { i } ^ { \mathrm { s a v e d } } .
$$

The estimated total token reduction percentage is then

$$
\mathrm { T o k e n ~ R e d u c t i o n } _ { i } = \frac { T _ { i } ^ { \mathrm { s a v e d } } } { T _ { i } ^ { \mathrm { b e f o r e } } } \times 1 0 0 .
$$

The efficiency table reports the mean of this quantity over reports. This metric measures withinmethod pruning effectiveness, namely how much of the method’s own pre-pruning token budget is removed by pruning.

## A.7 Savings vs. Baseline:

To quantify absolute savings relative to the unpruned baseline, we compare the mean total token count of each method against the baseline mean total token count. The percentage savings relative to baseline are computed as

$$
{ \mathrm { S a v i n g s ~ v s . ~ B a s e l i n e } } = \left( 1 - { \frac { \bar { T } _ { \mathrm { m e t h o d } } } { \bar { T } _ { \mathrm { b a s e l i n e } } } } \right) \times 1 0 0 .
$$

When reported in absolute units, mean savings vs. baseline are expressed in thousands of tokens:

$$
\mathrm { M e a n } \mathrm { S a v i n g s ~ v s . ~ B a s e l i n e } = \frac { \bar { T } _ { \mathrm { b a s e l i n e } } - \bar { T } _ { \mathrm { m e t h o d } } } { 1 0 0 0 } .
$$

Thus, this quantity answers how many fewer tokens, in k tokens per report, a method uses on average relative to baseline.

## A.8 Token breakdown and token share by stage.

For each token phase $\phi ,$ we aggregate stage-wise token usage as

$$
T _ { i , \phi } = T _ { i , \phi } ^ { \mathrm { i n } } + T _ { i , \phi } ^ { \mathrm { o u t } } .
$$

The token breakdown table reports $\bar { T } _ { \phi }$ for each stage. The token-share table reports the fraction of the total token budget consumed by each stage:

$$
{ \mathrm { S h a r e } } _ { i , \phi } = { \frac { T _ { i , \phi } } { T _ { i } } } \times 1 0 0 .
$$

This decomposition reveals whether a method’s cost is dominated by planning, query generation, pruning, result processing, or embedding. In most methods, result processing dominates the total token budget; in methods with explicit LLM-based query pruning, the pruning stage itself also contributes nontrivially.

## A.9 Runtime metrics:

For each report, total runtime is measured as wallclock time from the start to the end of the pipeline. The runtime breakdown table reports the mean and standard error of the total runtime and of each logged stage-level latency. Let $\boldsymbol { L } _ { i , \psi }$ denote the duration of stage ψ in report i, with reported stage values $L _ { \psi }$

We define the measured pipeline stages as follows:

• Total Runtime (s): The total elapsed wall-clock time for the entire report generation pipeline.

• Planning: Measures the time to generate the initial follow-up questions that decompose the user query into high-level research directions. This stage occurs once at the root and seeds all subsequent stages.

• Query Generation (Query Gen): Measures the time to convert follow-up questions—both from Planning and any additional questions generated during Result Processing—into candidate queries suitable for retrieval.

• Query Pruning: Measures any extra latency incurred when candidate queries are explicitly filtered before retrieval.

• Research / Scraping: Measures the retrievalheavy stage in which the system issues searches, visits sources, and gathers raw evidence for the candidate queries.

• Result Processing (Result Proc.): Measures the time spent processing retrieved context into structured learnings, follow-up questions if needed (i.e. max depth not yet reached), and citations. These follow-up questions later feed into Query Generation.

• Branch Pruning: Measures the overhead of post-retrieval pruning during recursive branch aggregation, where candidate branches are evaluated and potentially discarded based on incremental utility.

• Root Pruning: Measures the cost of presynthesis pruning of the final aggregated context before the report is produced.

## A.10 Pruning-stage effectiveness:

To characterize where pruning happens within the pipeline, we report stage-wise pruning ratios and stage-wise token reductions at pre-retrieval, postretrieval, and pre-synthesis levels. For a given stage $s ,$ let $n _ { i , s } ^ { \mathrm { b e f o r e } }$ and $n _ { i , s } ^ { \mathrm { a f t e r } }$ denote the number of candidate items before and after pruning. The stage-level pruning ratio is

$$
\mathrm { R a t i o } _ { i , s } = \frac { n _ { i , s } ^ { \mathrm { b e f o r e } } - n _ { i , s } ^ { \mathrm { a f t e r } } } { n _ { i , s } ^ { \mathrm { b e f o r e } } } \times 1 0 0 .
$$

Likewise, if $C _ { i , s } ^ { \mathrm { b e f o r e } }$ and $C _ { i , s } ^ { \mathrm { a f t e r } }$ denote the tokenized context sizes before and after pruning, the corresponding stage-level token reduction is

$$
\mathrm { T o k e n R e d } _ { i , s } = \frac { C _ { i , s } ^ { \mathrm { b e f o r e } } - C _ { i , s } ^ { \mathrm { a f t e r } } } { C _ { i , s } ^ { \mathrm { b e f o r e } } } \times 1 0 0 .
$$

For pre-retrieval pruning, the item count refers to candidate queries; for post-retrieval and presynthesis pruning, it refers to retained context items. Reporting both quantities is useful because pruning by item count and pruning by token mass are not identical: removing half of the items does not necessarily remove half of the tokens.

## B Additional Method Details

This section provides additional detail on the pruning strategies compared in the main paper. We first introduce shared notation, then describe each strategy in turn, including its intuition, mathematical form, and stage-specific role in our pipeline.

We adopt the notation from Section 3. Let $e ( \cdot ) \in$ $\mathbb { R } ^ { d }$ denote the embedding function, and let

$$
q = e ( Q )
$$

be the embedding of the root query Q. A candidate item is denoted by x. Depending on the pruning stage, x may represent a generated subquery, a retrieved context item, or a context block considered for final synthesis. We write C for the current retained context, with elements $c _ { i } \in \mathbb { R } ^ { d }$

Our default semantic similarity is cosine similarity:

$$
\mathrm { s i m } ( u , v ) = \frac { u ^ { \top } v } { \| u \| \| v \| } .
$$

When budget-awareness is needed, we write cost $( x )$ for the token cost of candidate x.

When a formula operates in embedding space, we write $e ( x )$ for the embedding of candidate x and $e ( c )$ for the embedding of a retained item $c \in C$ We use

$$
w ( x ) = \operatorname* { m a x } ( \sin ( e ( x ) , q ) , 0 )
$$

for the nonnegative scalar query-relevance weight, reserving q exclusively for the query embedding. For projection-based methods, $P _ { C } ( \cdot )$ denotes orthogonal projection onto the span of $\{ e ( c ) : c \in$ C}.

## B.1 Heuristic Strategies

Maximal Marginal Relevance (MMR) Maximal Marginal Relevance (MMR) is a classical information retrieval criterion that balances query relevance against redundancy with respect to already selected content (Carbonell and Goldstein, 2017). In our setting, the MMR score of candidate x is

$$
\begin{array} { l } { { \displaystyle V _ { \mathrm { M M R } } ( x \mid C , Q ) = \lambda \sin ( e ( x ) , q ) \ } \ ~ } \\ { { \displaystyle ~ - \ ( 1 - \lambda ) \operatorname* { m a x } _ { c \in C } \sin ( e ( x ) , e ( c ) ) . } \ ~ } \end{array}
$$

where $\lambda \in [ 0 , 1 ]$ controls the relevance–novelty trade-off.

The first term rewards semantic alignment with the research query, while the second penalizes overlap with previously retained evidence. A candidate receives a high score only if it is both relevant and non-redundant. Because cosine similarity lies in $[ - 1 , 1 ]$ , the effective score range is bounded by the mixture weights, shifting toward query relevance as λ increases.

Operationally, MMR is used for Post-Retrieval pruning and Pre-Synthesis pruning. At Post-Retrieval, the score is used as an information-gain test for deciding whether a branch should be retained. At Pre-Synthesis, MMR is used greedily to construct a compact final context under a budget by repeatedly selecting the item with the highest marginal MMR score.

Geometric Residual Novelty Geometric Residual Novelty (GRN) measures whether a candidate introduces a new semantic direction relative to the retained context. The idea is inspired by residualbased subspace selection methods such as orthogonal matching pursuit (Tropp and Gilbert, 2007). Let $\mathcal { U } ( C )$ denote the subspace spanned by the retained context embeddings, and let $P _ { C } ( e ( x ) )$ denote the orthogonal projection of e(x) onto that subspace.. We define the residual

$$
\mathrm { r e s } ( x ; C ) = e ( x ) - P _ { C } ( e ( x ) ) ,
$$

and the gain

$$
V _ { \mathrm { G R N } } ( x \mid C ) = \| \operatorname { r e s } ( x ; C ) \| _ { 2 }
$$

If the residual norm is small, then x lies largely within the span of existing evidence and contributes little novelty. If the residual norm is large, then x contributes a new semantic direction. With unitnormalized embeddings, $\mathcal { V } _ { \mathrm { G R N } } \in [ 0 , 1 ]$

In our pipeline, GRN is used at Post-Retrieval to determine whether a newly retrieved branch adds genuinely new information relative to already accepted branches, and at Pre-Synthesis to retain context items that maximize orthogonal novelty under a token or word budget.

Centroid Drift Centroid drift is an embeddingspace adaptation of centroid-based representativeness ideas from multi-document summarization (Radev et al., 2004). Rather than measuring redundancy through only pairwise similarity, it asks whether adding a candidate changes the semantic center of the retained context.

Let

$$
\mu _ { C } = { \frac { 1 } { | C | } } \sum _ { c \in C } e ( c )
$$

denote the centroid of the current context. After adding candidate $x ,$ the updated centroid becomes

$$
\mu _ { C \cup \{ x \} } = \frac { 1 } { | C | + 1 } \left( \sum _ { c \in C } e ( c ) + e ( x ) \right) .
$$

We define gain as

$$
\mathcal V _ { \mathrm { C D } } ( x \mid C ) = 1 - \sin ( \mu _ { C } , \mu _ { C \cup \{ x \} } ) .
$$

If adding x barely changes the centroid, then it is already well represented by the retained context and contributes little marginal value. If it produces a large drift, then it expands the semantic footprint of the context. Since cosine similarity lies in [−1, 1], the score lies in [0, 2], although in practice values are usually much smaller.

Centroid drift is used for Post-Retrieval pruning and Pre-Synthesis pruning. At Post-Retrieval, a branch is retained only if it shifts the semantic center of the accepted evidence enough to exceed a threshold. At Pre-Synthesis, we greedily select context items that most increase the semantic footprint of the final retained set.

Determinantal Point Processes (DPP) Determinantal Point Processes (DPPs) provide a probabilistic framework for selecting subsets that are both individually relevant and mutually diverse (Kulesza and Taskar, 2012). Let $L$ be a positive semidefinite kernel matrix over candidate items, with entries

$$
L _ { i j } = w ( i ) w ( j ) \langle \tilde { e } ( i ) , \tilde { e } ( j ) \rangle ,
$$

where $w ( i )$ is the nonnegative query-relevance weight defined above.

Because $L$ is constructed as a weighted Gram matrix over normalized embeddings, it is positive semidefinite by construction.

The gain of adding candidate x to context C is

$$
\mathcal { V } _ { \mathrm { D P P } } ( x \mid C ) = \frac { \operatorname* { d e t } ( L _ { C \cup \{ x \} } ) } { \operatorname* { d e t } ( L _ { C } ) } .
$$

Geometrically, this ratio measures how much additional volume the candidate contributes to the selected set. Highly redundant candidates contribute little new volume and therefore receive small gains. Because $L$ is positive semidefinite, the determinant ratio is nonnegative, with larger values indicating jointly relevant and diverse candidates.

We use DPP-style gain for Post-Retrieval pruning and Pre-Synthesis pruning. At Post-Retrieval, the determinant ratio acts as an information-gain score for deciding whether a branch is worth retaining. At Pre-Synthesis, it is used greedily to construct a diverse final synthesis context.

Submodular Coverage Submodular coverage is our most general budget-aware objective. It is based on query-focused facility-location style selection (Lin and Bilmes, 2011), and aims to retain a set of items that collectively covers the semantic space of the candidate pool while accounting for token cost.

We define the coverage function

$$
F ( C ) = \sum _ { j \in P } \operatorname* { m a x } _ { i \in C } \left[ w ( i ) \sin ( e ( i ) , e ( j ) ) \right] .
$$

Here, P denotes the full candidate pool available at the current pruning decision. It is fixed while scoring candidates for that decision step, and is recomputed only when the pipeline advances to a new pruning state. For example, if the current candidate pool is $P ~ = ~ \{ x _ { 1 } , x _ { 2 } , x _ { 3 } \}$ and the retained set is $C = \{ x _ { 1 } \}$ , then the marginal gain of adding x<sub>2</sub> is computed as $\Delta F ( x _ { 2 } \mid C ) =$ $F ( \{ x _ { 1 } , x _ { 2 } \} ) - F ( \{ x _ { 1 } \} )$ with coverage still evaluated over the same fixed pool P. Also, i and $j$ index candidate items, while $e ( i )$ and $e ( j )$ denote their embeddings. The marginal gain of adding item x is

$$
\Delta _ { F } ( x \mid C ) = F ( C \cup \{ x \} ) - F ( C ) .
$$

To incorporate budget awareness, we normalize by token cost:

$$
\mathcal { V } _ { \mathrm { S C } } ( x \mid C , Q ) = \frac { \Delta _ { F } ( x \mid C ) } { \mathrm { c o s t } ( x ) } .
$$

This objective favors candidates that cover large uncovered regions of the pool while remaining economical in token usage. Because redundant items add little new coverage, they receive low gain. Since the underlying objective is monotone submodular, greedy selection provides a natural and efficient approximation procedure.

Submodular coverage is used in our experiments at all three stages. At Pre-Retrieval, it scores generated subqueries before search. At Post-Retrieval, it prunes low-value retrieved branches. At Pre-Synthesis, it compresses the final context under a budget.

Combined Scoring Because relevance, novelty, and coverage capture different aspects of marginal value, we also study combined scoring functions that integrate multiple signals. A generic combined score can be written as

$$
\begin{array} { r l } & { \mathrm { S c o r e } ( x ) = \alpha \operatorname { R e l } ( x , Q ) } \\ & { \qquad + \beta \operatorname { N o v } ( x , C ) + \gamma \operatorname { C o v } ( x , C , Q ) , } \end{array}
$$

where $\alpha , \beta , \gamma \ge 0$ are mixture weights.

Our concrete hybrid instantiation combines direct query relevance, geometric novelty, and submodular coverage:

$$
\begin{array} { l } { \operatorname { S c o r e } ( x ) = \alpha \sin ( e ( x ) , q ) \medskip } \\ { \displaystyle - \beta V _ { \mathrm { G R N } } ( x \mid C ) + \gamma V _ { \mathrm { S C } } ( x \mid C , Q ) } \end{array}
$$

with $\alpha + \beta + \gamma = 1 .$

In the main Hybrid configuration used in our experiments, we set $( \alpha , \beta , \gamma ) = ( 0 . 4 0 , 0 . 3 0 , 0 . 3 0 )$

The first term rewards direct alignment with the research goal, the second rewards orthogonal novelty, and the third rewards global coverage under a token-aware objective. We use this hybrid to study whether explicitly combining local relevance, novelty, and global coverage yields a better qualityefficiency trade-off than any single criterion alone.

Lexical Pruning We additionally study a lexical pruning variant that replaces dense embedding similarity with a lightweight surface-form proxy. Each text is represented by a lexical profile consisting of TF–IDF weights and bigram sets. Similarity between two items is defined as a weighted combination of TF–IDF cosine similarity (Ramos, 2003) and bigram Jaccard overlap (Broder, 1997):

$$
\begin{array} { r l } & { \mathrm { L e x S i m } ( x , y ) = { w _ { \mathrm { t f i d f } } } \mathrm { C o s S i m } _ { \mathrm { t f i d f } } ( x , y ) } \\ & { \quad \quad \quad + \left. { w _ { \mathrm { j a c } } } \mathrm { J a c c a r d } _ { \mathrm { b i g r a m } } ( x , y ) , \right. } \end{array}
$$

with $w _ { \mathrm { t f i d f } } + w _ { \mathrm { j a c } } = 1$

Using this lexical similarity, we define a lexical MMR-style score

$$
\begin{array} { r l } & { \mathcal { V } _ { \mathrm { L e x } } ( x \mid C , Q ) = \lambda \operatorname { L e x S i m } ( x , Q ) } \\ & { \qquad - \left( 1 - \lambda \right) \underset { y \in C } { \operatorname* { m a x } } \operatorname { L e x S i m } ( x , y ) . } \end{array}
$$

This criterion preserves the same relevanceversus-redundancy structure as embedding-based MMR, but avoids the cost of dense embedding computation. We use it as a cheap baseline for studying how much pruning performance depends on richer semantic representations.

## B.2 LLM-Based Pruning

We evaluate LLM-based pruning variants in which a language model acts as a pruning judge. Given the current query, context, and candidate set, the model is prompted to assess the usefulness of each candidate and return structured outputs such as predicted gain, keep probability, or a keep-versusprune decision. The system then retains the highestscoring candidates or candidates above a threshold, depending on the stage.

Compared with fixed heuristic rules, this approach can incorporate richer semantic judgments and broader task context. However, it also introduces additional inference cost because pruning itself requires one or more model calls. In our experiments, LLM-based pruning is evaluated both as a standalone policy and in mixed-stage combinations with heuristic downstream pruning.

## B.3 Learning-Based Pruning

We also study a learned controller for Pre-Retrieval branch selection. Its role is intentionally narrow: given the current research state and a set of candidate subqueries, it predicts which candidate branches are most worth expanding before retrieval and downstream processing costs are incurred.

To train this controller, we reconstruct a candidate-level supervision dataset from raw generation logs. Each example corresponds to one candidate branch considered at one decision point, together with the local search state available when that decision is made. Across 34 complete runs, this yields 360 candidate branch decisions. This count is larger than the number of executed branches because each decision point may contain multiple candidate subqueries, only some of which are ultimately expanded. For consistency of targets, we derive labels only from the submodular-family runs (GPTResearcher\_sc), so the learned controller should be interpreted as a submodular-aligned pre-retrieval proxy rather than as a fully general learned pruning policy. We also partition the candidate-level dataset by run, so that decisions from the same execution trajectory do not appear in both training and evaluation splits.

Our model is a lightweight multitask neural value model built on the pretrained encoder $B A A I / b g e - s m a l l . e n - \nu I . S .$ One tower encodes the current search state, including the root query, local frame context, parent research goal, and previously retained branches, while a second tower encodes the candidate branch through its query and research goal. We keep the encoder frozen and feed the resulting semantic representations, together with structured search features such as depth, branch order, and prior retained count, into a small MLP. The model has two heads: a regression head that predicts a continuous utility score ${ \hat { g } } ( x )$ and a classification head that predicts a keep probability $p _ { \mathrm { k e e p } } ( x )$

At inference time, candidate x is retained if

$$
p _ { \mathrm { k e e p } } ( x ) \geq \tau ,
$$

where $\tau$ is the pruning threshold. In our experiments, we evaluate multiple thresholds, including $\tau = 0 . 5$ and $\tau = 0 . 7$ , to control pruning aggressiveness. If no candidate exceeds the threshold, we retain the top-scoring candidate as a fallback to avoid degenerate empty expansions.

In the full pipeline, learned Pre-Retrieval pruning is paired with submodular Post-Retrieval pruning and centroid-drift Pre-Synthesis pruning. Because the supervision set remains modest in size after restricting labels to a single target definition, we interpret the learned controller cautiously and treat it primarily as a proof-of-concept preretrieval proxy rather than as strong evidence that learned pruning currently outperforms welldesigned heuristics.

## B.4 Stage-Specific Application

Different pruning strategies are applied to different candidate types depending on the stage:

• Pre-Retrieval: candidate items are generated subqueries, represented by their query text and associated research goals. The objective is to avoid launching redundant or low-value search branches before retrieval cost is incurred.

• Post-Retrieval: candidate items are newly retrieved evidence blocks, evaluated against previously retained context. The objective is to stop low-value recursive expansion early.

• Pre-Synthesis: candidate items are the final aggregated context blocks. The objective is to construct a compact, diverse, and query-relevant synthesis context under a budget.

Operationally, Post-Retrieval pruning is typically used as a thresholded gating rule, while Pre-Retrieval and Pre-Synthesis often involve greedy selection under item or word budgets. This shared formulation lets us compare different notions of marginal value within a common stage-aware execution framework.

## B.5 Supplementary Cross-Benchmark Results on DeepResearch Bench

To assess whether our main efficiency findings transfer beyond DeepResearchGym, we also evaluate a subset of pruning configurations on Deep-Research Bench. We treat these experiments as supplementary cross-benchmark evidence rather than as a second primary benchmark. In particular, we use them to test whether the stage-ordering conclusion that earlier pruning yields substantially larger end-to-end savings than late-only pruning remains stable under a different evaluation setup. Table 13 summarizes overall report quality, and Table 14 summarizes token and runtime efficiency.

The DeepResearch Bench results support this efficiency trend: branch-only and two-stage pruning consistently yield much larger token and runtime savings than root-only pruning, and MMR remains one of the strongest compression-oriented heuristics. However, the method-level quality rankings are less aligned with those observed on Deep-ResearchGym. We interpret this discrepancy cautiously, as the two benchmarks differ both in task formulation and in how report quality is evaluated: DeepResearchGym uses an LLM-as-a-judge rubric over open-ended reports, whereas DeepResearch Bench reports RACE-style quality metrics. We therefore treat the DeepResearch Bench results as directional evidence for efficiency generalization, not as a full replication of the quality trade-offs in the main benchmark.

## B.6 Judge Sensitivity of Absolute Quality Scores

Appendix Table 4 shows that absolute rubric-based quality scores can vary substantially with judge choice, even for the same generated reports. We therefore interpret the reported quality values primarily as relative comparisons under a fixed independent judge, rather than as judge-invariant absolute measurements. This caveat affects qualitysensitive method comparisons more than the paper’s efficiency conclusions, which do not depend on rubric calibration. A stronger robustness analysis would test whether method-level rankings remain stable across multiple independent judges on a shared subset of examples; we leave that extension to future work.

## B.7 Sensitivity to Threshold Choice

To assess whether the reported operating points are unusually fragile, we perform a local threshold sweep for five representative post-retrieval pruning methods: MMR, GRN, Centroid Drift, DPP, and Submodular Coverage. For each method, we fix all non-threshold hyperparameters at the published setting and vary only the pruning threshold over at least five values around the main-paper configuration. We report the full sweep (Table 15).

We define a method’s stable sampled operating region as the largest contiguous threshold interval in the sampled sweep containing the published threshold $\tau _ { \mathrm { p u b } }$ for which overall quality remains within 2% of the published configuration, i.e.,

$$
Q ( \tau ) \geq 0 . 9 8 Q ( \tau _ { \mathrm { p u b } } ) .
$$

This criterion tests whether the reported hyperparameters lie in a locally stable quality–efficiency region rather than at a brittle point estimate. Values below are mean scores over a 10-query sensitivity subset and are intended to characterize local stability rather than fully re-rank methods on the full benchmark.

## C Prompts

![](images/da0304400e29ba1b4886ec94c687e25f52066127a6fada4e7598d8966b9f358c.jpg)  
(a) LLM-based query pruning instructions.  
(b) LLM-as-a-judge evaluation instructions.  
Figure 2: Prompts used in our pipeline.

Table 4: Judge-sensitivity analysis for rubric-based quality evaluation. Absolute scores vary substantially with judge choice, so main-paper quality results should be interpreted as relative comparisons under a fixed judge rather than as judge-invariant quality values.
<table><tr><td>Model</td><td>Judge</td><td>Overall</td><td>Clarity</td><td>Depth</td><td>Balance</td><td>Breadth</td><td>Support</td><td>Insight</td><td>KPR + KPC</td><td>Cit. Recall</td></tr><tr><td>gpt-5-mini</td><td>gpt-5-mini</td><td>58.47</td><td>57.20</td><td>61.50</td><td>59.90</td><td>67.40</td><td>41.50</td><td>63.30</td><td>61.70</td><td>92.98</td></tr><tr><td>gpt-4.1-mini</td><td>gpt-4.1-mini</td><td>91.50</td><td>89.00</td><td>96.00</td><td>90.00</td><td>96.60</td><td>91.00</td><td>87.00</td><td>72.24</td><td>99.52</td></tr><tr><td>gpt-4.1-mini</td><td>gpt-5-mini</td><td>47.67</td><td>50.00</td><td>45.00</td><td>48.00</td><td>61.00</td><td>32.00</td><td>50.00</td><td>72.25</td><td>90.56</td></tr><tr><td>Qwen2-7B-Instruct (32k)</td><td>gpt-5-mini</td><td>34.37</td><td>35.71</td><td>35.24</td><td>39.05</td><td>42.86</td><td>21.43</td><td>31.90</td><td>72.55</td><td>75.45</td></tr></table>

<table><tr><td></td><td colspan="2">Cost</td><td colspan="7">Quality</td><td>Relevance</td><td>Faithfulness</td></tr><tr><td>Method</td><td># Tokens</td><td>Runtime (s)</td><td>Overall</td><td>Clarity</td><td>Depth</td><td>Balance</td><td>Breadth</td><td>Support</td><td>Insight</td><td> $\mathrm { K P R + K P C }$ </td><td>Cit. Recall</td></tr><tr><td>Baseline</td><td> $3 7 5 . 4 \pm 2 . 6 \mathrm { k \Omega }$ </td><td> $3 4 2 2 . 6 \pm 1 4 0 . 6$ </td><td>57.83</td><td>55.05</td><td>62.12</td><td>57.27</td><td>65.96</td><td>45.45</td><td>61.11</td><td>70.23</td><td>95.54</td></tr><tr><td colspan="10">One-stage methods: Post-Retrieval Pruning</td><td></td><td></td></tr><tr><td>MMR(§ B.1)</td><td> $1 1 4 . 6 \pm 2 . 2 \mathrm { k \Omega }$ </td><td> $1 3 7 9 . 8 \pm 6 8 . 3$ </td><td>56.62</td><td>54.50</td><td>59.90</td><td>57.20</td><td>65.50</td><td>43.20</td><td>59.40</td><td>63.49</td><td>91.70</td></tr><tr><td>Geo. Residual Novelty(§ B.1)</td><td> $1 7 5 . 7 \pm 5 . 1 \bf { k }$ </td><td> $2 0 8 7 . 9 \pm 1 0 8 . 3$ </td><td>57.02</td><td>55.85</td><td>59.36</td><td>58.30</td><td>63.72</td><td>44.15</td><td>60.74</td><td>42.89</td><td>94.40</td></tr><tr><td>Centroid Drift(§ B.1)</td><td> $1 3 7 . 5 \pm 2 . 0 \mathrm { k \Omega }$ </td><td> $1 6 0 5 . 7 \pm 7 3 . 9$ </td><td>56.84</td><td>56.12</td><td>59.90</td><td>56.63</td><td>64.18</td><td>43.37</td><td>60.82</td><td>41.29</td><td>92.48</td></tr><tr><td>Submodular Coverage(§ B.1)</td><td> $1 4 1 . 9 \pm 3 . 3 \mathrm { k \Omega }$ </td><td> $1 7 3 7 . 2 \pm 1 0 8 . 1$ </td><td>55.22</td><td>53.54</td><td>57.68</td><td>55.86</td><td>62.42</td><td>42.63</td><td>59.19</td><td>47.94</td><td>91.97</td></tr><tr><td>DPP(§ B.1)</td><td> $1 2 9 . 6 \pm 2 . 5 \mathrm { k \Omega }$ </td><td> $1 5 2 0 . 6 \pm 7 3 . 3$ </td><td>54.51</td><td>53.19</td><td>56.81</td><td>55.71</td><td>62.75</td><td>41.10</td><td>57.47</td><td>43.33</td><td>95.62</td></tr><tr><td>Combined (§ B.1)</td><td> $1 1 7 . 8 \pm 2 . 5 \bf { k }$ </td><td> $1 3 9 9 . 5 \pm 7 1 . 0$ </td><td>56.03</td><td>54.90</td><td>58.30</td><td>57.20</td><td>64.20</td><td>42.60</td><td>59.00</td><td>64.13</td><td>92.59</td></tr><tr><td>Hybrid(§ B.1)</td><td> $1 2 1 . 3 \pm 2 . 8 \mathrm { k }$ </td><td> $1 4 4 9 . 5 \pm 7 5 . 6$ </td><td>57.63</td><td>57.40</td><td>59.70</td><td>57.90</td><td>65.10</td><td>45.10</td><td>60.60</td><td>65.23</td><td>92.16</td></tr><tr><td>LLM (§ B.1)</td><td> $2 1 1 . 8 \pm 7 . 9 \bf { k }$ </td><td> $2 3 1 0 . 7 \pm 1 3 7 . 8$ </td><td>59.65</td><td>57.40</td><td>63.00</td><td>60.00</td><td>66.80</td><td>48.20</td><td>62.50</td><td>49.65</td><td>93.54</td></tr><tr><td colspan="10">One-stage methods: Pre-Synthesis Pruning</td><td></td><td></td></tr><tr><td>MMR</td><td> $3 6 6 . 0 \pm 5 . 2 \mathrm { k \Omega }$ </td><td> $4 4 4 6 . 9 \pm 1 5 9 . 0$ </td><td>57.77</td><td>55.30</td><td>59.20</td><td>60.00</td><td>66.70</td><td>44.50</td><td>60.90</td><td>65.26</td><td>90.41</td></tr><tr><td>Geo. Residual Novelty</td><td> $3 7 4 . 1 \pm 3 . 1 \bf k$ </td><td> $4 4 9 3 . 3 \pm 1 5 5 . 2$ </td><td>55.98</td><td>53.70</td><td>59.35</td><td>56.41</td><td>64.67</td><td>41.96</td><td>59.78</td><td>41.67</td><td>90.38</td></tr><tr><td>Centroid Drift</td><td> $3 7 4 . 5 \pm 2 . 8 \mathrm { k \Omega }$ </td><td> $4 5 6 2 . 4 \pm 1 4 9 . 8$ </td><td>52.38</td><td>49.60</td><td>55.70</td><td>52.60</td><td>60.50</td><td>40.60</td><td>55.30</td><td>43.75</td><td>90.19</td></tr><tr><td>Submodular Coverage</td><td> $3 8 4 . 7 \pm 3 . 3 \mathrm { k \Omega }$ </td><td> $4 5 3 3 . 3 \pm 1 5 0 . 5$ </td><td>53.22</td><td>50.40</td><td>55.25</td><td>53.64</td><td>62.12</td><td>40.91</td><td>56.97</td><td>42.09</td><td>90.78</td></tr><tr><td>DPP</td><td> $3 7 4 . 4 \pm 3 . 2 \mathrm { k \Omega }$ </td><td> $4 5 1 2 . 4 \pm 1 4 9 . 8$ </td><td>54.70</td><td>52.12</td><td>57.37</td><td>56.26</td><td>63.64</td><td>41.52</td><td>57.27</td><td>54.07</td><td>90.24</td></tr><tr><td>Combined</td><td> $3 7 4 . 5 \pm 6 . 1 \bf { k }$ </td><td> $4 4 0 8 . 6 \pm 1 6 1 . 6$ </td><td>59.38</td><td>56.70</td><td>62.40</td><td>59.90</td><td>68.10</td><td>46.60</td><td>62.60</td><td>66.32</td><td>93.79</td></tr><tr><td>Hybrid</td><td> $3 3 2 . 3 \pm 1 0 . 6 \mathrm { k }$ </td><td> $3 8 3 4 . 1 \pm 2 0 5 . 6$ </td><td>60.68</td><td>57.50</td><td>64.10</td><td>61.80</td><td>70.00</td><td>46.40</td><td>64.30</td><td>65.62</td><td>95.07</td></tr><tr><td>LLM</td><td> $3 8 6 . 7 \pm 3 . 1 \bf { k }$ </td><td> $4 5 1 2 . 0 \pm 1 5 3 . 7$ </td><td>57.17</td><td>54.50</td><td>60.00</td><td>57.90</td><td>65.40</td><td>44.50</td><td>60.70</td><td>44.47</td><td>92.43</td></tr><tr><td colspan="10">Two-stage methods: Post-Retrieval + Pre-Synthesis Pruning</td><td></td></tr><tr><td>MMR</td><td> $1 1 4 . 6 \pm 2 . 2 \mathrm { k \Omega }$ </td><td> $1 3 8 1 . 3 \pm 6 8 . 2 $ </td><td>56.40</td><td>55.90</td><td>58.80</td><td>57.20</td><td>64.70</td><td>43.70</td><td>58.10</td><td>65.16</td><td>92.61</td></tr><tr><td>Geo. Residual Novelty</td><td> $1 7 5 . 2 \pm 5 . 3 \mathrm { k \Omega }$ </td><td> $2 0 8 0 . 2 \pm 1 0 7 . 5$ </td><td>57.94</td><td>54.37</td><td>60.52</td><td>58.33</td><td>64.48</td><td>43.02</td><td>60.94</td><td>64.47</td><td>93.06</td></tr><tr><td>Centroid Drift</td><td> $1 1 8 . 1 \pm 1 . 3 \mathbf { k }$ </td><td> $1 1 4 2 . 4 3 \pm 1 0 5 . 8 2$ </td><td>57.44</td><td>55.57</td><td>59.48</td><td>59.69</td><td>64.95</td><td>44.54</td><td>60.41</td><td>43.23</td><td>92.94</td></tr><tr><td>Submodular Coverage</td><td> $1 4 2 . 1 \pm 3 . 2 \mathrm { k \Omega }$ </td><td> $1 7 7 6 . 5 \pm 1 1 2 . 1$ </td><td>57.00</td><td>55.50</td><td>59.40</td><td>57.70</td><td>65.20</td><td>43.80</td><td>60.40</td><td>62.23</td><td>94.74</td></tr><tr><td>DPP</td><td> $1 2 2 . 8 \pm 1 . 8 \mathrm { k }$ </td><td> $1 4 6 2 . 3 1 \pm 2 3 6 . 5 5$ </td><td>56.73</td><td>56.60</td><td>58.40</td><td>56.90</td><td>64.60</td><td>44.50</td><td>59.40</td><td>62.33</td><td>93.62</td></tr><tr><td>CD + SC</td><td> $1 3 7 . 5 \pm 2 . 0 \mathrm { k \Omega }$ </td><td> $1 5 9 9 . 6 \pm 7 3 . 6$ </td><td>59.47</td><td>58.70</td><td>57.80</td><td>56.80</td><td>63.60</td><td>40.20</td><td>59.70</td><td>44.29</td><td>89.96</td></tr><tr><td> $\mathrm { C D } + \mathrm { L L M }$ </td><td> $1 3 6 . 0 \pm 3 . 2 \mathrm { k \Omega }$ </td><td> $1 5 8 9 . 3 \pm 7 6 . 2$ </td><td>58.65</td><td>56.80</td><td>62.10</td><td>58.90</td><td>66.20</td><td>46.00</td><td>61.90</td><td>63.40</td><td>94.34</td></tr><tr><td> $\mathbf { S C } + \mathbf { L L M }$ </td><td> $1 3 9 . 1 \pm 4 . 1 \bf { k }$ </td><td> $1 7 0 0 . 6 \pm 1 1 0 . 0$ </td><td>58.08</td><td>55.90</td><td>61.50</td><td>58.60</td><td>65.80</td><td>45.50</td><td>61.20</td><td>64.54</td><td>93.43</td></tr><tr><td>LLM</td><td> $2 2 6 . 6 \pm 8 . 1 \bf { k }$ </td><td> $2 6 3 2 . 3 \pm 1 4 8 . 2$ </td><td>58.27</td><td>55.05</td><td>61.11</td><td>58.89</td><td>66.26</td><td>46.06</td><td>62.22</td><td>48.61</td><td>93.01</td></tr><tr><td>Lexical</td><td> $1 1 5 . 1 \pm 1 . 2 \mathrm { k \Omega }$ </td><td> $1 4 1 4 . 3 \pm 6 4 . 8$ </td><td>54.62</td><td>53.40</td><td>57.20</td><td>57.30</td><td>61.70</td><td>40.30</td><td>57.80</td><td>46.13</td><td>91.38</td></tr><tr><td>Combined</td><td> $1 1 7 . 6 \pm 2 . 6 \mathrm { k \Omega }$ </td><td> $1 3 9 2 . 2 \pm 7 1 . 4$ </td><td>54.87</td><td>53.90</td><td>57.50</td><td>56.00</td><td>64.20</td><td>39.80</td><td>57.80</td><td>64.88</td><td>93.03</td></tr><tr><td>Hybrid</td><td> $1 2 1 . 7 \pm 2 . 8 \mathrm { k }$ </td><td> $1 4 5 7 . 4 \pm 7 5 . 3$ </td><td>56.92</td><td>56.30</td><td>60.20</td><td>57.30</td><td>65.40</td><td>43.20</td><td>59.10</td><td>64.25</td><td>93.51</td></tr><tr><td colspan="10">Three-stage methods: Pre-Retrieval + Post-Retrieval + Pre-Synthesis Pruning</td><td></td><td></td></tr><tr><td>MMR</td><td></td><td></td><td></td><td></td><td>58.90</td><td>56.30</td><td>63.70</td><td>42.60</td><td>59.90</td><td>63.43</td><td></td></tr><tr><td>Geo. Residual Novelty</td><td> $1 0 0 . 1 \pm 2 . 0 \mathrm { k }$   $1 5 0 . 0 \pm 4 . 5 \mathrm { k \Omega }$ </td><td> $1 1 5 7 . 7 \pm 6 3 . 6$   $1 9 3 7 . 7 \pm 3 4 4 . 0$ </td><td>55.90 56.68</td><td>54.00 54.55</td><td>58.69</td><td>58.18</td><td>64.85</td><td>43.64</td><td>60.20</td><td>45.13</td><td>91.84 91.33</td></tr><tr><td>Centroid Drift</td><td> $1 2 0 . 6 \pm 2 . 1 \bf { k }$ </td><td> $1 6 6 9 . 8 \pm 3 2 1 . 4$ </td><td>56.00</td><td>54.90</td></table>

Table 5: Performance comparison across all pruning strategies. Tokens are reported in thousands (k) (§ A.4). Values are mean ± standard error over 100 reports $( \ S \mathrm { { A . 1 } ) }$ . Quality, relevance, and faithfulness metrics were obtained from DeepResearchGym (§ A.2)  
Unless otherwise specified, the same pruning criterion is applied at both the post-retrieval and pre-synthesis stages. Method-specific hyperparameters are: MMR $( \lambda = 0 . 3 5$ , max 10 contexts); DPP $( \tau = 0 . 3 0$ , max 10 pre-synthesis contexts); centroid drift $( \delta = 0 . 0 3 ) ;$ geometric residual novelty $( \mathrm { G R N } , \tau = 0 . 8 5 ) ;$ ; and submodular coverage (prune when marginal gain per token < 0.05);LLM (δ = 0.3); Lexical $( \lambda = . 6 ,$ , threshold = .2).

Table 6: Efficiency comparison across all methods. Tokens are reported in thousands (k). Est. Token Reduction (%) denotes the estimated reduction relative to each method’s pre-pruning budget. Savings vs. Baseline (%) is computed from mean total tokens relative to the baseline mean token count.
<table><tr><td>Method</td><td># Nodes (§ A.3)</td><td># Tokens (§ A.4)</td><td>Est. Token Reduction (%)(§ A.6)</td><td>Savings vs. Baseline (%)(§ A.7)</td></tr><tr><td>Baseline</td><td> $2 9 . 0 \pm 0 . 0$ </td><td> $3 7 5 . 4 \pm 2 . 6 \mathrm { k \Omega }$ </td><td>0.0</td><td>0.0</td></tr><tr><td colspan="5">One-stage methods: Post-Retrieval Pruning</td></tr><tr><td>Geo. Residual Novelty (§ B.1)</td><td> $1 3 . 3 1 \pm 0 . 3 7$ </td><td> $1 7 5 . 7 \pm 5 . 1 \bf k$ </td><td> $2 2 . 0 2 \pm 0 . 2 6$ </td><td>53.2</td></tr><tr><td>Centroid Drift (§ B.1)</td><td> $1 0 . 4 7 \pm 0 . 1 3$ </td><td> $1 3 7 . 5 \pm 2 . 0 \mathrm { k }$ </td><td> $1 8 . 9 1 \pm 0 . 2 6$ </td><td>63.4</td></tr><tr><td>Submodular Coverage (§ B.1)</td><td> $1 0 . 8 2 \pm 0 . 2 8$ </td><td> $1 4 1 . 9 \pm 3 . 3 \mathrm { k \Omega }$ </td><td> $2 1 . 5 5 \pm 0 . 2 6$ </td><td>62.2</td></tr><tr><td>DPP (§ B.1)</td><td> $9 . 8 8 \pm 0 . 1 6$ </td><td> $1 2 9 . 6 \pm 2 . 5 \mathrm { k \Omega }$ </td><td> $2 1 . 9 9 \pm 0 . 2 2$ </td><td>65.5</td></tr><tr><td>LLM (§ B.2)</td><td> $1 4 . 9 0 \pm 0 . 5 3$ </td><td> $2 1 1 . 8 \pm 7 . 9 \bf { k }$ </td><td> $2 2 . 7 6 \pm 0 . 3 8$ </td><td>43.6</td></tr><tr><td>MMR(§ B.1)</td><td> $8 . 8 4 \pm 0 . 0 8$ </td><td> $1 1 4 . 6 \pm 2 . 2 \mathrm { k \Omega }$ </td><td> $2 2 . 1 1 \pm 0 . 4 8$ </td><td>69.5</td></tr><tr><td>Combined (§ B.1)</td><td> $9 . 0 0 \pm 0 . 1 1$ </td><td> $1 1 7 . 8 \pm 2 . 5 \bf { k }$ </td><td> $2 1 . 0 1 \pm 0 . 4 8$ </td><td>68.6</td></tr><tr><td>Hybrid(§ B.1)</td><td> $9 . 2 8 \pm 0 . 1 5$ </td><td> $1 2 1 . 3 \pm 2 . 8 \mathbf { k }$ </td><td> $2 0 . 9 2 \pm 0 . 4 9$ </td><td>67.7</td></tr><tr><td colspan="5">One-stage methods: Pre-Synthesis Pruning</td></tr><tr><td>Geo. Residual Novelty</td><td> $2 9 \pm 0 . 0 0$ </td><td> $3 7 4 . 1 \pm 3 . 1 \bf k$ </td><td> $2 8 . 1 2 \pm 0 . 1 7$ </td><td>0.4</td></tr><tr><td>Centroid Drift</td><td> $2 9 \pm 0 . 0 0$ </td><td> $3 7 4 . 5 \pm 2 . 8 \mathrm { k }$ </td><td> $2 8 . 1 2 \pm 0 . 1 7$ </td><td>0.2</td></tr><tr><td>Submodular Coverage</td><td> $2 9 \pm 0 . 0 0$ </td><td> $3 8 4 . 7 \pm 3 . 3 \mathrm { k \Omega }$ </td><td> $2 7 . 5 8 \pm 0 . 1 8$ </td><td>-2.5</td></tr><tr><td>DPP</td><td> $2 9 \pm 0 . 0 0$ </td><td> $3 7 4 . 4 \pm 3 . 2 \mathrm { k \Omega }$ </td><td> $2 8 . 1 5 \pm 0 . 1 8$ </td><td>0.3</td></tr><tr><td>LLM</td><td> $2 9 \pm 0 . 0 0$ </td><td> $3 8 6 . 7 \pm 3 . 1 \bf { k }$ </td><td> $2 7 . 6 6 \pm 0 . 2 5$ </td><td>-3.0</td></tr><tr><td>MMR</td><td> $2 9 \pm 0 . 0 0$ </td><td> $3 6 6 . 0 \pm 5 . 2 \mathrm { k \Omega }$ </td><td> $2 7 . 1 7 \pm 0 . 5 5$ </td><td>2.5</td></tr><tr><td>Hybrid</td><td> $2 9 \pm 0 . 0 0$ </td><td> $3 3 2 . 3 \pm 1 0 . 6 \mathrm { k }$ </td><td> $2 4 . 6 2 \pm 1 . 2 6$ </td><td>11.5</td></tr><tr><td>Combined</td><td> $2 9 \pm 0 . 0 0$ </td><td> $3 7 4 . 5 \pm 6 . 1 \mathrm { k \Omega }$ </td><td> $3 0 . 3 1 \pm 0 . 6 6$ </td><td>0.2</td></tr><tr><td colspan="5">Two-stage methods: Post-Retrieval + Pre-Synthesis Pruning</td></tr><tr><td>Centroid Drift</td><td> $9 . 0 2 \pm 0 . 0 2$ </td><td> $1 1 8 . 1 \pm 1 . 3 \mathbf { k }$ </td><td> $2 0 . 5 8 \pm 0 . 2 4$ </td><td>68.5</td></tr><tr><td>Submodular Coverage</td><td> $1 0 . 8 0 \pm 0 . 2 8$ </td><td> $1 4 2 . 1 \pm 3 . 2 \mathrm { k \Omega }$ </td><td> $2 2 . 9 2 \pm 0 . 2 9$ </td><td>62.1</td></tr><tr><td>CD + SC</td><td> $1 0 . 4 5 \pm 0 . 1 3$ </td><td> $1 3 7 . 5 \pm 2 . 0 \mathrm { k }$ </td><td> $2 0 . 6 9 \pm 0 . 3 3$ </td><td>63.4</td></tr><tr><td>LLM</td><td> $1 5 . 5 7 \pm 0 . 5 3$ </td><td> $2 2 6 . 6 \pm 8 . 1 \bf { k }$ </td><td> $2 3 . 5 7 \pm 0 . 3 1$ </td><td>39.7</td></tr><tr><td>Lexical</td><td> $9 . 0 0 \pm 0 . 0 0$ </td><td> $1 1 5 . 1 \pm 1 . 2 \mathrm { k \Omega }$ </td><td> $2 9 . 3 6 \pm 0 . 2 4$ </td><td>69.3</td></tr><tr><td>MMR</td><td> $8 . 8 4 \pm 0 . 0 8$ </td><td> $1 1 4 . 6 \pm 2 . 2 \mathrm { k \Omega }$ </td><td> $2 2 . 1 1 \pm 0 . 4 8$ </td><td>69.5</td></tr><tr><td>Hybrid</td><td> $9 . 3 0 \pm 0 . 1 5$ </td><td> $1 2 1 . 7 \pm 2 . 8 \bf { k }$ </td><td> $2 6 . 3 1 \pm 0 . 6 6$ </td><td>67.6</td></tr><tr><td>Combined</td><td> $8 . 9 8 \pm 0 . 1 1$ </td><td> $1 1 7 . 6 \pm 2 . 6 \bf { k }$ </td><td> $2 7 . 8 1 \pm 0 . 6 0$ </td><td>68.7</td></tr><tr><td> $\mathrm { C D } + \mathrm { L L M }$ </td><td> $1 0 . 2 4 \pm 0 . 1 7$ </td><td> $1 3 6 . 0 \pm 3 . 2 \mathrm { k \Omega }$ </td><td> $2 0 . 3 0 \pm 0 . 5 1$ </td><td>63.8</td></tr><tr><td> $\mathbf { S C } + \mathbf { L L M }$ </td><td> $1 0 . 5 4 \pm 0 . 3 0$ </td><td> $1 3 9 . 1 \pm 4 . 1 \bf { k }$ </td><td> $2 2 . 4 0 \pm 0 . 5 5$ </td><td>62.9</td></tr><tr><td colspan="5">Three-stage methods: Pre-Retrieval + Post-Retrieval + Pre-Synthesis Pruning</td></tr><tr><td>Geo. Residual Novelty</td><td> $1 1 . 5 7 \pm 0 . 3 3$ </td><td> $1 5 0 . 0 \pm 4 . 5 \mathrm { k \Omega }$ </td><td> $1 9 . 9 7 \pm 0 . 3 3$ </td><td>60.0</td></tr><tr><td>DPP</td><td> $8 . 6 8 \pm 0 . 1 5$ </td><td> $1 1 3 . 3 \pm 2 . 2 \mathrm { k \Omega }$ </td><td> $2 0 . 1 6 \pm 0 . 2 3$ </td><td>69.8</td></tr><tr><td>Centroid Drift</td><td> $9 . 2 7 \pm 0 . 1 3$ </td><td> $1 2 0 . 6 \pm 2 . 1 \bf { k }$ </td><td> $1 6 . 6 5 \pm 0 . 2 8$ </td><td>67.9</td></tr><tr><td>Lexical + CD + SC</td><td> $9 . 2 6 \pm 0 . 1 4$ </td><td> $1 2 0 . 1 \pm 2 . 3 \mathbf { k }$ </td><td> $1 7 . 8 2 \pm 0 . 3 9$ </td><td>68.0</td></tr><tr><td>Lexical</td><td> $7 . 9 0 \pm 0 . 0 4$ </td><td> $9 9 . 0 \pm 1 . 1 \mathrm { k \Omega }$ </td><td> $2 8 . 7 2 \pm 0 . 2 7$ </td><td>73.6</td></tr><tr><td>MMR</td><td> $7 . 8 2 \pm 0 . 0 8$ </td><td> $1 0 0 . 1 \pm 2 . 0 \mathbf { k }$ </td><td> $2 0 . 4 3 \pm 0 . 4 4$ </td><td>73.3</td></tr><tr><td>Hybrid</td><td> $8 . 1 6 \pm 0 . 1 4$ </td><td> $1 0 6 . 3 \pm 2 . 6 \mathrm { k }$ </td><td> $2 5 . 5 6 \pm 0 . 6 6$ </td><td>71.7</td></tr><tr><td>Combined</td><td> $7 . 9 0 \pm 0 . 1 0$ </td><td> $1 0 2 . 8 \pm 2 . 2 \mathrm { k \Omega }$ </td><td> $2 7 . 3 3 \pm 0 . 5 8$ </td><td>72.6</td></tr><tr><td>LLM</td><td> $1 0 . 1 2 \pm 0 . 4 5$ </td><td> $1 4 3 . 9 \pm 7 . 7 \bf { k }$ </td><td> $1 9 . 4 5 \pm 0 . 8 1$ </td><td>61.7</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 7: Runtime breakdown across all methods.(§ A.9) Values are reported as mean ± standard error over 100 reports (§ A.1). Research / Scraping denotes the retrieval-heavy stage (‘branch\_research‘ for the baseline and ‘scraping‘ for pruning methods). Pre-Retrieval, Post-Retrieval, and Pre-Synthesis report the latency of the corresponding pruning stages when separately instrumented.
<table><tr><td>Method</td><td>Total Runtime (s)</td><td>Research / Scraping</td><td> $\mathbf { R e s u l t P r o c . }$ </td><td> $\mathbf { Q u e r y 6 e n }$ </td><td>Planning</td><td>Pre-Retrieval</td><td>Post-Retrieval</td><td>Pre-Synthesis</td></tr><tr><td>Baseline</td><td> $3 4 2 2 . 6 \pm 1 4 0 . 6$ </td><td> $2 8 8 8 . 2 \pm 3 0 4 . 2$ </td><td> $1 1 5 8 . 5 \pm 5 4 . 3$ </td><td> $3 3 9 . 0 \pm 1 3 . 3$ </td><td> $1 3 . 3 \pm 0 . 7$ </td><td></td><td></td><td></td></tr><tr><td colspan="9">One-stage methods: Post-Retrieval Pruning</td></tr><tr><td>Geo. Residual Novelty</td><td> $1 8 8 9 . 0 5 \pm 4 0 0 . 0 4$ </td><td> $1 3 2 1 . 8 3 \pm 3 1 7 . 0 6$ </td><td> $4 5 6 . 6 1 \pm 5 8 . 9 9$ </td><td> $1 2 2 . 6 8 \pm 1 8 . 1 5$ </td><td> $1 3 . 4 7 \pm 0 . 6 5$ </td><td></td><td> $2 . 9 8 \pm 0 . 3 3$ </td><td></td></tr><tr><td>Centroid Drift</td><td> $1 5 8 7 . 4 8 \pm 2 2 0 . 5 0$ </td><td> $1 0 9 8 . 0 6 \pm 2 2 4 . 1 5$ </td><td> $3 7 7 . 1 3 \pm 2 3 . 3 2$ </td><td> $9 2 . 2 2 \pm 5 . 4 2$ </td><td> $1 3 . 6 2 \pm 0 . 6 9$ </td><td></td><td> $2 . 4 6 \pm 0 . 0 9$ </td><td></td></tr><tr><td>Submodular Coverage</td><td> $1 7 3 4 . 6 3 \pm 2 4 6 . 6 8$ </td><td> $1 2 0 9 . 7 4 \pm 2 3 5 . 5 8$ </td><td> $4 1 3 . 9 6 \pm 3 4 . 3 9$ </td><td> $1 0 5 . 5 9 \pm 1 2 . 6 3$ </td><td> $1 3 . 6 2 \pm 0 . 6 9$ </td><td></td><td> $3 . 0 3 \pm 0 . 3 9$ </td><td></td></tr><tr><td>DPP</td><td> $1 2 4 2 . 0 7 \pm 1 5 2 . 5 7$ </td><td> $8 2 3 . 0 6 \pm 1 1 4 . 4 7$ </td><td> $3 4 0 . 6 8 \pm 2 7 . 9 3$ </td><td> $8 0 . 7 7 \pm 7 . 5 8$ </td><td> $1 3 . 4 8 \pm 0 . 6 6$ </td><td></td><td> $2 . 5 9 \pm 0 . 1 5$ </td><td></td></tr><tr><td>LLM</td><td> $2 4 6 9 . 8 7 \pm 4 4 6 . 8 5$ </td><td> $1 4 9 9 . 3 0 \pm 3 3 3 . 1 9$ </td><td> $6 7 1 . 3 9 \pm 9 5 . 7 7$ </td><td> $1 8 7 . 3 7 \pm 3 0 . 6 2$ </td><td> $1 3 . 6 2 \pm 0 . 6 9$ </td><td> $1 0 5 . 6 4 \pm 1 8 . 9 5$ </td><td></td><td></td></tr><tr><td>Combined</td><td> $1 3 9 9 . 5 0 \pm 7 1 . 0 0$ </td><td> $9 9 9 . 0 9 \pm 6 4 . 9 8 $ </td><td> $3 1 0 . 6 8 \pm 8 . 1 8$ </td><td> $7 5 . 2 9 \pm 2 . 4 8$ </td><td> $1 4 . 0 3 \pm 0 . 4 7$ </td><td></td><td> $1 . 6 3 \pm 0 . 0 6$ </td><td></td></tr><tr><td>Hybrid</td><td> $1 4 4 9 . 5 0 \pm 7 5 . 6 1$ </td><td> $1 0 3 8 . 1 0 \pm 6 8 . 8 1$ </td><td> $3 1 8 . 9 6 \pm 8 . 3 1$ </td><td> $7 8 . 3 3 \pm 2 . 6 9$ </td><td> $1 4 . 0 3 \pm 0 . 4 7$ </td><td></td><td> $2 . 5 0 \pm 0 . 1 0$ </td><td></td></tr><tr><td>MMR</td><td> $1 1 2 1 . 7 8 \pm 1 3 6 . 9 8$ </td><td> $7 4 0 . 0 4 \pm 1 0 6 . 9 5$ </td><td> $3 0 4 . 0 2 \pm 2 8 . 5 4$ </td><td> $6 9 . 3 5 \pm 7 . 2 4$ </td><td> $1 2 . 9 2 \pm 0 . 7 9$ </td><td></td><td> $2 . 4 5 \pm 0 . 0 9$ </td><td></td></tr><tr><td colspan="9">One-stage methods: Pre-Synthesis Pruning</td></tr><tr><td>Geo. Residual Novelty</td><td> $4 2 1 6 . 3 1 \pm 4 7 3 . 9 1$ </td><td> $2 7 9 6 . 2 9 \pm 3 5 9 . 1 1$ </td><td> $1 1 1 1 . 7 1 \pm 8 8 . 8 3$ </td><td> $3 3 0 . 7 3 \pm 1 7 . 8 5$ </td><td> $1 3 . 4 3 \pm 0 . 6 8$ </td><td></td><td></td><td> $6 . 3 4 \pm 3 . 8 7$ </td></tr><tr><td>Centroid Drift</td><td> $4 4 0 6 . 0 7 \pm 3 3 4 . 4 4$ </td><td> $2 8 8 8 . 2 3 \pm 3 0 4 . 1 9$ </td><td> $1 1 5 8 . 5 4 \pm 5 4 . 3 4$ </td><td> $3 3 9 . 0 4 \pm 1 3 . 2 9$ </td><td> $1 3 . 2 7 \pm 0 . 7 0$ </td><td></td><td></td><td> $2 . 2 7 \pm 0 . 0 6$ </td></tr><tr><td>Submodular Coverage</td><td> $4 3 7 0 . 9 6 \pm 3 5 4 . 7 5$ </td><td> $2 9 0 4 . 3 0 \pm 2 9 6 . 4 7$ </td><td> $1 1 6 5 . 9 4 \pm 5 0 . 7 4$ </td><td> $3 4 3 . 3 2 \pm 1 2 . 5 4$ </td><td> $1 3 . 6 2 \pm 0 . 6 9$ </td><td></td><td></td><td> $1 . 2 9 \pm 0 . 1 2$ </td></tr><tr><td>DPP</td><td> $4 3 6 5 . 2 9 \pm 3 5 8 . 8 3$ </td><td> $2 8 9 6 . 4 7 \pm 3 0 0 . 1 5$ </td><td> $1 1 6 1 . 1 4 \pm 5 2 . 9 8$ </td><td> $3 4 2 . 9 1 \pm 1 2 . 5 5$ </td><td> $1 3 . 6 2 \pm 0 . 6 9$ </td><td></td><td></td><td> $2 . 2 3 \pm 0 . 1 5$ </td></tr><tr><td>LLM</td><td> $4 3 4 5 . 8 7 \pm 3 8 4 . 2 1$ </td><td> $2 8 6 2 . 3 6 \pm 3 1 7 . 9 1$ </td><td> $1 1 4 2 . 6 2 \pm 6 4 . 3 4$ </td><td>333.12 ± 16.24 13.62 ± 0.69</td><td></td><td></td><td></td><td>20.23 ± 2.65</td></tr><tr><td>Combined</td><td> $4 4 0 8 . 6 2 \pm 1 6 1 . 6 3$ </td><td> $3 0 0 2 . 8 2 \pm 1 4 3 . 5 8$ </td><td> $1 0 9 1 . 4 7 \pm 2 2 . 3 2$ </td><td> $3 2 2 . 5 1 \pm 6 . 6 9$ </td><td> $1 4 . 0 0 \pm 0 . 4 7$ </td><td></td><td></td><td> $1 . 2 6 \pm 0 . 0 6$ </td></tr><tr><td>Hybrid</td><td> $3 8 3 4 . 0 9 \pm 2 0 5 . 6 0$ </td><td> $2 6 2 5 . 2 2 \pm 1 6 4 . 3 0$ </td><td> $9 9 9 . 3 3 \pm 3 0 . 6 8$ </td><td> $3 0 3 . 3 2 \pm 8 . 3 2$ </td><td> $1 3 . 3 7 \pm 0 . 5 0$ </td><td></td><td></td><td> $2 . 3 8 \pm 0 . 0 9$ </td></tr><tr><td>MMR</td><td> $4 4 4 6 . 9 1 \pm 1 5 8 . 9 5$ </td><td> $3 0 2 8 . 7 6 \pm 1 4 2 . 7 2$ </td><td> $1 0 9 6 . 2 0 \pm 2 1 . 7 5$ </td><td> $3 2 3 . 4 9 \pm 6 . 6 5$ </td><td> $1 4 . 0 7 \pm 0 . 4 7$ </td><td></td><td></td><td> $2 . 5 2 \pm 0 . 0 6$ </td></tr><tr><td colspan="9">Two-stage methods: Post-Retrieval + Pre-Synthesis Pruning</td></tr><tr><td>Centroid Drift</td><td> $1 1 4 2 . 4 3 \pm 1 0 5 . 8 2$ </td><td> $7 4 8 . 4 8 \pm 8 8 . 4 9$ </td><td> $3 3 5 . 4 9 \pm 1 4 . 2 9$ </td><td> $7 7 . 8 5 \pm 5 . 5 8$ </td><td> $1 4 . 4 3 \pm 1 . 2 5$ </td><td></td><td> $1 . 3 3 \pm 0 . 1 5$ </td><td> $0 . 8 3 \pm 0 . 0 3$ </td></tr><tr><td>Submodular Coverage</td><td>1513.36 ± 163.23</td><td> $1 0 2 0 . 5 1 \pm 1 4 6 . 4 6$ </td><td>426.78 ± 37.31</td><td> $1 0 7 . 0 7 \pm 1 4 . 4 8$ </td><td> $1 3 . 8 7 \pm 0 . 8 3$ </td><td></td><td> $1 . 4 2 \pm 0 . 1 5$ </td><td>0.20 ± 0.05</td></tr><tr><td>Centroid Drift + Submodular</td><td> $1 5 8 8 . 0 1 \pm 2 2 0 . 4 4$ </td><td> $1 0 9 8 . 0 6 \pm 2 2 4 . 1 5$ </td><td> $3 7 7 . 1 3 \pm 2 3 . 3 2$ </td><td> $9 2 . 2 2 \pm 5 . 4 2$ </td><td> $1 3 . 6 2 \pm 0 . 6 9$ </td><td></td><td> $2 . 5 7 \pm 0 . 1 7$ </td><td> $0 . 2 9 \pm 0 . 0 1$ </td></tr><tr><td>LLM</td><td> $2 9 6 2 . 1 1 \pm 4 3 4 . 1 8$ </td><td> $1 7 6 3 . 2 1 \pm 3 3 7 . 5 6$ </td><td> $7 6 9 . 3 2 \pm 8 5 . 0 0$ </td><td> $2 1 0 . 6 8 \pm 2 8 . 7 3$ </td><td> $1 3 . 6 0 \pm 0 . 6 9$ </td><td></td><td> $1 7 7 . 2 4 \pm 2 1 . 1 0$ </td><td> $2 4 . 0 7 \pm 1 . 8 4$ </td></tr><tr><td>Lexical</td><td> $1 4 1 4 . 3 \pm 6 4 . 8$ </td><td> $1 0 1 4 . 8 \pm 6 4 . 2$ </td><td> $3 1 3 . 1 \pm 5 . 8$ </td><td> $7 6 . 0 \pm 1 . 8$ </td><td> $1 4 . 2 \pm 0 . 4$ </td><td></td><td> $0 . 0 4 \pm 0 . 0 0$ </td><td> $0 . 0 1 \pm 0 . 0 0$ </td></tr><tr><td>SC+LLM</td><td> $1 7 0 0 . 6 0 \pm 1 0 9 . 9 8 $ </td><td> $1 2 1 0 . 7 5 \pm 9 7 . 3 9$ </td><td> $3 6 8 . 6 5 \pm 1 3 . 6 7$ </td><td> $9 5 . 1 0 \pm 4 . 4 4$ </td><td> $1 3 . 9 7 \pm 0 . 4 7$ </td><td></td><td> $3 . 9 9 \pm 0 . 1 1$ </td><td> $9 . 6 1 \pm 0 . 3 3$ </td></tr><tr><td>CD+LLM</td><td> $1 5 8 9 . 3 3 \pm 7 6 . 1 8$ </td><td> $1 1 0 9 . 2 3 \pm 6 8 . 2 7$ </td><td> $3 6 2 . 0 2 \pm 1 0 . 2 8$ </td><td> $9 0 . 8 4 \pm 3 . 0 5$ </td><td> $1 3 . 9 7 \pm 0 . 4 7$ </td><td></td><td> $3 . 6 5 \pm 0 . 0 7$ </td><td> $1 0 . 8 0 \pm 0 . 2 4$ </td></tr><tr><td>Combined</td><td> $1 3 9 2 . 2 1 \pm 7 1 . 3 6$ </td><td> $9 9 4 . 3 8 \pm 6 5 . 2 2$ </td><td> $3 0 9 . 0 2 \pm 8 . 4 4$ </td><td> $7 4 . 7 8 \pm 2 . 4 5$ </td><td> $1 4 . 0 4 \pm 0 . 4 6$ </td><td></td><td> $1 . 6 0 \pm 0 . 0 6$ </td><td> $0 . 2 7 \pm 0 . 0 1$ </td></tr><tr><td>Hybrid MMR</td><td> $1 4 5 7 . 3 6 \pm 7 5 . 3 1$ </td><td> $1 0 4 3 . 0 3 \pm 6 8 . 6 0$ </td><td> $3 2 0 . 2 1 \pm 8 . 3 6$ </td><td> $7 8 . 9 3 \pm 2 . 6 9$ </td><td> $1 4 . 0 3 \pm 0 . 4 7$ </td><td></td><td> $2 . 3 4 \pm 0 . 0 8$ </td><td> $0 . 3 5 \pm 0 . 0 1$ </td></tr><tr><td></td><td> $1 3 8 1 . 3 0 \pm 6 8 . 2 2$ </td><td> $9 8 7 . 7 9 \pm 6 3 . 5 9$ </td><td> $3 0 5 . 0 4 \pm 6 . 6 9$ </td><td> $7 3 . 1 0 \pm 2 . 1 5$ </td><td> $1 4 . 0 9 \pm 0 . 4 6$ </td><td></td><td> $2 . 4 5 \pm 0 . 0 3$ </td><td> $1 . 0 0 \pm 0 . 0 3$ </td></tr><tr><td colspan="9">Three-stage methods: Pre-Retrieval + Post-Retrieval + Pre-Synthesis Pruning</td></tr><tr><td>Geo. Residual Novelty</td><td> $1 9 3 7 . 7 0 \pm 3 4 4 . 0 4$ </td><td> $1 4 1 7 . 2 9 \pm 3 1 8 . 6 2$ </td><td> $3 8 4 . 9 6 \pm 4 0 . 1 4$ </td><td> $1 1 2 . 5 8 \pm 1 4 . 2 4$ </td><td> $1 3 . 6 2 \pm 0 . 6 9$ </td><td> $0 . 8 5 \pm 0 . 0 7$ </td><td> $2 . 6 2 \pm 0 . 3 0$ </td><td> $1 . 3 3 \pm 0 . 0 7$ </td></tr><tr><td>DPP</td><td> $1 6 2 6 . 1 5 \pm 3 5 6 . 2 5$ </td><td> $1 1 8 9 . 8 3 \pm 3 3 3 . 6 2$ </td><td> $3 2 6 . 3 2 \pm 2 9 . 2 5$ </td><td> $8 8 . 1 6 \pm 6 . 0 3$ </td><td> $1 3 . 6 2 \pm 0 . 6 9$ </td><td> $0 . 7 8 \pm 0 . 0 5$ </td><td> $2 . 2 2 \pm 0 . 2 1$ </td><td> $1 . 2 4 \pm 0 . 1 9$ </td></tr><tr><td>Centroid Drift</td><td> $1 6 6 9 . 8 4 \pm 3 2 1 . 4 3$ </td><td> $1 2 1 2 . 2 2 \pm 3 1 7 . 9 3$ </td><td> $3 3 9 . 7 4 \pm 2 2 . 1 4$ </td><td> $9 5 . 9 5 \pm 5 . 3 1$ </td><td> $1 3 . 6 2 \pm 0 . 6 9$ </td><td> $0 . 8 9 \pm 0 . 1 1$ </td><td> $2 . 0 7 \pm 0 . 1 6$ </td><td> $1 . 3 6 \pm 0 . 1 5$ </td></tr><tr><td>Lexical  $+ \mathbf { C D } + \mathbf { S C }$ </td><td> $1 3 3 4 . 2 2 \pm 2 9 4 . 8 1$ </td><td> $9 0 8 . 7 9 \pm 2 7 9 . 2 5$ </td><td> $3 2 9 . 6 3 \pm 2 4 . 0 7$ </td><td> $9 1 . 3 6 \pm 6 . 2 7$ </td><td> $1 3 . 6 2 \pm 0 . 6 9$ </td><td> $0 . 0 0 \pm 0 . 0 0$ </td><td> $2 . 1 9 \pm 0 . 0 6$ </td><td> $0 . 2 9 \pm 0 . 0 1$ </td></tr><tr><td>Lexical</td><td> $1 1 6 6 . 4 \pm 5 6 . 7$ </td><td>806.3 ± 54.0</td><td> $2 7 3 . 6 \pm 5 . 8$ </td><td> $7 4 . 6 \pm 1 . 7$ </td><td> $1 4 . 1 \pm 0 . 4$ </td><td> $0 . 0 0 \pm 0 . 0 0$ </td><td> $0 . 0 3 \pm 0 . 0 0$ </td><td> $0 . 0 1 \pm 0 . 0 0$ </td></tr><tr><td>LLM</td><td> $1 5 7 5 . 2 3 \pm 1 2 0 . 2 2$ </td><td> $1 0 3 8 . 8 6 \pm 9 5 . 4 6$ </td><td> $3 5 3 . 6 9 \pm 2 1 . 1 3$ </td><td> $9 9 . 0 4 \pm 6 . 4 7$ </td><td> $1 3 . 4 5 \pm 0 . 5 0$ </td><td> $2 3 . 6 2 \pm 1 . 2 2$ </td><td> $5 1 . 9 9 \pm 2 . 6 1$ </td><td> $9 . 1 1 \pm 0 . 4 3$ </td></tr><tr><td>Combined</td><td> $1 2 2 7 . 6 0 \pm 7 6 . 6 2$ </td><td> $8 6 6 . 0 4 \pm 7 3 . 0 6$ </td><td> $2 7 1 . 1 5 \pm 7 . 0 3$ </td><td> $7 3 . 8 5 \pm 2 . 2 4$ </td><td> $1 4 . 1 1 \pm 0 . 4 7$ </td><td> $0 . 6 2 \pm 0 . 0 3$ </td><td> $1 . 3 0 \pm 0 . 0 3$ </td><td> $0 . 2 6 \pm 0 . 0 1$ </td></tr><tr><td>Hybrid</td><td> $1 2 8 2 . 7 2 \pm 8 1 . 1 8$ </td><td> $9 1 0 . 2 3 \pm 7 6 . 4 0$ </td><td> $2 7 8 . 6 8 \pm 8 . 2 0$ </td><td> $7 7 . 0 3 \pm 2 . 3 7$ </td><td> $1 4 . 0 1 \pm 0 . 4 7$ </td><td> $1 . 0 6 \pm 0 . 0 4$ </td><td> $1 . 7 2 \pm 0 . 0 6$ </td><td> $0 . 3 5 \pm 0 . 0 1$ </td></tr><tr><td>MMR</td><td> $1 1 5 7 . 7 0 \pm 6 3 . 6 0$ </td><td> $7 9 9 . 7 0 \pm 5 9 . 9 4$ </td><td> $2 6 7 . 5 4 \pm 6 . 8 8$ </td><td> $7 2 . 8 3 \pm 2 . 1 1$ </td><td> $1 4 . 1 0 \pm 0 . 4 7$ </td><td> $0 . 7 9 \pm 0 . 0 2$ </td><td> $2 . 0 3 \pm 0 . 0 3$ </td><td> $1 . 0 0 \pm 0 . 0 3$ </td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 8: Token breakdown across all methods (in thousands, k) (§ A.8). Values are mean ± standard error (§ A.1).

$$
3 7 5 . 4 \pm 2 . 6 \mathrm { k \Omega }
$$

$$
2 . 3 \pm 0 . 1 \mathrm { k \Omega }
$$

$$
1 1 . 9 \pm 0 . 1 \bf { k }
$$

$$
3 6 1 . 3 \pm 2 . 5 \mathrm { k }
$$

$$
1 7 5 . 7 \pm 5 . 1 \mathrm { k \Omega }
$$

$$
2 . 3 \pm 0 . 1 \mathrm { k \Omega }
$$

$$
4 . 8 \pm 0 . 2 \mathrm { k \Omega }
$$

$$
1 3 7 . 5 \pm 2 . 0 \mathrm { k }
$$

$$
1 6 3 . 2 \pm 3 . 8 \mathrm { k }
$$

$$
2 . 3 \pm 0 . 1 \mathrm { k \Omega }
$$

$$
3 . 5 \pm 0 . 1 \mathrm { k \Omega }
$$

$$
5 . 5 \pm 0 . 2 \mathrm { k \Omega }
$$

$$
1 4 1 . 9 \pm 3 . 3 \mathrm { k \Omega }
$$

$$
1 2 7 . 5 \pm 1 . 6 \mathrm { k }
$$

$$
2 . 3 \pm 0 . 1 \mathrm { k \Omega }
$$

$$
4 . 2 \pm 0 . 1 \mathrm { k \Omega }
$$

$$
3 . 6 \pm 0 . 1 \mathrm { k \Omega }
$$

$$
1 3 0 . 5 \pm 2 . 4 \mathrm { k }
$$

$$
5 . 5 \pm 0 . 2 \mathrm { k \Omega }
$$

$$
1 2 9 . 6 \pm 2 . 5 \mathrm { k \Omega }
$$

$$
2 . 3 \pm 0 . 1 \mathrm { k \Omega }
$$

$$
3 . 2 \pm 0 . 1 \mathrm { k \Omega }
$$

$$
1 1 9 . 3 \pm 1 . 8 \mathrm { k }
$$

$$
4 . 9 \pm 0 . 1 \mathrm { k \Omega }
$$

$$
2 1 1 . 8 \pm 7 . 9 \mathrm { k }
$$

$$
2 . 3 \pm 0 . 1 \mathrm { k \Omega }
$$

$$
5 . 5 \pm 0 . 3 \mathrm { k \Omega }
$$

$$
2 0 . 8 \pm 0 . 9 \mathrm { k \Omega }
$$

$$
1 8 3 . 3 \pm 5 . 4 \mathrm { k \Omega }
$$

$$
1 1 7 . 8 \pm 2 . 5 \mathrm { k \Omega }
$$

$$
2 . 2 \pm 0 . 1 \mathrm { k \Omega }
$$

$$
2 . 8 \pm 0 . 1 \mathrm { k \Omega }
$$

$$
7 5 . 1 \pm 1 . 7 \mathrm { k \Omega }
$$

$$
5 . 7 \pm 0 . 1 \mathrm { k \Omega }
$$

$$
1 2 1 . 3 \pm 2 . 8 \mathrm { k }
$$

$$
2 . 2 \pm 0 . 1 \mathrm { k \Omega }
$$

$$
2 . 9 \pm 0 . 1 \mathrm { k \Omega }
$$

$$
7 7 . 5 \pm 1 . 9 \mathrm { k }
$$

$$
5 . 9 \pm 0 . 2 \mathrm { k \Omega }
$$

$$
1 1 4 . 6 \pm 2 . 2 \mathrm { k \Omega }
$$

$$
2 . 2 \pm 0 . 1 \mathrm { k \Omega }
$$

$$
2 . 7 \pm 0 . 0 \mathrm { k \Omega }
$$

$$
7 4 . 0 \pm 1 . 6 \mathrm { k \Omega }
$$

$$
4 . 3 \pm 0 . 1 \mathrm { k \Omega }
$$

$$
3 7 4 . 1 \pm 3 . 1 \mathrm { k \Omega }
$$

$$
2 . 2 \pm 0 . 1 \mathrm { k \Omega }
$$

$$
1 1 . 9 \pm 0 . 1 \bf { k }
$$

$$
3 6 0 . 0 \pm 2 . 7 \mathrm { k \Omega }
$$

$$
3 7 4 . 5 \pm 2 . 8 \mathrm { k }
$$

$$
2 . 3 \pm 0 . 1 \mathrm { k \Omega }
$$

$$
1 1 . 9 \pm 0 . 1 \bf { k }
$$

$$
3 6 0 . 4 \pm 2 . 6 \mathrm { k \Omega }
$$

$$
3 8 4 . 7 \pm 3 . 3 \mathrm { k \Omega }
$$

$$
2 . 3 \pm 0 . 1 \mathrm { k \Omega }
$$

$$
1 1 . 9 \pm 0 . 1 \bf { k }
$$

$$
3 5 8 . 7 \pm 2 . 8 \mathrm { k }
$$

$$
1 1 . 8 \pm 0 . 1 \bf { k }
$$

$$
3 7 4 . 4 \pm 3 . 2 \mathrm { k \Omega }
$$

$$
2 . 3 \pm 0 . 1 \mathrm { k \Omega }
$$

$$
1 1 . 9 \pm 0 . 1 \bf { k }
$$

$$
3 6 0 . 3 \pm 3 . 0 \mathrm { k \Omega }
$$

$$
3 8 6 . 7 \pm 3 . 1 \mathrm { k \Omega }
$$

$$
2 . 3 \pm 0 . 1 \mathrm { k \Omega }
$$

$$
1 1 . 9 \pm 0 . 1 \bf { k }
$$

$$
1 3 . 2 \pm 0 . 2 \mathrm { k \Omega }
$$

$$
3 5 9 . 3 \pm 2 . 8 \mathrm { k \Omega }
$$

$$
3 7 4 . 5 \pm 6 . 1 \mathrm { k \Omega }
$$

$$
2 . 2 \pm 0 . 1 \mathrm { k \Omega }
$$

$$
1 1 . 8 \pm 0 . 1 \bf { k }
$$

$$
2 4 8 . 4 \pm 5 . 1 \mathrm { k \Omega }
$$

$$
3 3 2 . 3 \pm 1 0 . 6 \mathrm { k }
$$

$$
1 . 9 \pm 0 . 1 \mathrm { k \Omega }
$$

$$
1 1 . 6 \pm 0 . 1 \bf { k }
$$

$$
2 1 1 . 4 \pm 9 . 1 \bf { k }
$$

$$
3 6 6 . 0 \pm 5 . 2 \mathrm { k \Omega }
$$

$$
2 . 2 \pm 0 . 1 \mathrm { k \Omega }
$$

$$
1 1 . 8 \pm 0 . 1 \bf { k }
$$

$$
2 5 0 . 8 \pm 4 . 5 \mathrm { k \Omega }
$$

$$
1 1 8 . 1 \pm 1 . 3 \mathrm { k }
$$

$$
2 . 3 \pm 0 . 1 \mathrm { k \Omega }
$$

$$
2 . 8 \pm 0 . 0 \mathrm { k \Omega }
$$

$$
1 0 9 . 5 \pm 1 . 2 \mathrm { k \Omega }
$$

$$
3 . 6 \pm 0 . 0 \mathrm { k \Omega }
$$

$$
1 4 2 . 1 \pm 3 . 2 \mathrm { k \Omega }
$$

$$
2 . 2 \pm 0 . 1 \mathrm { k \Omega }
$$

$$
3 . 6 \pm 0 . 1 \mathrm { k \Omega }
$$

$$
1 3 0 . 7 \pm 2 . 4 \mathrm { k }
$$

$$
5 . 5 \pm 0 . 2 \mathrm { k \Omega }
$$

$$
\mathrm { C D } + \mathrm { S C }
$$

$$
1 3 7 . 5 \pm 2 . 0 \mathrm { k }
$$

$$
2 . 3 \pm 0 . 1 \mathrm { k \Omega }
$$

$$
3 . 4 \pm 0 . 1 \mathrm { k \Omega }
$$

$$
1 2 7 . 6 \pm 1 . 6 \mathrm { k }
$$

$$
2 2 6 . 6 \pm 8 . 1 \mathrm { k \Omega }
$$

$$
4 . 2 \pm 0 . 1 \mathrm { k \Omega }
$$

$$
2 . 3 \pm 0 . 1 \mathrm { k \Omega }
$$

$$
5 . 8 \pm 0 . 3 \mathrm { k \Omega }
$$

$$
2 5 . 7 \pm 0 . 9 \mathrm { k \Omega }
$$

$$
1 9 2 . 8 \pm 5 . 4 \mathrm { k \Omega }
$$

$$
1 1 5 . 1 \pm 1 . 2 \mathrm { k \Omega }
$$

$$
2 . 3 \pm 0 . 1 \mathrm { k \Omega }
$$

$$
2 . 8 \pm 0 . 0 \mathrm { k \Omega }
$$

$$
1 1 0 . 1 \pm 1 . 3 \mathrm { k }
$$

$$
1 3 9 . 1 \pm 4 . 1 \mathrm { k \Omega }
$$

$$
2 . 2 \pm 0 . 1 \mathrm { k \Omega }
$$

$$
3 . 5 \pm 0 . 1 \mathrm { k \Omega }
$$

$$
2 . 8 \pm 0 . 2 \mathrm { k \Omega }
$$

$$
8 8 . 1 \pm 2 . 7 \mathrm { k \Omega }
$$

$$
5 . 3 \pm 0 . 2 \mathrm { k \Omega }
$$

$$
1 2 1 . 7 \pm 2 . 8 \mathrm { k }
$$

$$
2 . 2 \pm 0 . 1 \mathrm { k \Omega }
$$

$$
2 . 9 \pm 0 . 1 \mathrm { k \Omega }
$$

$$
7 7 . 7 \pm 1 . 9 \mathrm { k }
$$

$$
5 . 9 \pm 0 . 2 \mathrm { k \Omega }
$$

$$
1 1 7 . 6 \pm 2 . 6 \mathrm { k \Omega }
$$

$$
2 . 2 \pm 0 . 1 \mathrm { k \Omega }
$$

$$
2 . 8 \pm 0 . 1 \mathrm { k \Omega }
$$

$$
7 4 . 9 \pm 1 . 8 \mathrm { k \Omega }
$$

$$
5 . 7 \pm 0 . 1 \mathrm { k \Omega }
$$

$$
1 3 6 . 0 \pm 3 . 2 \mathrm { k \Omega }
$$

$$
2 . 2 \pm 0 . 1 \mathrm { k \Omega }
$$

$$
3 . 4 \pm 0 . 1 \mathrm { k \Omega }
$$

$$
3 . 2 \pm 0 . 1 \mathrm { k \Omega }
$$

$$
8 6 . 6 \pm 2 . 1 \mathrm { k }
$$

$$
4 . 1 \pm 0 . 1 \mathrm { k \Omega }
$$

$$
1 1 4 . 6 \pm 2 . 2 \mathrm { k \Omega }
$$

$$
2 . 2 \pm 0 . 1 \mathrm { k \Omega }
$$

$$
2 . 7 \pm 0 . 1 \mathrm { k \Omega }
$$

$$
7 4 . 0 \pm 1 . 6 \mathrm { k \Omega }
$$

$$
4 . 3 \pm 0 . 1 \mathrm { k \Omega }
$$

$$
1 5 0 . 0 \pm 4 . 5 \mathrm { k \Omega }
$$

$$
2 . 2 \pm 0 . 1 \mathrm { k \Omega }
$$

$$
4 . 4 \pm 0 . 2 \mathrm { k \Omega }
$$

$$
1 3 8 . 6 \pm 3 . 3 \mathrm { k \Omega }
$$

$$
4 . 7 \pm 0 . 1 \mathrm { k \Omega }
$$

$$
1 1 3 . 3 \pm 2 . 2 \mathrm { k \Omega }
$$

$$
2 . 3 \pm 0 . 1 \mathrm { k \Omega }
$$

$$
3 . 1 \pm 0 . 1 \mathrm { k \Omega }
$$

$$
1 0 3 . 5 \pm 1 . 6 \mathrm { k }
$$

$$
4 . 4 \pm 0 . 1 \mathrm { k \Omega }
$$

$$
1 2 0 . 6 \pm 2 . 1 \mathrm { k \Omega }
$$

$$
2 . 3 \pm 0 . 1 \mathrm { k \Omega }
$$

$$
3 . 4 \pm 0 . 1 \mathrm { k \Omega }
$$

$$
1 1 1 . 2 \pm 1 . 6 \mathrm { k }
$$

$$
3 . 7 \pm 0 . 1 \mathrm { k \Omega }
$$

$$
_ \mathrm { L e x i c a l + C D + S C }
$$

$$
1 2 0 . 1 \pm 2 . 3 \mathrm { k }
$$

$$
2 . 3 \pm 0 . 1 \mathrm { k \Omega }
$$

$$
3 . 5 \pm 0 . 1 \mathrm { k \Omega }
$$

$$
1 1 0 . 7 \pm 1 . 7 \mathrm { k }
$$

$$
3 . 7 \pm 0 . 1 \mathrm { k \Omega }
$$

$$
9 9 . 0 \pm 1 . 1 \mathrm { k \Omega }
$$

$$
2 . 3 \pm 0 . 1 \mathrm { k \Omega }
$$

$$
2 . 8 \pm 0 . 0 \mathrm { k \Omega }
$$

$$
9 4 . 0 \pm 1 . 1 \mathrm { k \Omega }
$$

$$
1 4 3 . 9 \pm 7 . 7 \mathrm { k \Omega }
$$

$$
2 . 0 \pm 0 . 1 \mathrm { k \Omega }
$$

$$
3 . 8 \pm 0 . 2 \mathrm { k \Omega }
$$

$$
6 . 2 \pm 0 . 3 \mathrm { k \Omega }
$$

$$
8 2 . 5 \pm 4 . 6 \mathrm { k }
$$

$$
1 0 2 . 8 \pm 2 . 2 \mathrm { k \Omega }
$$

$$
2 . 2 \pm 0 . 1 \mathrm { k \Omega }
$$

$$
2 . 8 \pm 0 . 1 \mathrm { k \Omega }
$$

$$
6 5 . 1 \pm 1 . 5 \mathrm { k }
$$

$$
5 . 8 \pm 0 . 1 \mathrm { k \Omega }
$$

$$
1 0 6 . 3 \pm 2 . 6 \mathrm { k }
$$

$$
2 . 2 \pm 0 . 1 \mathrm { k \Omega }
$$

$$
2 . 9 \pm 0 . 1 \mathrm { k \Omega }
$$

$$
1 0 0 . 1 \pm 2 . 0 \mathrm { k }
$$

$$
2 . 2 \pm 0 . 1 \mathrm { k \Omega }
$$

$$
6 7 . 0 \pm 1 . 8 \mathrm { k }
$$

$$
2 . 7 \pm 0 . 0 \mathrm { k \Omega }
$$

$$
6 . 0 \pm 0 . 1 \mathrm { k \Omega }
$$

$$
6 4 . 5 \pm 1 . 4 \mathrm { k }
$$

$$
3 . 8 \pm 0 . 1 \mathrm { k \Omega }
$$

Table 9: Node-level pruning summary
<table><tr><td>Method</td><td># Nodes (§ A.3)</td><td>Avg. Pruned Nodes</td><td>Pruning Rate (%)</td></tr><tr><td>Baseline</td><td> $2 9 . 0 \pm 0 . 0$ </td><td> $0 . 0 \pm 0 . 0$ </td><td>0.0</td></tr><tr><td colspan="4">One-stage methods: Post-Retrieval Pruning</td></tr><tr><td>Geo. Residual Novelty</td><td> $1 3 . 3 1 \pm 0 . 3 7$ </td><td> $4 . 4 8 \pm 0 . 1 0$ </td><td>33.7</td></tr><tr><td>Centroid Drift</td><td> $1 0 . 4 7 \pm 0 . 1 3$ </td><td> $3 . 4 4 \pm 0 . 0 5$ </td><td>32.9</td></tr><tr><td>Submodular Coverage</td><td> $1 0 . 8 2 \pm 0 . 2 8$ </td><td> $4 . 8 2 \pm 0 . 0 7$ </td><td>44.5</td></tr><tr><td>DPP</td><td> $9 . 8 8 \pm 0 . 1 6$ </td><td> $4 . 9 9 \pm 0 . 0 6$ </td><td>50.5</td></tr><tr><td>LLM</td><td> $1 4 . 9 0 \pm 0 . 5 3$ </td><td> $6 . 3 0 \pm 0 . 1 9$ </td><td>42.3</td></tr><tr><td>Hybrid</td><td> $9 . 2 8 \pm 0 . 1 5$ </td><td> $4 . 8 2 \pm 0 . 0 5$ </td><td>51.9</td></tr><tr><td>Combined</td><td> $9 . 0 0 \pm 0 . 1 1$ </td><td> $4 . 7 5 \pm 0 . 0 5$ </td><td>52.8</td></tr><tr><td>MMR</td><td> $8 . 8 4 \pm 0 . 0 8$ </td><td> $4 . 9 3 \pm 0 . 0 3$ </td><td>55.8</td></tr><tr><td colspan="4">One-stage methods: Pre-Synthesis Pruning</td></tr><tr><td>Geo. Residual Novelty</td><td> $2 8 . 8 8 \pm 0 . 1 2$ </td><td> $0 . 0 0 \pm 0 . 0 0$ </td><td>0.0</td></tr><tr><td>Centroid Drift</td><td> $2 8 . 9 4 \pm 0 . 0 5$ </td><td> $0 . 0 0 \pm 0 . 0 0$ </td><td>0.0</td></tr><tr><td>Submodular Coverage</td><td> $2 8 . 8 5 \pm 0 . 1 3$ </td><td> $0 . 0 0 \pm 0 . 0 0$ </td><td>0.0</td></tr><tr><td>DPP</td><td> $2 8 . 9 0 \pm 0 . 1 0$ </td><td> $0 . 0 0 \pm 0 . 0 0$ </td><td>0.0</td></tr><tr><td>LLM</td><td> $2 8 . 9 8 \pm 0 . 0 2$ </td><td> $0 . 0 0 \pm 0 . 0 0$ </td><td>0.0</td></tr><tr><td>Combined</td><td> $2 8 . 9 0 \pm 0 . 0 8$ </td><td> $0 . 0 0 \pm 0 . 0 0$ </td><td>0.0</td></tr><tr><td>Hybrid</td><td> $2 8 . 9 8 \pm 0 . 0 2$ </td><td> $0 . 0 0 \pm 0 . 0 0$ </td><td>0.0</td></tr><tr><td>MMR</td><td> $2 8 . 9 8 \pm 0 . 0 2$ </td><td> $0 . 0 0 \pm 0 . 0 0$ </td><td>0.0</td></tr><tr><td colspan="4">Two-stage methods: Post-Retrieval + Pre-Synthesis Pruning</td></tr><tr><td>Centroid Drift</td><td> $9 . 0 2 \pm 0 . 0 2$ </td><td> $4 . 2 0 \pm 0 . 0 5$ </td><td>46.6</td></tr><tr><td>Submodular Coverage</td><td> $1 0 . 8 0 \pm 0 . 2 8$ </td><td> $4 . 8 0 \pm 0 . 0 7$ </td><td>44.4</td></tr><tr><td>CD + SC</td><td> $1 0 . 4 5 \pm 0 . 1 3$ </td><td> $3 . 4 3 \pm 0 . 0 5$ </td><td>32.8</td></tr><tr><td>LLM</td><td> $1 5 . 5 7 \pm 0 . 5 3$ </td><td> $6 . 3 0 \pm 0 . 1 6$ </td><td>40.5</td></tr><tr><td>Lexical</td><td> $9 . 0 0 \pm 0 . 0 0$ </td><td> $4 . 9 9 \pm 0 . 0 1$ </td><td>55.4</td></tr><tr><td> $\mathrm { S C } + \mathrm { L L M }$ </td><td> $1 0 . 5 4 \pm 0 . 3 0$ </td><td> $4 . 7 9 \pm 0 . 0 7$ </td><td>45.4</td></tr><tr><td> $\mathrm { C D } + \mathrm { L L M }$ </td><td> $1 0 . 2 4 \pm 0 . 1 7$ </td><td> $3 . 4 5 \pm 0 . 0 5$ </td><td>33.7</td></tr><tr><td>Combined</td><td> $8 . 9 8 \pm 0 . 1 1$ </td><td> $4 . 7 6 \pm 0 . 0 5$ </td><td>53.0</td></tr><tr><td>Hybrid</td><td> $9 . 3 0 \pm 0 . 1 5$ </td><td> $4 . 8 2 \pm 0 . 0 5$ </td><td>51.8</td></tr><tr><td>MMR</td><td> $8 . 8 4 \pm 0 . 0 8$ </td><td> $4 . 9 4 \pm 0 . 0 3$ </td><td>55.9</td></tr><tr><td colspan="4">Three-stage methods: Pre-Retrieval + Post-Retrieval + Pre-Synthesis Pruning</td></tr><tr><td>Geo. Residual Novelty</td><td> $1 1 . 5 7 \pm 0 . 3 3$ </td><td> $3 . 4 9 \pm 0 . 0 9$ </td><td>30.2</td></tr><tr><td>DPP</td><td> $8 . 6 8 \pm 0 . 1 5$ </td><td> $3 . 9 5 \pm 0 . 0 5$ </td><td>45.5</td></tr><tr><td>Centroid Drift</td><td> $9 . 2 7 \pm 0 . 1 3$ </td><td> $2 . 4 5 \pm 0 . 0 5$ </td><td>26.4</td></tr><tr><td> $_ \mathrm { L e x i c a l + C D + S C }$ </td><td> $9 . 2 6 \pm 0 . 1 4$ </td><td> $2 . 3 1 \pm 0 . 0 6$ </td><td>24.9</td></tr><tr><td>Lexical</td><td> $7 . 9 0 \pm 0 . 0 4$ </td><td> $3 . 8 8 \pm 0 . 0 4$ </td><td>49.1</td></tr><tr><td>LLM</td><td> $1 0 . 1 2 \pm 0 . 4 5$ </td><td> $4 . 5 7 \pm 0 . 1 3$ </td><td>45.2</td></tr><tr><td>Combined</td><td> $7 . 9 0 \pm 0 . 1 0$ </td><td> $3 . 7 5 \pm 0 . 0 5$ </td><td>47.5</td></tr><tr><td>Hybrid</td><td> $8 . 1 6 \pm 0 . 1 4$ </td><td> $3 . 8 1 \pm 0 . 0 5$ </td><td>46.7</td></tr><tr><td>MMR</td><td> $7 . 8 2 \pm 0 . 0 8$ </td><td> $3 . 9 7 \pm 0 . 0 2$ </td><td>50.8</td></tr></table>

This table summarizes how strongly each method contracts the research tree. # Nodes estimates the number of nodes actually explored per report, while Avg. Pruned Nodes measures how many of those explored nodes were discarded. Pruning Rate therefore captures the fraction of explored nodes removed by the pruning policy.

Table 10: Token accounting summary (k tokens; mean ± SE). Mean Savings vs. Baseline is computed from the reported mean total token counts
<table><tr><td>Method</td><td>Input</td><td>Output</td><td>Total</td><td>Est. Saved</td><td>Savings vs. Baseline (k)</td></tr><tr><td>Baseline</td><td> $2 6 0 . 7 \pm 2 . 7$ </td><td> $1 1 4 . 7 \pm 0 . 7$ </td><td> $3 7 5 . 4 \pm 2 . 6$ </td><td> $1 4 9 . 1 \pm 1 . 8$ </td><td>0.0</td></tr><tr><td colspan="6">One-stage methods: Post-Retrieval Pruning</td></tr><tr><td>Geo. Residual Novelty</td><td> $1 2 5 . 3 \pm { 3 . 7 }$ </td><td> $5 0 . 4 \pm 1 . 6$ </td><td> $1 7 5 . 7 \pm 5 . 1$ </td><td> $5 1 . 0 \pm 2 . 1$ </td><td>199.7</td></tr><tr><td>Centroid Drift</td><td> $9 9 . 0 \pm 1 . 6$ </td><td> $3 8 . 5 \pm 0 . 6$ </td><td> $1 3 7 . 5 \pm 2 . 0$ </td><td> $3 2 . 4 \pm 0 . 8$ </td><td>237.9</td></tr><tr><td>Submodular Coverage</td><td> $1 0 1 . 8 \pm 2 . 1$ </td><td> $4 0 . 1 \pm 1 . 3$ </td><td> $1 4 1 . 9 \pm 3 . 3$ </td><td> $3 8 . 7 \pm 0 . 9$ </td><td>233.4</td></tr><tr><td>DPP</td><td> $9 3 . 7 \pm { 1 . 8 }$ </td><td> $3 6 . 0 \pm 0 . 8$ </td><td> $1 2 9 . 6 \pm 2 . 5$ </td><td> $3 6 . 4 \pm 0 . 7$ </td><td>245.7</td></tr><tr><td>LLM</td><td> $1 5 4 . 0 \pm 5 . 7$ </td><td> $5 7 . 8 \pm 2 . 3$ </td><td> $2 1 1 . 8 \pm 7 . 9$ </td><td> $6 3 . 4 \pm 3 . 0$ </td><td>163.5</td></tr><tr><td>Hybrid</td><td> $8 7 . 7 \pm 2 . 2$ </td><td> $3 3 . 5 \pm 0 . 7$ </td><td> $1 2 1 . 3 \pm 2 . 8$ </td><td> $3 3 . 5 \pm 0 . 9$ </td><td>67.7</td></tr><tr><td>MMR</td><td> $8 2 . 9 \pm 1 . 9$ </td><td> $3 1 . 7 \pm 0 . 5$ </td><td> $1 1 4 . 6 \pm 2 . 2$ </td><td> $3 4 . 2 \pm 0 . 8$ </td><td>69.5</td></tr><tr><td>Combined</td><td> $8 5 . 5 \pm 2 . 1$ </td><td> $3 2 . 3 \pm 0 . 6$ </td><td> $1 1 7 . 8 \pm 2 . 5$ </td><td> $3 3 . 0 \pm 0 . 9$ </td><td>257.6</td></tr><tr><td colspan="6">One-stage methods: Pre-Synthesis Pruning</td></tr><tr><td>Geo. Residual Novelty</td><td> $2 5 9 . 5 \pm 2 . 8$ </td><td> $1 1 4 . 5 \pm 0 . 9$ </td><td> $3 7 4 . 1 \pm 3 . 1$ </td><td> $1 4 7 . 1 \pm 2 . 0$ </td><td>1.3</td></tr><tr><td>Centroid Drift</td><td> $2 5 9 . 7 \pm 2 . 7$ </td><td> $1 1 4 . 8 \pm 0 . 7$ </td><td> $3 7 4 . 5 \pm 2 . 8$ </td><td> $1 4 7 . 2 \pm 1 . 9$ </td><td>0.9</td></tr><tr><td>Submodular Coverage</td><td> $2 7 0 . 2 \pm 2 . 9$ </td><td> $1 1 4 . 4 \pm 0 . 9$ </td><td> $3 8 4 . 7 \pm 3 . 3$ </td><td> $1 4 7 . 3 \pm 2 . 0$ </td><td>-9.3</td></tr><tr><td>DPP</td><td> $2 5 9 . 8 \pm 3 . 0$ </td><td> $1 1 4 . 6 \pm 0 . 8$ </td><td> $3 7 4 . 4 \pm 3 . 2$ </td><td> $1 4 7 . 3 \pm 1 . 9$ </td><td>1.0</td></tr><tr><td>LLM</td><td> $2 7 1 . 3 \pm 3 . 1$ </td><td> $1 1 5 . 5 \pm 0 . 7$ </td><td> $3 8 6 . 7 \pm 3 . 1$ </td><td> $1 4 9 . 2 \pm 2 . 2$ </td><td>-11.3</td></tr><tr><td>Combined</td><td> $2 6 0 . 4 \pm 5 . 7$ </td><td> $1 1 4 . 1 \pm 0 . 9$ </td><td> $3 7 4 . 5 \pm 6 . 1$ </td><td> $1 7 1 . 6 \pm 4 . 2$ </td><td>0.2</td></tr><tr><td>Hybrid</td><td> $2 2 1 . 9 \pm 9 . 9$ </td><td> $1 1 0 . 4 \pm { 1 . 0 }$ </td><td> $3 3 2 . 3 \pm 1 0 . 6$ </td><td> $1 3 7 . 4 \pm 7 . 3$ </td><td>11.5</td></tr><tr><td>MMR</td><td> $2 5 1 . 5 \pm 5 . 0$ </td><td> $1 1 4 . 5 \pm 0 . 8$ </td><td> $3 6 6 . 0 \pm 5 . 2$ </td><td> $1 4 2 . 3 \pm 3 . 4$ </td><td>2.5</td></tr><tr><td colspan="6">Two-stage methods: Post-Retrieval + Pre-Synthesis Pruning</td></tr><tr><td>Centroid Drift</td><td> $8 5 . 9 \pm 1 . 3$ </td><td> $3 2 . 2 \pm 0 . 3$ </td><td> $1 1 8 . 1 \pm 1 . 3$ </td><td> $3 0 . 8 \pm 0 . 5$ </td><td>257.3</td></tr><tr><td>Submodular Coverage</td><td> $1 0 2 . 4 \pm 2 . 2$ </td><td> $3 9 . 8 \pm 1 . 3$ </td><td> $1 4 2 . 1 \pm 3 . 2$ </td><td> $4 1 . 9 \pm 0 . 9$ </td><td>233.3</td></tr><tr><td> $\mathrm { C D } + \mathrm { S C }$ </td><td> $9 9 . 1 \pm 1 . 6$ </td><td> $3 8 . 4 \pm 0 . 6$ </td><td> $1 3 7 . 5 \pm 2 . 0$ </td><td> $3 6 . 1 \pm 0 . 9$ </td><td>237.9</td></tr><tr><td>LLM</td><td> $1 6 5 . 6 \pm 5 . 8$ </td><td> $6 0 . 9 \pm 2 . 3$ </td><td> $2 2 6 . 6 \pm 8 . 1$ </td><td> $7 0 . 0 \pm 2 . 9$ </td><td>148.8</td></tr><tr><td>Lexical</td><td> $8 2 . 8 \pm 1 . 3$ </td><td> $3 2 . 4 \pm 0 . 3$ </td><td> $1 1 5 . 1 \pm 1 . 2$ </td><td> $4 8 . 0 \pm 0 . 7$ </td><td>260.3</td></tr><tr><td> $\mathbf { S C } + \mathbf { L L M }$ </td><td> $9 9 . 9 \pm 2 . 9$ </td><td> $3 9 . 2 \pm 1 . 4$ </td><td> $1 3 9 . 1 \pm 4 . 1$ </td><td> $4 1 . 7 \pm 1 . 3$ </td><td>236.3</td></tr><tr><td> $\mathrm { C o m b i n e d }$ </td><td> $8 5 . 3 \pm 2 . 1$ </td><td> $3 2 . 3 \pm 0 . 6$ </td><td> $1 1 7 . 6 \pm 2 . 6$ </td><td> $4 7 . 9 \pm 1 . 3$ </td><td>68.7</td></tr><tr><td> $\mathrm { C D } + \mathrm { L L M }$ </td><td> $9 8 . 1 \pm 2 . 5$ </td><td> $3 7 . 9 \pm 0 . 8$ </td><td> $1 3 6 . 0 \pm 3 . 2$ </td><td> $3 6 . 5 \pm 1 . 2$ </td><td>239.4</td></tr><tr><td>Hybrid</td><td> $8 8 . 0 \pm 2 . 2 $ </td><td> $3 3 . 6 \pm 0 . 7$ </td><td> $1 2 1 . 7 \pm 2 . 8$ </td><td> $4 6 . 1 \pm 1 . 5$ </td><td>67.6</td></tr><tr><td>MMR</td><td> $8 2 . 9 \pm 1 . 9$ </td><td> $3 1 . 7 \pm 0 . 5$ </td><td> $1 1 4 . 6 \pm 2 . 2$ </td><td> $3 4 . 2 \pm 0 . 8$ </td><td>69.5</td></tr><tr><td colspan="6">Three-stage methods: Pre-Retrieval + Post-Retrieval + Pre-Synthesis Pruning</td></tr><tr><td>Geo. Residual Novelty</td><td> $1 0 6 . 3 \pm 3 . 2$ </td><td> $4 3 . 7 \pm 1 . 4$ </td><td> $1 5 0 . 0 \pm 4 . 5$ </td><td> $3 8 . 7 \pm { 1 . 7 }$ </td><td>225.4</td></tr><tr><td>DPP Centroid Drift</td><td> $8 1 . 6 \pm { 1 . 7 }$ </td><td> $3 1 . 7 \pm 0 . 7$ </td><td> $1 1 3 . 3 \pm 2 . 2$ </td><td> $2 8 . 5 \pm 0 . 6$ </td><td>262.0</td></tr><tr><td> $_ \mathrm { L e x i c a l + C D + S C }$ </td><td> $8 6 . 4 \pm 1 . 7$ </td><td> $3 4 . 2 \pm 0 . 5$ </td><td> $1 2 0 . 6 \pm 2 . 1$ </td><td> $2 4 . 4 \pm 0 . 7$ </td><td>254.8</td></tr><tr><td></td><td> $8 6 . 0 \pm 1 . 7$ </td><td> $3 4 . 1 \pm 0 . 6$ </td><td> $1 2 0 . 1 \pm 2 . 3$ </td><td> $2 6 . 5 \pm 0 . 9$ </td><td>255.3</td></tr><tr><td>Lexical</td><td> $7 0 . 7 \pm 1 . 0$ </td><td> $2 8 . 3 \pm 0 . 3$ </td><td> $9 9 . 0 \pm 1 . 1$ </td><td> $4 0 . 2 \pm 0 . 7$ </td><td>276.4</td></tr><tr><td>LLM</td><td> $1 0 5 . 0 \pm 5 . 8$ </td><td> $3 8 . 9 \pm 2 . 0$ </td><td> $1 4 3 . 9 \pm 7 . 7$ </td><td> $3 8 . 9 \pm 2 . 2$ </td><td>231.5</td></tr><tr><td>Combined</td><td> $7 4 . 3 \pm 1 . 8$ </td><td> $2 8 . 5 \pm 0 . 5$ </td><td> $1 0 2 . 8 \pm 2 . 2$ </td><td> $4 0 . 9 \pm 1 . 1$ </td><td>72.6</td></tr><tr><td>Hybrid</td><td> $7 6 . 8 \pm 2 . 1$ </td><td> $2 9 . 5 \pm 0 . 6$ </td><td> $1 0 6 . 3 \pm 2 . 6$ </td><td> $3 8 . 9 \pm 1 . 4$ </td><td>71.7</td></tr><tr><td>MMR</td><td> $7 2 . 0 \pm 1 . 7$ </td><td> $2 8 . 2 \pm 0 . 5$ </td><td> $1 0 0 . 1 \pm 2 . 0$ </td><td> $2 7 . 0 \pm 0 . 7$ </td><td>73.3</td></tr><tr><td colspan="6">his table provides a more detailed view of token usage. In addition to total token cost, it separates input and outp kens, reports the estimated number of tokens saved by pruning, and shows the absolute mean token savings relative e baseline. This helps distinguish methods that achieve low total cost by aggressively pruning large prompt contex om those that reduce generation cost or avoid expanding large parts of the search tree altogether.</td></tr></table>

Table 11: Share of total token budget by pipeline stage $( \% ; \mathrm { m e a n } \pm \mathrm { S E } )$ . Dashes indicate stages that do not consume logged tokens for that method.
<table><tr><td>Method</td><td>Planning</td><td>Query Generation</td><td>Query Pruning</td><td>Result Proc.</td><td>Embedding</td></tr><tr><td>Baseline</td><td> $1 . 0 2 \pm 0 . 0 5$ </td><td> $4 . 5 8 \pm 0 . 1 4$ </td><td></td><td> ${ \bf 9 4 . 4 1 \pm 0 . 1 8 }$ </td><td></td></tr><tr><td colspan="6">One-stage methods: Post-Retrieval Pruning</td></tr><tr><td>Geo. Residual Novelty</td><td> $1 . 4 2 \pm 0 . 0 7$ </td><td> $2 . 6 8 \pm 0 . 0 4$ </td><td></td><td> $9 2 . 7 6 \pm 0 . 4 0$ </td><td> $3 . 1 4 \pm 0 . 0 3$ </td></tr><tr><td>Centroid Drift</td><td> $1 . 6 9 \pm 0 . 0 7$ </td><td> $2 . 5 4 \pm 0 . 0 4$ </td><td></td><td> $9 2 . 6 8 \pm 0 . 4 4$ </td><td> $3 . 1 0 \pm 0 . 0 3$ </td></tr><tr><td>Submodular Coverage</td><td> $1 . 6 7 \pm 0 . 0 7$ </td><td> $2 . 5 0 \pm 0 . 0 5$ </td><td></td><td> $9 1 . 9 8 \pm 0 . 4 5$ </td><td> $3 . 8 5 \pm 0 . 0 4$ </td></tr><tr><td>DPP</td><td> $1 . 8 1 \pm 0 . 0 7$ </td><td> $2 . 4 7 \pm 0 . 0 4$ </td><td></td><td> $9 1 . 9 0 \pm 0 . 4 2$ </td><td> $3 . 8 2 \pm 0 . 0 4$ </td></tr><tr><td>LLM</td><td> $1 . 2 3 \pm 0 . 0 6$ </td><td> $2 . 5 2 \pm 0 . 0 5$ </td><td> $9 . 5 5 \pm 0 . 1 4$ </td><td> $8 6 . 7 0 \pm 0 . 4 4$ </td><td></td></tr><tr><td>Combined</td><td> $1 . 9 0 \pm 0 . 0 7$ </td><td> $2 . 5 0 \pm 0 . 0 8$ </td><td></td><td> $6 2 . 5 1 \pm 0 . 7 7$ </td><td> $4 . 6 8 \pm 0 . 1 1$ </td></tr><tr><td>Hybrid</td><td> $1 . 8 6 \pm 0 . 0 7$ </td><td> $2 . 5 5 \pm 0 . 0 8$ </td><td></td><td> $6 2 . 8 4 \pm 0 . 6 7$ </td><td> $4 . 7 0 \pm 0 . 1 1$ </td></tr><tr><td>MMR</td><td> $1 . 9 3 \pm 0 . 0 6$ </td><td> $2 . 5 3 \pm 0 . 0 9$ </td><td></td><td> $6 3 . 4 2 \pm 0 . 7 2$ </td><td> $3 . 6 3 \pm 0 . 0 8$ </td></tr><tr><td colspan="6">One-stage methods: Pre-Synthesis Pruning</td></tr><tr><td>Geo. Residual Novelty</td><td> $0 . 6 0 \pm 0 . 0 2$ </td><td> $3 . 1 9 \pm 0 . 0 4$ </td><td></td><td> $9 6 . 2 1 \pm 0 . 3 9$ </td><td></td></tr><tr><td>Centroid Drift</td><td> $0 . 6 0 \pm 0 . 0 2$ </td><td> $3 . 1 9 \pm 0 . 0 4$ </td><td></td><td> $9 6 . 2 1 \pm 0 . 4 0$ </td><td></td></tr><tr><td>Submodular Coverage</td><td> $0 . 6 0 \pm 0 . 0 2$ </td><td> $3 . 1 0 \pm 0 . 0 4$ </td><td></td><td> $9 3 . 2 2 \pm 0 . 3 9$ </td><td> $3 . 0 9 \pm 0 . 0 3$ </td></tr><tr><td>DPP</td><td> $0 . 6 1 \pm 0 . 0 2$ </td><td> $3 . 1 9 \pm 0 . 0 4$ </td><td></td><td> $9 6 . 2 1 \pm 0 . 4 8$ </td><td></td></tr><tr><td>LLM</td><td> $0 . 6 0 \pm 0 . 0 2$ </td><td> $3 . 1 0 \pm 0 . 0 4$ </td><td> $3 . 4 3 \pm 0 . 0 6$ </td><td> $9 2 . 8 6 \pm 0 . 4 5$ </td><td></td></tr><tr><td>Combined</td><td> $0 . 5 8 \pm 0 . 0 2$ </td><td> $3 . 3 1 \pm 0 . 1 1$ </td><td></td><td> $6 5 . 2 1 \pm 0 . 8 3$ </td><td></td></tr><tr><td>Hybrid</td><td> $0 . 5 4 \pm 0 . 0 2$ </td><td> $4 . 1 0 \pm 0 . 2 0$ </td><td></td><td> $5 8 . 8 5 \pm 1 . 5 6$ </td><td></td></tr><tr><td>MMR</td><td> $0 . 6 0 \pm 0 . 0 2$ </td><td> $3 . 3 6 \pm 0 . 0 9$ </td><td></td><td> $6 7 . 7 0 \pm 0 . 7 6$ </td><td></td></tr><tr><td colspan="6">Two-stage methods: Post-Retrieval + Pre-Synthesis Pruning</td></tr><tr><td>Centroid Drift</td><td> $1 . 9 3 \pm 0 . 0 7$ </td><td> $2 . 4 0 \pm 0 . 0 4$ </td><td></td><td> $9 2 . 6 1 \pm 0 . 5 9$ </td><td> $3 . 0 6 \pm 0 . 0 3$ </td></tr><tr><td>Submodular Coverage</td><td> $1 . 6 7 \pm 0 . 0 7$ </td><td> $2 . 4 9 \pm 0 . 0 5$ </td><td></td><td> $9 1 . 9 9 \pm 0 . 5 8$ </td><td> $3 . 8 4 \pm 0 . 0 4$ </td></tr><tr><td> $\mathrm { C D } + \mathrm { S C }$ </td><td> $1 . 6 8 \pm 0 . 0 7$ </td><td> $2 . 5 3 \pm 0 . 0 4$ </td><td></td><td> $9 2 . 6 9 \pm 0 . 4 5$ </td><td> $3 . 1 0 \pm 0 . 0 3$ </td></tr><tr><td>LLM</td><td> $1 . 1 6 \pm 0 . 0 6$ </td><td> $2 . 4 8 \pm 0 . 0 4$ </td><td> $1 1 . 1 3 \pm 0 . 1 8$ </td><td> $8 5 . 2 5 \pm 0 . 4 8$ </td><td></td></tr><tr><td>Lexical</td><td> $1 . 9 8 \pm 0 . 0 7$ </td><td> $2 . 4 5 \pm 0 . 0 5$ </td><td></td><td> $9 5 . 5 7 \pm 0 . 0 9$ </td><td></td></tr><tr><td>SC+LLM</td><td> $1 . 6 8 \pm 0 . 0 7$ </td><td> $2 . 5 7 \pm 0 . 0 7$ </td><td> $1 . 8 7 \pm 0 . 0 6$ </td><td> $6 2 . 3 9 \pm 0 . 6 9$ </td><td> $3 . 6 1 \pm 0 . 0 8$ </td></tr><tr><td> $_ \mathrm { C D + L L M }$ </td><td> $1 . 6 7 \pm 0 . 0 6$ </td><td> $2 . 6 0 \pm 0 . 0 8$ </td><td> $2 . 2 9 \pm 0 . 0 6$ </td><td> $6 2 . 5 7 \pm 0 . 7 0$ </td><td> $2 . 9 0 \pm 0 . 0 7$ </td></tr><tr><td>Hybrid</td><td> $1 . 8 4 \pm 0 . 0 6$ </td><td> $2 . 5 3 \pm 0 . 0 8$ </td><td></td><td> $6 2 . 6 9 \pm 0 . 7 4$ </td><td> $4 . 7 3 \pm 0 . 1 1$ </td></tr><tr><td>Combined</td><td> $1 . 9 0 \pm 0 . 0 7$ </td><td> $2 . 5 2 \pm 0 . 0 9$ </td><td></td><td> $6 2 . 4 0 \pm 0 . 8 2$ </td><td> $4 . 6 7 \pm 0 . 1 1$ </td></tr><tr><td>MMR</td><td> $1 . 9 3 \pm 0 . 0 6$ </td><td> $2 . 5 1 \pm 0 . 0 8$ </td><td></td><td> $6 3 . 4 3 \pm 0 . 7 2$ </td><td> $3 . 6 3 \pm 0 . 0 8$ </td></tr><tr><td colspan="6">Three-stage methods: Pre-Retrieval + Post-Retrieval + Pre-Synthesis Pruning</td></tr><tr><td>Geo. Residual Novelty</td><td> $1 . 6 2 \pm 0 . 0 7$ </td><td> $2 . 9 5 \pm 0 . 0 4$ </td><td></td><td> $9 2 . 2 8 \pm 0 . 4 4$ </td><td> $3 . 1 5 \pm 0 . 0 3$ </td></tr><tr><td>DPP</td><td> $2 . 0 7 \pm 0 . 0 8$ </td><td> $2 . 7 8 \pm 0 . 0 4$ </td><td></td><td> $9 1 . 2 5 \pm 0 . 4 6$ </td><td> $3 . 9 0 \pm 0 . 0 4$ </td></tr><tr><td>Centroid Drift</td><td> $1 . 9 4 \pm 0 . 0 8$ </td><td> $2 . 8 5 \pm 0 . 0 4$ </td><td></td><td> $9 2 . 1 2 \pm 0 . 5 2$ </td><td> $3 . 0 9 \pm 0 . 0 3$ </td></tr><tr><td> $\mathrm { L e x i c a l + C D + S C }$ </td><td> $1 . 9 4 \pm 0 . 0 8$ </td><td> $2 . 9 2 \pm 0 . 0 6$ </td><td></td><td> $9 2 . 0 4 \pm 0 . 4 5$ </td><td> $3 . 1 0 \pm 0 . 0 3$ </td></tr><tr><td>Lexical</td><td> $2 . 3 0 \pm 0 . 0 8$ </td><td> $2 . 8 8 \pm 0 . 0 6$ </td><td></td><td> $9 4 . 8 2 \pm 0 . 1 1$ </td><td></td></tr><tr><td>LLM</td><td> $1 . 7 4 \pm 0 . 1 0$ </td><td> $3 . 0 4 \pm 0 . 1 5$ </td><td> $4 . 9 9 \pm 0 . 2 5$ </td><td> $5 3 . 7 2 \pm 1 . 1 7$ </td><td></td></tr><tr><td>Combined</td><td> $2 . 1 8 \pm 0 . 0 8$ </td><td> $2 . 9 1 \pm 0 . 1 2$ </td><td></td><td> $6 2 . 0 0 \pm 0 . 7 8$ </td><td> $5 . 7 5 \pm 0 . 0 7$ </td></tr><tr><td>Hybrid</td><td> $2 . 1 3 \pm 0 . 0 8$ </td><td> $2 . 9 1 \pm 0 . 1 0$ </td><td></td><td> $6 1 . 7 1 \pm 0 . 7 9$ </td><td> $5 . 7 2 \pm 0 . 0 6$ </td></tr><tr><td>MMR</td><td> $2 . 2 4 \pm 0 . 0 8$ </td><td> $2 . 9 6 \pm 0 . 1 2$ </td><td></td><td> $6 3 . 3 0 \pm 0 . 7 2$ </td><td> $3 . 6 8 \pm 0 . 0 8$ </td></tr><tr><td>is table decomposes each method&#x27;s total token budget into stage-wise shares, highlighting where token usage ncentrated within the pipeline. While most methods are dominated by result-processing costs, approaches with expli</td><td></td><td></td><td></td><td></td><td></td></tr></table>

This table decomposes each method’s total token budget into stage-wise shares, highlighting where token usage is concentrated within the pipeline. While most methods are dominated by result-processing costs, approaches with explicit LLM-based pre-retrieval pruning allocate a significant fraction of tokens to the pruning stage itself.

Table 12: Pruning-stage effectiveness (%; mean ± SE). Ratios denote the fraction of candidate items removed at each stage; token reduction denotes the corresponding decrease in context tokens.
<table><tr><td>Method</td><td>Q-Ratio</td><td></td><td></td><td>Q-Token Red. Branch Ratio Branch Token Red. Root Ratio Root Token Red.</td><td></td><td></td></tr><tr><td>Baseline</td><td></td><td>一</td><td></td><td></td><td></td><td></td></tr><tr><td colspan="7">One-stage methods: Post-Retrieval Pruning</td></tr><tr><td>Geo. Residual Novelty</td><td></td><td></td><td> $3 9 . 6 \pm 1 . 4$ </td><td> $4 0 . 6 \pm 1 . 4$ </td><td> $1 1 . 3 \pm { 1 . 0 }$ </td><td> $1 2 . 1 \pm 1 . 1$ </td></tr><tr><td>Centroid Drift</td><td></td><td></td><td> $3 7 . 5 \pm 1 . 0$ </td><td> $4 0 . 2 \pm 1 . 1$ </td><td> $4 . 7 \pm 0 . 5$ </td><td> $5 . 5 \pm 0 . 6$ </td></tr><tr><td>Submodular Coverage</td><td></td><td></td><td> $5 1 . 7 \pm 1 . 1$ </td><td> $5 4 . 1 \pm 1 . 1$ </td><td> $2 . 3 \pm 0 . 5$ </td><td> $2 . 5 \pm 0 . 6$ </td></tr><tr><td>DPP</td><td></td><td></td><td> $5 7 . 5 \pm 0 . 7$ </td><td> $5 8 . 3 \pm 0 . 8$ </td><td> $0 . 8 \pm 0 . 3$ </td><td> $1 . 0 \pm 0 . 3$ </td></tr><tr><td>LLM</td><td></td><td></td><td> $5 0 . 7 \pm 1 . 9$ </td><td> $4 9 . 7 \pm 1 . 9$ </td><td> $1 2 . 2 \pm 1 . 2$ </td><td> $1 2 . 9 \pm 1 . 2$ </td></tr><tr><td>Combined</td><td></td><td></td><td> $6 0 . 5 \pm 1 . 0$ </td><td> $6 1 . 4 \pm { 1 . 0 }$ </td><td> $0 . 2 \pm 0 . 1$ </td><td> $0 . 2 \pm 0 . 2$ </td></tr><tr><td>Hybrid</td><td></td><td></td><td> $5 9 . 9 \pm 1 . 1$ </td><td> $6 0 . 6 \pm 1 . 1$ </td><td> $0 . 4 \pm 0 . 2$ </td><td> $0 . 6 \pm 0 . 2$ </td></tr><tr><td>MMR</td><td></td><td></td><td> $6 3 . 9 \pm 0 . 8$ </td><td> $6 4 . 6 \pm 0 . 8$ </td><td> $0 . 0 \pm 0 . 0$ </td><td> $0 . 0 \pm 0 . 0$ </td></tr><tr><td colspan="7">One-stage methods: Pre-Synthesis Pruning</td></tr><tr><td>Geo. Residual Novelty</td><td></td><td></td><td></td><td></td><td> $7 5 . 8 \pm 0 . 4$ </td><td> $7 7 . 1 \pm 0 . 3$ </td></tr><tr><td>Centroid Drift</td><td></td><td></td><td></td><td></td><td> $7 5 . 9 \pm 0 . 3$ </td><td> $7 7 . 2 \pm 0 . 2$ </td></tr><tr><td>Submodular Coverage</td><td></td><td></td><td></td><td></td><td> $7 5 . 3 \pm 0 . 4$ </td><td> $7 7 . 5 \pm 0 . 3$ </td></tr><tr><td>DPP</td><td></td><td></td><td></td><td></td><td> $7 6 . 2 \pm 0 . 4$ </td><td> $7 7 . 5 \pm 0 . 2$ </td></tr><tr><td>LLM</td><td></td><td></td><td></td><td></td><td> $6 2 . 9 \pm 0 . 5$ </td><td> $6 1 . 8 \pm 0 . 5$ </td></tr><tr><td>Combined</td><td></td><td></td><td></td><td></td><td> $9 6 . 1 \pm 0 . 1$ </td><td> $9 6 . 0 \pm 0 . 1$ </td></tr><tr><td>Hybrid</td><td></td><td></td><td></td><td></td><td> $9 2 . 6 \pm 0 . 8$ </td><td> $9 2 . 5 \pm 0 . 8$ </td></tr><tr><td>MMR</td><td></td><td>一</td><td></td><td></td><td> $7 7 . 2 \pm 0 . 5$ </td><td> $7 6 . 2 \pm 0 . 5$ </td></tr><tr><td colspan="7">Two-stage methods: Post-Retrieval + Pre-Synthesis Pruning</td></tr><tr><td>Centroid Drift</td><td></td><td></td><td> $5 2 . 6 \pm 0 . 6$ </td><td> $5 4 . 4 \pm 0 . 7$ </td><td> $0 . 0 \pm 0 . 0$ </td><td> $0 . 0 \pm 0 . 0$ </td></tr><tr><td>Submodular Coverage</td><td></td><td></td><td> $5 1 . 6 \pm 1 . 1$ </td><td> $5 3 . 9 \pm 1 . 1$ </td><td> $1 6 . 3 \pm { 1 . 7 }$ </td><td> $1 7 . 0 \pm 1 . 9$ </td></tr><tr><td>CD + SC</td><td></td><td></td><td> $3 7 . 6 \pm 1 . 0$ </td><td> $4 0 . 2 \pm { 1 . 0 }$ </td><td> $2 0 . 3 \pm { 1 . 7 }$ </td><td> $2 1 . 0 \pm 1 . 7$ </td></tr><tr><td>LLM</td><td></td><td></td><td> $4 8 . 6 \pm 1 . 9$ </td><td> $4 8 . 0 \pm 1 . 9$ </td><td> $2 6 . 1 \pm 1 . 7$ </td><td> $2 4 . 7 \pm 1 . 8$ </td></tr><tr><td>Lexical MMR</td><td></td><td></td><td> $5 8 . 2 \pm 0 . 7$   $6 3 . 9 \pm 0 . 8$ </td><td> $5 7 . 7 \pm 0 . 9$ </td><td> $6 0 . 0 \pm 1 . 6$ </td><td> $5 7 . 9 \pm 1 . 7$ </td></tr><tr><td>Hybrid</td><td></td><td></td><td> $5 9 . 6 \pm 1 . 1$ </td><td> $6 4 . 6 \pm 0 . 8$ </td><td> $0 . 0 \pm 0 . 0$ </td><td> $0 . 0 \pm 0 . 0$ </td></tr><tr><td></td><td></td><td></td><td> $6 0 . 9 \pm 1 . 0$ </td><td> $6 0 . 4 \pm 1 . 1$ </td><td> $5 4 . 4 \pm 3 . 1$ </td><td> $5 3 . 8 \pm 3 . 1$ </td></tr><tr><td>Combined</td><td></td><td></td><td> $5 3 . 8 \pm 1 . 4$ </td><td> $6 1 . 6 \pm 1 . 0$   $5 6 . 0 \pm 1 . 4$ </td><td> $6 8 . 6 \pm 0 . 9$ </td><td> $6 7 . 1 \pm 1 . 1$ </td></tr><tr><td>SC+LLM CD+LLM</td><td></td><td></td><td></td><td></td><td> $2 3 . 7 \pm 1 . 8$ </td><td> $2 2 . 3 \pm 1 . 7$ </td></tr><tr><td></td><td></td><td></td><td> $4 0 . 0 \pm 1 . 6$ </td><td> $4 2 . 5 \pm 1 . 6$ </td><td> $2 5 . 9 \pm 0 . 6$ </td><td> $2 0 . 3 \pm 0 . 5$ </td></tr><tr><td colspan="7">Three-stage methods: Pre-Retrieval + Post-Retrieval + Pre-Synthesis Pruning</td></tr><tr><td>Geo. Residual Novelty</td><td> $2 5 . 0 \pm 0 . 0$ </td><td> $2 4 . 7 \pm 0 . 3$ </td><td> $3 6 . 0 \pm 1 . 3$ </td><td> $3 6 . 2 \pm { 1 . 3 }$ </td><td> $2 0 . 4 \pm 1 . 9$ </td><td> $2 1 . 2 \pm 2 . 0$ </td></tr><tr><td>DPP</td><td> $2 5 . 0 \pm 0 . 0$ </td><td> $2 4 . 2 \pm 0 . 2$ </td><td> $5 2 . 3 \pm 0 . 7$ </td><td> $5 2 . 4 \pm 0 . 8$ </td><td> $1 . 4 \pm 0 . 6$ </td><td> $1 . 4 \pm 0 . 6$ </td></tr><tr><td>Centroid Drift</td><td> $2 5 . 0 \pm 0 . 0$ </td><td> $2 4 . 8 \pm 0 . 3$ </td><td> $3 1 . 1 \pm 1 . 1$ </td><td> $3 2 . 5 \pm 1 . 1$ </td><td> $9 . 5 \pm 1 . 0$ </td><td> $9 . 6 \pm 1 . 1$ </td></tr><tr><td>Lexical + CD + SC</td><td> $2 7 . 8 \pm 1 . 0$ </td><td> $2 6 . 9 \pm 1 . 1$ </td><td> $2 9 . 1 \pm 1 . 0$ </td><td> $3 0 . 8 \pm { 1 . 0 }$ </td><td> $1 7 . 8 \pm { 1 . 5 }$ </td><td> $1 8 . 4 \pm { 1 . 5 }$ </td></tr><tr><td>Lexical</td><td> $2 7 . 5 \pm 1 . 0$ </td><td> $2 6 . 6 \pm 1 . 0$ </td><td> $5 4 . 5 \pm 0 . 6$ </td><td> $5 3 . 5 \pm 0 . 8$ </td><td> $5 9 . 0 \pm 1 . 6$ </td><td> $5 8 . 2 \pm { 1 . 8 }$ </td></tr><tr><td>MMR</td><td> $2 7 . 2 \pm 0 . 5$ </td><td> $2 1 . 2 \pm 0 . 4$ </td><td> $5 9 . 2 \pm 0 . 9$ </td><td> $5 8 . 8 \pm 0 . 9$ </td><td> $0 . 0 \pm 0 . 0$ </td><td> $0 . 0 \pm 0 . 0$ </td></tr><tr><td>LLM</td><td> $2 7 . 0 \pm 0 . 9$ </td><td> $2 1 . 3 \pm 0 . 8$ </td><td> $6 0 . 5 \pm 2 . 5$ </td><td> $5 9 . 2 \pm 2 . 5$ </td><td> $2 2 . 5 \pm 2 . 3$ </td><td> $2 1 . 1 \pm 2 . 3$ </td></tr><tr><td>Hybrid</td><td> $3 3 . 6 \pm 0 . 8$ </td><td> $2 6 . 8 \pm 0 . 7$ </td><td> $2 5 . 0 \pm 0 . 0$ </td><td> $2 4 . 0 \pm 0 . 2$ </td><td> $5 5 . 1 \pm 1 . 2$ </td><td> $5 5 . 0 \pm 1 . 2$ </td></tr><tr><td>Combined</td><td> $3 5 . 5 \pm 0 . 7$ </td><td> $2 8 . 5 \pm 0 . 6$ </td><td> $2 5 . 0 \pm 0 . 0$ </td><td> $2 4 . 0 \pm 0 . 2$ </td><td> $5 5 . 8 \pm 1 . 1$ </td><td> $5 5 . 7 \pm 1 . 2$ </td></tr></table>

This table decomposes pruning behavior across pre-retrieval, post-retrieval, and pre-synthesis stages, reporting both the fraction of items removed and the corresponding reduction in token load. This breakdown reveals where pruning occurs within the pipeline, distinguishing methods that act early at the pre-retrieval stage from those relying primarily on post-retrieval or pre-synthesis reduction.

![](images/f1a3ccc3f441b784aed1c66b3159aaa978323510ad07267712fc28637c81d8dd.jpg)  
Figure 3: Overall quality versus token usage in thousands (averaged over the 100 reports) for all pruning strategies reported in Table 5. Each point corresponds to one method configuration, with marker style and color indicating the pruning stage. The baseline is highlighted separately, and labels use shortened method names for readability.

Table 13: DeepResearch Bench overall quality across pruning criteria and stage placement. Scores are RACE-style report-quality metrics. Bold denotes the best pruned value in each stage column.
<table><tr><td>Criterion</td><td>Branch-only</td><td>Root-only</td><td>Two-stage</td><td>Three-stage</td></tr><tr><td>Baseline</td><td>0.4798</td><td>0.4798</td><td>0.4798</td><td>0.4798</td></tr><tr><td>DPP</td><td>0.4584</td><td>0.4774</td><td>0.4648</td><td>0.4591</td></tr><tr><td>Centroid Drift</td><td>0.4567</td><td>0.4759</td><td>0.4676</td><td>0.4590</td></tr><tr><td>GRN</td><td>0.4688</td><td>0.4794</td><td>0.4716</td><td>0.4612</td></tr><tr><td>MMR</td><td>0.4541</td><td>0.4766</td><td>0.4605</td><td>0.4523</td></tr><tr><td>Submodular</td><td>0.4557</td><td>0.4811</td><td>0.4561</td><td>0.4534</td></tr></table>

Table 14: DeepResearch Bench efficiency across pruning criteria and stage placement. Tokens are reported in thousands (k), and savings are relative to the unpruned baseline. Bold denotes the best pruned value in each stage column.
<table><tr><td></td><td colspan="3">Branch-only</td><td colspan="3">Root-only</td><td colspan="3">Two-stage</td><td colspan="3">Three-stage</td></tr><tr><td>Criterion</td><td>Tokens</td><td>Tok. Sav.</td><td>Run. Sav.</td><td>Tokens</td><td>Tok. Sav.</td><td>Run. Sav.</td><td>Tokens</td><td>Tok. Sav.</td><td>Run. Sav.</td><td>Tokens</td><td>Tok. Sav.</td><td>Run. Sav.</td></tr><tr><td>DPP</td><td>89.42</td><td>72.65%</td><td>81.14%</td><td>281.98</td><td>13.77%</td><td>32.54%</td><td>89.81</td><td>72.53%</td><td>80.95%</td><td>25.60</td><td>92.17%</td><td>93.60%</td></tr><tr><td>Centroid Drift</td><td>95.50</td><td>70.79%</td><td>78.01%</td><td>300.54</td><td>8.09%</td><td>28.62%</td><td>92.62</td><td>71.68%</td><td>80.32%</td><td>81.41</td><td>75.10%</td><td>79.93%</td></tr><tr><td>GRN</td><td>110.72</td><td>66.14%</td><td>74.77%</td><td>286.25</td><td>12.46%</td><td>28.75%</td><td>90.57</td><td>72.30%</td><td>82.89%</td><td>27.27</td><td>91.66%</td><td>93.44%</td></tr><tr><td>MMR</td><td>86.05</td><td>73.68%</td><td>81.94%</td><td>301.83</td><td>7.69%</td><td>28.49%</td><td>89.03</td><td>72.77%</td><td>81.76%</td><td>25.13</td><td>92.32%</td><td>94.30%</td></tr><tr><td>Submodular</td><td>96.62</td><td>70.45%</td><td>79.77%</td><td>312.15</td><td>4.54%</td><td>24.78%</td><td>97.68</td><td>70.13%</td><td>67.35%</td><td>84.03</td><td>74.30%</td><td>82.92%</td></tr></table>

Table 15: Local threshold sweeps for representative post-retrieval pruning methods. $\Delta Q _ { \mathrm { r e l } }$ is the percent change in overall quality relative to the published threshold setting for that method. Rows satisfying the 2% stability criterion are marked with $\checkmark .$ . Tokens are reported in thousands (k). Values are mean scores over a 10-query sensitivity subset.
<table><tr><td>Method</td><td>Threshold</td><td># Tokens</td><td>Runtime (s)</td><td>Overall</td><td> $\Delta Q _ { \mathrm { r e l } }$ </td><td>Stable?</td></tr><tr><td colspan="7">MMR (fixed λ = 0.35, published  $\tau _ { \mathrm { p u b } } = 0 . 3 5 )$ </td></tr><tr><td></td><td>0.10</td><td>121.3</td><td>1239.0</td><td>58.33</td><td>+3.24%</td><td>√</td></tr><tr><td></td><td>0.20</td><td>121.3</td><td>1238.9</td><td>57.50</td><td>+1.77%</td><td>√</td></tr><tr><td></td><td>0.30</td><td>121.3</td><td>1239.0</td><td>56.83</td><td>+0.58%</td><td> $\checkmark$ </td></tr><tr><td></td><td>0.35</td><td>121.3</td><td>1239.1</td><td>56.50</td><td>0.00%</td><td> $\checkmark$ </td></tr><tr><td></td><td>0.40</td><td>121.3</td><td>1239.3</td><td>56.50</td><td>0.00%</td><td> $\checkmark$ </td></tr><tr><td colspan="7">GRN (published  $\tau _ { \mathrm { p u b } } = 0 . 8 5 )$ </td></tr><tr><td>0.65</td><td></td><td>307.3</td><td>3034.3</td><td>61.30</td><td>+0.21%</td><td>√</td></tr><tr><td></td><td>0.75</td><td>268.7</td><td>2616.3</td><td>61.17</td><td>0.00%</td><td>√</td></tr><tr><td></td><td>0.80</td><td>231.4</td><td>2300.3</td><td>58.17</td><td>-4.90%</td><td></td></tr><tr><td></td><td>0.85</td><td>171.9</td><td>1756.2</td><td>61.17</td><td>0.00%</td><td> $\checkmark$ </td></tr><tr><td></td><td>0.90</td><td>138.3</td><td>1430.1</td><td>59.83</td><td>-2.19%</td><td></td></tr><tr><td colspan="7">Centroid Drift (published  $\tau _ { \mathrm { p u b } } = 0 . 0 3 )$ </td></tr><tr><td>0.005</td><td></td><td>227.0</td><td>2336.6</td><td>61.50</td><td>+0.28%</td><td>√</td></tr><tr><td></td><td>0.01</td><td>202.9</td><td>2068.9</td><td>60.50</td><td>-1.35%</td><td>√</td></tr><tr><td></td><td>0.03</td><td>137.4</td><td>1438.2</td><td>61.33</td><td>0.00%</td><td>√</td></tr><tr><td></td><td>0.05</td><td>126.1</td><td>1341.5</td><td>58.67</td><td>-4.34%</td><td></td></tr><tr><td></td><td>0.085</td><td>120.4</td><td>1237.7</td><td>58.83</td><td>-4.08%</td><td></td></tr><tr><td colspan="7">DPP (published  $\tau _ { \mathrm { p u b } } = 0 . 3 0 )$ </td></tr><tr><td></td><td>0.05</td><td>391.4</td><td>3723.4</td><td>61.33</td><td>+0.82%</td><td>√</td></tr><tr><td></td><td>0.10</td><td>363.1</td><td>3465.4</td><td>60.83</td><td>0.00%</td><td>√</td></tr><tr><td></td><td>0.20</td><td>216.0</td><td>2223.6</td><td>59.83</td><td>-1.64%</td><td>√</td></tr><tr><td></td><td>0.30</td><td>134.1</td><td>1422.9</td><td>60.83</td><td>0.00%</td><td>√</td></tr><tr><td></td><td>0.40</td><td>121.3</td><td>1237.6</td><td>57.33</td><td>-5.75%</td><td></td></tr><tr><td colspan="7">Submodular Coverage (published  $\tau _ { \mathrm { p u b } } = 0 . 0 5 )$ </td></tr><tr><td>0.01</td><td></td><td>383.1</td><td>3652.0</td><td>59.83</td><td>-0.83%</td><td>√</td></tr><tr><td></td><td>0.03</td><td>232.7</td><td>2303.3</td><td>59.83</td><td>-0.83%</td><td>√</td></tr><tr><td></td><td>0.05</td><td>143.7</td><td>1516.2</td><td>60.33</td><td>0.00%</td><td>√</td></tr><tr><td></td><td>0.08</td><td>126.3</td><td>1302.8</td><td>59.17</td><td>-1.92%</td><td>√</td></tr><tr><td></td><td>0.10</td><td>123.7</td><td>1261.1</td><td>57.67</td><td>-4.41%</td><td></td></tr></table>