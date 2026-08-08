# When Memory Lies: An Empirical Study of Spatial Memory Staleness in VLM Agents

Yushi Sun<sup>1∗</sup>, Yanjie Zhang<sup>2∗†</sup>

<sup>1</sup>LIGHTSPEED, Shenzhen, China

<sup>2</sup>The Hong Kong University of Science and Technology, Hong Kong, China ysunbp@connect.ust.hk, yzhangvj@connect.ust.hk

## Abstract

Memory-augmented VLM agents act on persistent spatial knowledge, yet that knowledge silently goes stale as the environment changes. We ask what happens when an agent must reconcile a confident memory claim with a contradicting observation, and whether current models can catch the conflict before it becomes a safety-relevant mistake. Using a dynamic FrozenLake testbed, we pair a staleness-detection task with a downstream navigation task across three closed-source models and three open-weight VLMs under both text and image inputs (1,800 detection runs, and 12,000 text-mode navigation episodes over four LLM navigators at a shared 50-seed scale). Three findings emerge. First, text solvability does not imply visual grounding: models that flag stale entries reliably from text nonetheless span vision F1 from 0.887 down to 0.067 on the identical grids, and the weakest keeps making fluent, confident decisions that ignore the image. Second, consuming stale memory without an audit is a safety liability: in our primary GPT-4o setting, an agent that trusts raw memory dies more than twice as often as the same agent given no memory at all. Third, auditing helps but does not close the gap: a transparent read-time filter removes much of the safety cost in text mode, yet even oracle stale labels bring no further significant gain on the current grid size, and when visual auditing is unreliable, filtering yields no consistent benefit. Together these results frame spatial-memory staleness as a safety failure mode and isolate reliable visual grounding and action selection under memory–observation conflict as the central open challenges for memory-augmented agents.

## 1 Introduction

Consider a VLM agent navigating a dynamic environment with a map built from prior experience. Its memory records that position (5, 6) is safe frozen ice, but the environment has since changed: that cell is now a lethal hole. The current observation shows the hole, yet the agent still holds the earlier, confident memory that the cell is safe. Which source does it act on? The answer carries a real cost: in our primary GPT-4o setting, an agent that trusts raw stale memory dies more than twice as often as the same agent given no memory at all. Figure 1 illustrates this failure mechanism and the read-time fix we study.

Persistent memory is a central design pattern for capable agents. Voyager (Wang et al. 2023) accumulates a skill library across Minecraft episodes; GITM (Zhu et al. 2023) maintains hierarchical trajectory memory; Reflexion (Shinn et al. 2023) stores feedback from prior failures; and MemGPT (Packer et al. 2023) pages structured memory in and out of context. Their evaluations principally study memory construction, retrieval, and reuse, rather than whether a stored spatial claim can silently become invalid before an agent acts on it. Adjacent work studies the temporal validity of textual factual knowledge (Mousavi, Alghisi, and Riccardi 2024; Chao et al. 2026; Vu et al. 2023) or edits outdated associations in model weights (Meng et al. 2022, 2023). Spatial-reasoning and hallucination benchmarks, meanwhile, measure VLM behavior on static observations (Wang et al. 2024; Chen et al. 2024; Cheng et al. 2024; Wu et al. 2024; Li et al. 2023; Guan et al. 2024). The missing intersection is a changed spatial memory that must be reconciled with the current observation before a safety-relevant action.

![](images/7fa375e4aa2ad16a191b605c832ac37eff65436a4a20f922e560c12e1e089d76.jpg)  
Figure 1: Schematic of the failure mechanism. Top: a per-step action policy, reasoning over stale memory, marches through (5, 6) after it has silently become a hole. Bottom: filtering the contradictory entry from the memory view enables a safe detour. The summary rates are from the GPT-4o navigation experiment over 50 L2 seeds, where OMCD reduces death from 74.4% to 31.6%.

We study this conflict in SpatialSTALE, a dynamic FrozenLake testbed. The agent is given a persistent memory that holds one textual claim per grid cell, each tagging the cell as safe or dangerous $\begin{array} { r l } { \mathrm { e . g . } , \ \mathrm { S A F E } } & { { } \mathrm { a t } \quad ( 2 , 5 ) : } \end{array}$ Frozen ice, safe to walk), while nonterminal cells can flip between safe frozen ice and lethal holes before or during navigation. The memory stays textual throughout; what varies is the current observation of the changed grid, which we present either as a coordinate-labeled text listing or as a rendered image, giving matched text and vision conditions. On paired instances, we first ask a model to judge whether each memory entry conflicts with the current observation, then measure goal-directed navigation using raw, filtered, or no memory. This design lets us ask five linked questions: Can current VLMs identify stale spatial claims? Does this ability transfer from text to vision? What safety cost follows from consuming raw stale memory? Can an explicit readtime audit remove that cost? Once an audit is reliable, are the residual failures due to stale-label accuracy or to action selection over filtered memory? We evaluate three closed-source models (GPT-4o, Claude-Sonnet-4.6, and Qwen3.6-Plus) and three open-weight VLMs (GLM-5.1, InternVL3-2B, and InternVL3-8B), with 50 shared detection seeds across models, modalities, and change regimes. For navigation, we compare NoMemory, NoFilter, SelfVerify, and OMCD, a transparent controlled read-time filtering intervention.

Our empirical results yield three core findings. First, text solvability does not ensure visual grounding. Capable models detect stale entries reliably from text, but the same grids produce vision F1 from 0.887 for Qwen to 0.067 for GLM; in the worst case, a model keeps issuing fluent stale decisions with little detectable sensitivity to image contradiction. Second, stale memory is a safety liability when consumed without an audit. On primary GPT-4o L2 text navigation, trusting raw stale memory raises death from 28.0% (no memory) to 74.4%, a gap that reflects direct “rushinto-hole” failures; a controlled filter removes much of this penalty in text mode. Third, reliable filtering does not fully resolve the problem. Oracle stale labels yield no significant gain over learned text filtering in the current 8×8 setting, and detection F1 does not explain residual navigation variation, localizing the remaining bottleneck to action selection over filtered memory. Conversely, when visual auditing is unreliable, filtering provides no consistent downstream benefit.

Our contributions are as follows:

• A paired empirical study of spatial-memory staleness that links entry-level text/vision auditing to safety-sensitive navigation across three closed-source and three openweight VLMs.

• An empirical characterization of modality-dependent auditing and the stale-memory safety tax, including outcome-conditioned trajectories that separate direct stale-memory failures from memoryless exploration failures.

• OMCD, a transparent controlled filtering intervention that identifies both when stale-memory harm is remediable and where filtering ceases to help: unreliable visual auditing upstream and action selection over reliable filtered memory downstream.

We release the code, all 50 seed map sets, full model traces, and reproducibility pipelines to support follow-up work.

## 2 Related Work

Memory agents and knowledge staleness. Persistentmemory agents such as Voyager, GITM, Reflexion, and MemGPT study how agents accumulate, retrieve, reflect on, or reuse past experience (Wang et al. 2023; Zhu et al. 2023; Shinn et al. 2023; Packer et al. 2023). Complementary work studies the temporal validity of factual knowledge or updates outdated associations through benchmarks and weight editing (Mousavi, Alghisi, and Riccardi 2024; Chao et al. 2026; Vu et al. 2023; Meng et al. 2022, 2023). Neither line evaluates a previously acquired spatial claim that becomes invalid and subsequently changes a safety-sensitive action.

Spatial reasoning and hallucination in VLMs. Spatial-Eval, SpatialVLM, SpatialRGPT, VSR, What’s Up, Spatial-Bot, VSP, and BLINK evaluate spatial perception, reasoning, planning, or low-level visual capability on fixed observations (Wang et al. 2024; Chen et al. 2024; Cheng et al. 2024; Liu, Emerson, and Collier 2023; Kamath, Hessel, and Chang 2023; Cai et al. 2024; Wu et al. 2024; Fu et al. 2024); POPE and HallusionBench probe hallucination under static visual inputs (Li et al. 2023; Guan et al. 2024). In contrast, we test whether a VLM can reconcile a stored spatial description of the past with a changed current text or image observation before acting.

Reflection, embodied planning, and change detection. Reflection methods revise model outputs from feedback (Shinn et al. 2023; Madaan et al. 2023); embodied planners combine language reasoning with actions or grounded feedback (Ahn et al. 2022; Song et al. 2023; Huang et al. 2022; Yao et al. 2023); and classical change detection and navigation benchmarks study image diferences or unknown environments (Hussain et al. 2013; Chevalier-Boisvert, Willems, and Pal 2018; Savva et al. 2019; Gu et al. 2022). These lines do not close the causal loop studied here: entry-level auditing of stale spatial memory, followed by its use in downstream navigation safety under environmental change.

## 3 Study Design

## 3.1 Experimental Paradigm and Overview

To study whether a language agent’s stored spatial memory can silently harm its actions, we need an environment in which memory, world, and outcome are all controllable and independently observable. Persistent natural language memory is by now a standard component of language agents (Wang et al. 2023; Zhu et al. 2023; Packer et al. 2023), so we adopt its per-entry claim format directly and treat each entry as a testable proposition about a location. To make such propositions falsifiable under change, we borrow the binary valid/stale annotation used in staleness benchmarks for stored knowledge (Mousavi, Alghisi, and Riccardi 2024;

Chao et al. 2026; Vu et al. 2023) and score entries with the balanced binary probing style of Li et al. (2023), so that recognition can be measured independently of downstream behavior. To then measure how recognition (or its failure) translates into action, we need a world where a single memory entry can be tied to a single, safety-relevant outcome; minimalist gridworlds are the established tool for isolating one such mechanism at a time (Chevalier-Boisvert, Willems, and Pal 2018; Chevalier-Boisvert et al. 2019; Leike et al. 2017), and FrozenLake, a standard Gymnasium benchmark (Towers et al. 2024), provides exactly the absorbing safe/unsafe structure that safe reinforcement learning uses to make unsafe actions observable as terminal outcomes. Combining these three ingredients yields SpatialSTALE: a fixed natural language memory, a controllably changed grid world, and a navigation task whose deaths and successes trace back to specific memory entries.

Each instance proceeds in four stages: (1) generate an original $8 \times 8$ grid ${ \bf g } _ { 0 }$ and its memory snapshot; (2) change selected frozen (F) and hole (H) cells before or during navigation; (3) ask the model to audit the memories against a text or image observation; and (4) navigate using raw, filtered, or no memory. Detection measures whether the model recognizes an invalid belief; navigation measures whether recognition changes behavior. Figure 1 illustrates this causal chain.

## 3.2 Experimental Procedure

Constructing the memory snapshot. The original grid contains approximately 25% holes, fixed start and goal cells, and at least one valid path. Before any change, we run 20 random walks on $\mathbf { g } _ { 0 } ,$ , each capped at 50 steps; the first visit to a position records its type. The benchmark generator then reads ${ \bf g } _ { 0 }$ and adds entries only for unvisited positions. This controlled completion is not a model prediction: it fixes coverage at one entry per cell $( N = { \bar { 6 } } 4 )$ Each entry exposes a natural language claim to the model, such as SAFE at (2,5): Frozen ice, safe to walk, while retaining its structured type only for evaluation.

Changing the world and auditing memory. After the snapshot is fixed, the environment flips nonterminal F and H cells while preserving a route to the goal. Given the unchanged memory and a current grid observation, the model returns a binary valid or stale judgment per entry. This follows binary probing in hallucination evaluation (Li et al. 2023), but compares past and present representations at the same coordinate.

Measuring consequences. The changed grid is then used for navigation. Stepping on H yields death and reward $- 1 ;$ reaching G yields +1; otherwise the episode times out. We record success, death, path length, and reward. Pairing both tasks on the same instances separates recognition from safe action after recognition.

## 3.3 Formalizing Staleness

Let $\mathcal { C } = \{ \mathrm { F } , \mathrm { H } , \mathrm { S } , \mathrm { G } \}$ and let entry $m _ { i }$ store position $p _ { i }$ and original structured type $c _ { i } ^ { 0 } \in \mathcal { C }$ . Define the hazard map

<table><tr><td>Regime</td><td>Pattern</td><td>Requested</td><td> $\mathrm { S t a l e \% }$ </td><td>Detection</td></tr><tr><td>L1</td><td>Random spot</td><td>5 to 7</td><td> $9 . 4 \pm 1 . 1$ </td><td>One shot</td></tr><tr><td>L2</td><td>Local clusters</td><td>12 to 16</td><td> $1 4 . 1 { \pm } 1 . 4 $ </td><td>One shot</td></tr><tr><td>L3</td><td>Online dynamic</td><td>2/event</td><td> $1 4 . 3 { \pm } 2 . 8 $ </td><td>Incremental</td></tr></table>

Table 1: Change regimes. Requested flips precede solvability checks. Stale% is measured over the 50 shared seeds; L3 uses the terminal detection window.

![](images/c6a5c8e4d5eb2df3ceaf5897786e1a95257f682f96f340a359a68bc36dcc4cde.jpg)  
Figure 2: Illustrative regimes. L1 and L2 are observed once after pregame changes; L3 is observed again after each realized online event.

h(H) = 1 and $h ( \mathrm { F } ) = h ( \mathrm { S } ) = h ( \mathrm { G } ) = 0$ . The evaluator assigns

$$
y _ { i } ( t ) = \mathbb { I } \big [ h ( c _ { i } ^ { 0 } ) \neq h ( \mathbf { g } _ { t } [ p _ { i } ] ) \big ] , \qquad \hat { y } _ { i } ( t ) \in \{ 0 , 1 \} ,\tag{1}
$$

where 1 means stale. Annotation therefore requires no subjective semantic matching: SAFE is stale exactly when the current cell is H, and DANGER is stale exactly when it is F. Start and goal never change; a rendered player marker P is treated as safe, while scoring uses its underlying cell. Since only F and H change, Equation 1 is equivalent to comparing stored and current types at $p _ { i }$ . We report entry level Precision, Recall, and F1.

## 3.4 Controlled Change Regimes

The three regimes vary when changes occur, how many are requested, and whether they are spatially localized (Table 1; examples in Figure 2).

L1 samples changes across the grid; L2 samples from two or three radius 2 neighborhoods; both run detection once after accepted pregame changes and before navigation. L3 begins unchanged, makes every fifth step eligible for an event (30% skip probability), and runs detection at the next decision point after each realized event. Its confusion matrix counts are pooled across detection windows. The stale ratio is $| \{ p : \mathbf { g } _ { t } { \dot { [ p ] } } \neq \mathbf { g } _ { 0 } [ p ] \} | / 6 4 $ ; reported values are means and standard deviations computed from the generated instances before any model call. Full sampling and rejection rules appear in Appendix A.

## 3.5 Inputs, Outcomes, and Comparisons

In text mode, the current full grid is a coordinate labeled list. In vision mode, the same full grid is a 384×384 color coded rendering with coordinate labels. Memory remains textual in both modes. Full prompt and rendering details are given in Appendix B. Fifty deterministic seeds generated from master seed 2024 are shared across models, modalities, and strategies, enabling paired Wilcoxon signed rank tests and cross level correlations between detection F1 and navigation success. The four navigation strategies compared on this shared seed set, and the pairwise contrasts they support, are defined in Section 4.

## 4 Memory Strategies and the OMCD Intervention

A navigator can receive two incompatible claims about one cell: a persistent memory may say that (5, 4) is safe, while the current observation shows a hole. We compare four strategies that difer only in how the memory store is presented to the navigator. All four share the same current full-grid observation, action space, and step budget. Three are baselines that isolate the efect of having memory, trusting stale memory, and applying a one-shot self-check; the fourth, OMCD, is our controlled filtering intervention.

## 4.1 Memory Strategies

NoMemory (memoryless control). The memory portion of the navigation prompt is empty. The model chooses each action from the current grid, its position, the goal, and the remaining step budget. NoMemory establishes how the same navigator behaves without a persistent belief store, and is the reference against which the efect of adding memory is read.

NoFilter (unfiltered baseline). All 64 original memory entries are passed to the navigator unchanged after the world changes, without a consistency check against the current observation. NoFilter measures what happens when a navigator consumes a persistent store that may contradict the visible grid. Its paired contrast with NoMemory directly measures the safety cost, if any, of trusting raw stale memory rather than acting from the current observation alone.

SelfVerify (one-shot self-check). Before navigation, the model receives the complete memory store and current observation in one query, then removes entries it judges inconsistent. It performs no further audit during the episode. This is the simplest read-time filter a memory-augmented navigator can apply. Compared with NoFilter, it tests whether one fullstore self-check mitigates the raw-memory cost; compared with NoMemory, it tests whether retaining a self-verified store is preferable to discarding memory entirely.

OMCD (batched, event-aware audit). OMCD also filters memory before navigation, but it audits entries in batches and, under L3, repeats the audit after realized environmental changes. Its comparison with NoFilter tests whether an explicit entry-level consistency gate removes the raw-memory tax. Its comparison with SelfVerify tests whether batched, event-triggered auditing improves on a one-shot full-store check. Comparing OMCD with NoMemory tests whether a filtered memory store provides value beyond the current observation. The next subsection specifies the procedure.

Oracle labels (label-quality ablation). Oracle supplies ground-truth stale labels through the same filtering interface as OMCD. The Oracle–OMCD contrast tests whether residual navigation error at the current scale is still limited by stale-label quality. Together, these conditions identify the safety cost of raw memory, the value of filtering it, and the point at which better labels cease to help, without claiming OMCD is universally preferable to memoryless reasoning.

## 4.2 OMCD Filtering Procedure

OMCD prevents detector-flagged conflicts from reaching the action prompt. It produces a filtered memory view, rather than requiring the navigator to arbitrate at every step between a confident past claim and the current observation.

Batched binary audit. OMCD partitions the 64 memory entries into batches of B = 10. For each batch, the model receives the current observation (a text grid or rendered image) and the natural-language memory entries, then returns a binary is\_stale judgment and a brief reason for each entry. Predicted-stale entries are removed; all other entries remain available to the navigator. We use binary judgments because continuous scores were poorly calibrated across thresholds (Appendix D). A sweep over $B \in \{ 1 , 5 , 1 0 , 2 0 \}$ finds that B = 10 reduces inference calls by 9× relative to per-entry querying at a 2.7 pp F1 cost (Appendix E).

Event-triggered re-auditing. For L1 and L2, OMCD audits once after pregame changes and before navigation. For L3, it stores the grid used for the preceding audit and audits again at the next decision point after every realized event. The active memory view is rebuilt from the latest judgments, so no audit occurs on steps where the world has not changed.

Algorithm 1 OMCD filtering intervention   
Require: Observation $o _ { t } ,$ memory store M, batch size B   
1: $M _ { \mathrm { a c t i v e } }  { \mathcal { M } }$   
2: for each batch $B \subset { \mathcal { M } }$ of size B do   
3: $\{ ( m , \hat { y } _ { m } ,$ reason $\mathsf { \Pi } _ { \mathsf { l m } } ) \} \gets \mathrm { L L M } ( o _ { t } , B )$   
4: for each $m \in B$ with $\hat { y } _ { m } =$ stale do   
5: $M _ { \mathrm { a c t i v e } }  M _ { \mathrm { a c t i v e } } \backslash \{ m \}$   
6: end for   
7: end for   
8: return $M _ { \mathrm { a c t i v e } }$

## 5 Results

## 5.1 Setup

Models. We evaluate three closed source models (GPT-4o, Claude-Sonnet-4.6, Qwen3.6-Plus) and three open weight VLMs (GLM-5.1, InternVL3-2B, InternVL3-8B), all at recommended temperatures.

Scale. Detection covers 6 models × 50 seeds × 3 regimes × 2 modalities, totaling 1,800 runs. Text-mode navigation is run at the same scale for each of the four LLM navigators reported in Table 3 (GPT-4o, Claude, Qwen, GLM): 50 shared seeds × 3 regimes × 4 strategies × 5 episodes per model, so every model–strategy–regime cell aggregates 250 episodes over the same seed set, for 12,000 text-mode episodes in total. We use paired Wilcoxon signed rank tests within shared seeds, at $\alpha = 0 . 0 0 1$ . The vision-navigation results in Table 2 are a smaller exploratory preview (10 seeds × 3 episodes).

![](images/e9bd9ea0ffb425b39b2120ac4c624fd973f4ae249e0a51f5b2973e9a6063eba1.jpg)  
Figure 3: Detection F1 across models, regimes, and modalities $( \mathrm { m e a n } \pm \mathrm { \ s t d }$ over $N = 5 0$ shared seeds). Rows are grouped by source: closed source above, open weight below.

Metric aggregation. Stale entries are the positive class throughout. For L1 and L2, each seed produces a single detection window, and Precision, Recall, and F1 are computed on that window’s confusion matrix over the 64 memory entries. For L3, a seed produces multiple detection windows as online events unfold; we sum the true positive, false positive, and false negative counts across a seed’s windows before computing per seed F1. Values reported in the main tables are the mean and standard deviation of these per seed F1 scores across the 50 shared seeds. We use F1 rather than balanced accuracy because the practical concern is recovering the rare stale entries that drive downstream deaths.

## 5.2 A Text-Solvable Audit Does Not Necessarily Transfer to Vision

Text staleness detection is near ceiling for capable models. In text mode, where observation and memory are both linguistic, Figure 3 shows that the three closed source models and open weight GLM-5.1 all achieve average text F1 above 0.88, though all degrade on L3 as changes arrive incrementally. The two smaller InternVL3 checkpoints behave diferently: the 2B checkpoint averages about 0.22 F1 in text and the 8B about 0.31. InternVL3-8B exceeds 2B in both modalities across every regime, so source availability alone does not explain detection quality; model family and capacity matter.

Visual transfer is strongly model-dependent. The same grids reveal a sharply diferent picture under vision. Qwen preserves text-level accuracy $( \bar { \Delta } \mathrm { F 1 } ~ = ~ - 0 . 0 1 1$ , not significant), Claude degrades moderately $( \Delta \mathrm { F 1 } ~ = ~ - 0 . 1 3 4$ $\textit { p } < 1 \mathrm { e } { - } 7 )$ , GPT-4o degrades severely $( \Delta \mathrm { F 1 } ~ = ~ - 0 . 5 9 8$ $p < 1 { \mathrm { e } } { \cdot } 9 ;$ failure profile in Appendix I), and GLM-5.1 collapses $( \Delta \mathsf { F } 1 = - 0 . 8 3 0 , p < \mathsf { l e } { - } 9 )$ . The best and worst vision models difer by a factor of thirteen (Qwen 0.887 versus GLM 0.067). Crucially, Qwen retains near-text accuracy on the same $3 8 4 \times 3 8 4$ renderings, which rules out unreadable input as the cause: the collapse of the others is a processing failure, not a rendering artifact, and the per-class analysis below traces it to the audit stage rather than low-level recognition. We thus attribute these failures to visual auditing under this rendering format, one axis a broader study should vary. Vision errors also change their safety profile and error balance. The aggregate F1 gap masks two finer-grained patterns. First, the precision–recall diagnostic in Appendix H shows a regime shift: snapshot settings L1/L2 generally overflag $( P \ < \ R )$ , while viable text detectors on L3 instead under-flag $( P > R )$ . Incremental change therefore shifts the dominant error from false alarms to missed stale entries. Second, change direction is safety-relevant: a thaw (F→H) turns a claimed-safe cell deadly, while a freeze (H→F) only makes a claimed-danger cell safe. In an exploratory per-class breakdown, Claude vision recalls 94.5% of deadly thaw entries $\left( N { = } 4 3 6 \right)$ but only 63.6% of freeze entries (N=121) on L2, while Qwen is near ceiling on both. This asymmetry is safety-aligned but model-specific, not a general VLM property.

![](images/2d9806b4e7e35c0e38d85cbc132451e9298ce33f8ef071ef0ecf7817882b2d54.jpg)

![](images/02f4cfa10a58ffd37ce3a388def03043a3fe5da431f13ba6f720b9f63101c45c.jpg)  
Figure 4: Rate at which each model flags a memory entry as stale under vision input, split by memory claim and ground truth. On the same 6,400 L1+L2 judgments per model, Claude flags deadly-stale entries at 92.5% versus 1.4% for correct safe entries, whereas GLM flags deadly-stale and correct safe entries at nearly the same rate (4.5% versus 5.0%).

GLM’s visual failure is consistent with memorydominated auditing. GLM-5.1 illustrates why this modality boundary is operationally important. Across 6,400 L1/L2 vision judgments, its reasoning strings contain no refusals and no lexical uncertainty markers such as “cannot,” “uncertain,” or “not sure,” yet often assert cell states absent from the rendered image (Appendix J). Figure 4 identifies the pattern behind those strings: GLM flags a memory-safe entry as stale at nearly the same rate whether the cell is truly safe or a lethal hole (5.0% versus $4 . 5 \% ,$ not significant), whereas Claude separates the two sharply. This constancy is itself a built-in perceptual control: unreadable coordinates or colors would make the flag rate swing with image content, but a rate that stays flat across opposite ground truths instead reflects a memory-dominated rule that is largely insensitive to the image, not a recognition deficit. Fluent reasoning is thus no evidence that the visual audit was grounded, and this isolates the failure to the audit stage without a separate recognition benchmark.

When the visual audit fails, filtering has no consistent downstream benefit. Keeping the rest of the pipeline fixed, we evaluate 10 seeds × 3 episodes per cell in Table 2. GPT-4o, Claude, and GLM reach at most 0.40 success and OMCD provides no consistent improvement, while Qwen retains high vision detection and success but leaves little headroom for filtering. This small-sample extension is exploratory, but supports the same boundary condition: a filtering policy cannot reliably improve navigation when its visual audit supplies corrupted labels.

<table><tr><td>Model</td><td>Reg</td><td>NoMem</td><td>NoFilt</td><td>SelfV</td><td>OMCD</td></tr><tr><td>Closed source</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>L1</td><td>.067</td><td>.200</td><td>.167</td><td>.133</td></tr><tr><td>GPT-40</td><td>L2</td><td>.033</td><td>.100</td><td>.133</td><td>.200</td></tr><tr><td></td><td>L3</td><td>.033</td><td>.100</td><td>.100</td><td>.133</td></tr><tr><td></td><td>L1</td><td>.100</td><td>.300</td><td>.400</td><td>.167</td></tr><tr><td>Claude-Sonnet-4.6</td><td>L2</td><td>.167</td><td>.233</td><td>.300</td><td>.367</td></tr><tr><td></td><td>L3</td><td>.300</td><td>.133</td><td>.433</td><td>.300</td></tr><tr><td></td><td>L1</td><td>.967</td><td>.867</td><td>.967</td><td>.967</td></tr><tr><td>Qwen3.6</td><td>L2</td><td>.833</td><td>.767</td><td>.833</td><td>.833</td></tr><tr><td></td><td>L3</td><td>.933</td><td>.933</td><td>.933</td><td>.967</td></tr><tr><td>Open weight</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>L1</td><td>.000</td><td>.233</td><td>.200</td><td>.167</td></tr><tr><td>GLM-5.1</td><td>L2</td><td>.000</td><td>.067</td><td>.100</td><td>.133</td></tr><tr><td></td><td>L3</td><td>.000</td><td>.233</td><td>.133</td><td>.167</td></tr><tr><td></td><td>L1</td><td>.000</td><td>.000</td><td>.000</td><td>.000</td></tr><tr><td>InternVL3-2B</td><td>L2</td><td>.000</td><td>.000</td><td>.000</td><td>.000</td></tr><tr><td></td><td>L3</td><td>.000</td><td>.000</td><td>.000</td><td>.000</td></tr><tr><td></td><td>L1</td><td>.000</td><td>.000</td><td>.000</td><td>.000</td></tr><tr><td>InternVL3-8B</td><td>L2</td><td>.000</td><td>.000</td><td>.000</td><td>.000</td></tr><tr><td></td><td>L3</td><td>.000</td><td>.033</td><td>.000</td><td>.000</td></tr></table>

Table 2: Vision navigation success (10 seeds × 3 episodes). Bold: best nonzero strategy within each model and regime; all zero ties are unbolded.

## 5.3 Unfiltered Stale Memory is a Safety Liability, but an Explicit Audit Removes Most of the Tax

Trusting stale memory can be substantially less safe than having no memory. The headline comparison in Table 3 is counterintuitive: on primary L2 GPT-4o, blindly trusting stale memory achieves 14.4% success and 74.4% death, while discarding memory entirely achieves 28.8% success and 28.0% death. In this setting, stale memory is 2.7 times deadlier than no memory. The gap is not a peculiarity of a single regime: on GPT-4o, OMCD improves on both memoryusing baselines (NoFilter and SelfVerify) in every regime, with success gains over NoFilter that grow from 14.4 pp on L1 to 24.4 pp on L3 (all $p < 0 . 0 0 1 )$ ). The other three models, run at the same scale, show the same efect, with average death-rate reductions relative to NoFilter of 38% (Claude), 45% (GLM), and 43% (Qwen).

The practical value of filtering is conditional on using memory at all. The relevant question is not whether a strong reasoner may choose to ignore memory, but what happens when an agent must consume a persistent store. NoMemory remains competitive or safer on Claude and Qwen and is a necessary baseline for measuring the tax; but conditional on using memory, an explicit entry-level audit is consistently safer than trusting it raw across all four models.

The stale-memory tax is remediable, and the learned text audit is close to label-saturating. The Oracle ablation in Figure 5 separates the value of filtering from the quality of the learned audit (Oracle uses ground-truth labels through OMCD’s pipeline). Applying Oracle rather than leaving memory raw raises L2/L3 success by 18–24 pp $( p < 1 0 ^ { - 5 } )$ showing the tax is remediable by a read-time consistency gate rather than by unavoidable environmental change. Replacing OMCD’s learned text labels with Oracle labels, however, produces no detectable diference in any regime $( | \Delta | \leq \bar { 0 . 0 2 } .$ $p > 0 . 2 )$ . We read this as a failure to detect a benefit at the current sample size rather than as proven equivalence: the small observed gaps are consistent with the learned audit being close to label-saturating for the downstream task in this fully observed 8×8 text setting, but a formal equivalence test would be needed to assert that the two are interchangeable.

![](images/0e1950cd56d1b115eaa0766fd25d691f8aebf59aba93ff81f603da97845647b5.jpg)  
Figure 5: Filtering intervention analysis (GPT-4o, text, success rate, $N = 5 0 { \bar { ) } }$ . Oracle supplies ground truth stale labels but uses the same filtering operation as OMCD. NoFilter → Oracle shows the tax is remediable; the absence of a detectable OMCD–Oracle gap indicates the learned audit is close to label-saturating in this setting, not that the two are formally equivalent.

![](images/53e5e4ebb89f666382a6e556049c900c306e93b7c1a59132894fad2629388a63.jpg)  
Figure 6: Outcome-conditioned trajectory length across strategies, on the two models with the largest death sample. NoFilter deaths on both models terminate in roughly half the steps of a successful trajectory (7.8 to 8.5 versus 14.5 to 14.6), the “rush-into-hole” signature of stale memory directing the agent to a lethal cell. Claude’s NoMemory deaths reverse this: at 16.1 steps they are the longest of any strategy, indicating exhaustive exploration before eventual failure. OMCD deaths sit between the two extremes.

## 5.4 After Filtering is Reliable, the Remaining Bottleneck is Acting on the Filtered Memory

Trajectory shape distinguishes direct stale-memory failures from memoryless exploration failures. Figure 6 explains why outcome-conditioned trajectories are more informative than a scalar death rate. On GPT-4o and Claude, NoFilter deaths terminate in 8.5 and 7.8 steps, sharply shorter than successful trajectories on the same runs (14.5 and 14.6 steps). These are not exploration accidents: stale memory directs the agent into a mislabelled lethal cell before it can accumulate on-path evidence. Claude’s NoMemory deaths show the mirror pattern, terminating at 16.1 steps, longer than its successful trajectories; without memory, exploration eventually reaches an unobserved hole. OMCD lies between these mechanisms, matching the memoryless success-path length on Claude (15.0 versus 14.8). Qwen’s log-recovered trajectories show the same NoFilter rush signature at 6.8 steps (Appendix F). Since comparable death rates can arise from either mechanism, outcome-conditioned trajectories are necessary for diagnosing how a navigation policy fails; Appendix G gives step-level traces of each failure mode on a shared seed.

<table><tr><td></td><td></td><td colspan="2">L1</td><td colspan="2">L2</td><td colspan="2">L3</td></tr><tr><td>Model</td><td>Strategy</td><td>SR↑</td><td>Death↓</td><td>SR↑</td><td>Death ↓</td><td>SR↑</td><td>Death ↓</td></tr><tr><td colspan="8">Closed source</td></tr><tr><td></td><td>OMCD</td><td>.512</td><td>.216</td><td>.328</td><td>.316</td><td>.504</td><td>.284</td></tr><tr><td>GPT-40</td><td>SelfVerify</td><td>.392</td><td>.424</td><td>.160</td><td>.660</td><td>.336</td><td>.508</td></tr><tr><td></td><td>NoFilter</td><td>.368</td><td>.468</td><td>.144</td><td>.744</td><td>.260</td><td>.524</td></tr><tr><td></td><td>NoMemory</td><td>.440</td><td>.292</td><td>.288</td><td>.280</td><td>.456</td><td>.368</td></tr><tr><td rowspan="4">Claude-Sonnet-4.6</td><td>OMCD</td><td>.600</td><td>.208</td><td>.496</td><td>.312</td><td>.604</td><td>.284</td></tr><tr><td>SelfVerify</td><td>.560</td><td>.256</td><td>.488</td><td>.348</td><td>.528</td><td>.332</td></tr><tr><td>NoFilter</td><td>.464</td><td>.368</td><td>.404</td><td>.496</td><td>.436</td><td>.428</td></tr><tr><td>NoMemory</td><td>.688</td><td>.160</td><td>.536</td><td>.300</td><td>.688</td><td>.228</td></tr><tr><td rowspan="4">Qwen3.6</td><td>OMCD</td><td>.960</td><td>.000</td><td>.820</td><td>.100</td><td>.920</td><td>.060</td></tr><tr><td>SelfVerify</td><td>.960</td><td>.000</td><td>.780</td><td>.100</td><td>.820</td><td>.120</td></tr><tr><td>NoFilter</td><td>.920</td><td>.020</td><td>.800</td><td>.120</td><td>.840</td><td>.140</td></tr><tr><td>NoMemory</td><td>.960</td><td>.000</td><td>.800</td><td>.080</td><td>.940</td><td>.040</td></tr><tr><td colspan="8">Open weight</td></tr><tr><td></td><td>OMCD</td><td>.884</td><td>.032</td><td>.832</td><td>.100</td><td>.932</td><td>.048</td></tr><tr><td rowspan="3">GLM-5.1</td><td>SelfVerify</td><td>.884</td><td>.032</td><td>.832</td><td>.100</td><td>.932</td><td>.048</td></tr><tr><td>NoFilter</td><td>.868</td><td>.084</td><td>.748</td><td>.216</td><td>.968</td><td>.032</td></tr><tr><td>NoMemory</td><td>.884</td><td>.016</td><td>.832</td><td>.016</td><td>.932</td><td>.048</td></tr></table>

Table 3: Text-mode navigation outcomes across change regimes. Cells report success rate (SR; higher is better) and death rate (Death; lower is better) for each model and memory strategy. Boldface marks the best value within each model–regime–metric combination, with ties also bolded. L1–L3 are the controlled change regimes defined in Section 3.

Once the audit is reliable, residual errors are not explained by stale-label accuracy. In the primary GPT-4o text study, per-seed detection F1 and OMCD success are essentially uncorrelated (Pearson $r \in [ + 0 . 0 0 5 , + 0 . 0 6 0 ]$ , all $p > 0 . 6 7 ;$ ; Spearman $| \rho | \le 0 . 0 5 )$ , despite F1 above 0.91 on L1/L2. Since aggregate F1 does not record whether the missed entries lie on the agent’s path, this null correlation is suggestive but not decisive; the direct evidence is stronger: every OMCD L2 death on a stale cell occurs in a run with F1 above 0.9, so the entry was detected and removed, yet the agent still stepped onto the hole (Appendix H). Taken with the absence of a detectable Oracle–OMCD gap, this is most consistent with a local bottleneck shift: once the audit is reliable, improving stale labels alone does not address the navigation policy’s remaining action-selection errors. We state this as a bounded observation for the current setting, not a general claim that action selection is always the limiting factor.

## 6 Conclusion

We examined how contemporary memory-augmented VLM agents behave when their spatial memory becomes stale. In the primary condition, trusting raw stale memory is less safe than having no memory, because stale beliefs route the agent directly into a hole. A controlled text audit removes much of the tax, but its value is conditional on reliable perception: text detection is near ceiling for capable models, whereas visual auditing on identical grids varies by over an order of magnitude and helps inconsistently when unreliable. Finally, Oracle labels yield no measurable further improvement in the current $8 \times 8$ text setting, pointing the open problem toward acting safely on filtered memory rather than detecting staleness alone.

Scope and limitations. We deliberately isolate the mechanism in a controlled 8×8 FrozenLake with symbolic changes and full observability, so that each death traces to a specific stale entry; the cost is that our numbers are lower bounds on a clean setting, not field estimates. Text navigation covers all four navigators at the same 50-seed scale, while the mechanistic analyses (Oracle, trajectory decomposition, F1– success correlation) and vision navigation are reported on GPT-4o and a 10-seed preview respectively. We therefore do not claim OMCD is universally preferable to memoryless reasoning, or that the model-specific asymmetries generalize. We expect the qualitative mechanisms, a stale-memory safety tax and a modality-dependent audit, to transfer wherever an agent conditions actions on persistent spatial claims; larger worlds, ambiguous changes, and distractor memories are the natural next tests.

## References

Ahn, M.; Brohan, A.; Brown, N.; Chebotar, Y.; Cortes, O.; David, B.; Finn, C.; Fu, C.; Gopalakrishnan, K.; Hausman, K.; et al. 2022. Do As I Can, Not As I Say: Grounding Language in Robotic Afordances. In Conference on Robot Learning (CoRL).

Cai, W.; Ponomarenko, Y.; Yuan, J.; Li, X.; Yang, W.; Dong, H.; and Zhao, B. 2024. SpatialBot: Precise Spatial Understanding with Vision Language Models. arXiv preprint arXiv:2406.13642.

Chao, H.; Bai, Y.; Sheng, R.; Li, T.; and Sun, Y. 2026. STALE: Can LLM Agents Know When Their Memories Are No Longer Valid? arXiv preprint arXiv:2605.06527.

Chen, B.; Xu, Z.; Kirmani, S.; Ichter, B.; Driess, D.; Florence, P.; Sadigh, D.; Guibas, L.; and Xia, F. 2024. SpatialVLM: Endowing Vision-Language Models with Spatial Reasoning Capabilities. arXiv preprint arXiv:2401.12168.

Cheng, A.-C.; Yin, H.; Fu, Y.; Guo, Q.; Yang, R.; Kautz, J.; Wang, X.; and Liu, S. 2024. SpatialRGPT: Grounded Spatial Reasoning in Vision-Language Models. In Advances in Neural Information Processing Systems (NeurIPS).

Chevalier-Boisvert, M.; Bahdanau, D.; Lahlou, S.; Willems, L.; Saharia, C.; Nguyen, T. H.; and Bengio, Y. 2019. BabyAI: A Platform to Study the Sample Eficiency of Grounded Language Learning. In International Conference on Learning Representations (ICLR).

Chevalier-Boisvert, M.; Willems, L.; and Pal, S. 2018. Minimalistic Gridworld Environment for OpenAI Gym. https: //github.com/maximecb/gym-minigrid.

Fu, X.; Hu, Y.; Li, B.; Feng, Y.; Wang, H.; Lin, X.; Roth, D.; Smith, N. A.; Ma, W.-C.; and Krishna, R. 2024. BLINK: Multimodal Large Language Models Can See but Not Perceive. In European Conference on Computer Vision (ECCV).

Gu, J.; Stefani, E.; Wu, Q.; Thomason, J.; and Wang, X. E. 2022. Vision-and-Language Navigation: A Survey of Tasks, Methods, and Future Directions. arXiv preprint arXiv:2203.12667.

Guan, T.; Liu, F.; Wu, X.; Xian, R.; Li, Z.; Liu, X.; Wang, X.; Chen, L.; Huang, F.; Yacoob, Y.; Manocha, D.; and Zhou, T. 2024. HallusionBench: An Advanced Diagnostic Suite for Entangled Language Hallucination and Visual Illusion in Large Vision-Language Models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

Huang, W.; Xia, F.; Xiao, T.; Chan, H.; Liang, J.; Florence, P.; Zeng, A.; Tompson, J.; Mordatch, I.; Chebotar, Y.; et al. 2022. Inner Monologue: Embodied Reasoning through Planning with Language Models. In Conference on Robot Learning (CoRL).

Hussain, M.; Chen, D.; Cheng, A.; Wei, H.; and Stanley, D. 2013. Change Detection from Remotely Sensed Images: From Pixel-Based to Object-Based Approaches. ISPRS Journal ofPhotogrammetry and Remote Sensing, 80: 91–106.

Kamath, A.; Hessel, J.; and Chang, K.-W. 2023. What’s “Up” with Vision-Language Models? Investigating Their Struggle

with Spatial Reasoning. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP), 9161–9175.

Leike, J.; Martic, M.; Krakovna, V.; Ortega, P. A.; Everitt, T.; Lefrancq, A.; Orseau, L.; and Legg, S. 2017. AI Safety Gridworlds. arXiv preprint arXiv:1711.09883.

Li, Y.; Du, Y.; Zhou, K.; Wang, J.; Zhao, W. X.; and Wen, J.-R. 2023. Evaluating Object Hallucination in Large Vision-Language Models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP), 292–305.

Liu, F.; Emerson, G.; and Collier, N. 2023. Visual Spatial Reasoning. Transactions of the Association for Computational Linguistics, 11: 635–651.

Madaan, A.; Tandon, N.; Gupta, P.; Hallinan, S.; Gao, L.; Wiegrefe, S.; Alon, U.; Dziri, N.; Prabhumoye, S.; Yang, Y.; et al. 2023. Self-Refine: Iterative Refinement with Self-Feedback. In Advances in Neural Information Processing Systems (NeurIPS).

Meng, K.; Bau, D.; Andonian, A.; and Belinkov, Y. 2022. Locating and Editing Factual Associations in GPT. In Advances in Neural Information Processing Systems (NeurIPS).

Meng, K.; Sharma, A. S.; Andonian, A. J.; Belinkov, Y.; and Bau, D. 2023. Mass-Editing Memory in a Transformer. In International Conference on Learning Representations (ICLR).

Mousavi, S. M.; Alghisi, S.; and Riccardi, G. 2024. DyKnow: Dynamically Verifying Time-Sensitive Factual Knowledge in LLMs. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2024.

Packer, C.; Wooders, S.; Lin, K.; Fang, V.; Patil, S. G.; Stoica, I.; and Gonzalez, J. E. 2023. MemGPT: Towards LLMs as Operating Systems. arXiv preprint arXiv:2310.08560.

Savva, M.; Kadian, A.; Maksymets, O.; Zhao, Y.; Wijmans, E.; Jain, B.; Straub, J.; Liu, J.; Koltun, V.; Malik, J.; Parikh, D.; and Batra, D. 2019. Habitat: A Platform for Embodied AI Research. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision (ICCV).

Shinn, N.; Cassano, F.; Berman, E.; Gopinath, A.; Narasimhan, K.; and Yao, S. 2023. Reflexion: Language Agents with Verbal Reinforcement Learning. In Advances in Neural Information Processing Systems (NeurIPS).

Song, C. H.; Wu, J.; Washington, C.; Sadler, B. M.; Chao, W.-L.; and Su, Y. 2023. LLM-Planner: Few-Shot Grounded Planning for Embodied Agents with Large Language Models. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV).

Towers, M.; Kwiatkowski, A.; Terry, J.; Balis, J. U.; De Cola, G.; Deleu, T.; Goulão, M.; Kallinteris, A.; Krimmel, M.; KG, G.; Deleu, T.; Goulão, M.; Kallinteris, A.; Krimmel, M.; KG,

A.; Pérez-Vicente, R.; Pierré, A.; Schulhof, S.; Tai, J. J.; Tan, H.; and Younis, O. G. 2024. Gymnasium: A Standard Interface for Reinforcement Learning Environments. arXiv preprint arXiv:2407.17032.

Vu, T.; Iyyer, M.; Wang, X.; Constant, N.; Wei, J.; Wei, J.; Tar, C.; Sung, Y.-H.; Zhou, D.; Le, Q.; and Luong, T. 2023. Fresh-LLMs: Refreshing Large Language Models with Search Engine Augmentation. arXiv preprint arXiv:2310.03214.

Wang, G.; Xie, Y.; Jiang, Y.; Mandlekar, A.; Xiao, C.; Zhu, Y.; Fan, L.; and Anandkumar, A. 2023. Voyager: An Open-Ended Embodied Agent with Large Language Models. arXiv preprint arXiv:2305.16291.

Wang, J.; Ming, Y.; Shi, Z.; Vineet, V.; Wang, X.; Li, Y.; and Joshi, N. 2024. Is a Picture Worth a Thousand Words? Delving into Spatial Reasoning for Vision Language Models. In Advances in Neural Information Processing Systems (NeurIPS).

Wu, Q.; Zhao, H.; Saxon, M.; Bui, T.; Wang, W. Y.; Zhang, Y.; and Chang, S. 2024. VSP: Assessing the Dual Challenges of Perception and Reasoning in Spatial Planning Tasks for VLMs. arXiv preprint arXiv:2407.01863.

Yao, S.; Zhao, J.; Yu, D.; Du, N.; Shafran, I.; Narasimhan, K.; and Cao, Y. 2023. ReAct: Synergizing Reasoning and Acting in Language Models. In International Conference on Learning Representations (ICLR).

Zhu, X.; Chen, Y.; Tian, H.; Tao, C.; Su, W.; Yang, C.; Huang, G.; Li, B.; Lu, L.; Wang, X.; Qiao, Y.; Zhang, Z.; and Dai, J. 2023. Ghost in the Minecraft: Generally Capable Agents for Open-World Environments via Large Language Models with Text-based Knowledge and Memory. arXiv preprint arXiv:2305.17144.

## A Dificulty Level Configuration

This appendix documents the full parameterization of the three dificulty levels in SpatialSTALE. All values are deterministic functions of the per-instance seed.

Common parameters. Grid size $H { = } W { = } 8 ;$ initial hole fraction 0.25; start position $S = ( 0 , 0 )$ ; goal position $G =$ (H−1, W−1). A valid path from S to G is guaranteed by rejection sampling: instances without a path are regenerated. Memory store contains exactly N=64 entries (one per cell).

L1 (Spot). Number of changes drawn uniformly from {5, 6, 7}. Each change picks a random cell (uniform over all 64), applies a Thaw or Freeze with equal probability, subject to the constraint that the start cell S and goal cell G are never modified. Changes are applied before the episode begins.

L2 (Cluster). Number of changes drawn uniformly from {12, 13, 14, 15, 16}. A cluster center c is sampled uniformly from non-{S, G} cells; all changes are then sampled from a 2-radius (Manhattan) neighborhood of c. Operations are Thaw/Freeze with equal probability. Path validity is enforced post-hoc by retrying the cluster center if blocked.

L3 (Online dynamic). Changes occur during the episode. At every k=5 steps, a dynamic-event trigger fires with probability 0.7 (i.e., 30% skip); when it fires, 2 cells are flipped (Thaw or Freeze with equal probability, never S or G). For each episode, the original grid ${ \bf g } _ { 0 }$ is fixed and all subsequent grids $\mathbf { g } _ { t }$ are derived from it. Ground-truth staleness at time t is defined cumulatively: $m _ { i }$ is stale if $\mathbf { g } _ { t } [ \mathbf { p o s } _ { i } ] \neq \mathbf { g } _ { 0 } [ \mathbf { p o s } _ { i } ]$ Detection is invoked at each event trigger, and metrics are aggregated across all detection windows within the episode.

Stale-ratio statistics. Across 50 seeds:

• L1: stale ratio mean 0.094 ± 0.011, range [0.078, 0.109].

• L2: stale ratio mean 0.141 ± 0.014, range [0.109, 0.172].

• L3: end-of-episode stale ratio mean 0.143 ± 0.028, range [0.078, 0.219].

The L3 ratio is reported at the final detection window; intermediate windows have proportionally smaller ratios.

## B Game Setup and Rendering

State and observation. The grid is encoded as an $8 \times 8$ matrix of cell types {S, F, H, G} (start, frozen, hole, goal). The agent occupies a single cell and has four discrete actions: UP, DOWN, LEFT, RIGHT. Stepping into H terminates the episode with reward −1 (death); reaching G terminates with reward +1 (success); the maximum episode length is 50 steps, beyond which the episode times out with reward 0.

Text-mode observation. The full grid is presented as a coordinate-labeled list:

Row 0: (0,0)=S (0,1)=F (0,2)=H (0,3)=F ...   
Row 1: (1,0)=F (1,1)=F (1,2)=H ...   
Agent at (3,4). Goal at (7,7).

Each cell is fully observable (we deliberately do not introduce partial observability in this benchmark, as our focus is on staleness rather than discovery).

Vision-mode observation. The grid is rendered as a 384 × 384 RGB image (48 px per cell). Cell colors: $\textrm { F } =$ light blue (#A8C8E8), H = near-black (#202020), S = green (#2E8B57), G = gold (#DAA520), agent = red triangle overlay. Each cell carries its coordinate as a small text label in the top-left corner $( \mathrm { e . g . , } ^ { \ast } ( 3 , 4 ) ^ { \ast } )$ . A 1-px white grid line separates cells. Memory entries remain as text in all conditions.

Memory format. Each entry is a single line of natural language anchored to a coordinate:

[mem\_023] SAFE at (2,5): Frozen ice - safe to   
walk   
[mem\_041] DANGER at (4,7): Hole - dangerous,   
do not step

The mem\_<id> prefix is a stable identifier used to score detections against ground truth.

Memory accumulation. Before each episode, we simulate 20 random-walk exploration episodes on the original (prechange) grid. Each visited cell produces a memory entry of the corresponding type. After exploration, any unvisited cell receives a synthetic entry of the correct type. This guarantees exactly 64 entries per instance and removes coverage variance as a confound.

Strategy interface. All navigation strategies share an identical action interface (the four discrete moves above) and identical observation pipelines, and all select each action with the same per-step LLM navigator. They difer only in (i) what memory text is placed in the prompt and (ii) whether/how staleness detection runs. The full decision procedure is documented in Appendix C.

Implementation environment. The closed-source models (GPT-4o, Claude-Sonnet-4.6, Qwen3.6-Plus) are accessed through their oficial inference APIs; no model weights are trained or fine-tuned in this work. The open-weight VLMs (GLM-5.1, InternVL3-2B, InternVL3-8B) are run locally on a single Ubuntu 22.04 machine (Intel Xeon Platinum CPU, 512 GB RAM) with 2×NVIDIA H20 GPUs (96 GB each, CUDA 12.4), served with vLLM 0.6.3 for batched inference. The benchmark generator, environment dynamics, and evaluation harness are implemented in Python 3.11 using gymnasium 1.0.0, numpy 1.26.4, torch 2.4.0, and transformers 4.45.2; all library versions are additionally pinned in the released code repository.

## C Navigation Decision Procedure

Per-step LLM action selection. All four strategies (NoMemory, NoFilter, SelfVerify, OMCD) and the Oracle ablation use the same navigator: at every step, the model is prompted with the current full-grid observation, its own position, the goal position, and the remaining step budget, and it returns a single discrete action (UP, DOWN, LEFT, RIGHT). We deliberately use no external symbolic planner: the point of the study is how the language model itself acts when a persistent memory claim conflicts with what it currently observes. The strategies difer only in the memory text placed in the prompt alongside the observation:

• NoMemory: no memory entries are included; the model decides from the current observation alone.

• NoFilter: all 64 original memory entries are included unchanged, even where they now contradict the observation.

• SelfVerify: the model first removes entries it judges inconsistent in a single pre-navigation pass (Section 4); the surviving entries are then included for every step.

• OMCD / Oracle: entries flagged stale (by the OMCD detector, or by ground truth for Oracle) are removed before navigation; the remaining entries are included.

Observation is always current and truthful. The observation shown to the navigator at each step is rendered directly from the live environment state, so it always reflects the current grid, including any dynamic changes already applied under L3. Memory is therefore never used as a stand-in for the map: it is auxiliary text that may agree or disagree with the observation. Consequently, a stale “SAFE” entry does not silently hide a hole from the model; rather, it supplies a confident but wrong claim that competes with the correct observation for the model’s decision. This is why removing misleading entries (OMCD, Oracle) can reduce deaths without any change to the observation itself.

Dynamic re-auditing. Under L3, where cells change during the episode, OMCD and Oracle re-run detection after each realized event and rebuild the active memory view from the latest judgments; NoMemory, NoFilter, and SelfVerify keep their memory view fixed across the episode. Episodes terminate on reaching the goal (reward +1), stepping onto a hole (reward −1), or exhausting the 50-step budget (reward 0).

Action parsing and fallback. Actions are parsed from the model’s structured output. On the rare occasions when the output cannot be parsed into a legal action, we fall back to a single greedy step toward the goal so that the episode can continue; such fallbacks are logged and are infrequent.

## D Binary vs. Continuous Staleness Output

Setup. An alternative to the binary is\_stale output used in OMCD is to ask the LLM for a continuous staleness score $s \in [ 0 , 1 ]$ and threshold it post-hoc. We compared this on GPT-4o (L1, N=50 seeds, text mode) at three thresholds (Table 4).

Table 4: Continuous staleness score vs. binary classification (GPT-4o, L1, N=50 seeds, text mode).
<table><tr><td>Output type</td><td>Precision</td><td>Recall</td><td>F1</td></tr><tr><td>Continuous, τ=0.3</td><td>.603</td><td>.987</td><td>.749</td></tr><tr><td>Continuous, τ=0.5</td><td>.742</td><td>.961</td><td>.838</td></tr><tr><td>Continuous, τ=0.7</td><td>.891</td><td>.872</td><td>.881</td></tr><tr><td>Binary (used in OMCD)</td><td>.856</td><td>.984</td><td>.914</td></tr></table>

## Observations.

1. No single threshold on continuous output dominates binary. The best continuous result (τ=0.7, F1 = 0.881) trails binary by 3 points.

2. Continuous scores have heterogeneous semantics across models. In a cross-model comparison (N=50 seeds per model), the same score s=0.6 corresponded to ∼80% true-stale rate for GPT-4o but only ∼35% for Claude. There is no global threshold that works for all models.

3. Binary judgments collapse this calibration problem: each model is asked the same yes/no question, and we aggregate yes/no outcomes rather than numerical values.

Conclusion. We use binary output throughout the main paper. This decision afects OMCD alone; the benchmark itself is agnostic to output type (any detector producing $\hat { y } _ { i } \in$ {stale, valid} is admissible).

## E Batch Size Ablation

We ablate the batch size B used in OMCD on GPT-4o, L1, text mode, N=50 seeds. Each instance has exactly 64 memory entries, so the number of LLM calls per detection is $\lceil 6 4 / B \rceil$ . Table 5 reports the resulting detection quality and cost.

Table 5: Efect of batch size B on detection quality and inference cost.
<table><tr><td>B</td><td>#calls/inst</td><td>Precision</td><td>Recall</td><td>F1</td><td>Tokens/inst</td></tr><tr><td>1</td><td>64</td><td>.892</td><td>.995</td><td>.941</td><td>18.7k</td></tr><tr><td>5</td><td>13</td><td>.871</td><td>.990</td><td>.926</td><td>5.2k</td></tr><tr><td>10</td><td>7</td><td>.856</td><td>.984</td><td>.914</td><td>3.1k</td></tr><tr><td>20</td><td>4</td><td>.812</td><td>.955</td><td>.878</td><td>2.2k</td></tr></table>

## Observations.

• F1 decreases monotonically with batch size as attention dilutes across more concurrent entries.

• However, the drop from B=1 (per-entry queries, expensive) to B=10 is small (−2.7pp F1), while inference cost drops by 9×.

• B=20 shows a sharper drop (−3.6pp from B=10), suggesting we are crossing a regime where the LLM loses track of which entry is which within the batch.

Conclusion. B=10 is a reasonable cost and quality trade of for N=64. The same trend should hold for larger memory stores; we leave a scaling study (where B may need to grow sublinearly with N) to future work.

## F Trajectory Length Decomposition

Table 6 decomposes episode length by outcome for the three closed source models that record per outcome traces. Two properties of this decomposition are used in the main text (Section 5, discussion around Table 3).

First, on GPT-4o and Claude the fatal trajectories under NoFilter are the shortest of any strategy (8.5 and 7.8 steps, both marked in bold). Short fatal trajectories under NoFilter are consistent with stale memory entries directing the navigator into holes early, rather than an accumulation of exploration errors. Qwen is an exception: its few OMCD deaths occur earliest (3.1 steps), so the "NoFilter dies fastest" signature is not universal.

Second, the successful trajectory column is nearly flat across strategies within each model (|∆| < 0.5 steps between the best and worst strategy on every model). This rules out a path length artifact for the reversal reported in the main text where NoMemory equals or exceeds OMCD on Claude and Qwen: those wins are not driven by NoMemory taking a longer, luckier route.

Table 6: Trajectory length by outcome (mean steps, closed source models with per outcome traces). Bold: shortest death trajectory within each model, a diagnostic rather than desirable performance.

<table><tr><td>Model</td><td>Strategy</td><td>Steps|Success</td><td>Steps|Death</td></tr><tr><td rowspan="5">GPT-40</td><td>OMCD</td><td>16.1</td><td>12.4</td></tr><tr><td>SelfVerify</td><td>14.5</td><td>9.5</td></tr><tr><td>NoFilter</td><td>14.5</td><td>8.5</td></tr><tr><td>NoMemory</td><td>16.4</td><td>12.7</td></tr><tr><td>OMCD</td><td>15.0</td><td>8.3</td></tr><tr><td rowspan="4">Claude-4.6</td><td>SelfVerify</td><td>14.8</td><td>8.6</td></tr><tr><td>NoFilter</td><td>14.6</td><td>7.8</td></tr><tr><td>NoMemory</td><td>14.8</td><td>16.1</td></tr><tr><td>OMCD</td><td>14.8</td><td>3.1</td></tr><tr><td rowspan="3">Qwen3.6</td><td>SelfVerify</td><td>14.7</td><td>4.1</td></tr><tr><td>NoFilter</td><td>14.9</td><td>6.8</td></tr><tr><td>NoMemory</td><td>15.1</td><td>4.4</td></tr></table>

## G Navigation Trace Studies

We present three condensed navigation traces from the L2 evaluation (GPT-4o, text mode, seed 4810). All three start from the same initial state; they difer only in the memory strategy. We use > for an action that succeeds and X for the death step.

## Trace A: NoFilter (dies at step 9).

S=(0,0) Goal=(7,7)   
Stale entries in memory: 14. Agent does   
not know which.   
Memory (incl. stale SAFE at   
(5,4)) steers the model toward   
(3,4),(4,4),(5,4)   
step 1: > (0,0)→(1,0) F   
step 2: > (1,0)→(2,0) F   
step 3: > (2,0)→(3,0) F   
step 4 to 8: > model follows the   
memory-suggested route   
step 9: X (5,4) had thawed to H during   
change   
Outcome: DEATH, reward −1.

## Trace B: OMCD (succeeds at step 26).

Detection round: flagged mem\_044 (SAFE   
at (5,4)) as stale.   
With the misleading entry removed, the   
model detours via (5,3)→(6,3)   
step 1 to 8: > matches Trace A   
step 9 to 14: > detour (avoids (5,4))   
step 15 to 26: > continues to goal   
Outcome: SUCCESS, reward +1.

## Trace C: NoMemory (timeout at step 50).

No memory in the prompt; the model   
decides from the current observation   
only.   
step 1 to 14: > wanders generally   
toward (7,7)   
step 15: avoids (3,5) H, which is   
visible in the current grid   
step 16 to 49: > slow exploration   
step 50: timeout, agent at (6,6).   
Outcome: TIMEOUT, reward 0.

## Aggregate observations. Across 50 seeds at L2:

• NoFilter dies 74.4% of the time, almost always within the first 15 steps as the agent walks confidently into a newly-thawed hole.

• OMCD succeeds 32.8% of the time. Failures mostly come from late-game deaths where a stale entry was missed, or from the model taking a long detour and running out of steps.

• NoMemory has the highest timeout rate (∼32%) because exploration is slow, but the lowest death rate (28.0%) because, without a confident stale claim to trust, the model rarely commits to an unobserved cell.

## H Detection Diagnostics

This appendix reports three diagnostics that refine the aggregate F1 results in Figure 3. They support the main-text interpretation but are not additional primary endpoints.

## H.1 Precision–Recall Error Regimes

![](images/aa5f869026948ec924079ee8d30f1195d1fc145c6b8c852aaeee75b6735cb27d.jpg)  
Figure 7: Per-run mean precision and recall for every (model, mode, regime), with iso-F1 contours. Filled markers are text mode, open markers are vision. L1/L2 points generally lie above the P=R diagonal (over-flagging), viable L3 text points lie below it (under-flagging), and GLM vision collapses near the lower left.

Figure 7 separates errors that aggregate F1 obscures. Snapshot regimes L1/L2 generally have P < R: models recover stale cells aggressively, accepting more false alarms. On L3, the viable text-mode series instead have P > R: online changes shift the dominant residual error toward missed stale entries. GLM vision is distinct from both patterns because both precision and recall collapse below 0.15 and 0.10, respectively.

## H.2 Recall by Change Direction

![](images/817b9360f9a8ef470b67ea17294cd023c64cc8ca0cc603267e238612a2fc1452.jpg)  
Figure 8: Per-entry recall on truly stale entries, split by change type. A thaw is F→H: memory says safe but the current cell is a lethal hole. A freeze is H→F: memory says danger but the current cell is safe. Claude vision L1 shows a 17.2 pp gap in favour of the deadly class; the corresponding L2 gap is 30.9 pp.

The costs of the two stale classes difer: trusting thaw is fatal, while trusting freeze is overcautious. As Figure 8 shows, the observed Claude vision asymmetry favours recall of the deadly class; on L2, recall is 0.945 for thaw (N=436) versus 0.636 for freeze (N=121). This is a descriptive, modelspecific observation. Qwen is near ceiling on both classes, while GLM vision has too little recall on either class for a meaningful directional comparison.

## H.3 Detection F1 and Navigation Success

![](images/ee2e216b72f2a260c3afffd57bd2e7eaa8b2d6497fd97e09cd03e629d897eba6.jpg)  
Figure 9: Per-seed detection F1 versus OMCD navigation success on GPT-4o text (N = 50 seeds per regime). All three per-regime correlations are near zero $( p > 0 . 6 7 )$ . The wider F1 spread on L3 still yields no success gradient.

As Figure 9 shows, the primary GPT-4o text data contain no per-seed association between detection F1 and OMCD success: Pearson r ranges from +0.005 to +0.060, and all three p values exceed 0.67. This null relationship complements the Oracle ablation: after the text audit is reliable, label accuracy is not the observable source of residual navigation variation in this setting.

## I Additional Detection Results

GPT-4o vision failure profile. GPT-4o’s vision-mode detection sits between Claude’s moderate degradation and

GLM-5.1’s total collapse: average ∆F1 = −0.598 relative to text mode, a significantly larger drop than Claude’s −0.134 (the second-largest gap among non-collapsing models). Figure 3 summarizes the per-dificulty numbers; here we characterize the failure mode.

Unlike GLM-5.1, GPT-4o does not produce fluent uniform hallucinations at confidence 1.0. Instead, it tends to (i) refuse or hedge when asked to enumerate cell-level state from the rendered 8×8 image (roughly 30% of the L1 vision responses contain phrases such as “I cannot reliably determine the color at that coordinate”), and (ii) produce partially-correct enumerations in which nearby cells’ F/H labels are systematically swapped. The result is that GPT-4o’s precision is competitive but its recall on stale cells drops by a factor of ∼3, driving the F1 collapse. This is a distinct failure category from GLM-5.1’s silent hallucination and further demonstrates that diferent frontier VLMs fail at spatial change detection in qualitatively diferent ways.

Detection error breakdown. Across all 300 GPT-4o detection runs (50 seeds × 3 regimes × 2 modalities), we further categorize errors as:

• False negatives on Thaw events (memory says SAFE, cell now H): 63.2% of all FNs.

• False negatives on Freeze events (memory says DAN-GER, cell now F): 36.8% of all FNs.

• False positives: predominantly on cells adjacent to a true stale cell, consistent with propagation behavior.

The Thaw/Freeze asymmetry is important for the safety story: Thaw-stale memories are exactly the ones whose miss kills the agent (memory says “safe,” cell is actually a hole). Future detectors should up-weight Thaw recall.

## J GLM-5.1 Vision Failure: Verbatim Cases

This appendix reproduces three GLM-5.1 vision-mode detection responses verbatim (lightly trimmed for length). All three correspond to memory entries that are ground-truth stale in the corresponding image, and all three are predicted as is\_stale: false with maximum confidence.

Case 1 (L1, seed 1734). The image shows cell (2, 5) rendered as a dark hole (H). The memory entry says SAFE at (2,5), which is stale (the cell originally was F and has been flipped to H).

{   
"memory\_id": "mem\_021",   
"is\_stale": false,   
"confidence": 1.0,   
"reasoning": "The image clearly   
shows that cell (2,5) is light blue,   
indicating frozen ice (F). The memory   
states SAFE at (2,5), which matches   
what I observe. No conflict, memory is   
valid."   
}

The model asserts the cell is F when it is rendered H.

Case 2 (L2, seed 4810). The image shows cell (4, 3) rendered as a hole (H), part of a cluster of three nearby flipped cells. Memory says SAFE at (4,3).

{   
"memory\_id": "mem\_035",   
"is\_stale": false,   
"confidence": 1.0,   
"reasoning": "Looking at row 4,   
position 3 shows a frozen tile   
consistent with the memory. The agent   
has not encountered any change in this   
region."   
}

“Row 4, position 3” was a dark hole tile.

Case 3 (L1, seed 7822). The image shows cell (0, 2) flipped from H to F (Thaw). Memory says DANGER at (0,2), which is now stale.

{   
"memory\_id": "mem\_002",   
"is\_stale": false,   
"confidence": 1.0,   
"reasoning": "The image confirms (0,2)   
is a hole. Memory marking it as DANGER   
is accurate."   
}

The model again confirms the wrong cell state.

Aggregate statistics (all 6,400 GLM vision judgments).

• Refusal rate: 0.0% (no judgment returns unknown or a refusal string).

• Reported confidence = 1.0 on 100.0% of judgments.

• Modal hallucination pattern: “image shows X” where X equals the memory’s content rather than the image’s actual content. This is consistent with image bias: when text and image inputs both reference the same coordinate, the model resolves the conflict by trusting the textual prior and verbally describing the image as agreeing.

Implication. Standard confidence-based filtering would discard zero of these failures. Token-level uncertainty (e.g., logit entropy of the is\_stale token) might catch some, but is not exposed by current commercial API endpoints. SpatialSTALE thus ofers a concrete, reproducible setting for studying detection of silent vision hallucinations.

## K Planned Benchmark Extensions

We outline several extensions of SpatialSTALE that future work can pursue to increase task dificulty and broaden coverage.

Larger grids. Scaling from 8 × 8 to 16 × 16 or 32 × 32 increases the number of memory entries linearly with area and is expected to magnify the efect of spatial propagation (Finding 8). It also stresses the LLM’s ability to localize references in longer text grids; a 16 × 16 text observation occupies ∼1,000 tokens.

Distractor entries. The current memory store contains one entry per cell. A more challenging variant adds non-spatial distractor entries (e.g., “the rules of the game are X”, “previously the agent observed Y”). This tests whether detectors can correctly ignore non-stale-able entries and reduces the efective signal-to-noise ratio.

Multi-step memory edits. Rather than annotating staleness binarily, future versions could ask the detector to propose an edit, e.g., “replace ‘DANGER at (3,4)’ with ‘SAFE at (3,4)’ ”. This converts the task from classification to generation and exposes whether models can articulate what changed, not just detect that something did.

Multi-agent staleness. A separate axis of dificulty is staleness caused by other agents’ actions, rather than by environment dynamics. This requires modeling other agents’ likely behavior and is closer to real multi-agent deployments.

Continuous-control variants. Beyond grids, the same staleness setup can be embedded in continuous-action environments (Habitat, AI2-THOR), where memory entries reference object locations or pose. Detection there must additionally cope with perception noise.

Vision-only memory. Currently memory entries are always textual. A vision-only variant stores past observation crops as memory, asking the detector to compare crops directly. This isolates pure visual matching from textual reasoning.

## L Declaration of generative AI and AI-assisted technologies in the writing process

During the preparation of this work, the authors used Workbuddy in order to improve the readability and language of the manuscript. After using this tool, the authors reviewed and edited the content as needed and take full responsibility for the content of the published article.