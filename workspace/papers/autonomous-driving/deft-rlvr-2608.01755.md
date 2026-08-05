# Deferred Exposure of Future Trajectories for Verifiable Reasoning in Autonomous Driving VLMs

Zixuan Huang<sup>1,2,†</sup>, Yang Zhou<sup>2,3,†</sup>, Kaixuan Wang<sup>2</sup>, Guli Zhang<sup>2</sup>, Hongyan Xie<sup>1</sup>, Yakun Zhu<sup>4</sup>, Hao Geng<sup>1</sup>, Xiaozhi Chen<sup>2</sup>, Yikun Ban<sup>1,∗</sup>, Deqing Wang<sup>1,</sup>

<sup>1</sup>Beihang University, <sup>2</sup>Zhuoyu Technology, Shenzhen, China, <sup>3</sup>Zhejiang University, <sup>4</sup>Shanghai Jiao Tong University

<sup>†</sup>Equal contribution, <sup>∗</sup>Corresponding author

Recent Vision-Language-Action (VLA) models for autonomous driving (AD) increasingly utilize chainof-thought (CoT) supervision to enhance the reasoning capabilities of their Vision-Language Model (VLM) components, yet existing annotation pipelines commonly expose the teacher model to the logged ground-truth (GT) future trajectory. We empirically show that this induces trajectory anchoring bias: teacher models rationalize the revealed outcome rather than infer a decision from scene evidence, producing less causally faithful CoTs and substantially more severe hallucinations, especially in causally challenging scenes. Removing the GT trajectory eliminates this shortcut, but open-ended trajectory generation entangles high-level decision-making with precise geometric synthesis and lowlevel dynamics. To make trajectory-level driving decisions verifiable without requiring open-ended trajectory synthesis, we introduce Autonomous-Driving Multiple-Choice Question (AD-MCQ), which casts planning as selection among explicit trajectory candidates. Taking this a step further, we propose Deferred Exposure of Future Trajectories for RLVR (DEFT-RLVR) to transform future trajectories from pre-decision anchors into post-decision verification targets. Experimental results show that DEFT-RLVR improves AD reasoning while preserving or even enhancing general visual capabilities. With VLM-only inference and controllable dificulty through candidate construction, AD-MCQ provides a flexible, scalable, and extensible foundation for future research on verifiable AD reasoning.

Date: 2026.7.31 Code: https://github.com/hzx122/DEFT-RLVR Model: https://huggingface.co/hzxllll/DEFT-RLVR-model-HF Dataset: https://huggingface.co/datasets/hzxllll/AD-MCQ Correspondence: huang\_zx@buaa.edu.cn

云ZYT

## 1 Introduction

Mainstream Vision-Language-Action (VLA) models for autonomous driving (AD) typically couple a large Vision-Language Model (VLM) with a substantially smaller action expert [27, 59]. The action expert is typically specialized for geometric prediction, whereas high-level reasoning and decision making fall to the VLM, making its AD-specific reasoning capability critical to downstream planning.

Recent work seeks to enhance this ability through CoT supervision [12, 59, 62, 82, 87]. However, when reasoning must resolve into a concrete driving decision [87], rather than scene understanding alone [25, 39, 42, 64], its CoT supervision is typically conditioned on the ground-truth (GT) trajectory. Given the logged future trajectory, the VLM CoT annotator rationalizes a known outcome rather than inferring a decision from scene evidence. This mirrors anchoring bias in cognitive psychology, whereby initially supplied information can disproportionately shape subsequent judgments [76].

We empirically validate trajectory anchoring bias through the controlled study in Figure 1. GT-conditioned CoTs exhibit lower causal faithfulness than causal-planning CoTs, with the degradation primarily concentrated in hard causal scenarios where reliable reasoning is most critical. Moreover, exposing the model to the GT trajectory substantially increases the incidence of severe hallucinations, indicating that trajectory conditioning can inject fabricated causal evidence into the CoT supervision used for subsequent training.

![](images/c8561f69b32907d60457ae60ba8e6d7186f10f77556498f72878c1985f7930ab.jpg)  
Figure 1 GT conditioning induces post-hoc rationalization. The illustrated CoT invents a mandatory-turn sign absent from the scene; aggregate results show lower causal faithfulness and preference, with more severe hallucinations.

Given this, a natural remedy is to hide the GT trajectory while the teacher derives both its rationale and driving decision from the observed scene, and to verify the predicted future only afterward. This restores the solve-then-verify paradigm used in reasoning-model distillation [51, 72], rather than revealing the answer before constructing its rationale.

For AD, however, open-ended trajectory synthesis is poorly matched to this paradigm because it entangles high-level decision making with precise continuous geometry and low-level dynamics. We therefore seek a language-model-compatible interface through which AD reasoning can emerge from the VLM’s general reasoning capability.

To make trajectory-level driving decisions verifiable without open-ended geometric synthesis, we introduce Autonomous-Driving Multiple-Choice Question (AD-MCQ), a candidate-trajectory benchmark that casts AD planning as selection among scene-specific explicit trajectories. Unlike coarse meta actions, its candidates preserve distinctions in braking time, speed profile, and lateral geometry, grounding each answer in a concrete explicit plan.

AD-MCQ makes trajectory-level driving decisions verifiable, but revealing candidate trajectories before reasoning can simply replace the GT-trajectory anchor with a candidate-set anchor: the policy model inevitably focuses on comparing the relative quality of trajectories, thereby taking shortcuts in reasoning. We therefore propose Deferred Exposure of Future Trajectories for RLVR (DEFT-RLVR), which first commits the policy to a scene-derived decision and only then reveals candidates for explicit grounding. In this way, trajectories supervise reasoning as post-decision targets rather than pre-decision premises.

Across multiple VLM backbones, DEFT-RLVR consistently strengthens autonomous-driving reasoning and decision making while slightly improving aggregate general visual capability.

In summary, our contributions in this work are as follows:

• We identify and empirically validate trajectory anchoring bias: exposing the demonstrated future trajectory produces action-consistent but causally unfaithful rationales, especially in hard causal scenes.

• We introduce AD-MCQ, a verifiable candidate-trajectory benchmark that preserves explicit trajectorylevel distinctions and supports both exact selection and candidate-blind reasoning evaluation without open-ended coordinate generation.

• We propose DEFT-RLVR, which defers candidate-trajectory exposure until after the policy has committed to a scene-derived decision and uses exact trajectory correctness and question-specific process supervision. This design improves AD reasoning while preserving the general visual capability of the base policy.

<table><tr><td>GT Exposure</td><td>Severe Halluc. ↓</td><td>Pairwise Win ↑</td></tr><tr><td>No (Causal Planning)</td><td>29.0%</td><td>60.5%</td></tr><tr><td>Yes (GT-Conditioned)</td><td>50.0%</td><td>24.0%</td></tr></table>

Table 1 Human-rated efect of pre-reasoning GT-trajectory exposure on CoT quality. Exposing the trajectories increases severe hallucinations and reduces pairwise preference.

## 2 Trajectory Anchoring Bias in AD VLMs

Our motivation begins with a simple research question: does revealing the GT future trajectory help a teacher infer a faithful driving rationale, or merely make an already known outcome easier to justify? The AD-VLM must infer the appropriate trajectory from the evidence available in the historical scene. By contrast, the GT trajectory can act as an anchor, allowing teachers to reverse inference and construct a post hoc explanation of the revealed outcome [1, 68]. Consequently, the resulting CoT may be geometrically consistent with the GT trajectory while failing to faithfully identify the scene evidence that genuinely supports the action [3, 71].

We examine this anchoring hypothesis through a human-scored study summarized in Figure 2 and Table 1. GT-conditioned CoTs exhibit lower causal faithfulness, a substantially higher incidence of severe hallucination, and lower pairwise preference than causal-planning CoTs. The study design and detailed analysis are provided in Appendix B.

![](images/cde3bd25b7645c1e4ea16a762370b1767565b0da60e17c31ebe7a41071993a50.jpg)  
Figure 2 Human-rated causal faithfulness comparison. GT-conditioned CoTs receive lower scores than causal-planning CoTs across grounding (GND), absence of hallucination (NO-HALL), specificity (SPEC), causal coherence (COH), and aggregate causal-faithfulness score (CFS).

These results expose a severe supervision-direction mismatch: for post hoc chain-of-thought annotation of trajectory decisions, access to the future trajectory serves as a reasoning shortcut rather than a decision target. We therefore retain trajectories as verifiable targets while excluding them from the premises of causal reasoning: the model must first infer a plan from the scene and only then ground it in an explicit future.

## 3 AD-MCQ: A Verifiable Candidate-Trajectory Benchmark

As shown in Figure 3, AD-MCQ formulates autonomous-driving planning as selecting a future trajectory from a small, scene-specific candidate set. This formulation preserves trajectory-level granularity while replacing open-ended coordinate generation with an exactly verifiable decision.

Task Formulation. Each AD-MCQ instance consists of multi-view scene-history frames $V _ { i } ,$ ego history and current motion state $H _ { i } ,$ a navigation instruction $I _ { i } ,$ and a shufled set of explicit candidate trajectories $\mathcal { A } _ { i } = ( \mathbf { P } _ { i , 1 } , \ldots , \mathbf { P } _ { i , M } )$ . Exactly one candidate corresponds to the quantized logged future, and its shufled position $a _ { i } ^ { \star } \in \{ 1 , \ldots , M \}$ serves as the exactly verifiable target.

![](images/5a36f4fa6b782a896afc620421ba39bc8dbc31654ac2d3c444ed5a100364807f.jpg)  
Figure 3 The framework of AD-MCQ and DEFT-RLVR. AD-MCQ turns explicit trajectory selection into an exactly verifiable decision; DEFT-RLVR defers candidate exposure and combines outcome correctness with rubric supervision of candidate-blind reasoning.

Discrete Trajectory Prototypes. Let a fixed-horizon ego trajectory be $\mathbf { P } = ( \mathbf { p } _ { 1 } , \dots , \mathbf { p } _ { T } ) \in \mathbb { R } ^ { T \times 2 }$ , where $\mathbf { p } _ { t } ~ = ~ ( x _ { t } , y _ { t } )$ denotes longitudinal and lateral displacement from the current ego pose. We flatten each trajectory in a corpus of N logged futures and apply K-means to obtain a codebook $\mathcal { C } = \{ \mathbf { C } _ { 1 } , \dots , \mathbf { C } _ { K } \}$ where each prototype $\mathbf { C } _ { k } = ( \mathbf { c } _ { k , 1 } , \ldots , \mathbf { c } _ { k , T } )$ represents a complete future motion. We quantize P by nearestprototype assignment:

(1)

![](images/351835e005e620f5aa9b9cd7370f41bdb6a94b910bea7e7a2b349a8a8b9c88a7.jpg)  
Figure 4 Trajectory-codebook scaling. Reconstruction error versus (a) codebook size K and (b) clustering-corpus size N; solid/dashed curves indicate in-/out-of-sample trajectories.

To determine an appropriate codebook configuration, we study how the number of clustering trajectories N and prototypes K afect reconstruction fidelity and codebook utilization. Figure 4 shows that K=8192 provides a favorable balance between out-of-sample reconstruction fidelity and codebook utilization. We provide full construction and scaling analyses in Appendix C.

Candidate-trajectory Construction. We measure the distance between two prototypes by

$$
d _ { i j } = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \Vert \mathbf { c } _ { i , t } - \mathbf { c } _ { j , t } \Vert _ { 2 } , \quad \rho _ { i j } = 1 - \frac { d _ { i j } - d _ { \operatorname* { m i n } } } { d _ { \operatorname* { m a x } } - d _ { \operatorname* { m i n } } } ,\tag{2}
$$

where larger $\rho _ { i j }$ indicates more similar decoded trajectories. For each driving scene i, we first map the GT trajectory $\mathbf { P } _ { i } ^ { \mathrm { g t } }$ to its nearest codebook entry $z _ { i } ^ { \star } = z ( \mathbf { P } _ { i } ^ { \mathrm { g t } } )$ . Based on Eq. 2, we define the hard-negative pool

$$
\begin{array} { r } { \mathcal { H } ( z _ { i } ^ { \star } ) = \left\{ z \in [ K ] \setminus \left\{ z _ { i } ^ { \star } \right\} : \rho _ { \operatorname* { m i n } } \leq \rho _ { z , z _ { i } ^ { \star } } \leq \rho _ { \operatorname* { m a x } } \right\} . } \end{array}\tag{3}
$$

We construct split-specific distractors from $\mathcal { H } ( z _ { i } ^ { \star } )$ as detailed in Appendix D and summarized algorithmically in Appendix D.1, combine $M - 1$ distinct negatives with $z _ { i } ^ { \star }$ , and randomly shufle the M candidates. Finally, we decode the candidate indices $( z _ { i , 1 } , \ldots , z _ { i , M } )$ into the waypoint-option set shown to the model:

$$
\begin{array} { r } { \mathcal { A } _ { i } : = ( \mathbf { P } _ { i , m } ) _ { m = 1 } ^ { M } = \left( \mathbf { C } _ { z _ { i , m } } \right) _ { m = 1 } ^ { M } \in ( \mathbb { R } ^ { T \times 2 } ) ^ { M } . } \end{array}\tag{4}
$$

The VLM then selects one shufled option, and its decision is evaluated by the deterministic verifier. Complete instances are provided in Appendix G.

Notably, the codebook ultimately retrieves textual waypoint trajectory candidates [29]. Unlike direct waypoint retrieval, the codebook maps continuous futures to a finite motion vocabulary [43, 44, 67], enabling controlled hard-negative construction without sacrificing explicit geometry.

## 4 DEFT-RLVR: Deferred Exposure of Future Trajectories

Building on AD-MCQ, DEFT-RLVR mitigates trajectory-induced anchoring bias by deferring candidate exposure until after the policy commits to a scene-derived decision, while jointly optimizing the two interaction stages with rubric-based reasoning supervision.

## 4.1 DEFT: Deferred Exposure of Future Trajectories

By deferring candidate exposure, DEFT reserves candidate geometry for grounding an already formed scenederived decision rather than shaping the decision itself.

Specifically, for question i, let $X _ { i } = ( V _ { i } , H _ { i } , I _ { i } )$ denote the scene context and $\mathcal { A } _ { i } = ( \mathbf { P } _ { i , m } ) _ { m = 1 } ^ { M }$ the candidate trajectories with option label set $\mathcal { L } _ { i }$

Turn 1: Causal Decision Reasoning. Conditioned solely on the scene context $X _ { i } ,$ , the causal-reasoning prompt $p _ { 1 }$ (provided in Appendix F.1) elicits causal reasoning over scene evidence before the candidate trajectories are revealed:

$$
u _ { 1 , i } = p _ { 1 } ( X _ { i } ) , \quad y _ { 1 , i } \sim \pi _ { \theta } ( { \cdot } \mid u _ { 1 , i } ) .\tag{5}
$$

The resulting response $y _ { 1 , i }$ explicitly traces how scene evidence leads to driving implications and commits to a complete high-level decision (HLD) before candidate exposure.

Turn 2: Explicit-Trajectory Grounding. Only after this decision has been formed do we reveal $\mathbf { \mathcal { A } } _ { i }$ through the trajectory-matching prompt $p _ { 2 }$ (provided in Appendix F.1), which treats the recorded decision as binding and uses candidate geometry only to identify the closest explicit realization of that decision. With $u _ { 2 , i } = p _ { 2 } ( \mathcal { A } _ { i } )$ we sample:

$$
y _ { 2 , i } \sim \pi _ { \theta } ( \cdot \mid u _ { 1 , i } , y _ { 1 , i } , u _ { 2 , i } ) .\tag{6}
$$

The selected option is $\widehat { a } _ { i } = \mathrm { p a r s e } ( y _ { 2 , i } ) \in \mathcal { L } _ { i } \cup \{ \bot \}$ , where ⊥ denotes an invalid output. This design prevents candidate geometry from conditioning the initial reasoning process while retaining exact trajectory-level verification.

## 4.2 Joint Optimization of the Two-Stage Interaction

Although DEFT separates candidate-free decision formation from trajectory grounding, we optimize them jointly as a single rollout using Group Relative Policy Optimization (GRPO) [14]. For each question i, we sample G two-turn rollouts under the interaction defined in Section 4.1. We serialize rollout $j$ as the complete two-turn sequence $s _ { i , j } = u _ { 1 , i } \oplus y _ { 1 , i , j } \oplus u _ { 2 , i } \oplus y _ { 2 , i , j }$ . During optimization, we perform a single forward pass over $s _ { i , j }$ to compute the token likelihoods used for importance sampling, while masking the prompt tokens so that the policy objective is applied only to the generated tokens in $y _ { 1 , i , j }$ and $y _ { 2 , i , j }$ . Both turns share the rollout reward $R _ { i , j }$ and its group-normalized advantage.

## 4.3 Structured Rubric Rewards for Reasoning-Trace Supervision

AD-MCQ provides a verifiable outcome reward:

$$
R _ { i , j } ^ { \mathrm { M C Q } } = \mathbb { I } [ \widehat { a } _ { i , j } = a _ { i } ^ { \star } ] ,\tag{7}
$$

where $a _ { i } ^ { \star }$ is the oracle option, but this signal alone cannot distinguish grounded reasoning from rationalization.

To prevent reinforcing reasoning trajectories that arrive at the correct answer through shortcut exploitation or random guessing [15], we apply rubric-based reasoning rewards to rollouts with correct MCQ answers. For each answer-correct rollout, we form the normalized Turn-1 reasoning trace $\widetilde { y } _ { 1 , i , j } : = \mathrm { N o r m a l i z e } ( y _ { 1 , i , j } )$ and asynchronously submit $\widetilde { y } _ { 1 , i , j }$ to a text grader for evaluation. Specifically, DEFT-RLVR generates an instancespecific rubric once ofline using a vision-language rubric generator $\mathcal { G }$ conditioned on the fixed generation prompt $p _ { \mathrm { r u b } }$ (provided in Appendix F.2):

$$
\mathcal { C } _ { i } = \mathcal { G } ( p _ { \mathrm { r u b } } ( X _ { i } ) ) .\tag{8}
$$

The resulting rubric $\mathcal { C } _ { i } = \{ ( c _ { i , k } , w _ { i , k } ) \} _ { k = 1 } ^ { K _ { i } }$ contains atomic, positively weighted criteria that explicitly encode scene evidence verified by the ofline generator and is reused across rollouts [13, 48, 49].

During RL rollouts, we prompt a shared VLM judge J with $p _ { \mathrm { t x t } }$ (provided in Appendix F.3) to evaluate each reasoning trace against the rubric criteria set $\mathcal { C } _ { i }$

$$
\mathbf { b } _ { i , j } = \mathcal { I } \big ( p _ { \mathrm { t x t } } ( \mathcal { C } _ { i } , \widetilde { y } _ { 1 , i , j } ) \big ) \in \{ 0 , 1 \} ^ { K _ { i } } ,\tag{9}
$$

where $b _ { i , j , k } = 1$ indicates that the CoT satisfies criterion $c _ { i , k }$ . With $\mathbf { w } _ { i } = ( w _ { i , 1 } , \ldots , w _ { i , K _ { i } } )$ , the rubric reward is:

$$
R _ { i , j } ^ { \mathrm { R U B } } = \frac { \mathbf { w } _ { i } ^ { \top } \mathbf { b } _ { i , j } } { \Vert \mathbf { w } _ { i } \Vert _ { 1 } } \in [ 0 , 1 ] .\tag{10}
$$

The final rollout reward is

$$
R _ { i , j } = R _ { i , j } ^ { \mathrm { M C Q } } R _ { i , j } ^ { \mathrm { R U B } } .\tag{11}
$$

Thus, trajectory correctness determines whether a rollout receives process supervision, while neither the GT trajectory nor the candidate set serves as input to the Turn-1 causal reasoning process. Crucially, $\mathbf { P } _ { i } ^ { \mathrm { g t } }$ is never directly provided to ${ \mathcal { G } } , \pi _ { \theta } \ \circ \ r { \mathcal { I } } .$ . The judge evaluates the reasoning process solely according to the predefined rubric criteria, withoutdirectaccesstothevisualinput,candidateoptions,orGTtrajectory. Compared with directly providing a VLM-based judge with the full visual context, this text-only grading scheme is substantially more eficient. Moreover, it allows the judge to focus on assessing the quality of the textual reasoning, without its attention being diluted by a large number of visual tokens [10, 86].

## 5 Experiments

## 5.1 Experimental Setup

Data and benchmark. We divide scenes from Waymo Open E2E [69] and an internal driving corpus into Train, Dev, and AD-MCQ-500 Test splits, containing 5,000, 100, and 500 scenes, respectively, with K=8192 and $N { = } 4 8 9 , 0 4 2$ . Each visual input $V _ { i }$ contains four frames sampled at 2 Hz from three cameras: front-left, front, and front-right. Train follows the natural scene distribution, whereas Dev and Test focus on causally demanding scenes with structured hard distractors. Dev is curated as the harder of the two evaluation splits. Details are shown in Appendix D.1.

Evaluation. For AD-specific evaluation, we report ACC (AD-MCQ-500 accuracy) and two complementary CoT metrics: CFS (Normalized Causal-Faithfulness Score) and HLD (High-Level-Decision Consistency).

To assess general-capability retention, we use 12 vision-language benchmarks covering four capability groups [9, 77]: basic visual perception (Basic Visual) [54, 60, 74], embodied spatial reasoning (Embodied Spatial) [8, 36, 53, 57], 3D and multi-view reasoning (3D/Multi-View) [28, 37, 61, 75], and referring-expression-based spatial grounding (RefSpatial) [84]. We additionally evaluate cross-domain AD transfer on an external 500- scene nuScenes set. Complete evaluation settings are provided in Appendix D.2.

<table><tr><td rowspan="2">Method</td><td colspan="3">AD-Specific Reasoning</td><td colspan="5">General Visual Capability(%)</td></tr><tr><td>ACC(%) ↑ CFS ↑ HLD ↑</td><td></td><td></td><td>Basic ↑</td><td>Embodied ↑ 3D/MV ↑ RefSpatial ↑</td><td></td><td></td><td>Avg. ↑</td></tr><tr><td>Qwen3-VL-8B-Instruct</td><td>28.1</td><td></td><td></td><td rowspan="2">81.60</td><td rowspan="2">56.63</td><td rowspan="2">42.48</td><td rowspan="2">38.52</td><td rowspan="2">54.81</td></tr><tr><td>+ DEFT</td><td>56.6</td><td>0.431</td><td>0.425</td></tr><tr><td> $+ \ \mathrm { J E F T } + \mathrm { R L V R } \ ( R ^ { \mathrm { M C Q } } )$ </td><td>61.1</td><td>0.428</td><td>0.427</td><td>81.39</td><td>56.67</td><td>41.99</td><td>39.56</td><td>54.90</td></tr><tr><td> $+ \ \mathrm { D E F T } + \mathrm { R L V R } \ ( R ^ { \mathrm { M C Q } } )$ </td><td>76.4</td><td>0.442</td><td>0.462</td><td>81.36</td><td>57.15</td><td>42.80</td><td>44.26</td><td>56.39</td></tr><tr><td> $+ \ \mathrm { D E F T } + \mathrm { R L V R } \ ( R ^ { \mathrm { M C Q } } R ^ { \mathrm { G E N } } )$ </td><td>75.2</td><td>0.580</td><td>0.487</td><td>81.59</td><td>56.31</td><td>42.81</td><td>44.79</td><td>56.37</td></tr><tr><td> $+ \mathrm { \ D E F T - R L V R }$ </td><td>77.9</td><td>0.658</td><td>0.501</td><td>81.33</td><td>57.41</td><td>42.27</td><td>43.35</td><td>56.09</td></tr><tr><td>+ JEFT Distillation</td><td>64.0</td><td>0.620</td><td>0.480</td><td>74.20</td><td>52.80</td><td>39.60</td><td>32.60</td><td>49.80</td></tr><tr><td>+ DEFT Distillation (Plan Only)</td><td>68.2</td><td>0.925</td><td>0.560</td><td>78.00</td><td>53.49</td><td>40.20</td><td>33.46</td><td>51.29</td></tr><tr><td>+ DEFT Distillation (Full Interaction)</td><td>82.4</td><td>0.909</td><td>0.591</td><td>74.94</td><td>54.04</td><td>40.38</td><td>34.56</td><td>50.98</td></tr><tr><td>+ DEFT Distillation (Mixed Targets)</td><td>84.1</td><td>0.934</td><td>0.627</td><td>76.73</td><td>53.51</td><td>40.20</td><td>36.76</td><td>51.80</td></tr><tr><td>Qwen3.5-4B</td><td>34.0</td><td></td><td></td><td>80.31</td><td>53.00</td><td>40.73</td><td>36.19</td><td>52.56</td></tr><tr><td>+ DEFT</td><td>65.6</td><td>0.738</td><td>0.481</td><td rowspan="2">80.23</td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2"></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td> $+ \ \mathrm { J E F T } + \mathrm { R L V R } \ ( R ^ { \mathrm { M C Q } } )$   $+ \mathrm { \ D E F T } + \mathrm { R L V R } \ ( R ^ { \mathrm { M C Q } } )$ </td><td>72.3 79.0</td><td>0.740 0.805</td><td>0.511 0.529</td><td>80.76</td><td>53.34 52.75</td><td>40.16 41.83</td><td>31.32 38.72</td><td>51.26 53.52</td></tr><tr><td> $+ \ \mathrm { D E F T } + \mathrm { R L V R } \ \overset { \cdot } { ( } R ^ { \mathrm { M C Q } } R ^ { \mathrm { G E N } } )$ </td><td>79.4</td><td>0.819</td><td>0.540</td><td>80.41</td><td>53.37</td><td>40.67</td><td>38.62</td><td>53.27</td></tr><tr><td>+ DEFT-RLVR</td><td>82.2</td><td>0.822</td><td>0.582</td><td>80.88</td><td>55.20</td><td>40.94</td><td>35.93</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>53.24</td></tr></table>

Table 2 Main results on AD reasoning and general visual capability. Base models use JEFT for AD evaluation. CFS and HLD are scored by Qwen3.5-397B-A17B, with strong agreement with human annotations demonstrated in Appendix E.

Models. We use Qwen3-VL-8B-Instruct [2] and Qwen3.5-4B [46] as the base models. Qwen3.5-397B-A17B provides supervision targets for distillation, while Qwen3.6-35B-A3B generates instance-specific rubrics ofline and serves as the reasoning-process judge.

Candidate-Exposure Settings. DEFT (Deferred Exposure of Future Trajectories) first elicits a pre-exposure plan and reveals the candidate trajectories only in the subsequent selection turn. Conversely, JEFT (Joint Exposure of Future Trajectories) is the matched ablation that presents the same scene context and candidate trajectories jointly with the same prompts. For a fair comparison, both settings use $T { \mathrm { = } } 1 . 0 ,$ top- $\cdot p { = } 0 . 9 5$ , and the same total token budget of 24,576 tokens, with DEFT capped at 12,000 tokens per turn.

RLVR Variants. JEFT + RLVR (R<sup>MCQ</sup>) and DEFT + RLVR $\pmb { ( R ^ { \mathrm { M C Q } } ) }$ use the same exact-choice reward and difer in candidate-exposure order. DEFT + RLVR (R<sup>MCQ</sup>R<sup>GEN</sup>) uses a shared rubric whose score $R ^ { \mathrm { G E N } }$ is assigned online by a scene-conditioned VLM grader. DEFT-RLVR uses the instance-specific reward $R ^ { \mathrm { M C Q } } R ^ { \mathrm { R U B } }$ defined in Eq. 11.

Distillation Variants. We prompt Qwen3.5-397B-A17B under the corresponding exposure setting and use its responses as supervised fine-tuning targets for the student. JEFT Distillation imitates the single-turn reasoning-and-selection response generated with candidates exposed from the outset. DEFT Distillation (Plan Only) imitates only the Turn-1 plan generated before candidate exposure. DEFT Distillation (Full Interaction) supervises the complete two-turn plan-then-match interaction. DEFT Distillation (Mixed Targets) uses an equal mixture of plan-only and full-interaction targets. Detailed implementations of the RLVR and distillation variants are provided in Appendix D.4.

## 5.2 Candidate-Grounded Training Improves Generalizable AD Reasoning

DEFT mitigates candidate anchoring bias across inference and training. As illustrated in Table 2, Compared with training-free JEFT, DEFT raises ACC from 28.1% to 56.6% on Qwen3-VL-8B and from 34.0% to 65.6% on Qwen3.5-4B. Under the same correctness-only reward, ${ \mathsf { D E F T } } + { \mathsf { R L V R } } ( R ^ { \mathrm { M C Q } } )$ exceeds its matched JEFT + RLVR $\pmb { ( R ^ { \mathrm { M C Q } } ) }$ ablation by 15.3 and 6.7 percentage points on Qwen3-VL-8B and Qwen3.5-4B, respectively.

Across both backbones, DEFT-RLVR jointly improves MCQ accuracy, CFS, and HLD over training-free DEFT, demonstrating gains in reasoning quality and decision consistency rather than final-choice accuracy alone. Under equal-data distillation, DEFT Distillation (Full Interaction) outperforms JEFT Distillation, raising MCQ accuracy from 64.0% to 82.4%, while improving CFS by 0.289 and HLD by 0.111.

These consistent performance gaps can be attributed to the same information-order mechanism: when futuretrajectory candidates are visible during decision formation, the policy may organize the reasoning around a favored answer, creating a shortcut consistent with the anchoring bias shown in Section 2. DEFT prevents this shortcut by requiring the model to derive its decision from scene evidence before grounding it in a concrete trajectory.

Fine-grained trajectory grounding strengthens generalizable AD reasoning. Compared with the Plan Only variant, DEFT Distillation (Mixed Targets) incorporates full two-turn targets and raises accuracy from 68.2% to 84.1%, CFS from 0.925 to 0.934, and HLD from 0.560 to 0.627. These simultaneous gains show that Turn-2 trajectory grounding is more than a mechanism for providing RL with an exact, verifiable reward: the required fine-grained discrimination among trajectory candidates also improves reasoning quality and trajectory-decision accuracy.

Candidate-grounded training delivers AD reasoning gains that generalize to an OOD driving domain. On the out-of-distribution (OOD) nuScenes domain [4], both DEFT training variants still significantly outperform training-free DEFT on all three AD metrics. As shown in Table 3, DEFT-RLVR raises candidate accuracy from 39.6% to 49.5%, while improving CFS by 0.114 and HLD by 0.073.

Together, these gains indicate that our training paradigm helps the policy learn transferable scene-to-decision reasoning and subsequent explicit-trajectory grounding, rather than rely on source-specific visual cues.

VisualEmbodied3D/MVRefGround
<table><tr><td>Method</td><td>ACC ↑</td><td>CFS ↑</td><td>HLD ↑</td></tr><tr><td>DEFT (Training-Free)</td><td>39.6</td><td>0.522</td><td>0.286</td></tr><tr><td>DEFT Distillation (Mixed Targets)</td><td>55.8</td><td>0.655</td><td>0.370</td></tr><tr><td>DEFT-RLVR</td><td>49.5</td><td>0.636</td><td>0.359</td></tr></table>

Table 3 Cross-domain results on 500 nuScenes scenes using Qwen3-VL-8B-Instruct. Both candidate-grounded training variants outperform training-free DEFT across accuracy (ACC), causal-faithfulness score (CFS), and high-level decision consistency (HLD), demonstrating that the learned scene-to-decision reasoning transfers beyond the training domain. Mixed-target distillation achieves the strongest overall results, while DEFT-RLVR also delivers consistent gains using verifiable reward supervision.

![](images/5108f74f352b902c50ca55a7be2f3a97848d1516674c792f60bb96cf0354d54f.jpg)

![](images/b27a31bdce284941f058975fe4783ba57ea3d524ed53f42a39799d56bc6f414e.jpg)  
Figure 5 Cold-start SFT trades general capability for AD specialization. (a) Hard-100 Dev accuracy. (b) First-epoch performance changes across four general-capability groups relative to the Base VLM.

![](images/740bd1f313916e9a6fcd87a2a22d90ff7b6011b052968d6f5ed1812442030e8d.jpg)  
Figure 6 Training dynamics of the Qwen3-VL-8B-Instruct RLVR variants. From left to right, the panels report development-set accuracy, response length, actor entropy, and actor KL loss. Candidate-visible JEFT exhibits by far the largest policy drift while remaining the least accurate. Deferred exposure substantially improves accuracy, and rubric-supervised DEFT-RLVR attains the strongest late-stage performance while keeping responses shorter and entropy lower than correctness-only DEFT, indicating more controlled and productive exploration.

## 5.3 RLVR Improves AD Reasoning without Sacrificing General Visual Capability

Distillation improves AD specialization at the cost of general capability. As shown in Table 2, JEFT Distillation reduces the average of general visual capability from 54.81% to 49.80%, while the equal-data DEFT variants retain 50.98%–51.80%. This is because token-level SFT supervision pushes the student toward an AD-specific response distribution generated by an external teacher rather than selectively reinforcing correct behavior. Although DEFT-based distillation mitigates this policy shift relative to shortcut-prone JEFT distillation, dense teacher imitation still trades general capability for AD performance.

Cold-start SFT causes an early decline in general visual capability. To assess whether RLVR should start from an AD-specialized policy, we first apply SFT to teacher-generated responses as a cold-start stage, with experimental details shown in Appendix D.5. As shown in Figure 5, although cold-start SFT improves Dev accuracy, all four general-capability groups decline from the first epoch. Initializing RL from this policy would additionally anchor KL regularization to an already shifted policy. Thus, we start DEFT-RLVR from the unmodified Base VLM.

DEFT-RLVR improves AD reasoning while preserving general visual capability. As illustrated in Table 2, DEFT-RLVR raises the average of general visual capability from 54.81% to 56.09% on Qwen3-VL-8B and from 52.56% to 53.24% on Qwen3.5-4B. Unlike SFT, RL-based variants learn from responses sampled from the current or a recent policy. The resulting policy gradients merely increase or decrease the probability of each sampled token conditioned on its corresponding context [88], thereby constituting a more fine-grained form of policy optimization than SFT [11]. Meanwhile, the causal reasoning process partially exercises visualspatial reasoning shared with the general benchmarks, which may explain the modest gains of general visual capability. The evaluation results of all 12 benchmarks are provided in Appendix H.

## 5.4 Ablation of the RLVR Design

JEFT + RLVR $\pmb { ( R ^ { \mathrm { M C Q } } ) }$ vs. DEFT + RLVR $\mathbf { \Gamma } ( R ^ { \mathrm { M C Q } } ) { \mathrm { : } }$ deferred exposure avoids shortcut-driven optimization. Figure 6 shows that the JEFT variant exhibits substantially larger policy drift while remaining less accurate than DEFT variants. As shown in Table 2, DEFT improves accuracy from 61.1% to 76.4% on Qwen3-VL-8B and from 72.3% to 79.0% on Qwen3.5-4B, while also achieving higher CFS and HLD. These results suggest that deferred exposure efectively reduces candidate-visible shortcuts and promotes more efective scene-derived reasoning.

DEFT+RLVR (R<sup>MCQ</sup>)vs.DEFT-RLVR: rubric supervisionprunesunproductive exploration. As shown in Figure 6, DEFT + RLVR $( R ^ { \mathrm { M C Q } } )$ produces the longer and higher-entropy responses than rubric-supervised variants. Meanwhile, Table 2 shows that the introduction of rubric supervision consistently enhances the reasoning and

<table><tr><td>Method</td><td>Step ↓</td><td>Rollout↓</td><td>Scoring ↓</td></tr><tr><td> $\overline { { \mathsf { D E F T } + \mathsf { R L V R } ( R ^ { \mathrm { M C Q } } ) } }$ </td><td>424.5</td><td>171.1</td><td>0.03</td></tr><tr><td> $\mathsf { D E F T } + \mathsf { R L V R } ( R ^ { \mathrm { M C Q } } R ^ { \mathrm { G E N } } )$ </td><td>724.4</td><td>489.8</td><td>156.8</td></tr><tr><td> $\mathsf { D E F T - R L V R }$ </td><td>426.5</td><td>174.6</td><td>4.12</td></tr></table>

Table 4 Per-step runtime of Qwen3-VL-8B-Instruct DEFT RLVR variants in seconds. DEFT-RLVR adds only 0.5% overhead over correctness-only training while being 41.1% faster than the online-rubric variant, demonstrating its superior eficiency for rubric-supervised optimization.

![](images/d587ad06274bad2dfeb5f5b7c2fd9af5ca896c0cfbd10521f861fd4be0360d8b.jpg)  
Figure 7 Direct trajectory-token generation introduces coupled prediction and capability-retention bottlenecks. $^ { ( \mathrm { a } , \mathrm { b } ) }$ Despite increasing fit under ${ \mathrm { S F T } } ,$ both in-distribution and out-of-distribution trajectory-prediction ADEs remain well above the codebook reconstruction floor of 0.279 m, indicating that most of the error arises from token inference rather than trajectory quantization. (c) Direct token supervision also substantially degrades general visual capability, and incorporating CoT does not prevent this degradation. Together, these results motivate externalizing planning as selection over explicit trajectory candidates rather than internalizing a large trajectory-token vocabulary.

HLD quality on both backbones. This improvement can be attributed to the rubric’s fine-grained supervision, which efectively steers the reasoning process toward greater faithfulness while suppressing unproductive exploration.

DEFT-RLVR vs. $\mathsf { D E F T } + \mathsf { R L V R } ( R ^ { \mathrm { M C Q } } R ^ { \mathrm { G E N } } ) { \mathrm { : } }$ stronger reasoning capacity with marginal time cost. As shown in Table 4, compared with DEFT + RLVR $( R ^ { \mathrm { M C Q } } )$ , DEFT-RLVR substantially improves reasoning quality while introducing marginal training cost, increasing total step time by only 0.5%, from 424.5 to 426.5 seconds. DEFT-RLVR likewise achieves higher reasoning quality than the online-rubric variant, DEFT + RLVR $( R ^ { \mathrm { M C Q } } R ^ { \mathrm { G E N } } )$ , despite incurring substantially lower training costs. Specifically, by constructing instancespecific criteria ofline and retaining only text-based grading online, DEFT-RLVR reduces total step time from 724.4 to 426.5 seconds (41.1% faster).

## 5.5 Why Formulate AD Planning as a Candidate-Grounded MCQ?

Candidate grounding avoids the dual bottleneck of trajectory error and general-capability degradation. Using a shared codebook, we fine-tune Qwen3-VL-8B-Instruct via SFT to predict trajectory tokens with or without trajectory-conditioned CoT; full details are provided in Appendix D.6. Figure 7(a,b) shows that, even after SFT begins to overfit, both in- and out-of-distribution prediction ADEs remain substantially above the codebook’s 0.279 m reconstruction ADE, identifying token generation as the primary error source. Figure 7(c) further indicates that direct trajectory generation severely degrades general visual capability even with CoT. Both failures arise because training forces the VLM to internalize a large trajectory-token inventory within its original vocabulary, substantially perturbing the pretrained token distribution. AD-MCQ instead uses the codebook only to retrieve waypoint candidates and externalizes generation as selection among scene conditioned explicit trajectories, thereby avoiding both bottlenecks.

The gains of DEFT-RLVR generalize across diverse MCQ option constructions. Holding scenes and oracle trajectories fixed, we randomly resample distractors across five candidate counts and four hard-negative similarity bounds. Figure 8 shows that DEFT-RLVR outperforms training-free DEFT in all 20 settings. Thus, the capability learned by DEFT-RLVR under a fixed MCQ configuration transfers to new candidate-set constructions rather than relying on a particular distractor geometry.

![](images/985730efe31403178b5db22ba073ea97f577c09226f586d374c710664a0737bd.jpg)  
Figure 8 Robustness to candidate-set construction. Holding scenes and oracle trajectories fixed, we resample distractors across five candidate-set sizes and four hard-negative similarity bounds. DEFT-RLVR consistently outperforms training-free DEFT in all 20 configurations, indicating that its gains transfer across candidate constructions rather than depending on a fixed distractor geometry.

Meanwhile, highly similar future trajectories and larger candidate sets remain the most challenging regimes for fine-grained candidate grounding. Through controlled option construction, AD-MCQ thus provides a simple yet dificulty-controllable experimental paradigm for future research. Appendix D.7 provides further experimental details.

## 6 Conclusion

We identify anchoring bias in AD VLMs, propose AD-MCQ and leverage DEFT-RLVR for training. This framework improves generalizable AD reasoning while preserving and even enhancing the model’s general visual capabilities. Since AD-MCQ relies solely on the VLM and allows dificulty to be controlled through option construction, it provides a highly deployable and scalable foundation for future research.

## References

[1] Iván Arcuschin, Jett Janiak, Robert Krzyzanowski, Senthooran Rajamanoharan, Neel Nanda, and Arthur Conmy. Chain-of-thought reasoning in the wild is not always faithful. arXiv preprint arXiv:2503.08679, 2025.

[2] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025.

[3] Sriram Balasubramanian, Samyadeep Basu, and Soheil Feizi. A closer look at bias and chain-of-thought faithfulness of large (vision) language models. In Christos Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng, editors, Findings of the Association for Computational Linguistics: EMNLP 2025, pages 13406– 13439, Suzhou, China, November 2025. Association for Computational Linguistics. ISBN 979-8-89176-335-7. doi: 10.18653/v1/2025.findings-emnlp.723. https://aclanthology.org/2025.findings-emnlp.723/.

[4] Holger Caesar, Varun Bankiti, Alex H Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Giancarlo Baldan, and Oscar Beijbom. nuscenes: A multimodal dataset for autonomous driving. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11621–11631, 2020.

[5] Nikhil Chandak, Shashwat Goel, Ameya Prabhu, Moritz Hardt, and Jonas Geiping. Answer matching outperforms multiple choice for language model evaluation. arXiv preprint arXiv:2507.02856, 2025. doi: 10.48550/arXiv.2507. 02856.

[6] Yanda Chen, Joe Benton, Ansh Radhakrishnan, Jonathan Uesato, Carson Denison, John Schulman, Arushi Somani, Peter Hase, Misha Wagner, Fabien Roger, et al. Reasoning models don’t always say what they think. arXiv preprint arXiv:2505.05410, 2025.

[7] Kashyap Chitta, Aditya Prakash, Bernhard Jaeger, Zehao Yu, Katrin Renz, and Andreas Geiger. Transfuser: Imitation with transformer-based sensor fusion for autonomous driving. IEEE transactions on pattern analysis and machine intelligence, 45(11):12878–12895, 2022.

[8] Mengfei Du, Binhao Wu, Zejun Li, Xuan-Jing Huang, and Zhongyu Wei. Embspatial-bench: Benchmarking spatial understanding for embodied tasks with large vision-language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pages 346–355, 2024.

[9] Haodong Duan, Xinyu Fang, Junming Yang, Xiangyu Zhao, Zerun Ma, Yuxuan Qiao, Mo Li, Tianhao Liang, Lin Zhu, Amit Agarwal, et al. Vlmevalkit: An open-source toolkit for evaluating large multi-modality models. arXiv preprint arXiv:2407.11691, 2024.

[10] Senyu Fei, Siyin Wang, Junhao Shi, Zihao Dai, Jikun Cai, Pengfang Qian, Li Ji, Xinzhe He, Shiduo Zhang, Zhaoye Fei, et al. Libero-plus: In-depth robustness analysis of vision-language-action models. arXiv preprint arXiv:2510.13626, 2025.

[11] Yuqian Fu, Tinghong Chen, Jiajun Chai, Xihuai Wang, Songjun Tu, Guojun Yin, Wei Lin, Qichao Zhang, Yuanheng Zhu, and Dongbin Zhao. Srft: A single-stage method with supervised and reinforcement fine-tuning for reasoning. arXiv preprint arXiv:2506.19767, 2025.

[12] Yi Gu, Yan Wang, Yuxiao Chen, Yurong You, Wenjie Luo, Yue Wang, Wenhao Ding, Boyi Li, Heng Yang, Boris Ivanovic, et al. Accelerating structured chain-of-thought in autonomous vehicles. arXiv preprint arXiv:2602.02864, 2026.

[13] Anisha Gunjal, Anthony Wang, Elaine Lau, Vaskar Nath, Bing Liu, and Sean M. Hendryx. Rubrics as rewards: Reinforcement learning beyond verifiable domains. arXiv preprint arXiv:2507.17746, 2025.

[14] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

[15] Xu Guo, Qiming Ge, Jian Tong, Kedi Chen, Jin Zhang, Xiaogui Yang, Xuan Gao, Haijun Lv, Zhihui Lu, Yicheng Zou, et al. Rethinking multiple-choice questions for rlvr: Unlocking potential via distractor design. In Findings of the Association for Computational Linguistics: ACL 2026, pages 20092–20113, 2026.

[16] Helia Hashemi, Jason Eisner, Corby Rosset, Benjamin Van Durme, and Chris Kedzie. Llm-rubric: A multidimensional, calibrated approach to automated evaluation of natural language texts. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 13806–13834, 2024.

[17] Yihan Hu, Jiazhi Yang, Li Chen, Keyu Li, Chonghao Sima, Xizhou Zhu, Siqi Chai, Senyao Du, Tianwei Lin, Wenhai Wang, et al. Planning-oriented autonomous driving. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 17853–17862, 2023.

[18] Zhiyu Huang, Haochen Liu, and Chen Lv. Gameformer: Game-theoretic modeling and learning of transformerbased interactive prediction and planning for autonomous driving. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3903–3913, 2023.

[19] Zhiyu Huang, Haochen Liu, Jingda Wu, and Chen Lv. Diferentiable integrated motion prediction and planning with learnable cost function for autonomous driving. IEEE transactions on neural networks and learning systems, 2023.

[20] Zhiyu Huang, Xinshuo Weng, Maximilian Igl, Yuxiao Chen, Yulong Cao, Boris Ivanovic, Marco Pavone, and Chen Lv. Gen-drive: Enhancing difusion generative driving policies with reward modeling and reinforcement learning fine-tuning. arXiv preprint arXiv:2410.05582, 2024.

[21] Zixuan Huang, Yikun Ban, Lean Fu, Xiaojie Li, Zhongxiang Dai, Jianxin Li, and Deqing Wang. Adaptive sample scheduling for direct preference optimization. arXiv preprint arXiv:2506.17252, 2025.

[22] Zixuan Huang, Xin Xia, Yuxi Ren, Jianbin Zheng, Xuanda Wang, Zhixia Zhang, Hongyan Xie, Songshi Liang, Zehao Chen, Xuefeng Xiao, et al. Does your reasoning model implicitly know when to stop thinking? arXiv preprint arXiv:2602.08354, 2026.

[23] Zixuan Huang, Xin Xia, Yuxi Ren, Jianbin Zheng, Xuefeng Xiao, Hongyan Xie, Huaqiu Li, Songshi Liang, Zhongxiang Dai, Fuzhen Zhuang, Jianxin Li, Yikun Ban, and Deqing Wang. Real-time aligned reward model beyond semantics. 2026. https://api.semanticscholar.org/CorpusID:285240754.

[24] Jyh-Jing Hwang, Runsheng Xu, Hubert Lin, Wei-Chih Hung, Jingwei Ji, Kristy Choi, Di Huang, Tong He, Paul Covington, Benjamin Sapp, et al. Emma: End-to-end multimodal model for autonomous driving. arXiv preprint arXiv:2410.23262, 2024.

[25] Ayesha Ishaq, Jean Lahoud, Ketan More, Omkar Thawakar, Ritesh Thawkar, Dinura Dissanayake, Noor Ahsan, Yuhao Li, Fahad Shahbaz Khan, Hisham Cholakkal, et al. Drivelmm-o1: A step-by-step reasoning dataset and large multimodal model for driving scenario understanding. In 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 20501–20508. IEEE, 2025.

[26] Bo Jiang, Shaoyu Chen, Qing Xu, Bencheng Liao, Jiajie Chen, Helong Zhou, Qian Zhang, Wenyu Liu, Chang Huang, and Xinggang Wang. Vad: Vectorized scene representation for eficient autonomous driving. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 8340–8350, 2023.

[27] Bo Jiang, Shaoyu Chen, Bencheng Liao, Xingyu Zhang, Wei Yin, Qian Zhang, Chang Huang, Wenyu Liu, and Xinggang Wang. Senna: Bridging large vision-language models and end-to-end autonomous driving. arXiv preprint arXiv:2410.22313, 2024.

[28] Dingming Li, Hongxing Li, Zixuan Wang, Yuchen Yan, Hang Zhang, Siqi Chen, Guiyang Hou, Shengpei Jiang, Wenqi Zhang, Yongliang Shen, et al. Viewspatial-bench: Evaluating multi-perspective spatial localization in vision-language models. arXiv preprint arXiv:2505.21500, 2025.

[29] Pengxiang Li, Yinan Zheng, Yue Wang, Huimin Wang, Hang Zhao, Jingjing Liu, Xianyuan Zhan, Kun Zhan, and Xianpeng Lang. Discrete difusion for reflective vision-language-action models in autonomous driving. arXiv preprint arXiv:2509.20109, 2025.

[30] Yiheng Li, Cunxin Fan, Chongjian Ge, Zhihao Zhao, Chenran Li, Chenfeng Xu, Huaxiu Yao, Masayoshi Tomizuka, Bolei Zhou, Chen Tang, Mingyu Ding, and Wei Zhan. Womd-reasoning: A large-scale dataset for interaction reasoning in driving, 2025. https://arxiv.org/abs/2407.04281.

[31] Bencheng Liao, Shaoyu Chen, Haoran Yin, Bo Jiang, Cheng Wang, Sixu Yan, Xinbang Zhang, Xiangyu Li, Ying Zhang, Qian Zhang, et al. Difusiondrive: Truncated difusion model for end-to-end autonomous driving. arXiv preprint arXiv:2411.15139, 2024.

[32] Weiting Liu, Jieyi Bi, Wanqi Zhou, Jianfeng Feng, Yining Ma, Ai Han, and Wenlian Lu. Toolanchor: Anchoring counterfactual context to boost agentic tool-use capability. arXiv preprint arXiv:2607.14145, 2026.

[33] Weiting Liu, Han Wu, Yufei Kuang, Xiongwei Han, Tao Zhong, Jianfeng Feng, and Wenlian Lu. Automated optimization modeling via a localizable error-driven perspective. arXiv preprint arXiv:2602.11164, 2026.

[34] Hongbo Ma, Fei Shen, Hongbin Xu, Xiaoce Wang, Gang Xu, Jinkai Zheng, Liangqiong Qu, and Ming Li. Styletailor: Towards personalized fashion styling via hierarchical negative feedback, 2025. https://arxiv.org/abs/2508. 06555.

[35] Weijian Ma, Ruoxin Chen, Keyue Zhang, Shuang Wu, and Shouhong Ding. Instruct where the model fails: Generative data augmentation via guided self-contrastive fine-tuning. Proceedings of the AAAI Conference on Artificial Intelligence, 39(6):5991–5999, Apr. 2025. doi: 10.1609/aaai.v39i6.32640. https://ojs.aaai.org/index. php/AAAI/article/view/32640.

[36] Weijian Ma, Shizhao Sun, Tianyu Yu, Ruiyu Wang, Tat-Seng Chua, and Jiang Bian. Thinking with blueprints: Assisting vision-language models in spatial reasoning via structured object representation, 2026. https://arxiv. org/abs/2601.01984.

[37] Wufei Ma, Haoyu Chen, Guofeng Zhang, Yu-Cheng Chou, Jieneng Chen, Celso de Melo, and Alan Yuille. 3dsrbench: A comprehensive 3d spatial reasoning benchmark. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 6924–6934, 2025.

[38] Jiageng Mao, Yuxi Qian, Junjie Ye, Hang Zhao, and Yue Wang. Gpt-driver: Learning to drive with gpt. arXiv preprint arXiv:2310.01415, 2023.

[39] Ana-Maria Marcu, Long Chen, Jan Hünermann, Alice Karnsund, Benoit Hanotte, Prajwal Chidananda, Saurabh Nair, Vijay Badrinarayanan, Alex Kendall, Jamie Shotton, et al. Lingoqa: Visual question answering for autonomous driving. In European Conference on Computer Vision, pages 252–269. Springer, 2024.

[40] Ming Nie, Renyuan Peng, Chunwei Wang, Xinyue Cai, Jianhua Han, Hang Xu, and Li Zhang. Reason2drive: Towards interpretable and chain-based reasoning for autonomous driving. In European Conference on Computer Vision, pages 292–308. Springer, 2024.

[41] Long Ouyang, Jefrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744, 2022.

[42] Sung-Yeon Park, Can Cui, Yunsheng Ma, Ahmadreza Moradipari, Rohit Gupta, Kyungtae Han, and Ziran Wang. Nuplanqa: A large-scale dataset and benchmark for multi-view driving scene understanding in multi-modal large language models. arXiv preprint arXiv:2503.12772, 2025.

[43] Karl Pertsch, Kyle Stachowicz, Brian Ichter, Danny Driess, Suraj Nair, Quan Vuong, Oier Mees, Chelsea Finn, and Sergey Levine. Fast: Eficient action tokenization for vision-language-action models. arXiv preprint arXiv:2501.09747, 2025.

[44] Jonah Philion, Xue Bin Peng, and Sanja Fidler. Trajeglish: Trafic modeling as next-token prediction. arXiv preprint arXiv:2312.04535, 2023.

[45] Tianwen Qian, Jingjing Chen, Linhai Zhuo, Yang Jiao, and Yu-Gang Jiang. Nuscenes-qa: A multi-modal visual question answering benchmark for autonomous driving scenario. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 4542–4550, 2024.

[46] Qwen Team. Qwen3.5: Towards native multimodal agents, February 2026. https://qwen.ai/blog?id=qwen3.5.

[47] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36:53728–53741, 2023.

[48] Delip Rao and Chris Callison-Burch. Autorubric: Unifying rubric-based llm evaluation. arXiv preprint arXiv:2603.00077, 2026.

[49] MohammadHossein Rezaei, Robert Vacareanu, Zihao Wang, Clinton Wang, Bing Liu, Yunzhong He, and Afra Feyza Akyürek. Online rubrics elicitation from pairwise comparisons. arXiv preprint arXiv:2510.07284, 2025.

[50] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

[51] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

[52] Chonghao Sima, Katrin Renz, Kashyap Chitta, Li Chen, Hanxue Zhang, Chengen Xie, Jens Beißwenger, Ping Luo, Andreas Geiger, and Hongyang Li. Drivelm: Driving with graph visual question answering. In European Conference on Computer Vision, pages 256–274. Springer, 2024.

[53] Chan Hee Song, Valts Blukis, Jonathan Tremblay, Stephen Tyree, Yu Su, and Stan Birchfield. Robospatial: Teaching spatial understanding to 2d and 3d vision-language models for robotics. arXiv preprint arXiv:2411.16537, 2024.

[54] Haoran Sun, Bingyang Wang, Suyang Yu, Yijiang Li, Qingying Gao, Haiyun Lyu, Lianyu Huang, Zelong Hong, Jiahui Ge, Qianli Ma, Hang He, Yifan Zhou, Lingzi Guo, Lantao Mei, Maijunxian Wang, Dezhi Luo, and Hokin Deng. Probing perceptual constancy in large vision-language models, 2026. https://arxiv.org/abs/2502.10273. ES-Reasoning Workshop at ICLR 2026.

[55] Wenchao Sun, Xuewu Lin, Yining Shi, Chuang Zhang, Haoran Wu, and Sifa Zheng. Sparsedrive: End-to-end autonomous driving via sparse scene representation. arXiv preprint arXiv:2405.19620, 2024.

[56] Tianyi Tan, Yinan Zheng, Ruiming Liang, Zexu Wang, Kexin Zheng, Jinliang Zheng, Jianxiong Li, Xianyuan Zhan, and Jingjing Liu. Flow matching-based autonomous driving planning with advanced interactive behavior modeling. Advances in Neural Information Processing Systems, 38:38310–38335, 2025.

[57] Gemini Robotics Team, Saminda Abeyruwan, Joshua Ainslie, Jean-Baptiste Alayrac, Montserrat Gonzalez Arenas, Travis Armstrong, Ashwin Balakrishna, Robert Baruch, Maria Bauza, Michiel Blokzijl, et al. Gemini robotics: Bringing ai into the physical world. arXiv preprint arXiv:2503.20020, 2025.

[58] Kexin Tian, Jingrui Mao, Yunlong Zhang, Jiwan Jiang, Yang Zhou, and Zhengzhong Tu. Nuscenes-spatialqa: A spatial understanding and reasoning benchmark for vision-language models in autonomous driving. arXiv preprint arXiv:2504.03164, 2025.

[59] Xiaoyu Tian, Junru Gu, Bailin Li, Yicheng Liu, Yang Wang, Zhiyong Zhao, Kun Zhan, Peng Jia, Xianpeng Lang, and Hang Zhao. Drivevlm: The convergence of autonomous driving and large vision-language models. arXiv preprint arXiv:2402.12289, 2024.

[60] Shengbang Tong, Ellis L Brown II, Penghao Wu, Sanghyun Woo, Adithya Jairam Iyer, Sai Charitha Akula, Shusheng Yang, Jihan Yang, Manoj Middepogu, Ziteng Wang, et al. Cambrian-1: A fully open, vision-centric exploration of multimodal llms. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024.

[61] Maijunxian Wang, Ruisi Wang, Juyi Lin, Ran Ji, Thaddäus Wiedemer, Qingying Gao, Dezhi Luo, Yaoyao Qian, Lianyu Huang, Zelong Hong, Jiahui Ge, Qianli Ma, Hang He, et al. A very big video reasoning suite. In Proceedings of the 43rd International Conference on Machine Learning, 2026. https://openreview.net/forum? id=AwC77yHpP6.

[62] Tianqi Wang, Enze Xie, Ruihang Chu, Zhenguo Li, and Ping Luo. Drivecot: Integrating chain-of-thought reasoning with end-to-end driving. arXiv preprint arXiv:2403.16996, 2024.

[63] Wenhai Wang, Jiangwei Xie, ChuanYang Hu, Haoming Zou, Jianan Fan, Wenwen Tong, Yang Wen, Silei Wu, Hanming Deng, Zhiqi Li, et al. Drivemlm: Aligning multi-modal large language models with behavioral planning states for autonomous driving. arXiv preprint arXiv:2312.09245, 2023.

[64] Zhaoyang Wei, Chenhui Qiang, Bowen Jiang, Xumeng Han, Xuehui Yu, and Zhenjun Han. Adˆ 2-bench: A hierarchical cot benchmark for mllm in autonomous driving under adverse conditions. arXiv preprint arXiv:2506.09557, 2025.

[65] Di Wu, Xin Lu, Yanyan Zhao, and Bing Qin. Separate the wheat from the chaf: A post-hoc approach to safety re-alignment for fine-tuned language models. arXiv preprint arXiv:2412.11041, 2024.

[66] Di Wu, Yanyan Zhao, Xin Lu, Mingzhe Li, and Bing Qin. Star-s: Improving safety alignment through self-taught reasoning on safety rules. arXiv preprint arXiv:2601.03537, 2026.

[67] Wei Wu, Xiaoxin Feng, Ziyan Gao, and Yuheng Kan. Smart: scalable multi-agent real-time motion generation via next-token prediction. Advances in Neural Information Processing Systems, 37:114048–114071, 2024.

[68] Rongwu Xu, Zehan Qi, and Wei Xu. Preemptive answer “attacks” on chain-of-thought reasoning. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar, editors, Findings of the Association for Computational Linguistics: ACL 2024, pages 14708–14726, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.findings-acl.876. https://aclanthology.org/2024.findings-acl.876/.

[69] Runsheng Xu, Hubert Lin, Wonseok Jeon, Hao Feng, Yuliang Zou, Liting Sun, John Gorman, Kate Tolstaya, Sarah Tang, Brandyn White, et al. Wod-e2e: Waymo open dataset for end-to-end driving in challenging long-tail scenarios. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3709–3718, 2026.

[70] Zhenhua Xu, Yujia Zhang, Enze Xie, Zhen Zhao, Yong Guo, Kwan-Yee K Wong, Zhenguo Li, and Hengshuang Zhao. Drivegpt4: Interpretable end-to-end autonomous driving via large language model. IEEE Robotics and Automation Letters, 2024.

[71] Zhongxing Xu, Chengzhi Liu, Qingyue Wei, Juncheng Wu, James Zou, Xin Wang, Yuyin Zhou, and Sheng Liu. More thinking, less seeing? assessing amplified hallucination in multimodal reasoning models. Advances in Neural Information Processing Systems, 38:82878–82905, 2025.

[72] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

[73] Bangji Yang, Hongbo Ma, Jiajun Fan, and Ge Liu. Batched contextual reinforcement. In Forty-third International Conference on Machine Learning, 2026. https://openreview.net/forum?id=8Oc3Mx754M.

[74] Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything v2. Advances in Neural Information Processing Systems, 37:21875–21911, 2024.

[75] Sihan Yang, Runsen Xu, Yiman Xie, Sizhe Yang, Mo Li, Jingli Lin, Chenming Zhu, Xiaochen Chen, Haodong Duan, Xiangyu Yue, et al. Mmsi-bench: A benchmark for multi-image spatial intelligence. arXiv preprint arXiv:2505.23764, 2025.

[76] Taha Yasseri and Jannie Reher. Fooled by facts: quantifying anchoring bias through a large-scale experiment. Journal of Computational Social Science, 5(1):1001–1021, 2022.

[77] Haorui Yu, Diji Yang, Hang He, Fengrui Zhang, and Qiufeng Yi. Vulca-bench: A multicultural vision-language benchmark for evaluating cultural understanding, 2026. https://arxiv.org/abs/2601.07986.

[78] Chuhuai Yue, Chengqi Dong, Yinan Gao, Hang He, Jiajun Chai, Wei Lin, and Guojun Yin. Promoting eficient reasoning with verifiable stepwise reward. Proceedings of the AAAI Conference on Artificial Intelligence, 40(41): 34530–34538, 2026. doi: 10.1609/aaai.v40i41.40752. https://doi.org/10.1609/aaai.v40i41.40752.

[79] Songyan Zhang, Wenhui Huang, Zihui Gao, Hao Chen, and Chen Lv. Wisead: Knowledge augmented end-to-end autonomous driving with vision-language model. arXiv preprint arXiv:2412.09951, 2024.

[80] Zhejun Zhang, Peter Karkus, Maximilian Igl, Wenhao Ding, Yuxiao Chen, Boris Ivanovic, and Marco Pavone. Closed-loop supervised fine-tuning of tokenized trafic models. arXiv preprint arXiv:2412.05334, 2024.

[81] Zhixia Zhang, Zixuan Huang, Xin Xia, Deqing Wang, Fuzhen Zhuang, Shuai Ma, Ning Ding, Yaodong Yang, Jianxin Li, and Yikun Ban. Heterogeneous agent collaborative reinforcement learning. arXiv preprint arXiv:2603.02604, 2026.

[82] Qingqing Zhao, Yao Lu, Moo Jin Kim, Zipeng Fu, Zhuoyang Zhang, Yecheng Wu, Zhaoshuo Li, Qianli Ma, Song Han, Chelsea Finn, et al. Cot-vla: Visual chain-of-thought reasoning for vision-language-action models. arXiv preprint arXiv:2503.22020, 2025.

[83] Wenzhao Zheng, Ruiqi Song, Xianda Guo, Chenming Zhang, and Long Chen. Genad: Generative end-to-end autonomous driving. In European Conference on Computer Vision, pages 87–104. Springer, 2024.

[84] Enshen Zhou, Jingkun An, Cheng Chi, Yi Han, Shanyu Rong, Chi Zhang, Pengwei Wang, Zhongyuan Wang, Tiejun Huang, Lu Sheng, et al. Roborefer: Towards spatial referring with reasoning in vision-language models for robotics. Advances in Neural Information Processing Systems, 38:28404–28481, 2026.

[85] Xingcheng Zhou, Xuyuan Han, Feng Yang, Yunpu Ma, and Alois C Knoll. Opendrivevla: Towards end-to-end autonomous driving with large vision language action model. arXiv preprint arXiv:2503.23463, 2025.

[86] Xueyang Zhou, Yangming Xu, Guiyao Tie, Yongchao Chen, Guowen Zhang, Duanfeng Chu, Pan Zhou, and Lichao Sun. Libero-pro: Towards robust and fair evaluation of vision-language-action models beyond memorization. arXiv preprint arXiv:2510.03827, 2025.

[87] Zewei Zhou, Tianhui Cai, Seth Zhao, Yun Zhang, Zhiyu Huang, Bolei Zhou, and Jiaqi Ma. Autovla: A visionlanguage-action model for end-to-end autonomous driving with adaptive reasoning and reinforcement fine-tuning. Advances in Neural Information Processing Systems, 38:27920–27956, 2026.

[88] Xinyu Zhu, Mengzhou Xia, Zhepei Wei, Wei-Lin Chen, Danqi Chen, and Yu Meng. The surprising efectiveness of negative reinforcement in llm reasoning. Advances in Neural Information Processing Systems, 38:126546–126573, 2025.

## Appendix

## Contents

1 Introduction 1   
2 Trajectory Anchoring Bias in AD VLMs 3   
3 AD-MCQ: A Verifiable Candidate-Trajectory Benchmark 3   
4 DEFT-RLVR: Deferred Exposure of Future Trajectories 5   
4.1 DEFT: Deferred Exposure of Future Trajectories 5   
4.2 Joint Optimization of the Two-Stage Interaction 5   
4.3 Structured Rubric Rewards for Reasoning-Trace Supervision 6   
5 Experiments 6   
5.1 Experimental Setup 6   
5.2 Candidate-Grounded Training Improves Generalizable AD Reasoning . 7   
5.3 RLVR Improves AD Reasoning without Sacrificing General Visual Capability 9   
5.4 Ablation of the RLVR Design 9   
5.5 Why Formulate AD Planning as a Candidate-Grounded MCQ? 10   
6 Conclusion 11   
A Related Work 20   
A.1 Vision–Language Reasoning for Autonomous Driving 20   
A.2 Trajectory Representations and End-to-End Planning 20   
A.3 Verifiable Post-Training and Faithful Reasoning . 20   
B Causal Faithfulness under Future-Trajectory Exposure 21   
B.1 Study Design 21   
B.2 Human Evaluation 22   
C Trajectory Codebook Construction and Representation Analysis 23   
C.1 Trajectory Representation and Quantization 23   
C.2 Reconstruction Evaluation Setup 23   
C.3 Resolution as the Codebook Scales 24   
C.4 Data Scale and Generalization 24   
C.5 Codebook Selection . 25   
D Detailed Experimental Settings 25   
D.1 Data Sources and AD-MCQ Construction 25   
D.2 Evaluation Settings and Benchmarks 28   
D.3 Models, Distillation, and RLVR Optimization 29   
D.4 Baselines and Controlled Variants 31   
D.5 Cold-Start SFT 32   
D.6 Direct Trajectory-Token SFT Diagnostic 33   
D.7 Candidate-Set Dificulty and Construction Robustness 34   
E Human Validation of AD CoT Evaluation 35   
F Prompt Templates 35   
F.1 Two-Turn Candidate-Grounded Policy 35   
F.2 Ofline Question-Specific Rubric Generation 37   
F.3 Online Text-Only Rubric Grader 38   
F.4 Image-Conditioned Rubric Reward for the Controlled Variant 38   
F.5 Joint-Exposure Policy Prompt (JEFT) 39   
F.6 CFS and HLD Evaluation-Judge Prompts 41   
G Detailed Candidate-Trajectory MCQ Case Studies 42   
H Full General Visual Capability Results 44

## A Related Work

## A.1 Vision–Language Reasoning for Autonomous Driving

Language-conditioned driving models have evolved from using language as an auxiliary source of supervision to placing a VLM directly in the perception, reasoning, and planning loop. Early systems formulate driving as graph-based visual question answering, language-conditioned behavior prediction, or interpretable trajectory generation [38, 52, 63, 70]. Subsequent work expands this direction through multi-view scene reasoning, knowledge augmentation, behavioral planning states, and unified vision–language–action architectures [24, 27, 59, 79, 85]. Driving-oriented question-answering and reasoning benchmarks complement these models by measuring scene understanding, spatial reasoning, and interpretable decision making [30, 39, 40, 42, 45, 58].

More recent methods explicitly supervise chain-of-thought reasoning or combine reasoning traces with action learning [12, 25, 62, 64, 82, 87]. These approaches establish the value of explicit intermediate reasoning for driving. Our work addresses a distinct question concerning the direction of that supervision: when a rationale is generated with access to the logged future trajectory, the trajectory can become a premise from which the teacher works backward. We instead require a scene-grounded decision before exposing future-trajectory candidates, retaining trajectory-level supervision while preventing the target future from anchoring the initial reasoning process.

## A.2 Trajectory Representations and End-to-End Planning

End-to-end driving has been studied through sensor-fusion policies, planning-oriented representations, sparse scene abstractions, and integrated prediction–planning architectures [7, 17–19, 26, 55]. Generative planners further model multimodal futures with autoregressive, difusion, or flow-based objectives [20, 31, 56, 83]. In parallel, discretized action and trajectory representations make continuous behavior compatible with tokenbased sequence models [29, 43, 44, 67, 80]. Such representations reduce the mismatch between language-model decoding and continuous control, but direct full-vocabulary trajectory generation still requires the VLM to synthesize precise geometry and can encourage task-specific memorization.

AD-MCQ uses trajectory prototypes diferently. The prototypes define a scene-specific set of decoded, explicit candidates rather than a global action vocabulary that the VLM must generate. This formulation preserves diferences in lateral geometry, braking time, and speed profile while turning trajectory-level planning into exact candidate selection. It is therefore closer to a verification interface than to a replacement for a downstream continuous planner. Our candidate-construction and held-out representation analyses further separate codebook suficiency from the dificulty of full-vocabulary trajectory-token prediction.

## A.3 Verifiable Post-Training and Faithful Reasoning

Supervised instruction tuning, preference optimization, and reinforcement learning provide complementary mechanisms for adapting foundation models [21, 34, 35, 41, 47, 50, 65, 66]. For reasoning models, reinforcement learning with automatically checkable outcomes can elicit capabilities without requiring imitation of every intermediate step [14, 22, 33, 51, 72, 73, 78, 81]. Recent work extends this principle beyond exact symbolic answers through multidimensional rubrics and rubric-derived rewards [13, 16, 23, 48, 49], while studies of multiple-choice RLVR show that distractor construction and answer format materially shape the learned behavior [5, 15]. DEFT-RLVR combines exact candidate correctness with instance-specific rubric rewards, but gates process rewards on outcome correctness and grades the candidate-blind reasoning trace without visual or answer-related information.

This design is also motivated by evidence that a chain of thought need not be a faithful account of the evidence that produced an answer. Preemptively revealing an answer can distort subsequent reasoning, and both language and vision–language models may rationalize cues or hallucinate support for an already favored conclusion [1, 3, 6, 32, 68, 71]. Our controlled annotation study instantiates this issue in trajectory-level driving decisions, and our deferred-exposure formulation converts the future trajectory from a pre-reasoning cue into a post-decision verification target.

## B Causal Faithfulness under Future-Trajectory Exposure

We provide the study design and detailed analysis for the comparison summarized in Section 2. We examine whether revealing the logged GT future trajectory helps a teacher infer a faithful driving rationale or merely makes an already known outcome easier to justify.

## B.1 Study Design

We evaluate the efect of GT-conditioned annotation on 100 strong-causal driving scenes, including 70 Waymo scenes and 30 internal scenes. We select scenes in which the logged future trajectory substantially deviates from constant-velocity extrapolation, covering hard braking, stopping from motion, and sharp turns. We detail the construction of the hard causal evaluation set in Appendix D.1. For each scene, we construct paired annotations using the same teacher. Across the two experimental settings, we fix Qwen3.5-397B-A17B as the teacher, the same 12 visual frames, ego-state and navigation text, the complete system prompt and all user instructions outside the intervention block, the four-item causal-reasoning body, the HIGH\_LEVEL\_- DECISION output contract, temperature 0, and disabled thinking. The two arms use a byte-identical system prompt and an otherwise identical user template. The sole intervention is one contiguous block containing the logged GT future as 10 raw ego-frame waypoints: the causal-planning arm omits this block, whereas the GT-conditioned arm inserts it before the shared reasoning instructions. Both arms still ask the teacher to infer and commit to a high-level decision from the scene; the GT-conditioned prompt does not ask the teacher to justify a known action, repeat an action label, or select among candidates. Consequently, the paired comparison changes only the availability of future-trajectory information while holding the task wording, inputs, model, decoding, and response format fixed. The complete role-separated templates below expose the single insertion point directly.

![](images/e228e9e33aa237c16f7208118846c651475929389339f3ef4ee6f64b810178a1.jpg)  
Figure 9: Complete causal-planning chat template. The teacher receives the scene history and ego state but no logged future. Per-scene inputs are shown as variables.

![](images/e1fa6dab272b2455db9aed64f64112555a04bdf5977087859c59edadd20c5f73.jpg)  
Figure 10: Complete GT-conditioned chat template. It is identical to Figure 9 except for the highlighted rawwaypoint GT block; no derived action label or rationalization-specific instruction is introduced.

## B.2 Human Evaluation

We conduct a human evaluation of the two experimental settings. Across the 100 scenes and two settings, we obtain 200 CoTs in total. We ask two annotators to independently score every CoT along four dimensions: grounding (GND), absence of hallucination (NO-HALL), specificity (SPEC), and causal coherence (COH). Each dimension is scored on a three-point ordinal scale, where 0 denotes a clear failure with a consequential error, 1 denotes partial satisfaction with an omission or minor error, and 2 denotes full satisfaction without a substantive error. We thus collect 200 × 4 × 2 = 1,600 dimension-level ratings. For each CoT, we average the two annotations on each dimension and sum the four averaged scores to obtain a causal-faithfulness score (CFS) from 0 to 8. The annotators work independently, and disagreements are retained rather than resolved through discussion or adjudication.

We instantiate the three-point scale with dimension-specific observable criteria. For GND, 0 indicates that the stated rationale conflicts with or is unsupported by the visible scene, 1 indicates that it uses some relevant scene evidence but omits or misinterprets a non-critical cue, and 2 indicates that its decision-relevant claims are supported by the observed scene. For NO-HALL, 0 indicates a consequential fabricated object, event, trafic control, or interaction, 1 indicates an unsupported but non-critical detail, and 2 indicates no unsupported factual claim. We count a CoT as a severe hallucination for an annotator when its NO-HALL score is 0; we compute the reported severe-hallucination rate by averaging this binary indicator over annotators and CoTs within each experimental setting. For SPEC, 0 indicates a generic rationale that could apply to unrelated scenes, 1 indicates limited use of scene-specific actors or geometry, and 2 indicates suficient reference to the particular actors, spatial relations, and trafic context that determine the maneuver. For COH,

0 indicates that the conclusion does not follow from the stated evidence or contains a major contradiction, 1 indicates a broadly plausible causal chain with a missing link or minor inconsistency, and 2 indicates a complete and internally consistent connection from scene evidence to the proposed maneuver.

For dimension-level scoring, we anonymize the experimental settings by replacing their names with random identifiers and randomizing presentation order. We present the scene and a single CoT in each item and never show its paired counterpart alongside it. The annotators are not informed of our anchoring hypothesis or which setting produced an item, and we withhold the GT future trajectory during scoring. We use the same teacher model, scene inputs, reasoning template, decoding settings, and output format for the two settings, as specified in the study design above.

After completing the independent dimension-level scoring, each annotator separately evaluates all 100 scenematched CoT pairs. For each scene, we present the two CoTs together in randomized order without experimental-setting labels, and ask the annotator to select the better rationale or record a tie using the same grounding, absence-of-hallucination, specificity, and causal-coherence criteria defined above. The two annotators therefore provide 200 independent pairwise judgments: causal planning is preferred in 121 judg ments (60.5%), GT-conditioned reasoning is preferred in 48 (24.0%), and 31 judgments (15.5%) are ties.

## C Trajectory Codebook Construction and Representation Analysis

In this section, we present the trajectory-codebook construction and representation analysis underlying AD-MCQ. Notably, the codebook serves a specific role in our benchmark: (1) the codebook is a purely kinematic aggregation of motion trajectories and is aggregated independently of scene observations; (2) we use it only to retrieve waypoint candidates when constructing AD-MCQ. The codebook itself and its indices are never exposed to the policy, which receives only the corresponding decoded waypoint candidates. We first formalize this representation, then describe the reconstruction evaluation setup and analyze how codebook size and data scale determine reconstruction fidelity and prototype utilization.

## C.1 Trajectory Representation and Quantization

A logged future is represented as a 5 s ego-frame trajectory sampled at 2 Hz, $\mathbf { P } = ( \mathbf { p } _ { 1 } , \dots , \mathbf { p } _ { T } ) \in \mathbb { R } ^ { T \times 2 }$ with $T = 1 0$ . Flattening P yields a 20-dimensional vector. Given N such trajectories, we apply K-means to obtain $\mathcal { C } = \{ \mathbf { C } _ { 1 } , \ldots , \mathbf { C } _ { K } \}$ , where each centroid $\mathbf { C } _ { k } \in \mathbb { R } ^ { T \times 2 }$ represents a complete speed and lateral-motion profile. As defined in Eq. 1, each trajectory is assigned to its nearest centroid under cumulative squared waypoint distance.

For implementation, we associate each codebook index k with a symbolic identifier $\tau _ { k } \equiv < \mathrm { t r a j \_ k } >$ . Encoding returns the identifier of the nearest prototype, whereas decoding retrieves the complete waypoint sequence, $\mathrm { D e c } ( \tau _ { k } ) = { \bf C } _ { k }$ . These identifiers provide a discrete indexing space for candidate construction; AD-MCQ serializes the decoded coordinates rather than the identifiers themselves. Consequently, quantization remains external to the VLM and does not require modifying its vocabulary or generating dense coordinates autoregressively.

## C.2 Reconstruction Evaluation Setup

We pool approximately $5 . 1 \times 1 0 ^ { 5 }$ logged trajectories from the Waymo Open E2E corpus and an internal driving corpus. Waymo ego future\_states are subsampled from 4 Hz to 2 Hz; both sources use the same 5 s, 10-waypoint, ego-frame, meter convention and are therefore clustered jointly. We reserve 5% (25,739 trajectories) from codebook fitting for same-distribution out-of-sample evaluation and use an independent Waymo validation split (106,360 trajectories) for cross-source evaluation.

Clustering uses MiniBatchKMeans with batch size 10,000, 300 iterations, and three initializations. For each $( N , K )$ configuration, we repeat K-means with three initialization seeds and report the mean over seeds separately for the fitting corpus, the held-out split, and the independent Waymo split. We measure average displacement error (ADE), final displacement error (FDE), their $\mathrm { p 5 0 / p 9 5 / p 9 9 }$ tail statistics, and codebook utilization, defined as the fraction of prototypes assigned at least one evaluation trajectory.

## C.3 Resolution as the Codebook Scales

Table 5 reports the resolution sweep at the full clustering-set size. Increasing K reduces ADE and FDE smoothly rather than producing a sharp saturation point; each doubling lowers in-sample ADE by approximately 15%. The improvement follows the approximate trend ADE ≈ $C K ^ { - 0 . 2 5 }$ , but increasingly fine codebooks allocate prototypes to sparse motions that are not recovered on held-out trajectories.

<table><tr><td>K</td><td>In ADE</td><td>Out ADE</td><td>Cross ADE</td><td>In FDE</td><td>Out FDE</td><td>In Util.</td><td>Out Util.</td></tr><tr><td>256</td><td>0.680</td><td>0.675</td><td>0.704</td><td>1.281</td><td>1.273</td><td>100.0%</td><td>100.0%</td></tr><tr><td>512</td><td>0.567</td><td>0.563</td><td>0.592</td><td>1.063</td><td>1.057</td><td>100.0%</td><td>100.0%</td></tr><tr><td>1024</td><td>0.480</td><td>0.478</td><td>0.505</td><td>0.890</td><td>0.887</td><td>100.0%</td><td>100.0%</td></tr><tr><td>2048</td><td>0.399</td><td>0.399</td><td>0.432</td><td>0.729</td><td>0.729</td><td>100.0%</td><td>100.0%</td></tr><tr><td>4096</td><td>0.333</td><td>0.336</td><td>0.374</td><td>0.601</td><td>0.605</td><td>100.0%</td><td>98.7%</td></tr><tr><td>8192</td><td>0.283</td><td>0.290</td><td>0.331</td><td>0.504</td><td>0.516</td><td>99.7%</td><td>92.0%</td></tr><tr><td>16384</td><td>0.242</td><td>0.252</td><td>0.297</td><td>0.426</td><td>0.446</td><td>99.4%</td><td>73.7%</td></tr></table>

Table 5 Trajectory-codebook resolution at N=489,042 clustering trajectories. Errors are measured in meters and per-cell ADE standard deviation is at most 0.005. Larger codebooks improve reconstruction fidelity but reduce outof-sample utilization beyond K=8192.

<table><tr><td> $K$ </td><td>N</td><td>In ADE</td><td>Out ADE</td><td>Gap</td><td>Out Util.</td></tr><tr><td rowspan="5">1024</td><td>5,000</td><td>0.407</td><td>0.528</td><td>+0.121</td><td>99.8%</td></tr><tr><td>10,000</td><td>0.442</td><td>0.509</td><td>+0.067</td><td>99.8%</td></tr><tr><td>20,000</td><td>0.464</td><td>0.496</td><td>+0.032</td><td>100.0%</td></tr><tr><td>50,000</td><td>0.476</td><td>0.489</td><td>+0.013</td><td>100.0%</td></tr><tr><td>100,000 200,000</td><td>0.476 0.477</td><td>0.479 0.477</td><td>+0.003 ~0</td><td>100.0% 100.0%</td></tr><tr><td rowspan="5">8192</td><td>489,042 50,000</td><td>0.480</td><td>0.478</td><td>~0</td><td>100.0%</td></tr><tr><td></td><td>0.243</td><td>0.303</td><td>+0.060</td><td>89.6%</td></tr><tr><td>100,000</td><td>0.266</td><td>0.297</td><td>+0.031</td><td>90.4%</td></tr><tr><td>200,000</td><td>0.274</td><td>0.290</td><td>+0.016</td><td>91.6%</td></tr><tr><td>489,042</td><td>0.283</td><td>0.290</td><td>+0.007</td><td>92.0%</td></tr></table>

Table 6 Reconstruction generalization as the clustering corpus grows. At fixed K, additional trajectories reduce the out-of-sample gap and stabilize prototype utilization.

The resolution–coverage trade-of is visible in both Table 5 and Figure 4a. Out-of-sample utilization remains complete through K=2048, is 98.7% at $K { = } 4 0 9 6$ and 92.0% at $K { = } 8 1 9 2$ , but falls to 73.7% at $K { = } 1 6 3 8 4$ . The independent Waymo split exhibits the same monotonic resolution trend with a consistent 0.02–0.05 m ADE ofset, showing that the comparison across K is not specific to a single held-out split.

## C.4 Data Scale and Generalization

We next vary the number of clustering trajectories while fixing K. As summarized in Table 6 and Figure 4b, small clustering sets yield an artificially low in-sample error but a larger held-out error because their centroids specialize to incidental sample positions. Increasing N closes this gap, with in-sample and out-of-sample reconstruction approaching convergence once the number of trajectories per prototype becomes suficiently large.

For K=1024, the gap is efectively closed once N reaches approximately $1 0 ^ { 5 } ;$ ; for K=8192, it decreases to 0.007 m at the full data scale. The long tail remains the main source of quantization error: at full data, ADE p99 decreases from 2.74 m at K=256 to 1.14 m at K=16384, and FDE p99 decreases from 5.82 m to 2.33 m. Thus, larger codebooks improve rare-motion reconstruction but do not eliminate long-tail error by themselves.

## C.5 Codebook Selection

Our choice of K=8192 follows from the joint behavior of reconstruction fidelity, held-out utilization, and data support. At the full clustering scale, it achieves 0.290 m out-of-sample ADE and 0.516 m out-of-sample FDE while retaining 92.0% utilization. Doubling the codebook further improves displacement error, but the fraction of prototypes exercised out of sample drops by more than 18%. Conversely, smaller codebooks retain nearly complete utilization but provide coarser trajectory distinctions. We therefore use K=8192 throughout AD-MCQ as the operating point that preserves fine-grained speed and lateral-motion patterns without allocating a large fraction of the codebook to unsupported prototypes.

## D Detailed Experimental Settings

## D.1 Data Sources and AD-MCQ Construction

We construct AD-MCQ from 514,781 scene–trajectory pairs drawn from Waymo Open E2E and an internal driving corpus. For each scene, every front-left, front, and front-right camera stream is decoded into four historical frames sampled at 2 Hz, together with the ego state, navigation context, and logged future trajectory. Qwen3-VL groups each adjacent pair of frames into one temporal patch, so the rendered prompt displays two temporal-patch timestamps (<0.2 seconds><1.2 seconds>) per camera stream even though the model input contains four decoded frames; Appendix F.1 shows the resulting prompt representation.

We then construct the three scene-disjoint downstream splits in Table 7. Training follows the natural scene distribution, whereas Dev and Test emphasize causally dificult long-tail maneuvers. Following Section 3, each question contains M=6 decoded options retrieved from a K=8192-prototype codebook. Every option is a 5 s ego-frame trajectory with 10 waypoints sampled at 2 Hz. The decoded candidates are logged-data-derived waypoint prototypes.

Notation and Precomputation. We write $X _ { i } ~ = ~ ( V _ { i } , H _ { i } , I _ { i } )$ for the visual history, ego history and current motion state, and navigation instruction of scene i. This factorization preserves three complementary signals required for driving: $V _ { i }$ provides spatial coverage and short-term temporal evidence about scene dynamics, $H _ { i }$ supplies the ego-motion context needed to interpret those observations, and $I _ { i }$ specifies route-level intent when multiple futures are geometrically feasible. It also follows the established VLA input interface of multi-view, multi-frame images, ego-vehicle states, and high-level navigation instructions [87]. For split $s ,$ Algorithm 1 maps the input scene–trajectory pairs to $\mathcal { D } _ { s } = \{ ( X _ { i } , \mathcal { A } _ { i } , a _ { i } ^ { \star } ) \}$ }, where $A _ { i }$ is the shufled six-option set and $a _ { i } ^ { \star }$ is the position of the quantized logged future after shufling.

We precompute one $K \times K$ similarity matrix for the entire codebook and reuse it for every scene and split. Specifically, $d _ { \mathrm { m i n } }$ and $d _ { \mathrm { m a x } }$ in Eq. 2 are the minimum and maximum over all $K ^ { 2 }$ pairwise trajectory ADE values, rather than statistics of an individual split or candidate pool. We clip the resulting similarities to [0, 1] and set the diagonal to one. Thus, candidate sampling only indexes the fixed row associated with the oracle token; it does not renormalize similarities per instance.

<table><tr><td>Split</td><td>Num</td><td>Scene Preference</td><td>Distractors</td></tr><tr><td>Train</td><td>5,000</td><td>Natural distribution.</td><td>Random</td></tr><tr><td>Dev</td><td>100</td><td>55 stop-from-motion; 30 hard brakes; 15 sharp turns. Structured</td><td></td></tr><tr><td>Test</td><td>500</td><td>250 straight/brake/stop; 125 left; 125 right.</td><td>Structured</td></tr></table>

Table 7 Split-specific AD-MCQ distractors: 5 random hard negatives for Train; 2 scale-matched + 1 constant-velocity + 2 hard negatives for Dev/Test.

Training Split: Scene and Candidate Construction. We randomly sample 5,000 training scenes from the natural scene distribution. For each scene, we instantiate the oracle with the nearest codebook prototype to the logged future and sample five distinct distractors at random from the hard-negative pool in Eq. 3. For the main experiments, this pool uses $\rho _ { \mathrm { m i n } } { = } 0 . 3 0$ and $\rho _ { \mathrm { m a x } } { = } 0 . 8 5$ . We sample uniformly without replacement within this band using a deterministic per-record random-number generator whose seed is derived from the base seed and record key as SHA1(base\_seed:record\_key). Consequently, repeated construction with the same base seed produces the same candidates. Sampling candidates broadly within this dificulty range avoids teaching the policy a fixed distractor template.

Dev and Test: Hard-Causal Scene Construction. Dev and Test evaluate whether a model can identify the scene evidence that causally supports a driving decision. We apply the ordered classifier in Table 8. Let $v _ { 0 }$ and $v _ { f }$ denote the current and final logged speeds, $L _ { \mathrm { { g t } } }$ the logged-future path length, $L _ { \mathrm { c v } }$ the path length under constant-velocity extrapolation, and $\Delta \psi$ the net heading change of the logged future. We define the braking ratio as $b = 1 - L _ { \mathrm { g t } } / L _ { \mathrm { c v } }$ . The minimum current-speed threshold is $v _ { \mathrm { m i n } } { = } 3 . 0 \mathrm { m } / \mathrm { s }$ for Waymo and $\mathrm { 1 . 5 m / s }$ for the internal corpus. Scenes classified as low-speed or routine straight driving are discarded; the remaining stop-from-motion, hard-braking, and sharp-turn scenes form the hard-causal pool. For ranking within each retained class, we use $g _ { \mathrm { c v } } = \lVert p _ { T } ^ { \mathrm { c v } } - p _ { T } ^ { \mathrm { g t } } \rVert _ { 2 }$ , the endpoint gap between constant-velocity extrapolation and the logged future, and retain up to the top 800 scenes per upstream pool and class. Each resulting question includes the constant-velocity trajectory as an explicit distractor. We retain only instances for which its quantized token is distinct from the oracle and belongs to the evaluation hard-negative pool defined below.

<table><tr><td>Ordered Class</td><td>Condition</td><td>Disposition</td></tr><tr><td>Low speed</td><td> $v _ { 0 } < v _ { \mathrm { m i n } }$ </td><td>Discard</td></tr><tr><td>Stop from motion</td><td> $v _ { f } < 0 . 5 \mathrm { m / s }$  and  $b > 0 . 5 5$ </td><td>Retain</td></tr><tr><td>Hard brake</td><td> $b > 0 . 4 5$  and  $v _ { f } < 0 . 6 v _ { 0 }$ </td><td>Retain</td></tr><tr><td>Sharp turn</td><td> $| \Delta \psi | > 2 5 ^ { \circ }$ </td><td>Retain</td></tr><tr><td>Routine straight</td><td>Otherwise</td><td>Discard</td></tr></table>

Table 8 Ordered hard-causal scene classifier. The first satisfied condition determines the class.

For Dev, we manually review the candidate pool and retain the most extreme 55 stop-from-motion, 30 hard braking, and 15 sharp-turn cases. Test broadens directional coverage with 250 straight braking or stopping scenes, 125 left-turn scenes, and 125 right-turn scenes; the turns are primarily sharp. The median gap between the logged future and constant-velocity extrapolation is 28.5 m on Dev and 27.0 m on Test, confirming that the larger test set preserves the long-tail focus. Dev is used for the motivation study and reward development; the scene-disjoint AD-MCQ-500 Test split is used only for final evaluation.

Dev and Test: Structured Candidate Construction. For each Dev or Test scene, we instantiate the oracle with the nearest codebook prototype to the logged future and construct five distractors as a structured mixture. All five distractors come from the split-specific evaluation hard-negative pool

$$
\mathcal { H } _ { i } ^ { \mathrm { e v a l } } = \left\{ z \in [ K ] \setminus \left\{ z _ { i } ^ { \star } \right\} : 0 . 3 0 \leq \rho _ { z , z _ { i } ^ { \star } } \leq 0 . 9 2 \right\} .\tag{12}
$$

We select two scale-matched hard negatives; they match the oracle’s endpoint-displacement scale while difering in trajectory shape, countering shortcuts based only on displacement magnitude. For prototype $k ,$ define its scale as

$$
q _ { k } = \lVert \mathbf { c } _ { k , T } \rVert _ { 2 } .\tag{13}
$$

We define ScaleMatch<sub>0.</sub> ${ _ { 1 5 } } \big ( z , z _ { i } ^ { \star } \big )$ by

$$
| q _ { z } - q _ { z _ { i } ^ { \star } } | \leq 0 . 1 5 q _ { z _ { i } ^ { \star } } .\tag{14}
$$

Among unused prototypes in $\mathcal { H } _ { i } ^ { \mathrm { e v a l } }$ satisfying this constraint, we take the two with the smalles $| q _ { z } - q _ { z _ { i } ^ { \star } } |$ Thus, “scale” denotes the Euclidean displacement of the final 5-s waypoint from the current ego origin, not path length or speed.

For the constant-velocity candidate, we use the recorded planar ego velocity $\begin{array} { r l } { \mathbf { v } _ { i } } & { { } = } \end{array}$ record["velocity"][:2], rather than estimating velocity from the sampled ego-history positions. With $\Delta t = 0 . 5 \mathrm { s } ,$ , we construct

$$
\begin{array} { r } { \mathbf { p } _ { i , t } ^ { \mathrm { c v } } = t \Delta t \mathbf { v } _ { i } , \qquad t = 1 , \ldots , 1 0 , } \end{array}\tag{15}
$$

and retrieve its nearest codebook prototype using Eq. 1, countering momentum-based extrapolation. For every retained instance, this prototype is distinct from the oracle and the scale-matched candidates and lies in $\mathcal { H } _ { i } ^ { \mathrm { e v a l } }$ . Two additional distinct samples satisfying $0 . 3 0 \leq \rho \leq 0 . 8 5$ complete the five distractors. All random draws from this band are uniform without replacement.

<table><tr><td>Candidate Component</td><td>Count</td><td>Eligibility Rule</td></tr><tr><td>Oracle</td><td>1</td><td>Nearest codebook prototype to the logged future.</td></tr><tr><td>Scale-matched</td><td>2</td><td>In  $\mathcal { H } _ { i } ^ { \mathrm { e v a l } }$  , with endpoint-displacement mismatch at most 0.15 relative to the oracle; smallest mismatch first.</td></tr><tr><td>Constant velocity</td><td>1</td><td>In  $\mathcal { H } _ { i } ^ { \mathrm { e v a l } }$  , nearest to  $\mathbf { p } _ { t } ^ { \mathrm { c v } } = t ( 0 . 5 \mathrm { s } ) \mathbf { v } _ { i } .$ </td></tr><tr><td>General hard negative</td><td>2</td><td>Distinct prototypes satisfying  $0 . 3 0 \leq \rho \leq 0 . 8 5 .$ </td></tr></table>

Table 9 Structured six-candidate construction used for Dev and Test. The oracle and previously selected prototypes are excluded during each distractor-selection step.

Candidate Validity and Deduplication. We maintain an exclusion set containing the oracle and every selected distractor, so scale-matched, constant-velocity, and hard-negative candidates cannot duplicate one another. Every retained Train, Dev, and Test instance has enough eligible prototypes to obtain the required five distinct distractors; no sampling fallback outside its split-specific hard-negative pool is used. We randomly shufle the oracle and five distractors and record $a _ { i } ^ { \star }$ as the oracle’s shufled position.

The positive option is consistent with the logged future, route intent, and map constraints, whereas negative options remain plausible explicit futures. When possible, distractors match the positive option in displacement, speed range, endpoint distance, or temporal horizon but difer in lane choice, yielding behavior, braking timing, obstacle clearance, or route compliance. We randomize option order to reduce position bias. This split-specific construction deliberately tests whether behavior learned from diverse random negatives transfers to targeted endpoint-displacement- and momentum-based distractors.

Algorithm 1 summarizes how we convert each scene–trajectory pair into an AD-MCQ instance. The splitspecific sampling operator follows the settings in Appendix D.1: Train uses five samples from the broad hard-negative pool, whereas Dev and Test use two scale-matched negatives, one constant-velocity negative, and two additional hard negatives. In the algorithm, ScaleMatch<sub>0.15</sub> compares the endpoint displacement magnitudes of two trajectory prototypes, as defined in Appendix D.1.

<table><tr><td>Quantity</td><td>Mean</td><td>P50</td><td>P75</td><td>P90</td><td>P95</td></tr><tr><td>Oracle ADE (m)</td><td>0.45</td><td>0.227</td><td>0.483</td><td>0.933</td><td>1.630</td></tr><tr><td>Oracle FDE (m)</td><td>0.79</td><td>0.319</td><td>0.684</td><td>1.576</td><td>2.671</td></tr><tr><td>Oracle ADE  $/ \ L _ { \mathrm { g t } } \ ( \% )$ </td><td>3.0</td><td>2.4</td><td>3.7</td><td>5.8</td><td>7.4</td></tr><tr><td>All distractors ADE (m)</td><td>19.77</td><td>17.94</td><td>24.93</td><td>35.27</td><td>44.25</td></tr><tr><td>Nearest distractor ADE (m)</td><td>7.16</td><td>7.05</td><td>8.73</td><td>10.27</td><td>11.86</td></tr></table>

Table10 Oracle reconstruction fidelity and candidate separation on AD-MCQ-500. Distractor ADE is measured against the logged future; the nearest distractor is selected independently for each instance. P50, P75, P90, and P95 denote the 50th, 75th, 90th, and 95th percentiles, respectively, of each quantity across evaluation instances.

Algorithm 1: Split-Specific AD-MCQ Construction   
Input: scene–trajectory pairs ${ \cal { S } } = \{ ( { \cal { X } } _ { i } , { \bf { P } } _ { i } ^ { \mathrm { { g t } } } ) \}$ assigned to split $s \in$ {Train, Dev, Test}; codebook ${ \mathcal C } = \{ { \bf C } _ { k } \} _ { k = 1 } ^ { K } ;$   
option count $M = 6 ;$ recorded planar ego velocity ${ \bf v } _ { i } ;$ similarities $\rho$ from $\operatorname { E q . 2 } .$   
Output: $\mathcal { D } _ { s } = \{ ( X _ { i } , \mathcal { A } _ { i } , a _ { i } ^ { \star } ) \}$   
1 Initialize $\mathcal { D } _ { s }  \mathcal { D } .$   
2 for each $( X _ { i } , \mathbf { P } _ { i } ^ { \mathrm { g t } } ) \in \mathcal { S }$ do   
3 Quantize the logged future: $z _ { i } ^ { \star } \gets \arg \operatorname* { m i n } _ { k \in [ K ] } \sum _ { t = 1 } ^ { T } \| \mathbf { p } _ { i , t } ^ { \mathrm { g t } } - \mathbf { c } _ { k , t } \| _ { 2 } ^ { 2 } .$   
4 Initialize the selected distractors $\mathcal { N } _ { i }  \emptyset .$   
5 if s = Train then   
6 Form $\mathcal { H } _ { i } ^ { \mathrm { t r } } = \{ z \neq z _ { i } ^ { \star } : 0 . 3 0 \leq \rho _ { z , z _ { i } ^ { \star } } \leq 0 . 8 5 \}$ and uniformly sample $M - 1$ distinct indices into ${ \mathcal { N } } _ { i } .$   
7 else $( s \in$ {Dev, Test})   
8 Form ${ \mathcal H } _ { i } ^ { \mathrm { e v a l } } ~ = ~ \{ z ~ \neq ~ z _ { i } ^ { \star } ~ : ~ 0 . 3 0 ~ \leq ~ \rho _ { z , z _ { i } ^ { \star } } ~ \leq ~ 0 . 9 2 \}$ ; add the two indices in this pool satisfying   
ScaleMatch<sub>0.1</sub> $\mathinner { 5 \mathopen { \left( z , z _ { i } ^ { \star } \right) } }$ with the smallest endpoint-scale mismatches.   
9 Form $\mathbf { p } _ { i , t } ^ { \mathrm { c v } } = t ( 0 . 5 \mathrm { s } ) \mathbf { v } _ { i }$ for $t = 1 , \ldots , 1 0 ,$ , quantize it as $z _ { i } ^ { \mathrm { c v } } \gets z ( \mathbf { P } _ { i } ^ { \mathrm { c v } } )$ , and add $z _ { i } ^ { \mathrm { c v } } \in \mathcal { H } _ { i } ^ { \mathrm { e v a l } }$ , which is   
distinct from the two scale-matched indices.   
10 Uniformly sample two additional unused indices in $\mathcal { H } _ { i } ^ { \mathrm { e v a l } }$ satisfying $\rho _ { z , z _ { i } ^ { \star } } \leq 0 . 8 5$   
11 end if   
12 Randomly permute $\{ z _ { i } ^ { \star } \} \cup \mathcal { N } _ { i }$ to obtain $( z _ { i , 1 } , \ldots , z _ { i , M } )$ , and record the oracle position $a _ { i } ^ { \star }$ such that   
$z _ { i , a _ { i } ^ { \star } } = z _ { i } ^ { \star }$   
13 Decode the indices into explicit waypoint options: $\mathcal { A } _ { i } \gets ( \mathbf { C } _ { z _ { i , 1 } } , \ldots , \mathbf { C } _ { z _ { i , M } } )$   
14 Add $( X _ { i } , \mathcal { A } _ { i } , a _ { i } ^ { \star } )$ to $\mathcal { D } _ { s }$   
15 end for   
16 return $\mathcal { D } _ { s } .$

Oracle Fidelity and Candidate Separation. The oracle provides a high-fidelity representation of the logged motion, while the alternative candidates constitute geometrically distinct plans rather than duplicate quantizations or small coordinate perturbations. Table 10 supports this conclusion: the oracle prototype accurately reconstructs the logged future for most AD-MCQ-500 instances, and its displacement error is typically small relative to the scale of the trajectory space. By contrast, even the nearest distractor in each instance remains substantially farther from the logged future. Its error exceeds the oracle error in all 500 instances, with a median oracle–distractor gap of 6.74 m. We do not assume that the logged future is the unique safe trajectory: AD-MCQ operationalizes planning as discrimination among explicit plan hypotheses relative to demonstrated behavior, not as an exhaustive certification of every feasible future. Nevertheless, the cross-domain gains in Table 3 and the consistent gains under resampled candidate counts and similarity bands in Figure 8 and Table 11 show that the learned capability transfers across both driving domains and candidate constructions, rather than merely recovering a fixed recorded action or exploiting one particular distractor geometry.

## D.2 Evaluation Settings and Benchmarks

AD-Specific Evaluation. All AD-specific evaluations use a 65,536-token context window and the following decoding configuration: T=1.0, top-p=0.95, thinking enabled, and a generation limit of 12,000 tokens per turn for DEFT inference and 24,576 tokens in total. For JEFT, we directly set the token budget to 24,576.

We evaluate candidate selection on AD-MCQ-500 using strict option accuracy. Each reported model is evaluated eight times under the same configuration, and we report the mean over these runs. On the same 500 scenes, we assess candidate-blind Turn-1 outputs with two complementary metrics. Normed-CFS (Normalized Causal-Faithfulness Score) is a GT-blind automatic score over the same four dimensions used in the human study: grounding, absence of hallucination, specificity, and causal coherence. The judge assigns a binary value $b _ { d } \in \{ 0 , 1 \}$ to each dimension d ∈ {GND, NO-HALL, SPEC, COH}, and we compute Normed-CFS $\begin{array} { r } { \frac { 1 } { 4 } \sum _ { d } b _ { d } \in \mathsf { \bar { \Gamma } } [ 0 , 1 ] } \end{array}$ This difers deliberately from the $0 / 1 / 2$ ordinal scale used by human annotators in Section 2. Human annotators can reliably distinguish partial from full satisfaction, whereas this intermediate category is less stable for a model judge; we therefore ask the automatic evaluator only for mechanically defined binary decisions [16, 48]. We report this normalized automatic score as CFS in the result tables. HLD (High-Level-Decision Consistency) measures agreement between the predicted high-level decision and the GT trajectory, requiring both direction and speed to match the oracle action induced by the GT waypoints. Qwen3.5-397B-A17B judges these two open-ended metrics; strict candidate accuracy uses exact option matching and no LLM judge. The complete CFS and HLD evaluation-judge prompts are provided in Appendix F.6.

General-CapabilityEvaluation. To measure capability retention, we evaluate 12 benchmarks with VLMEvalKit and report four category averages: basic visual perception, embodied spatial reasoning, 3D/multi-view reasoning, and RefSpatial grounding. The category averages are computed over the following benchmark triplets: Basic Visual comprises CV-Bench-2D, CV-Bench-3D, and DA-2K; Embodied Spatial comprises EmbSpatial-Bench, RoboSpatialHome, and ERQA; 3D/Multi-View comprises 3DSRBench, MMSIBench, and ViewSpatialBench; and RefSpatial comprises the Location, Placement, and Unseen splits of RefSpatial-Bench. Their macro-average is reported as General AVG. All scores are percentages obtained with greedy decoding and without a chain-of-thought prompt, using the same configuration across main-table evaluations.

Cross-Domain AD Evaluation. We construct an external 500-scene set from nuScenes val using the same trajectory codebook, six-candidate construction, ego-motion inputs, the two-turn deferred-exposure interface, and eight independently sampled evaluation runs, matching AD-MCQ-500. The resulting evaluation preserves the task and output contract while changing the driving domain. We use Qwen3.5-397B-A17B to judge the two external-set open metrics, matching the evaluator used for the in-domain evaluation.

## D.3 Models, Distillation, and RLVR Optimization

Models. We use Qwen3-VL-8B-Instruct and Qwen3.5-4B as the base policies. Qwen3.5-397B-A17B [46] supplies CoT annotations, and Qwen3.6-35B-A3B serves as the ofline vision-language rubric generator and online text-only rubric grader. The cold-start SFT configuration is detailed in Appendix D.5.

Controlled Distillation Baselines. DEFT Distillation (Plan Only), DEFT Distillation (Full Interaction), and DEFT Distillation (Mixed Targets) each use 5,000 teacher-labeled scenes and start from the unmodified Qwen3-VL-8B-Instruct policy. Their targets are respectively all Turn-1-only, all complete two-turn, or a fixed 2,500/2,500 split of the two formats. All three runs use TP=2, global batch size 32, micro-batch size 1, maximum sequence length 20,480, and a constant learning rate of $1 \times 1 0 ^ { - 5 }$ . The visual encoder is frozen and the language model is updated.

Algorithm 2 summarizes the controlled teacher-target construction and student fine-tuning pipeline. All vari ants use the same teacher-labeled scene budget and student initialization; they difer only in when candidate trajectories are exposed to the teacher and which generated turns are retained as supervised targets.

Algorithm 2: Unified Target Construction and Training for the Distillation Variants   
Input: training instances $\mathcal { D } = \{ ( X _ { i } , \mathcal { A } _ { i } , a _ { i } ^ { \star } ) \} _ { i = 1 } ^ { 5 , 0 0 0 }$ ; teacher π (Qwen3.5-397B-A17B); unmodified student initial  
ization $\pi _ { \boldsymbol { \theta } _ { 0 } } ;$ variant $v ;$ candidate-blind prompt $_ { p 1 } ;$ trajectory-matching prompt $p _ { 2 }$   
Variants: JEFT Distillation; DEFT Distillation (Plan Only); DEFT Distillation (Full Interaction); DEFT Distillation (Mixed   
Targets).   
Output: fine-tuned student $\pi _ { \theta } .$   
1 Initialize the supervised target set $\mathcal { T } _ { v }  \mathcal { D } .$   
2 If v = Mixed, fix an equal partition $\mathcal { D } = \mathcal { D } _ { \mathrm { p l a n } } \dot { \cup } \mathcal { D } _ { \mathrm { f u l l } }$ , with $| \mathcal { D } _ { \mathrm { p l a n } } | = | \mathcal { D } _ { \mathrm { f u l l } } | = 2 { , } 5 0 0 .$   
3 for each $( X _ { i } , \mathcal { A } _ { i } , a _ { i } ^ { \star } ) \in \mathcal { D }$ do   
JEFT Distillation: reveal $\mathbf { \mathcal { A } } _ { i }$ together with $X _ { i }$ at the outset, query π<sub>T</sub> for one joint reasoning-and-selection   
4   
response $y _ { i } ^ { \mathrm { j o i n t } }$ , and add the resulting single-turn example to $\mathcal { T } _ { v } .$   
5 All DEFT distillation variants: query $\pi _ { \mathrm { T } }$ with $p _ { 1 } ( X _ { i } )$ , without $\mathcal { A } _ { i } ,$ to obtain the candidate-blind plan $y _ { 1 , i } .$   
6 DEFT Distillation (Plan Only): retain only the Turn-1 pair $( p _ { 1 } ( X _ { i } ) , y _ { 1 , i } )$ in $\mathcal { T } _ { v } ;$ do not include candidate   
matching in the target.   
DEFT Distillation (Full Interaction): after $y _ { 1 , i }$ is complete, reveal $\mathcal { A } _ { i } ,$ query π with $p _ { 2 }$ for the matching   
7   
response $^ { y _ { 2 } , i , }$ and add the complete plan-then-match interaction to $\mathcal { T } _ { v } .$   
DEFT Distillation (Mixed Targets): ${ \mathrm { i f ~ } } i \in \mathcal { D } _ { \operatorname { p l a n } }$ , retain only $y _ { 1 , i }$ as in Plan Only; otherwise reveal $\begin{array} { r } { A _ { i } , } \end{array}$ generate   
8   
$y _ { 2 , i } ,$ and retain the complete interaction as in Full Interaction.   
9 end for   
10 Initialize π<sub>θ</sub> $ \pi _ { \theta _ { 0 } }$ and perform supervised fine-tuning on $\mathcal { T } _ { v } ,$ freezing the visual encoder and updating the   
language model under the shared optimization configuration   
11 return $\pi _ { \theta } .$

RLVR Optimization. We optimize the complete two-turn sequence with GRPO and one trajectory-level scalar advantage shared by the generated tokens of both turns. For each optimizer step we sample 16 rollouts for each of 64 questions, giving 1,024 trajectories per step. The actor uses a constant learning rate of $1 \times 1 0 ^ { - 6 }$ without warmup. The policy loss uses token-mean aggregation and gradient clipping at 1.0, with the normalized advantage clipped to [−3.75, 3.75]. A KL loss with coeficient 0.001 regularizes the policy but is not added to the reward. Training is on-policy (one update per sampled batch) and uses no critic or entropy bonus. Rollouts are sampled with $T { = } 1 . 0 .$ , top- $\scriptstyle \cdot p = 0 . 9 5$ , and top-k= − 1. We use prompt and response limits of 12,288 and 24,576 tokens, respectively, with a 12,000-token generation limit per turn and a 65,536-token rollout context limit. Rollout inference uses a device-memory utilization of 0.9.

The Qwen3-VL-8B and Qwen3.5-4B experiments use the same optimization configuration. All main RLVR runs start directly from their respective base models.

Reward and Grader Configuration. The exact verifier parses the final occurrence of FINAL\_CHOICE: [A–F] and assigns one only when it matches the shufled oracle option; malformed and incorrect responses receive zero. For DEFT-RLVR, each question has 6–10 positive atomic rubric criteria with integer weights in [1, 10]. Qwen3.6-35B-A3B generates these criteria ofline with temperature 0.3, top-p = 0.9, a 1,024-token output limit, and thinking disabled. The same model grades each correct normalized Turn-1 trace online without images, candidate trajectories, the oracle, or answer letters, using temperature 1.0, top-p = 0.7, a 256-token output limit, and thinking disabled. It is served in bf16 with tensor parallelism 16 and a 65,536-token context window. If grading fails after retry handling, we fall back to the exact-correctness reward; an incorrect final choice always receives zero.

Before either rubric-based grader is called, we isolate the candidate-blind Turn-1 trace by truncating the serialized interaction at the injected Part-2 option block and removing any residual injected option text or Turn-2 response. We then remove reasoning-wrapper tags, the HIGH\_LEVEL\_DECISION line, and structural headers such as PART, REASONING, and ===, while retaining the substantive evidence and causal reasoning. We denote the resulting trace by $\widetilde { y } _ { 1 , i , j } : = \mathrm { N o r m a l i z e } ( y _ { 1 , i , j } )$ . This normalization prevents the grader from using the committed maneuver, candidate options, or final answer as a proxy for reasoning quality.

Controlled SFT comparisons use equal numbers of teacher-annotated examples. Controlled RLVR comparisons share the base VLM, training scenes, prompts, rollout budget, and core optimization settings unless explicitly stated otherwise.

## D.4 Baselines and Controlled Variants

We compare DEFT-RLVR against the following task-matched baselines and controlled DEFT variants, all using the same AD-MCQ task and candidate representation.

• Base VLM (Direct MCQ). We run the base model in a single turn with candidates visible from the outset and evaluate only its exact-choice accuracy.

• DEFT (Training-Free). We run the same base model with our two-turn evaluation prompts: it first produces a candidate-blind plan and then selects among the revealed trajectories.

• JEF ${ \displaystyle { \mathsf { T } } + { \mathsf { R L V R } } { \mathsf { ( } } { \mathit { R } } ^ { \mathrm { M C Q } } } )  .$ . GRPO optimizes the single-turn JEFT response using only the binary exactchoice reward in Eq. 7. Its reasoning and selection instructions match the corresponding DEFT turns. It uses the same candidate sets, training scenes, verifier, and per-step rollout group size as the other RLVR runs.

• DEFT+RLVR $\scriptstyle ( R ^ { \mathrm { M C Q } } ) .$ This baseline uses the same candidate-blind Turn 1 and candidate-revealed Turn 2 as DEFT-RLVR, jointly optimizes both generated turns, and assigns the same exact-choice reward to their tokens.

• DEFT ${ \mathsf { \bar { \tau } } } + { \mathsf { R L V R } } ( R ^ { \mathrm { M C Q } } R ^ { \mathrm { G E N } } ) .$ This variant preserves the two-turn interface and applies a shared imageconditioned rubric to outcome-correct rollouts. We provide $\widetilde { y } _ { 1 , i , j }$ as defined above, together with twelve scene frames, to a Qwen3.6-35B-A3B grader. The grader is decoded with temperature $1 . 0 , \mathrm { t o p } . p = 0 . 7$ a 128-token output limit. It returns four binary indicators b<sup>ground</sup>, b<sup>no-hall</sup>, $b ^ { \mathrm { s p e c } }$ , and $b ^ { \mathrm { c o h } }$ for grounding, absence of hallucination, specificity, and coherence, respectively. We define the shared-rubric score as

$$
\begin{array} { r l r } & { } & { R ^ { \mathrm { G E N } } = 0 . 3 0 b ^ { \mathrm { g r o u n d } } + 0 . 3 0 b ^ { \mathrm { n o - h a l l } } } \\ & { } & { + 0 . 1 0 b ^ { \mathrm { s p e c } } + 0 . 3 0 b ^ { \mathrm { c o h } } , } \end{array}\tag{16}
$$

and assign the rollout reward $R = R ^ { \mathrm { M C Q } } R ^ { \mathrm { G E N } }$ . Thus, an invalid or incorrect final choice receives zero before rubric grading. The rubric, decision rules, and weights are fixed across scenes rather than generated per instance; the exact grader prompt is shown in Figure 15.

• JEFT Distillation. We supervise the student with teacher responses produced under JEFT.

• DEFT Distillation (Plan Only). The student imitates the Qwen3.5-397B-A17B teacher’s candidate-blind Turn-1 plan on every annotated scene. At evaluation, candidate matching is elicited from the resulting policy without having been included in its SFT targets.

• DEFT Distillation (Full Interaction). The student imitates both the candidate-blind plan and the subsequent candidate-selection response.

• DEFT Distillation (Mixed Targets). We divide the same annotated-scene budget approximately equally between Turn-1-only and complete two-turn targets.

The three completed deferred-exposure distillation runs use the same 5,000 teacher-labeled scenes, raw Qwen3- VL-8B-Instruct initialization, and optimization settings described above. All principal RLVR variants start from their respective base models; their shared GRPO configuration is described above.

Runtime Accounting. Table 4 reports median values over deduplicated main-trainer events. All policy runs use four nodes with 16 PPU-ZW810E accelerators per node (96 GB per accelerator; 64 accelerators in total). The actor and reference model are fully sharded over these 64 accelerators, and policy rollouts use vLLM 0.18.0 with 64 tensor-parallel-size-1 engines, bf16 inference, a 65,536-token context limit, 0.9 device-memory utilization, chunked prefill, and CUDA graphs. The $R ^ { \mathrm { M C Q } } \mathrm { - o n l y }$ run uses this policy pool alone. Both rubricbased variants use an additional, separate pool of 64 PPU-ZW810E accelerators for Qwen3.6-35B-A3B grading (bf16, tensor parallelism 16). For $\bar { R } ^ { \mathrm { G E N } }$ , the online image-conditioned grader receives the normalized trace $\widetilde { y } _ { 1 , i , j }$ and 12 scene frames; for DEFT-RLVR, the online grader receives the same normalized trace without images. Thus, the table compares end-to-end wall-clock latency under our deployed configuration, rather than total accelerator-hours.

A step comprises policy rollout, reward scoring, reference-model log probabilities, the GRPO forward/backward update, and policy-weight synchronization; validation and pre-training ofline rubric generation are excluded. The Rollout column measures policy generation only, whereas Scoring includes reward queuing, communication, input parsing, and grader inference. We aggregate 1,234, 910, and 1,141 logged steps for $R ^ { \mathrm { M C Q } } , R ^ { \mathrm { M C Q } } R ^ { \mathrm { G E \hat { N } } }$ , and DEFT-RLVR, respectively, merging resumed logs and retaining the latest event for each duplicated global step. No logged warm-up or anomalous steps are manually removed.

Algorithm 3 integrates the four RLVR configurations into a shared on-policy training loop. Color-coded branches isolate their candidate-exposure interfaces and reward computations, while all black steps use the same rollout grouping and GRPO update.

Algorithm 3: Unified On-Policy Training for the RLVR Variants   
Input: training instances $\mathcal { D } = \{ ( X _ { i } , \mathcal { A } _ { i } , a _ { i } ^ { \star } ) \}$ ; variant v; policy $\pi _ { \boldsymbol { \theta } } ;$ rubric generator $\mathcal { G } ;$ text grader $\mathcal { I } ;$ prompts   
$p _ { 1 } , p _ { 2 } , p _ { \mathrm { r u b } } , p _ { \mathrm { t x t } } ;$ rollouts per instance $G .$   
Variants: JEFT + RLVR $( R ^ { \mathrm { M C Q } } ) ;$ DEFT + RLVR $( R ^ { \mathrm { M C Q } } ) ;$ DEFT + RLVR $\begin{array} { r l } {  { ( R ^ { \mathrm { M C Q } } R ^ { \mathrm { G E N } } ) ; } } \end{array}$ DEFT-RLVR.   
Output: optimized policy $\pi _ { \theta } .$   
1 Generate and cache $\mathcal { C } _ { i }  \mathcal { G } ( p _ { \mathrm { r u b } } ( X _ { i } ) )$ for every $i \in \mathcal { D } .$   
2 for each on-policy RL iteration $t = 1 , 2 , \ldots$ . do   
3 Sample a fresh minibatch $\boldsymbol { B } _ { t } \subset \mathcal { D }$ and set the rollout policy π<sub>old</sub> $ \pi _ { \theta _ { t } }$   
4 for each $i \in { \cal B } _ { t }$ and $j \in \{ 1 , \ldots , G \}$ do   
5 JEFT + RLVR $( R ^ { \mathrm { M C Q } } ) { : }$ Reveal $\mathbf { \mathcal { A } } _ { i }$ at the outset and sample one joint reasoning-and-selection response $y _ { i , j }$   
from $\pi _ { \mathrm { o l d } } .$   
6 All DEFT variants [DEFT + RLVR $( R ^ { \mathrm { M C Q } } ) ;$ DEFT + RLVR $( R ^ { \mathrm { M C Q } } R ^ { \mathrm { G E N } } ) ;$ DEFT-RLVR]: sample the candidate  
free $_ { 3 1 , i , j } ;$ then reveal $\mathbf { \mathcal { A } } _ { i }$ and sample the matching response $y _ { 2 , i , j } .$   
7 Parse $\widehat { a } _ { i , j }$ from the final response and set $\overline { { R _ { i , j } ^ { \mathrm { M C Q } }  \mathbb { I } [ \widehat { a } _ { i , j } = a _ { i } ^ { \star } ] } }$   
8 JEFT + RLVR $( R ^ { \mathrm { M C Q } } )$ $R _ { i , j } \gets R _ { i , j } ^ { \mathrm { M C Q } } .$   
9 DEFT + RLVR $( R ^ { \mathrm { M C Q } } ) { : }$ $R _ { i , j } \gets R _ { i , j } ^ { \mathrm { M C Q } } ;$ assign it to both turns.   
DEFT + RLVR $\begin{array} { r l } {  { ( R ^ { \mathrm { M C Q } } R ^ { \mathrm { G E N } } ) \colon } } & { { } } \end{array}$ If $R _ { i , j } ^ { \mathrm { M C Q } } = 1$ , grade $\widetilde { y } _ { 1 , i , j }$ together with the scene frames using the shared   
10   
image-conditioned rubric to obtain $R _ { i , j } ^ { \mathrm { G E N } }$ ; otherwise set $R _ { i , j } ^ { \mathrm { G E N } } = 0 .$ . Then set $R _ { i , j } \gets R _ { i , j } ^ { \mathrm { M C Q } } R _ { i , j } ^ { \mathrm { G E N } }$   
DEFT-RLVR: If $R _ { i , j } ^ { \mathrm { M C Q } } ~ = ~ 1$ , grade $\widetilde { y } _ { 1 , i , j }$ against $\mathcal { C } _ { i }$ with the text-only grader and compute $R _ { i , j } ^ { \mathrm { R U B } } ~ =$   
11 $\mathbf { w } _ { i } ^ { \top } \mathbf { b } _ { i , j } / \Vert \mathbf { w } _ { i } \Vert _ { 1 } ;$ otherwise set $R _ { i , j } ^ { \mathrm { R U B } } ~ = ~ 0$ If grading fails after retries, set $R _ { i , j } ^ { \mathrm { R U B } } ~  ~ 1$ Then set   
$R _ { i , j } \gets R _ { i , j } ^ { \mathrm { M C Q } } R _ { i , j } ^ { \mathrm { R U B } }$   
12 Serialize the generated response or complete two-turn interaction as $s _ { i , j } .$   
13 end for   
14 Normalize $\{ R _ { i , j } \} _ { j = 1 } ^ { G }$ within each rollout group to obtain the clipped advantages $\{ \widehat { A } _ { i , j } \} _ { j = 1 } ^ { G }$   
15 Compute generated-token importance ratios between $\pi _ { \theta }$ and $\pi _ { \mathrm { o l d } }$ over $\left\{ s _ { i , j } \right\}$ , with prompt tokens masked   
and $\widehat { A } _ { i , j }$ shared by all generated turns.   
16 Take exactly one GRPO update $\theta _ { t }  \theta _ { t + 1 }$ , applying KL regularization separately from the reward.   
17 Discard the rollout batch; do not reuse it after the policy update.   
18 end for when the training budget is exhausted.

## D.5 Cold-Start SFT

The cold-start ablation uses 1,941 deduplicated examples: 1,600 causal-planning examples (an equal mixture of Turn-1-only and complete two-turn targets) and 341 general multimodal reasoning examples relabeled by the teacher. We hold the preprocessing seed fixed at 20260625. This SFT stage freezes the visual encoder and updates the language model without resizing the original 151,936-entry vocabulary. Optimization uses a constant learning rate of $1 \times 1 0 ^ { - 5 }$ with 10 warmup steps, a global batch size of 64, a per-rank micro batch size of 1, a maximum sequence length of 24,576, and tensor parallelism of 2. The main run trains for five epochs, comprising 155 optimizer steps (31 per epoch), and saves every 31 steps. For the dense initialization study, we use the same configuration for one epoch and save every 3 steps.

## D.6 Direct Trajectory-Token SFT Diagnostic

This controlled experiment underlies Figure 7 and separates three possible bottlenecks: trajectory quantization, inference over the trajectory-token vocabulary, and retention of the base VLM’s general capabilities. It is distinct from the adaptation setting in Section 5.1 and uses its own matched SFT configuration.

Model, Data, and Targets. We extend Qwen3-VL-8B-Instruct with the K=8192 trajectory tokens analyzed in Appendix C. Each token decodes to a 5 s ego-frame trajectory containing 10 waypoints at 2 Hz. Both settings use the same 100,000 examples: 88,636 Waymo-E2E training scenes and 11,364 internal driving scenes. SFT wo/ CoT directly emits the oracle trajectory token. SFT w/ CoT first emits a four-part rationale—scene description, critical object, reasoning, and best action—annotated by Qwen3.5-397B-A17B with the GT action available, and then emits the same oracle token. Accordingly, this setting tests trajectory-anchored rationalization rather than the scene-first reasoning used by DEFT-RLVR.

The teacher annotation limit is 2048 tokens. Of the raw CoT annotations, 6,796 (6.8%) end before the requested “Best Driving Action” conclusion and 16,104 (16.1%) lack a parseable structured action field. Preprocessing repairs the output wrapper so that every final training target contains a closed reasoning segment and answer segment; the underlying truncation remains a limitation of this diagnostic.

Optimization. We train both settings for 14 epochs with global batch size 128, micro-batch size 1, maximum sequence length 2048, a constant learning rate of $1 \times 1 0 ^ { - 5 }$ , and 100 warmup steps. One epoch corresponds to 782 optimizer steps. All optimization and data settings other than the target sequence are shared.

Trajectory Evaluation. We distinguish memorization, held-out in-domain generalization, and zero-shot crossdomain transfer. The memorization tier uses seen internal moving scenes; the in-domain tier uses Waymo-E2E validation scenes excluded from training; and the OOD tier uses nuScenes, which is absent from the training mixture. We decode each predicted special token without skipping special tokens and report ADE, FDE, and parse-failure rate over the full 5 s horizon. The epoch curves use fixed n=100 subsets. At epoch 14, larger n=2000 evaluations give 2.136 m ADE for NoCoT and 2.257 m for CoT on Waymo-E2E validation; the available NoCoT nuScenes endpoint is 5.297 m. These larger endpoints agree with the trends in Figure 7a–b.

General-Capability Evaluation. We use greedy decoding with one sample, no CoT prompt, and model thinking disabled. We evaluate CV-Bench-2D/3D, DA-2K, ERQA, EmbSpatialBench, RoboSpatialHome, MMSI-Bench, RefSpatial-Bench Location/Placement/Unseen, 3DSRBench, and ViewSpatialBench, and report the arithmetic mean of their 12 primary metrics as AVG(12).

Although training loss continues to decrease, held-out Waymo-E2E ADE reaches its minimum near epoch 8 and then rises, while zero-shot nuScenes ADE plateaus after epoch 4. The best held-out ADE is 1.963 m, compared with the subset-specific 0.279 m codebook quantization floor (0.290 m on the larger held-out representation split in Appendix C). Adding CoT is slightly worse in domain and provides only a small out-ofdomain bufer, with additional parse failures. The two settings therefore reach similar trajectory accuracy despite substantially diferent general-capability retention.

For SFT wo/ CoT, the 12-benchmark mean falls from 53.49% at epoch 1 to 0.67% at epoch 8 and approaches zero thereafter. CoT slows but does not prevent forgetting: its mean decreases from 50.93% to 21.50% by epoch 14. These curves show that rationale supervision primarily delays destructive specialization rather than improving trajectory precision.

We additionally compare the output interfaces directly using Qwen3.5-397B-A17B on the hard-causal development set. Full-vocabulary prediction obtains approximately zero strict accuracy and 2.5 m ADE, whereas six-way candidate selection reaches 0.61 greedy accuracy and 0.88 pass@64. This is not a matched downstream evaluation, but it isolates the output interface and supports the conclusion that candidate restriction substantially reduces search dificulty.

## D.7 Candidate-Set Difficulty and Construction Robustness

This diagnostic tests whether the advantage of DEFT-RLVR persists when candidate-set dificulty changes. We hold the scene, question, and oracle trajectory fixed and rebuild only the distractors. Unlike the structured Dev/Test construction in Appendix D.1, which combines scale-matched, constant-velocity, and hard-negative candidates, this diagnostic disables the first two sources and samples every distractor from the hard-negative similarity band. It therefore isolates candidate count and band width rather than reproducing the main AD-MCQ-500 candidate sets.

We vary the number of candidates as $M \ \in \ \{ 2 , 4 , 6 , 8 , 1 0 \}$ and the upper similarity bound as $\rho _ { \mathrm { m a x } } ~ \in$ $\{ 0 . 5 0 , 0 . 7 0 , 0 . 8 5 , 0 . 9 5 \}$ , while fixing $\rho _ { \mathrm { m i n } } ~ = ~ 0 . 3 0$ . Similarity is derived from the pairwise trajectory ADE within the codebook: $\rho = 1 - ( \mathrm { A D E } - d _ { \operatorname* { m i n } } ) / ( d _ { \operatorname* { m a x } } - d _ { \operatorname* { m i n } } )$ . For each setting, we evaluate DEFT (Training-Free), the DEFT-RLVR checkpoint, and DEFT Distillation (Mixed Targets) with two candidate-blind rounds of eight samples per question. The adapted checkpoints are DEFT-RLVR step 570 and DEFT Distillation (Mixed Targets) iteration 5750; decoding follows the AD-specific evaluation configuration in Appendix D.2. One malformed item is excluded consistently, leaving 499 paired questions.

<table><tr><td>M</td><td> $\rho _ { \mathrm { m a x } }$ </td><td>DEFT-TF</td><td>DEFT-RLVR</td><td>Mixed Distill.</td><td>Δ</td></tr><tr><td>2</td><td>0.50</td><td>88.3</td><td>95.8</td><td>97.6</td><td>+7.5</td></tr><tr><td>2</td><td>0.70</td><td>88.1</td><td>95.2</td><td>96.9</td><td>+7.1</td></tr><tr><td>2</td><td>0.85</td><td>84.8</td><td>90.8</td><td>94.5</td><td>+6.0</td></tr><tr><td>2</td><td>0.95</td><td>75.3</td><td>79.7</td><td>90.5</td><td>+4.4</td></tr><tr><td>4</td><td>0.50</td><td>73.3</td><td>88.7</td><td>98.8</td><td>+15.4</td></tr><tr><td>4</td><td>0.70</td><td>73.5</td><td>89.2</td><td>98.3</td><td>+15.7</td></tr><tr><td>4</td><td>0.85</td><td>67.8</td><td>83.4</td><td>95.0</td><td>+15.6</td></tr><tr><td>4</td><td>0.95</td><td>51.3</td><td>61.0</td><td>84.7</td><td>+9.7</td></tr><tr><td>6</td><td>0.50</td><td>65.6</td><td>85.7</td><td>99.0</td><td>+20.0</td></tr><tr><td>6</td><td>0.70</td><td>71.3</td><td>88.4</td><td>97.7</td><td>+17.0</td></tr><tr><td>6</td><td>0.85</td><td>67.3</td><td>81.0</td><td>94.3</td><td>+13.7</td></tr><tr><td>6</td><td>0.95</td><td>45.8</td><td>55.6</td><td>81.2</td><td>+9.8</td></tr><tr><td>8</td><td>0.50</td><td>57.3</td><td>82.1</td><td>98.7</td><td>+24.8</td></tr><tr><td>8</td><td>0.70</td><td>68.3</td><td>85.5</td><td>98.3</td><td>+17.2</td></tr><tr><td>8</td><td>0.85</td><td>60.6</td><td>78.1</td><td>93.7</td><td>+17.5</td></tr><tr><td>8</td><td>0.95</td><td>39.2</td><td>49.5</td><td>77.8</td><td>+10.3</td></tr><tr><td>10</td><td>0.50</td><td>44.3</td><td>64.1</td><td>79.5</td><td>+19.8</td></tr><tr><td>10</td><td>0.70</td><td>51.3</td><td>67.4</td><td>78.0</td><td>+16.1</td></tr><tr><td>10</td><td>0.85</td><td>47.0</td><td>60.7</td><td>73.3</td><td>+13.7</td></tr><tr><td>10</td><td>0.95</td><td>29.8</td><td>36.0</td><td>59.7</td><td>+6.2</td></tr></table>

Table 11 Candidate-set ablation accuracy (%). ∆: DEFT-RLVR gain over DEFT (Training-Free). Pure hard-negative distractors are resampled; each entry averages two candidate-blind rounds (eight samples per question).

DEFT-RLVR improves over DEFT (Training-Free) in all 20 settings, with gains ranging from 4.4% to 24.8%. The two-candidate setting places both models near a ceiling and ofers limited discrimination. At $\rho _ { \mathrm { m a x } } = 0 . 9 5$ both models are compressed by closely matched distractors, and the gain also narrows. Even at $M { = } 1 0$ and $\rho _ { \mathrm { m a x } } { = } 0 . 9 5$ , both models remain above the 10% chance level, so the hardest setting remains discriminative rather than collapsing to random choice. Intermediate thresholds exhibit small non-monotonic variation, so we do not interpret $\rho _ { \mathrm { m a x } }$ as a perfectly calibrated scalar measure of realized dificulty. DEFT Distillation (Mixed Targets) attains the highest accuracy throughout this grid, but its training targets and objective difer from those of DEFT-RLVR; the comparison is therefore descriptive rather than a controlled RL-versus-SFT attribution.

## E Human Validation of AD CoT Evaluation

We validate the automatic evaluation used for the two AD CoT metrics in the main results. We first pool candidate-blind Turn-1 CoTs from all methods evaluated on AD-MCQ-500 and then randomly sample 200 outputs from this combined pool. This audit is designed to measure human–judge agreement rather than compare individual methods, so the sample is not stratified by method. We randomly partition the sampled outputs into two disjoint subsets and assign one subset to each of two human annotators. The annotators work independently on their assigned subsets without seeing the model identity or the scores produced by the automatic judge, and each output receives exactly one human annotation. For CFS, the human annotators use the four-dimension $0 / 1 / 2$ ordinal rubric defined in Appendix B, summing grounding, absence of hallucination, specificity, and causal coherence to a score in [0, 8] and then normalizing it to [0, 1]. The automatic judge evaluates the same four conceptual dimensions but makes the binary decisions defined in Appendix F.6; its four $0 / 1$ outputs are averaged to obtain the automatic Normed-CFS in [0, 1]. We compute CFS agreement after placing both scores on this common normalized scale. For HLD, both the human annotator and automatic judge require the predicted direction and speed decision to agree with the action induced by the GT trajectory. We pool the resulting 200 non-overlapping human–judge pairs to compute the agreement statistics in Table 12. The Qwen3.5-397B-A17B judge and its prompt were fixed before the human labels were examined.

<table><tr><td>Metric</td><td>Agreement Measure</td><td>Result</td></tr><tr><td>CFS</td><td>Spearman&#x27;s ρ</td><td>0.78</td></tr><tr><td>CFS</td><td>Mean absolute error</td><td>0.067</td></tr><tr><td>CFS</td><td>Within 0.125 agreement</td><td>88.5%</td></tr><tr><td>HLD</td><td>Exact agreement</td><td>92.0%</td></tr><tr><td>HLD</td><td>Cohen&#x27;s κ</td><td>0.84</td></tr></table>

Table 12 Agreement between human annotations and the automatic judge on a random sample of 200 candidate-blind AD reasoning outputs. The two annotators evaluate disjoint subsets, so each output contributes one human–judge pair. The 0.125 tolerance corresponds to one point on the human annotator’s unnormalized 0–8 CFS scale.

Across both metrics, the human annotations are highly consistent with the automatic scores: CFS exhibits strong rank agreement and small absolute error, while HLD decisions show high exact and chance-corrected agreement. This audit supports the use of the automatic judge for the AD CoT metrics in the main table; candidate-selection accuracy remains exact-match based and is therefore outside the scope of this validation.

## F Prompt Templates

## F.1 Two-Turn Candidate-Grounded Policy

Figure 11 gives the first-turn message exactly as presented to the policy, up to example-specific variables. Each of the three video streams is decoded online into four historical frames at 2 fps. Qwen3-VL groups adjacent frames into temporal patches, so the rendered message displays two temporal-patch timestamps, 0.2 and 1.2 s; these two markers still correspond to four input frames. Candidate trajectories are deliberately absent from this message. The system message is shared by both turns, and generation uses temperature 1.0, top-p 0.95, and a maximum of 12,000 tokens.

Turn-1 Candidate-Blind Policy Prompt   
System Prompt   
You are a helpful assistant.   
User Message — Turn 1: Plan   
Visual input: Video 1 (front-left camera): <0.2 seconds><1.2 seconds>   
Video 2 (front camera): <0.2 seconds><1.2 seconds>   
Video 3 (front-right camera): <0.2 seconds><1.2 seconds>   
Multiple forward-facing camera streams are mounted on the ego vehicle; the frames above are ordered from oldest to newest.   
The ego vehicle behavior in the recent history is {EGO\_HISTORY}. The ego vehicle’s current velocity is {VEL\_X} m/s at   
x-direction and {VEL\_Y} m/s at y-direction. The ego vehicle’s current acceleration is {ACC\_X} m/s<sup>2</sup> at x-direction and   
{ACC\_Y} m/s<sup>2</sup> at y-direction. The current driving command instruction of ego vehicle is: {DRIVING\_CMD}, indicating   
the intended route direction. Note that the left and right driving commands cover turns, lane changes and sharp curves   
driving behavior.   
Trajectory coordinates are in meters in the ego frame: +x is forward, +y is to the left. Plan the ego’s 5-second future   
motion (t = 0.5s .. 5.0s).   
This is a multiple-choice problem, but solve it in TWO option-independent parts. In Part 1 reason CAUSALLY, as if you   
must plan the ego trajectory yourself; ignore that options exist until Part 2. Use only history-visible evidence; never cite   
unseen future events.   
PART 1 — PLAN (no options are shown to you; plan the ego maneuver from the scene alone)   
In 1–2 concise sentences each:   
1. Evidence: from the history-visible scene, identify only the few facts that causally constrain the future motion — the   
road/route structure and any element whose state changes what the ego can safely, legally, or feasibly do next.   
2. Causal chain: for each constraint reason scene → consequence (what becomes unsafe, illegal, infeasible, or of-route) →   
the maneuver it forces. Treat the ego’s CURRENT motion (its speed and acceleration) as only the starting condition to   
be acted upon, NEVER as evidence that the motion should continue unchanged: the plan follows from the scene, not from   
the current velocity. Unless the scene positively shows that continuing unchanged is safe and on-route, let the constraints   
— not the momentum — decide the maneuver.   
3. SPEED first, and firmly: decide whether to keep speed, slow, or stop from the hazards alone (signals, a lead or stopped   
vehicle, a crossing agent, a stop line, a tight curve or intersection). This decision holds REGARDLESS of which way the   
road goes — never default to holding speed just because the path looks clear or because you are unsure where the road   
leads.   
Then commit to a high-level decision on its own line in EXACTLY this format:   
HIGH\_LEVEL\_DECISION: <one firmly committed speed profile (e.g. decelerate to a stop / hold ∼Xm/s / slow then   
proceed) + exactly one firmly committed direction from straight / left / right; one sentence, with the main cause>

Figure 11: Turn-1 policy chat template. The policy observes the scene, ego state, and navigation command, but no candidate trajectory. The role-separated panel preserves the production message order, and variables in braces are instantiated per scene.  
Turn-2 Candidate-Trajectory Matching Prompt   
System Prompt   
Continuation of Turn 1; no new system message.   
User Message — Turn 2: Option Match   
PART 2 — OPTION MATCH (only now use the options)   
Below are the candidate future trajectories. Exactly one is the best choice   
Option A   
waypoints xy: [(x0,y0), ... 10 points ...]   
Option B   
waypoints xy: [...]   
Option C   
waypoints xy: [...]   
Option D   
waypoints xy: [...]   
Option E   
waypoints xy: [...]   
Option F   
waypoints xy: [...]   
Pick the candidate that best realizes your HIGH\_LEVEL\_DECISION. Both the committed SPEED profile (keep speed /   
slow / stop) and the single committed DIRECTION (straight / left / right) are binding: rule out any candidate inconsistent   
with either decision. Use the candidate waypoints only to match the already committed plan, never to resolve uncertainty   
or revise Part 1. If no candidate matches perfectly, choose the closest realization of the fixed speed and direction decision   
and briefly identify the residual mismatch.   
End your answer with a final line in EXACTLY this format (nothing after it):   
FINAL\_CHOICE: <one letter from [A, B, C, D, E, F]>  
Figure 12: Turn-2 policy chat template. The environment reveals six deterministically shufled candidate trajectories only after the policy has committed to its first-turn plan. The parser uses the last FINAL\_CHOICE field.

## F.2 Offline Question-Specific Rubric Generation

For DEFT-RLVR, a fixed Qwen3.6-35B-A3B rubric generator receives twelve scene frames, the first-turn task, and the ego-state summary, but no logged future, candidate trajectory, oracle label, or statistic derived from the future trajectory. The only scalar repeated outside the first-turn task is the current ego-speed norm, computed from the current planar velocity. The generator runs once ofline with temperature 0.3, top-p 0.9, and a maximum of 1,024 tokens. The resulting six to ten scene-specific criteria are stored with the training example and subsequently applied by the online text-only grader.

## Ofline Question-Specific Rubric-Generation Prompt

System Prompt — Rubric Generation   
Your task is to generate a self-contained rubric for autonomous-driving causal reasoning. You are given only the history visible driving scene (multi-view frames + ego state) and the task question. You are NOT given the logged future trajectory, candidate trajectories, or an oracle answer. Generate a set of binary, checkable evaluation criteria describing what a scene grounded, well-reasoned chain-of-thought (CoT) should contain for this scene. The grader who scores a candidate CoT against your criteria will see only the criterion text and the CoT — not the frames or any answer-related information. Each criterion must therefore state the relevant scene fact explicitly and be verifiable from the CoT text alone.   
CRITICAL CONSTRAINTS:   
1. HISTORY-ONLY: Construct every criterion solely from facts clearly supported by the provided history-visible frames, ego state, and navigation instruction. Never infer an unseen future event, hidden trafic-control state, or hypothetical agent behavior. If a cue is visually ambiguous, either omit it or state the uncertainty explicitly; never turn ambiguity into a definite fact.   
2. PROCESS, NOT OUTCOME: Evaluate how the CoT identifies visible constraints and reasons about their driving implications. Do NOT prescribe an oracle maneuver, exact future speed, trajectory, waypoint sequence, or option letter. Final maneuver correctness is evaluated separately by an exact outcome verifier. A criterion may require the CoT to explain how a visible cue constrains safe, legal, feasible, or on-route motion, but must not assume access to the future that actually occurred. 3. SCENE-GROUNDED & CONCRETE: Each criterion must reference a concrete cue clearly visible in these frames — a specific agent, vehicle, pedestrian, lane, trafic light, road feature, navigation instruction, or the ego speed/heading. This is what makes grading mechanical and stable. Generic criteria like “is specific” or “is coherent” are FORBIDDEN; replace them with a concrete cue check.   
4. CAUSAL RELEVANCE, NOT OBJECT LISTING: Reward a cue only when the CoT connects it to a driving consequence or constraint. Merely mentioning an object is insuficient. Prefer criteria of the form visible scene fact → safety/legal/feasibil ity/route implication; do not reward exhaustive scene description or unrelated object lists.   
5. REASONING FIDELITY: Criteria should reward reasoning that is grounded in visible cues, free of invented or unsupported facts, internally consistent, and relevant to the immediate driving decision — not just fluent or confident language. 6. ANTI-GAMING: Include 1–2 criteria that guard against reward-hacking, phrased as GOOD behavior with a POSITIVE weight — e.g., “avoids asserting a trafic-light state that is not clearly visible,” “avoids treating current momentum as suficient evidence to continue unchanged,” “avoids filler self-praise such as I am confident,” or “avoids switching language mid-response.” These are PRESENT (earn the weight) when the CoT does NOT do the bad thing, and NOT\_PRESENT when it does.   
RUBRIC PRINCIPLES (from OnlineRubrics): Mutually Exclusive & Collectively Exhaustive; each criterion Atomic (one idea); Binary (yes/no); Self-contained. 6–10 criteria. Weights are POSITIVE integers 1–10 (NO negative weights the paper’s synthetic rubrics are positive-only; anti-gaming is phrased as positive good-behavior criteria above). Output ONLY a JSON object, no code fence, no extra prose:   
{"initial\_reasoning": "brief", "rubrics": [{"criterion": "text", "weight": int}, ...]}   
User Message — Rubric Generation   
Task question (turn-1, the student has NOT seen any options yet):   
Video 1 (front-left camera): <0.2 seconds><1.2 seconds>   
Video 2 (front camera): <0.2 seconds><1.2 seconds>   
Video 3 (front-right camera): <0.2 seconds><1.2 seconds>   
<ego-state text and the complete PART 1 prompt in Figure 11>   
Ego state: velocity\_norm={VEL}.   
Now generate the rubric criteria for evaluating a candidate CoT about this scene, following the system constraints. Output the JSON object.   
Camera stream 1, oldest to newest.   
[image: <media\_root>/<token>/4\_resize/00.jpg]   
[image: <media\_root>/<token>/4\_resize/01.jpg]   
[image: <media\_root>/<token>/4\_resize/02.jpg]   
[image: <media\_root>/<token>/4\_resize/03.jpg]   
Camera stream 2, oldest to newest.   
[image: <media\_root>/<token>/2/00.jpg]   
... (four frames from stream 2) ...   
Camera stream 3, oldest to newest.   
[image: <media\_root>/<token>/7/00.jpg]   
(four frames from stream 3) ...

Figure 13: Ofline question-specific rubric-generation chat template. Criteria are constructed solely from historyvisible scene context, without the logged future, candidate trajectories, or an oracle label. Each criterion explicitl encodes a concrete scene constraint and its driving implication so that the image- and answer-blind online grader can check it from the CoT alone. The abbreviated repeated Part-1 block is exactly the text in Figure 11.

## F.3 Online Text-Only Rubric Grader

At rollout time, the grader sees only the stored criteria and the normalized first-turn CoT ye . It receives no image, oracle trajectory, option list, or high-level-decision line. We use temperature 1.0, top-p 0.7, and a maximum of 256 tokens.

Online Text-Only Rubric-Grader Prompt   
System Prompt — Text-Only Grader   
You are a strict text verifier for autonomous-driving reasoning. You see ONLY the numbered criteria and ONE candidate   
chain-of-thought (CoT) — you do NOT see any frames. You are NOT told the correct maneuver and must NOT assume any   
maneuver is correct. You will evaluate the CoT against the criteria by checking what the CoT TEXT states or omits. For   
EACH criterion: first state a short objective fact about what the CoT text claims or fails to claim, then derive PRESENT   
or NOT\_PRESENT mechanically from that fact. Treat any cue the CoT asserts as a textual claim only — do not assume   
it is grounded in the real scene, since you cannot see the scene. Never let fluency override what the CoT text actually says.   
PRESENT semantics: mark PRESENT if the criterion is satisfied by the CoT text, NOT\_PRESENT otherwise. If a   
criterion has multiple sub-conditions, NOT\_PRESENT unless ALL are met, EXCEPT “such as”/“for example”/“including”   
lists are illustrative (meeting any one sufices). For anti-gaming criteria phrased as good behavior (e.g., “avoids inventing   
X,” “avoids filler self-praise”): PRESENT when the CoT does NOT do the bad thing, NOT\_PRESENT when it does.   
User Message — Text-Only Grader   
Evaluate the candidate chain-of-thought against EACH numbered criterion below. For each, first state a one-line objec   
tive fact, then derive PRESENT or NOT\_PRESENT. Output ONLY a JSON object mapping each criterion number to   
PRESENT or NOT\_PRESENT. No prose around the JSON, no code fence.   
1. {criterion text} (weight {w1})   
2. {criterion text} (weight {w2})   
... (6–10 criteria) ...   
Output exactly: {"1":"PRESENT","2":"NOT\_PRESENT",...}   
<candidate chain-of-thought>   
{normalized turn-1 CoT}  
Figure 14: Online text-only grader chat template. The grader returns one binary decision per stored criterion; the weighted present rate supplies the process score for an outcome-correct rollout.

## F.4 Image-Conditioned Rubric Reward for the Controlled Variant

For DEFT + RLVR $\mathfrak { ( R ^ { \mathrm { M C Q } } } R ^ { \mathrm { G E N } } )$ , a Qwen3.6-35B-A3B judge scores $\widetilde { y } _ { 1 }$ from twelve scene frames using temperature 1.0, top-p 0.7, and at most 128 tokens. This controlled variant retains the two-turn DEFT interface but replaces DEFT-RLVR’s question-specific rubric and text-only grader with a shared image-conditioned four-axis rubric.

Image-Conditioned Rubric-Judge Prompt   
System Prompt — Image-Conditioned Judge   
You are a strict perception verifier for autonomous-driving reasoning. You see the scene (multi-view frames + ego state)   
and ONE candidate chain-of-thought. You are NOT told the correct maneuver and must NOT assume any maneuver is   
correct. For each axis you will first state a short objective fact about the frames, then derive the verdict mechanically from   
that fact. State only what is clearly visible; if a cue is not clearly visible, treat it as absent. Never let fluency override what   
the frames show.   
User Message — Image-Conditioned Judge   
For each axis output ONE line in EXACTLY this form: ’<AXIS>: <fact-tag> -> <YES|NO>’. No other text. Use these   
fact-tags and rules:   
GROUNDING: fact-tag = ’cited-visible=all’ if every cited object/agent/road-feature is clearly visible, else ’cited  
visible=some/none’. Rule: all → YES; some/none → NO.   
NO\_HALLUCINATION: fact-tag = ’unsupported-claims=none’ if no claim contradicts/is invented by the frames, else   
’unsupported-claims=any’. Rule: none → YES; any → NO.   
SPECIFICITY: fact-tag = ’concrete-cue-named=yes’ if the reasoning names at least one concrete visible cue by type/po  
sition, else ’no’. Rule: yes → YES; no → NO.   
COHERENCE: fact-tag = ’self-contradictions=none’ if no two statements assert incompatible facts, else ’any’. Rule:   
none → YES; any → NO.   
Output EXACTLY these four lines, nothing else:   
GROUNDING: <tag> -> <YES|NO>   
NO\_HALLUCINATION: <tag> -> <YES|NO>   
SPECIFICITY: <tag> -> <YES|NO>   
COHERENCE: <tag> -> <YES|NO>   
<twelve frames from the three camera streams>   
<candidate chain-of-thought>   
{normalized turn-1 CoT}  
Figure 15: Image-conditioned four-axis rubric-judge template for the controlled variant (axis weights: 0.30/0.30/0.10/0.30).

## F.5 Joint-Exposure Policy Prompt (JEFT)

Figure 16 gives JEFT’s matched single-turn prompt: the same scene context, ego state, navigation command, candidates, reasoning requirements, and outputs as DEFT, but with all six candidates preceding both reasoning and the high-level decision. JEFT uses temperature 1.0, top-p 0.95, and at most 24,576 tokens, matching DEFT’s total generation budget.

![](images/780088eb3056654bd3b312cb598d120a881c790e90d4f912b6e4877d9cf0b56d.jpg)  
Figure 16: JEFT joint-exposure policy template with all candidate trajectories visible before reasoning.

## F.6 CFS and HLD Evaluation-Judge Prompts

We use the following fixed prompts to evaluate the two open-ended AD metrics reported in the main results. The CFS judge is GT-blind and evaluates the candidate-blind Turn-1 CoT against the visible scene along grounding, absence of hallucination, specificity, and coherence. The HLD judge receives the predicted highlevel decision and GT maneuver and requires agreement in both direction and speed.

CFS Evaluation Judge Prompt   
System Prompt   
You are a strict perception verifier for autonomous-driving reasoning. You see the scene (multi-view frames + ego state)   
and ONE candidate chain-of-thought. You are NOT told the correct maneuver and must NOT assume any maneuver is   
correct. For each axis you will first state a short objective fact about the frames, then derive the verdict mechanically from   
that fact. State only what is clearly visible; if a cue is not clearly visible, treat it as absent. Never let fluency override what   
the frames show.   
User Message   
For each axis output ONE line in EXACTLY this form: ’<AXIS>: <fact-tag> -> <YES|NO>’. No other text. Use these   
fact-tags and rules:   
GROUNDING: fact-tag = ’cited-visible=all’ if every cited object/agent/road-feature is clearly visible, else ’cited  
visible=some/none’. Rule: all → YES; some/none → NO.   
NO\_HALLUCINATION: fact-tag = ’unsupported-claims=none’ if no claim contradicts/is invented by the frames, else   
’unsupported-claims=any’. Rule: none → YES; any → NO.   
SPECIFICITY: fact-tag = ’concrete-cue-named=yes’ if the reasoning names at least one concrete visible cue by type/po  
sition, else ’no’. Rule: yes → YES; no → NO.   
COHERENCE: fact-tag = ’self-contradictions=none’ if no two statements assert incompatible facts, else ’any’. Rule:   
none → YES; any → NO.   
Output EXACTLY these four lines, nothing else:   
GROUNDING: <tag> -> <YES|NO>   
NO\_HALLUCINATION: <tag> -> <YES|NO>   
SPECIFICITY: <tag> -> <YES|NO>   
COHERENCE: <tag> -> <YES|NO>   
<multi-view scene frames and ego state>   
<candidate chain-of-thought>   
{candidate-blind Turn-1 CoT}

HLD Evaluation Judge Prompt   
System Prompt   
You are a strict trajectory-matching verifier for autonomous-driving decisions. You are given ONE predicted high-level   
decision (free text: direction + speed regime + cause) and the GROUND-TRUTH future maneuver (a semantic action label   
+ the gt waypoint xy sequence). Your job is to decide whether the PREDICTED decision matches the GROUND-TRUTH   
maneuver along two independent axes: DIRECTION and SPEED. For each axis first state a short objective fact, then   
derive the verdict mechanically. Judge the predicted decision ONLY against the ground truth, not against what would be   
a safe/legal drive.   
User Message   
GROUND-TRUTH maneuver label: {oracle action label}   
GROUND-TRUTH waypoints xy: {first 10 oracle waypoints}   
gt summary: {net displacement, number of points, and mean lateral displacement}   
PREDICTED high-level decision:   
{HIGH\_LEVEL\_DECISION text}   
For each axis output ONE line in EXACTLY this form: ’<AXIS>: <fact-tag> -> <YES|NO>’. No other text. Use these   
fact-tags and rules:   
DIRECTION: fact-tag = ’pred-dir=<L|R|S|LC>’ derived from the predicted decision’s lateral intent (left=L, right=R,   
straight/forward=S, lane-change=LC, unsure=U); compare to the gt maneuver’s lateral intent (from the label + the sign   
of gt waypoints’ y). Rule: same lateral intent → YES; diferent (e.g., pred straight vs. gt left) → NO; either unsure → NO.   
SPEED: fact-tag = ’pred-spd=<stop|decel|constant|accel>’ derived from the predicted speed regime; compare to the gt   
maneuver’s speed regime (from the label; use gt waypoints as confirmation: a gt trajectory that keeps moving far =   
constant/accel, one that halts = stop, one that shortens = decel). Rule: same regime → YES; diferent (e.g., pred stop vs.   
gt acceleration) → NO; unsure → NO.   
Output EXACTLY these two lines, nothing else:   
DIRECTION\_MATCH: <tag> -> <YES|NO>   
SPEED\_MATCH: <tag> -> <YES|NO>

## G Detailed Candidate-Trajectory MCQ Case Studies

Figures 17–18 show two representative DEFT-RLVR rollouts at signal- and stop-controlled intersections.

![](images/7b770aca2937beadc6928d1025d9e8a8bbf0c82c70894117f78b89e40d2e6091.jpg)  
Figure 17 Complete two-turn MCQ case for a red-light stop on a wet, construction-constrained approach. The policy identifies the signal before seeing options and then matches that commitment to the stopping trajectory A.

## Case 2: Stop-Sign Compliance under a Constant-Speed History

![](images/7ef663f4675798c68c0a299bd275893c7cd9fdb1b899bb6b2ae9cc0a00d4090f.jpg)

![](images/9b975bdf030187cb4e871a2d37a71db33572b80f0d5e9689808265d8f5612cc6.jpg)  
Figure 18 Complete two-turn MCQ case for a stop-controlled intersection. Despite a constant-speed history, the candidate-blind plan is governed by the visible stop control, and trajectory C is selected only after this commitment.

## H Full General Visual Capability Results

Tables 13–14 expand the four category aggregates in Table 2 into the complete 12-benchmark evaluation. JEFT-based methods are task-matched conventional baselines; DEFT-based rows are variants of our framework.
<table><tr><td></td><td colspan="3">Basic Visual</td><td colspan="3">Embodied Spatial</td></tr><tr><td>Method</td><td>CV2D</td><td>CV3D</td><td>DA2K</td><td>ERQA</td><td>EmbSpat</td><td>RoboSpat</td></tr><tr><td>Qwen3-VL-8B-Instruct</td><td>81.88</td><td>93.83</td><td>69.10</td><td>43.00</td><td>77.75</td><td>49.14</td></tr><tr><td>十  $\mathrm { J E F T } + \mathrm { R L V R } \ ( R ^ { \mathrm { M C Q } } )$ </td><td>80.97</td><td>93.92</td><td>69.29</td><td>44.00</td><td>78.30</td><td>47.71</td></tr><tr><td>十  $\mathsf { D E F T } + \mathsf { R L V R } ( R ^ { \mathrm { M C Q } } )$ </td><td>81.41</td><td>93.58</td><td>69.10</td><td>44.75</td><td>78.41</td><td>48.29</td></tr><tr><td>十  $\mathsf { D E F T } + \mathsf { R L V R } ( R ^ { \mathrm { M C Q } } R ^ { \mathrm { G E N } } )$  + DEFT-RLVR (ours)</td><td>81.29</td><td>94.08</td><td>69.39</td><td>42.75</td><td>78.19</td><td>48.00</td></tr><tr><td></td><td>80.85</td><td>94.00</td><td>69.15</td><td>45.00</td><td>77.80</td><td>49.43</td></tr><tr><td>+DEFT Distillation (Plan Only)</td><td>78.12</td><td>88.67</td><td>67.21</td><td>39.25</td><td>74.09</td><td>47.14</td></tr><tr><td>+DEFT Distillation (Full Interaction)</td><td>75.96</td><td>88.50</td><td>60.35</td><td>39.00</td><td>75.69</td><td>47.43</td></tr><tr><td>+DEFT Distillation (Mixed Targets)</td><td>76.25</td><td>90.83</td><td>63.10</td><td>42.50</td><td>74.04</td><td>44.00</td></tr><tr><td>Qwen3.5-4B</td><td>82.09</td><td>91.58</td><td>67.26</td><td>47.25</td><td>74.04</td><td>37.71</td></tr><tr><td> $+ \ \mathrm { J E F T } + \mathrm { R L V R } \ ( R ^ { \mathrm { M C Q } } )$ </td><td>81.55</td><td>91.58</td><td>67.55</td><td>47.75</td><td>72.83</td><td>39.43</td></tr><tr><td>十  $\mathsf { D E F T } + \mathsf { R L V R } ( R ^ { \mathrm { M C Q } } )$ </td><td>82.27</td><td>91.50</td><td>68.52</td><td>45.75</td><td>73.65</td><td>38.86</td></tr><tr><td> $+ \ \mathsf { D E F T } + \mathsf { R L V R } ( R ^ { \mathrm { M C Q } } R ^ { \mathrm { G E N } } )$ </td><td>81.81</td><td>92.00</td><td>67.41</td><td>47.50</td><td>73.74</td><td>38.86</td></tr><tr><td>+ DEFT-RLVR (ours)</td><td>82.13</td><td>92.42</td><td>68.09</td><td>50.00</td><td>73.60</td><td>42.00</td></tr></table>

Table 13 Benchmark-level results for Basic Visual and Embodied Spatial capabilities. Relative to the corresponding base model, DEFT-RLVR improves five of the six benchmarks for both backbones, with mean gains of 0.26% for Qwen3-VL 8B and 1.39% for Qwen3.5-4B; the largest gains are 2.00% on ERQA and 4.29% on RoboSpat, respectively.

<table><tr><td rowspan="2"></td><td colspan="3">3D/Multi-View</td><td colspan="3">RefSpatial</td></tr><tr><td>3DSR</td><td>MMSI</td><td>ViewSpat</td><td>RefLoc</td><td>RefPlc</td><td>RefUns</td></tr><tr><td>Qwen3-VL-8B-Instruct</td><td>55.21</td><td>30.70</td><td>41.54</td><td>55.00</td><td>32.00</td><td>28.57</td></tr><tr><td> $+ \ \mathrm { J E F T } + \mathrm { R L V R } \ ( R ^ { \mathrm { M C Q } } )$ </td><td>54.88</td><td>29.70</td><td>41.39</td><td>57.00</td><td>37.00</td><td>24.68</td></tr><tr><td> $+ \mathsf { D E F T } + \mathsf { R L V R } ( R ^ { \mathrm { M C Q } } )$ </td><td>54.88</td><td>31.50</td><td>42.02</td><td>56.00</td><td>43.00</td><td>33.77</td></tr><tr><td> $+ \ \mathsf { D E F T } + \mathsf { R L V R } ( R ^ { \mathrm { M C Q } } R ^ { \mathrm { G E N } } )$ </td><td>55.09</td><td>32.00</td><td>41.33</td><td>57.00</td><td>41.00</td><td>36.36</td></tr><tr><td>+ DEFT-RLVR (ours)</td><td>55.17</td><td>30.00</td><td>41.65</td><td>54.00</td><td>41.00</td><td>35.06</td></tr><tr><td>+DEFT Distillation (Plan Only)</td><td>50.67</td><td>27.50</td><td>42.42</td><td>45.00</td><td>32.00</td><td>23.38</td></tr><tr><td>+DEFT Distillation (Full Interaction)</td><td>49.62</td><td>27.30</td><td>44.22</td><td>44.00</td><td>35.00</td><td>24.68</td></tr><tr><td>+DEFT Distillation (Mixed Targets)</td><td>50.42</td><td>26.80</td><td>43.38</td><td>48.00</td><td>35.00</td><td>27.27</td></tr><tr><td>Qwen3.5-4B</td><td>45.98</td><td>33.50</td><td>42.72</td><td>51.00</td><td>29.00</td><td>28.57</td></tr><tr><td> $+ \ \mathrm { J E F T } + \mathrm { R L V R } \ ( R ^ { \mathrm { M C Q } } )$ </td><td>45.69</td><td>32.10</td><td>42.68</td><td>44.00</td><td>24.00</td><td>25.97</td></tr><tr><td> $+ \mathsf { D E F T } + \mathsf { R L V R } ( R ^ { \mathrm { M C Q } } )$ </td><td>48.30</td><td>32.80</td><td>44.38</td><td>49.00</td><td>36.00</td><td>31.17</td></tr><tr><td> $+ \ \mathsf { D E F T } + \mathsf { R L V R } ( R ^ { \mathrm { M C Q } } R ^ { \mathrm { G E N } } )$ </td><td>46.09</td><td>32.90</td><td>43.03</td><td>55.00</td><td>31.00</td><td>29.87</td></tr><tr><td>+ DEFT-RLVR (ours)</td><td>46.50</td><td>33.30</td><td>43.01</td><td>53.00</td><td>34.00</td><td>20.78</td></tr></table>

Table 14 Benchmark-level results for 3D/Multi-View and RefSpatial capabilities. Relative to the corresponding base model, DEFT-RLVR changes the Qwen3-VL-8B 3D/Multi-View mean by −0.21% while improving its RefSpatial mean by 4.83%, including gains of 9.00% on RefPlc and 6.49% on RefUns. For Qwen3.5-4B, the six-benchmark mean is largely preserved (−0.03%), with changes of +0.20% on 3D/Multi-View and −0.26% on RefSpatial.