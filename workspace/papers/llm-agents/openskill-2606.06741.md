# OPENSKILL: Open-World Self-Evolution for LLM Agents

Zhiling Yan1,∗, Dingjie Song1,∗, Hanrong Zhang2, Wei Liang1, Yuxuan Zhang3,4, Yutong Dai5, Lifang He1, Philip S. Yu2, Ran Xu5, Xiang Li6, Lichao Sun1,†

1Lehigh University, 2University of Illinois Chicago, 3University of British Columbia, 4Vector Institute, 5Salesforce AI Research, 6Massachusetts General Hospital and Harvard Medical School ∗Equal contribution, †Corresponding author

Self-evolving agents requires adaptation after deployment, but existing approaches assume a usable learning loop, such as curated skills, successful trajectories, or verifier signals. Real open-world deployments may provide none of these, offering only a task prompt. In this work, we study openworld self-evolution, where an agent must build both its skills and its own verification signals from scratch, using open-world resources but no target-task supervision. We propose OpenSkill, a framework that bootstraps this loop: it acquires grounded knowledge and verification anchors from documentation, repositories, and the web, synthesizes them into transferable skills, and refines those skills against self-built virtual tasks grounded in the anchors rather than in target answers. The open world thus supplies both the knowledge to be learned and a supervision-independent practice environment, with target-task supervision reserved for final evaluation. Across three benchmarks and two target agents, OpenSkill attains the best automated pass rate while satisfying the no-supervision constraint. Analysis shows its skills transfer across models without model-specific adaptation, and its self-built verifier aligns with ground-truth outcomes despite never accessing them.

Code: https://github.com/OpenLAIR/OpenSkill Website: https://openlair.github.io/openskill/

![](images/3c2c1df9ae14f788a4bb86720ae15d23db377f32ac2a209826f5cf0fc6ae03a4.jpg)

## 1 Introduction

LLM agents can use tools and external resources to solve open-ended tasks beyond text generation (Yao et al., 2022; Schick et al., 2023; Zhou et al., 2023; Nakano et al., 2021; Yang et al., 2024). Because such tasks change across environments, self-evolving agents aim to improve after deployment by accumulating reusable knowledge or behavior (Shinn et al., 2023; Wang et al., 2023; Zhang et al., 2026b; Xia et al., 2026).

Existing self-evolving agents often assume a usable learning loop, such as curated skills, successful traces, or task feedback (Zhang et al., 2026b; Xia et al., 2026). Real open-world deployments may provide only a seed task prompt, with no initial skills or verifier for judging improvement.

We call this capability open-world self-evolution (Figure 1). Here the agent starts from only a seed task prompt and open-world resources, and must build both its skills and its verification signals from scratch. These resources include independently accessible evidence such as documentation, repositories, papers, tutorials, and web pages, but exclude hidden target answers, rewards, verifier outputs, or solution traces. Facts alone are not enough: the same evidence must supply the learning loop itself. This means building two coupled components: skill content that captures what to learn, and a verification signal that can improve those skills without target-task supervision.

Limitation 1: skill construction. Existing approaches often rely on human-written skills, model-generated knowledge, or skills distilled from successful trajectories (Wang et al., 2023; Zhang et al., 2026b; Huang et al., 2025a; Ma et al., 2026; Yang et al., 2026). These sources are costly, bounded by prior knowledge, or unavailable before successful task attempts. In an open-world setting, the agent must instead infer what to learn, acquire external evidence, and turn it into reusable skills. This matters because many tasks require current or domain-specific knowledge, and recent benchmarks show that skill quality is often the limiting

![](images/be2ff7e3f845ffe93e39efa2623908e29e3b64cc0fd19b75f23cdeb0bbccec10.jpg)  
Figure 1 Paradigms for self-evolving agent skills. Unlike human-curated, LLM-generated, or supervised selfevolution paradigms, OpenSkill (ours) acquires skills from the open world and verifies them with self-built virtual tasks, making it simultaneously scalable, grounded, and supervision-free.

factor (Li et al., 2026; Liu et al., 2026b).

Limitation 2: verification construction. Existing self-improvement loops often revise behavior using tasklevel feedback, self-feedback, or verifier outputs (Shinn et al., 2023; Zhou et al., 2023; Madaan et al., 2023; Zhang et al., 2026b,a; Wang et al., 2025). This works in curated benchmarks, but open-world deployment may expose no reliable feedback during learning. The agent must therefore construct a separate practice environment whose supervision comes from open-world knowledge rather than hidden target-task answers.

This points to a sharper central question:

## Can an LLM agent self-evolve in the open world?

To answer this, we propose OpenSkill, a framework for open-world self-evolution. Given only a task prompt, a base model, tool access, and open-world resources, OpenSkill bootstraps a learning loop from scratch. It proceeds in three stages: open-world knowledge acquisition retrieves grounded knowledge and verification anchors from the open world; leakage-free skill evolution drafts skills and refines them against self-built virtual tasks rather than target answers; and zero-shot target evaluation deploys the refined skill to the target agent. Open-world resources thus supply both the knowledge to be learned and a supervision-independent practice environment, and target-task supervision is reserved for final evaluation alone. Empirically, OpenSkill delivers the best automated pass rate in every benchmark–agent setting (e.g., +8.9 / +8.8 over the strongest closed-world baseline on SkillsBench), transfers across models without adaptation, and builds a verifier covering 88.9% of ground-truth test intents—all without target-task supervision during learning. This paper makes three contributions:

• We define open-world self-evolution: from only a task prompt, an agent must build both its skills and its own verification signals from open-world resources, with no target-task supervision.

• We propose OpenSkill, which bootstraps this loop and yields skills that transfer across models, with a practice environment for refining them.

• We show that OpenSkill achieves the best automated pass rate across three benchmarks and two model families, and transfers across models—without target-task supervision during learning.

## 2 Open-World Self-Evolution

We introduce open-world self-evolution, a setting in which an LLM agent must improve from only a task prompt and open-world resources, with no initial skills, demonstrations, rewards, or verifiers. We formalize it and its supervision constraint (Section 2.1), then present OpenSkill, a three-stage pipeline that acquires open-world knowledge, refines skills against self-generated virtual tasks, and deploys them zero-shot (Section 2.2).

![](images/e1f76bec8611e2a931c1c59a9c889962452e6b8d7823c9f2f0e05e579b8d309e.jpg)  
Figure 2 Overview of the OpenSkill framework. A base agent acquires open-world knowledge from external resources to build a skill plan, then iteratively generates, executes, and refines the skill in a sandbox, using a virtual-task verifier and diagnostic retriever to fix bugs and knowledge gaps. A leakage barrier keeps target supervision out of skill construction, unlocking it only for final evaluation.

## 2.1 Problem Setting

Consider a set of $n$ target tasks $\{ ( \mathcal { T } _ { i } , \mathcal { E } _ { i } ) \} _ { i = 1 } ^ { n }$ where $\mathcal { T } _ { i }$ is a natural-language instruction and $\mathcal { E } _ { i }$ is an execution environment. An LLM agent $\pi _ { \theta }$ executes in $\mathcal { E } _ { i }$ and produces a terminal state $x _ { i } = \pi _ { \theta } ( \mathcal { T } _ { i } , \mathcal { E } _ { i } )$ . Each task is paired with a ground-truth test suite $\mathcal { T } _ { i } ^ { \mathrm { G T } } \in \{ 0 , 1 \}$

An agent can self-evolve either by modifying its weights $\theta \ ( \mathrm { e . g . }$ , via fine-tuning or reinforcement learning) or by augmenting its context with external knowledge artifacts. We adopt the latter, as it is computationally cheap, transferable across models, and inspectable, whereas weight modification is expensive, model-specific, and opaque. We formalize such an artifact as a skill set $S _ { i } = \{ s _ { i , 1 } , \ldots , s _ { i , m } \}$ that conditions the agent’s behavior, $x _ { i } = \pi _ { \theta } ( \mathbb { Z } _ { i } , \mathbb { S } _ { i } , \mathcal { E } _ { i } )$ , without altering θ. Self-evolution then reduces to constructing, for each task, a skill set under which the augmented agent passes its test suite: $\mathcal { T } _ { i } ^ { \mathrm { G T } } ( \pi _ { \theta } ( \mathbb { Z } _ { i } , \mathcal { S } _ { i } , \mathcal { E } _ { i } ) ) = 1$

The difficulty of the open-world setting is that this construction must proceed without target-task supervision. The agent observes only the instruction $\mathcal { T } _ { i }$ and the environment $\mathcal { E } _ { i } ;$ the ground-truth test suite $\mathcal { T } _ { i } ^ { \mathrm { G T } }$ , reference solutions, and human feedback are hidden. Observing the input $( { \mathcal { T } } _ { i } , { \mathcal { E } } _ { i } )$ is not supervision: we reserve the term supervision for dependence on the hidden signal $\mathcal { T } _ { i } ^ { \mathrm { G T } }$ (a gold answer, reward, or verifier output). Inspecting the input is thus permitted, while any dependence on $\overset { \sim } { \mathcal { T } _ { i } ^ { \mathrm { G T } } }$ is not. We call a construction procedure $f$ supervision-free when it neither observes $\mathcal { T } _ { i } ^ { \mathrm { G T } }$ during skill construction nor reverse-engineers it to build the virtual tests used for refinement (Section 2.2.2). The skill set is then built purely from the observable input:

$$
\hat { S } _ { i } = f ( { \mathbb T } _ { i } , { \mathcal E } _ { i } ) .\tag{1}
$$

The observable input alone, however, is rarely sufficient to construct competent skills. We therefore let the agent interact with the open world and acquire open-world knowledge $\kappa .$ such as public documentation, code repositories, papers, and tutorials, none of which reveals target-task supervision. The construction problem then expands to:

$$
\begin{array} { r } { \hat { S } _ { i } = f ( { \mathbb T } _ { i } , { \mathcal E } _ { i } , { \cal K } ) , } \end{array}\tag{2}
$$

where $f$ is the proposed OpenSkill pipeline.

## 2.2 The OPENSKILL Pipeline

OpenSkill includes three stages (Figure 2): acquiring domain knowledge from the open world, refining skills against self-generated virtual tasks, and deploying the final skill zero-shot to the target agent. The stages are

connected by the artifacts they pass on: Stage 1 produces $k _ { i } , p _ { i } .$ , and $k _ { i } ^ { v } ;$ Stage 2 drafts skills from $( p _ { i } , k _ { i } )$ and refines them against virtual tests built from $k _ { i } ^ { v }$ , emitting a frozen $\hat { S } _ { i } ^ { * }$ that Stage 3 deploys. A leakage barrier keeps the hidden ${ \mathcal { T } } _ { i } ^ { \mathrm { G T } }$ out of Stages 1–2.

## 2.2.1 Stage 1: Open-World Knowledge Acquisition

Prior skill creation methods (Anthropic, 2025; Yang et al., 2026) construct skills entirely from the LLM’s parametric knowledge $\kappa _ { \theta } .$ , the knowledge stored in the frozen weights $\theta ,$ as opposed to the open-world knowledge K of Section 2.1. This limits the resulting skills to what the model already knows, which is insufficient for tasks that require up-to-date APIs, project-specific conventions, or niche domain rules.

OpenSkill expands the knowledge base by querying the open world. Given $( { \mathcal { T } } _ { i } , { \mathcal { E } } _ { i } )$ the pipeline first retrieves task-relevant knowledge from the open world $k _ { i } = { \mathcal { D } } ( { \mathcal { T } } _ { i } , { \mathcal { K } } ) , k _ { i } \subset { \mathcal { K } } .$ , where $\mathcal { D }$ is an open-world retrieval function that traverses K and returns knowledge documents containing background concepts, best practices, API documentation, and source citations. A structured skill plan $p _ { i }$ is then synthesized based on $( \mathbb { Z } _ { i } , \mathcal { E } _ { i } , k _ { i } )$ , specifying the skill architecture, key procedures, and domain rules (implementation in Appendix B.1).

In addition to knowledge for skill construction, the pipeline retrieves verification knowledge $k _ { i } ^ { v } = \mathcal { D } ^ { v } ( \mathcal { T } _ { i } , \mathcal { K } )$ $k _ { i } ^ { v } \subset { \cal K } .$ which provides independently verifiable anchors for later quality assessment, including reference values from official documentation, statistical invariants of well-known datasets, cross-validation procedures from domain standards, and expected output formats. $k _ { i } ^ { v }$ is used in Stage 2 to ground virtual test generation (implementation in Appendix B.2).

To prevent answer leakage, all queries issued by D and $\mathcal { D } ^ { v }$ are filtered to exclude the benchmark name and any identifiers that could lead to $\mathcal { T } ^ { \mathrm { G T } }$ ; we audit this information isolation in Appendix F.

## 2.2.2 Stage 2: Leakage-Free Skill Evolution

Given $( \mathbb { Z } _ { i } , \mathcal { E } _ { i } , p _ { i } , k _ { i } )$ , the base agent $\pi _ { \theta }$ generates an initial skill set $\hat { S } _ { i } ^ { ( 0 ) } = \{ \hat { s } _ { i , 1 } ^ { ( 0 ) } , \ldots , \hat { s } _ { i , m } ^ { ( 0 ) } \}$ , whose size m $( 1 \leq m \leq 4 )$ is fixed by the Stage-1 plan $p _ { i }$ , and refines all m skills jointly within a single agent session.

To assess the skill without ground-truth feedback, the pipeline constructs a virtual test suite $\tilde { \mathcal { T } } _ { i } = \{ \tilde { t } _ { i , 1 } , \dots , \tilde { t } _ { i , K } \}$ grounded in the verification knowledge $k _ { i } ^ { v }$ obtained in Stage 1:

$$
\begin{array} { r } { \tilde { \mathcal { T } } _ { i } = g ( \mathcal { T } _ { i } , \mathcal { E } _ { i } , k _ { i } ^ { v } ) , } \end{array}\tag{3}
$$

$\tilde { \mathcal { T } } _ { i }$ serves as a proxy for $\mathcal { T } _ { i } ^ { \mathrm { G T } }$ to guide skill refinement. The pipeline $\textit { f } \left( \mathrm { E q . \ 2 } \right)$ calls $g$ internally: it scores each round’s skills with $g \mathrm { ^ s }$ tests to drive refinement, never observing $\mathcal { T } _ { i } ^ { \mathrm { G T } }$ Each virtual test $\tilde { t } _ { i , k } \in \{ 0 , 1 \}$ is a deterministic assertion anchored to independently verifiable facts rather than guessing what the ground-truth tests might check. For example, it checks the known row count of a public dataset, the expected range of a standard metric, or the documented output format of a library function. The generator $g$ is realized as an isolated verifier LLM session that emits a deterministic pytest suite (Appendix B.3).

The pipeline iterates for up to J rounds. At each round $j ,$ the current skill set $\hat { S } _ { i } ^ { ( j ) }$ is executed and evaluated against $\tilde { \tau } _ { i } .$ . The virtual pass rate:

$$
\tilde { r } ^ { ( j ) } = \frac { 1 } { | \tilde { \mathcal { T } } _ { i } | } \sum _ { k = 1 } ^ { K } \tilde { t } _ { i , k } \big ( \pi _ { \theta } ( \mathcal { T } _ { i } , \hat { { S } } _ { i } ^ { ( j ) } , \mathcal { E } _ { i } ) \big )\tag{4}
$$

serves as a proxy for skill quality. This proxy is reliable only if r˜ aligns with the hidden $\mathcal { T } _ { i } ^ { \mathrm { G T } }$ ; otherwise refinement may reward skills that overfit the virtual tests. We treat this alignment as an empirical question and measure it directly in Section 4.2.

Diagnostic-driven refinement. When $\tilde { r } ^ { ( j ) } < 1$ , the pipeline produces a structured failure diagnostic $\mathcal { F } ^ { ( j ) }$ comprising per-assertion results, root-cause analysis, and revision suggestions, then refines the skill:

$$
\hat { S } _ { i } ^ { ( j + 1 ) } = \pi _ { \theta } ( \hat { S } _ { i } ^ { ( j ) } , \mathcal { F } ^ { ( j ) } \mid \mathcal { T } _ { i } , \mathcal { E } _ { i } , p _ { i } , k _ { i } ) .\tag{5}
$$

When the diagnostic indicates a knowledge gap rather than an implementation bug, the pipeline triggers a targeted retrieval $k _ { i } ^ { ( \mathrm { g a p } ) } = \mathcal { D } ( \mathcal { F } ^ { ( j ) } , \mathcal { K } )$ and injects the result into the refinement context. The gap-versus-bug decision is made by an LLM classifier; we detail it and the targeted-retrieval budget in Appendix B.5.

The loop terminates when $\tilde { r } ^ { ( j ) } = 1$ or after at most J refinement rounds, whichever comes first; if $\tilde { r } ^ { ( j ) } < 1$ throughout, it exhausts the J-round budget (we set J=3). Auxiliary stall- and budget-based early stops are detailed in Appendix B.4.

## 2.2.3 Stage 3: Zero-Shot Target Evaluation

After evolution, the final skill set $\hat { S } _ { i } ^ { * }$ —the last refined version at loop termination, edited in place rather than a best-of-N snapshot (Appendix B.4, B.6)—is deployed to the target agent $\pi _ { \theta ^ { \prime } }$ in a zero-shot setting: the agent executes the target tasks with $\hat { S } _ { i } ^ { * }$ , and $\mathcal { T } _ { i } ^ { \mathrm { G T } } \in \{ 0 , 1 \}$ determines pass or fail. Because the skill is an explicit artifact rather than model weights, it can be deployed to any target agent without retraining.

We evaluate the performance of $\pi _ { \theta ^ { \prime } }$ by the average pass rate on the hidden ground-truth tests:

$$
\mathrm { P a s s R a t e } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \mathcal { T } _ { i } ^ { \mathrm { G T } } \big ( \pi _ { \theta ^ { \prime } } ( \mathcal { T } _ { i } , \hat { S } _ { i } ^ { * } , \mathcal { E } _ { i } ) \big ) .\tag{6}
$$

Note that the target agent $\pi _ { \theta ^ { \prime } }$ need not be the construction agent $\pi _ { \theta } \colon$ because $\hat { S } _ { i } ^ { * }$ is a portable artifact, skills built with one model can be deployed on another, and the hidden $\mathcal { T } _ { i } ^ { \mathrm { G T } }$ enters the pipeline only here, at final evaluation.

## 3 Experiment

## 3.1 Experimental Setup

Benchmarks. We evaluate on three agentic benchmarks. SkillsBench (Li et al., 2026) is our primary benchmark, spanning 11 task domains where skill quality is the limiting factor; SocialMaze (Xu et al., 2025) (social reasoning) and ScienceWorld (Wang et al., 2022a) (interactive science) add two distinct task types. All are run under the open-world protocol of Section 2, with the ground-truth tests $\mathcal { T } _ { i } ^ { \mathrm { G T } }$ hidden during construction and consulted only at final evaluation; full dataset details are given in Appendix A.2.

Target agents. We report two target agents from different model families, Opus 4.6 (Claude Code) and GPT 5.2 (Codex). For each, the pipeline is run end-to-end so that skills are constructed and deployed with the same model.

Baselines. We compare against seven automated baselines, all closed-world—none retrieves open-world knowledge or builds a self-verifier: No Skill; Self-Gen and CoT, which self-generate skills in a single pass from parametric knowledge (CoT adds a structured chain-of-thought workflow); Skill Creator (Anthropic, 2025) and AutoSkill (Yang et al., 2026), skill-synthesis methods that iteratively refine skills from prior knowledge or interaction traces; and Memento (Zhou et al., 2026), a memory-/experience-based baseline (plus SkillNet (Liang et al., 2026) on ScienceWorld). A Human upper bound is excluded from the comparison. Appendix A.3 details the baselines.

Metric and protocol. All methods share the same target agents and report average reward on the hidden test suite. A per-task cost breakdown of the OpenSkill pipeline (skill creation vs. evaluation) is reported in Appendix E. Detailed baseline definitions, prompts, the evaluation protocol, and full implementation configuration are deferred to Appendices A.3, G, A.4, A.5, and B.

## 3.2 Main Results

Table 1 reports the average reward (pass rate) on SkillsBench per domain and overall, for each target agent (exact model versions in Appendix A.1).

<table><tr><td>Domain</td><td>No Skill</td><td>Self-Gen</td><td>CoT</td><td>Skill-Creator AutoSkill</td><td></td><td>Memento</td><td>OPENSKILL</td><td>Human</td></tr><tr><td colspan="9">Opus 4.6 (Claude Code)</td></tr><tr><td>Software</td><td>32.6</td><td>37.9</td><td>34.9</td><td>51.3</td><td>36.0</td><td>34.4</td><td>59.9</td><td>38.8</td></tr><tr><td>Office</td><td>17.0</td><td>16.7</td><td>17.1</td><td>21.4</td><td>25.7</td><td>31.4</td><td>50.0</td><td>50.0</td></tr><tr><td>Science</td><td>25.6</td><td>31.3</td><td>30.0</td><td>36.2</td><td>33.3</td><td>35.0</td><td>35.0</td><td>46.7</td></tr><tr><td>Media</td><td>36.1</td><td>27.9</td><td>20.4</td><td>38.5</td><td>23.6</td><td>21.8</td><td>39.6</td><td>36.4</td></tr><tr><td>Cybersecurity</td><td>17.8</td><td>18.8</td><td>20.4</td><td>24.6</td><td>16.6</td><td>28.8</td><td> 44.1</td><td>55.0</td></tr><tr><td>Finance</td><td>17.5</td><td>16.7</td><td>20.0</td><td>27.5</td><td>25.0</td><td>25.0</td><td>25.0</td><td>30.0</td></tr><tr><td>Robotics</td><td>27.6</td><td>13.3</td><td>16.0</td><td>36.0</td><td>4.0</td><td>32.0</td><td>36.0</td><td>36.0</td></tr><tr><td>Energy</td><td>41.2</td><td>11.1</td><td>40.0</td><td>60.0</td><td>33.3</td><td>60.0</td><td>60.0</td><td>66.7</td></tr><tr><td>Manufacturing</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>46.7</td></tr><tr><td>Health</td><td>24.8</td><td>19.8</td><td>19.2</td><td>31.2</td><td>14.5</td><td>25.0</td><td>69.6</td><td>80.0</td></tr><tr><td>Math</td><td>43.2</td><td>30.0</td><td>30.0</td><td>50.0</td><td>0.0</td><td>30.0</td><td>50.0</td><td>50.0</td></tr><tr><td>Overall </td><td>25.5</td><td>23.9</td><td>23.9</td><td> 34.7</td><td>24.7</td><td>30.1</td><td>43.6</td><td>44.5</td></tr><tr><td>△ us. No Skill</td><td>1</td><td>-1.6</td><td>-1.6</td><td>+9.2</td><td>-0.8</td><td>+4.6</td><td>+18.1</td><td>+19.0</td></tr><tr><td colspan="9">GPT 5.2 (Codex)</td></tr><tr><td>Software</td><td>33.2</td><td>48.4</td><td>47.2</td><td>44.4</td><td>16.7</td><td>19.5</td><td>49.1</td><td>42.5</td></tr><tr><td>Office</td><td>32.9</td><td>31.0</td><td>26.2</td><td>26.2</td><td>9.4</td><td>14.3</td><td> 44.3</td><td>48.6</td></tr><tr><td>Science</td><td>30.4</td><td>30.3</td><td>29.8</td><td>21.9</td><td>5.5</td><td>13.8</td><td>48.6</td><td>48.3</td></tr><tr><td>Media</td><td>31.3</td><td>31.0</td><td>31.8</td><td>30.9</td><td>15.2</td><td>18.2</td><td>30.4</td><td>58.2</td></tr><tr><td>Cybersecurity</td><td>25.0</td><td>20.8</td><td>34.7</td><td>36.8</td><td>4.1</td><td>12.5</td><td>52.5</td><td>42.5</td></tr><tr><td>Finance</td><td>0.0</td><td>29.2</td><td>25.0</td><td>20.8</td><td>8.4</td><td>12.5</td><td>25.0</td><td>27.5</td></tr><tr><td>Robotics</td><td>16.0</td><td>26.7</td><td>40.0</td><td>20.0</td><td>13.4</td><td>26.6</td><td>40.0</td><td>40.0</td></tr><tr><td>Energy</td><td>0.0</td><td>33.3</td><td>55.6</td><td>22.2</td><td>11.0</td><td>22.3</td><td>80.0</td><td>53.3</td></tr><tr><td>Manufacturing</td><td>0.0</td><td>11.1</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>Math</td><td>30.0</td><td>33.3</td><td>33.3</td><td>50.0</td><td>33.5</td><td>0.0</td><td>50.0</td><td>40.0</td></tr><tr><td>Health</td><td>29.2</td><td>30.2</td><td>30.2</td><td>24.3</td><td>20.0</td><td>16.5</td><td>27.9</td><td>90.0</td></tr><tr><td>Overall </td><td>25.0</td><td>32.2</td><td>33.3</td><td>29.2</td><td>11.2</td><td>15.6</td><td>42.1</td><td>44.8</td></tr><tr><td>△ vs. No Skill</td><td></td><td>+7.2</td><td>+8.3</td><td>+4.2</td><td>-13.8</td><td>-9.4</td><td>+17.1</td><td>+19.8</td></tr></table>

Table 1 Main results on SkillsBench (11 domains): average reward (pass rate, %) by domain for two target agents. Best automated method per row in bold, second best underlined; the OpenSkill column is shaded. Human is a reference upper bound (set off on the right) and is excluded from the best-method comparison. The ∆ vs. No Skill row gives each method’s overall pass-rate gain over the No-Skill floor (in points); negative values fall below it.

Best overall pass rate on both agents. OpenSkill achieves the best automated overall pass rate on both target agents—43.6% on Opus 4.6 and 42.1% on GPT 5.2—beating the strongest baseline by +8.9 and +8.8 points (Skill-Creator and CoT, respectively) and landing within 1–3 points of the Human upper bound (44.5% 44.8%). It is also the only method strong on both agents: the single-pass variants (Self-Gen, CoT) help on GPT 5.2 but not Opus 4.6, while the iterative methods do the reverse and collapse on GPT 5.2 (AutoSkill 24.7%→11.2%, Memento 30.1%→15.6%, both below the 25.0% no-skill agent). Open-world acquisition and leakage-free verification thus help regardless of the underlying model.

Broad per-domain gains. OpenSkill is best or tied-best in 8 of 11 domains on Opus 4.6 and 7 of 11 on GPT 5.2, with the largest gains in knowledge-intensive domains (Opus: Health 69.6%, Software 59.9%; GPT: Energy 80.0%, Cybersecurity 52.5%, both above Human). It ties for second in Science and Finance on Opus 4.6, and Manufacturing collapses to 0.0% for all automated methods—a failure open-world acquisition alone does not resolve.

## 3.3 Beyond SkillsBench: Other Task Types

We repeat the evaluation on SocialMaze and ScienceWorld with the same two agents and closed-world baselines . As Table 2 shows, OpenSkill is the best automated method in all four columns—SocialMaze 82.7% (Opus) / 70.7% (GPT) and ScienceWorld 90.0% / 85.3%—improving over the strongest baseline by +0.9 to +2.2 points, with larger gains on GPT 5.2 as on SkillsBench. A per-subtask SocialMaze breakdown is in Appendix D.

<table><tr><td colspan="3"> SocialMaze</td><td colspan="2">ScienceWorld</td></tr><tr><td>Method</td><td>Opus 4.6</td><td>GPT 5.2</td><td>Opus 4.6</td><td>GPT 5.2</td></tr><tr><td>No Skill</td><td>81.6</td><td>66.6</td><td>87.1</td><td>78.0</td></tr><tr><td>Self-Gen</td><td>80.4</td><td>67.6</td><td>88.0</td><td>75.9</td></tr><tr><td>CoT</td><td>79.4</td><td>66.7</td><td>88.0</td><td>80.0</td></tr><tr><td>Skill Creator</td><td>81.0</td><td>69.8</td><td>88.3</td><td>83.1</td></tr><tr><td>AutoSkill</td><td>77.2</td><td>65.9</td><td>1</td><td>1</td></tr><tr><td>Memento</td><td>80.3</td><td>67.4</td><td>1</td><td>1</td></tr><tr><td>SkillNet</td><td>一</td><td>二</td><td>88.7</td><td>77.8</td></tr><tr><td> OPENSKILL</td><td>82.7</td><td>70.7</td><td>90.0</td><td>85.3</td></tr></table>

Table 2 Average reward (%) on SocialMaze and ScienceWorld for both target agents. Best automated score per column in bold, second best underlined; “–” marks methods not run on a benchmark.

## 4 Analysis

We probe OpenSkill along three questions:

RQ1 (Transferability): Do its skills transfer across models without model-specific adaptation?

RQ2 (Verifier quality): Without ground-truth tests, do the virtual verifier’s proxy tests align with and cover ground-truth test intents?

RQ3 (Component contribution): How much does each design element contribute?

## 4.1 RQ1: Skill Generalization

A key advantage of explicit, reusable skills is that skills generated by one model can be transferred to other models without retraining or regeneration. We evaluate this cross-model transferability by deploying the skill libraries produced by Opus 4.6 under three generation methods—OpenSkill (ours), AutoSkill, and Memento—directly onto four weaker models: Haiku 4.5,1 Qwen 3 Coder,2 DeepSeek V3 (Liu et al., 2024), and Mistral Large 3.3 Each model is evaluated on SkillsBench.

Figure 3 shows that OpenSkill-generated skills consistently yield the highest reward across all four target models, improving over the no-skill baseline by 5.5%–14.8% points. Notably, these gains are

![](images/848783be548d89351d539cbae8b453fccccfaa39f39e09ea89d307cbb41960c0.jpg)  
Figure 3 Average reward (%) when transferring Opus 4.6- generated skills to other models on SkillsBench.

achieved without any model-specific adaptation: the same skill files produced by Opus 4.6 are used as-is. Memento skills also transfer reasonably well to four models. AutoSkill performs worse than the no-skill baseline, suggesting that its generated skills are tightly coupled to the originating model and fail to generalize.

Takeaway (RQ1)

OpenSkill encodes task-relevant knowledge in a model-agnostic form, so the same skill files transfer effectively across models—without any model-specific adaptation—even to substantially weaker ones.

## 4.2 RQ2: Virtual Verifier Quality

A central design element of OpenSkill is the Virtual Verifier—a separate LLM agent that generates proxy test suites from task specifications alone, without access to ground-truth (GT) tests. This proxy provides iterative feedback to the task executor during open-loop deployment where no GT oracle is available. We evaluate the quality of surrogate-generated tests along two axes: alignment with GT evaluation outcomes, and coverage of GT test intents.

Alignment with GT outcomes. Table 3 reports the agreement between virtual verifier decisions and GT evaluation outcomes. The virtual verifier achieves 56.9% precision and 80.5% recall, with an overall agreement rate of 60.7%. The association between virtual verifier decisions and GT reward is statistically significant (Fisher’s exact test $\mathrm { O R } = 2 . 9 7 , p = 0 . 0 3 5 ;$ point-biserial $r = 0 . 2 4 2 \ :$ $p = 0 . 0 2 7 )$ , confirming that the virtual verifier pro-

<table><tr><td colspan="2">Reward &gt;0</td><td>Reward = 0</td><td>Total</td></tr><tr><td>Proxy Pass</td><td>39.29%</td><td>29.76%</td><td>69.05%</td></tr><tr><td>Proxy Fail</td><td>9.52%</td><td>21.43%</td><td>30.95%</td></tr><tr><td>Total</td><td>48.81%</td><td>51.19%</td><td>100.00%</td></tr></table>

Table 3 Percentage distribution of proxy results and GT rewards (N = 84).

vides a meaningful quality signal despite operating without access to ground-truth tests. We analyze the remaining disagreement cases and their failure modes in Appendix C.

Coverage of GT test intents. We further analyze how well the virtual verifier’s generated tests cover the evaluation intents of human-authored GT tests. We randomly sample 15 tasks and use an LLM judge, i.e., opus 4.6 to perform semantic matching between each GT test function and the surrogate test suite, determining whether the GT test’s purpose (e.g., “check output file exists,” “verify numerical accuracy within tolerance”) is addressed by at least one virtual test.

Across the 15 sampled tasks, the virtual verifier covers 88.9% of GT test intents (120 out of 135). The uncovered 11.1% cluster in two categories: (1) anti-cheat meta-validation checks specific to the benchmark infrastructure, and (2) deep semantic quality properties (e.g., taxonomy coherence, lemmatization correctness) that require domain expertise beyond what the task specification provides. Meanwhile, the virtual verifier generates a median of 3.4× more test functions per task than the GT suite, contributing 15.3 additional assertions per task on average—primarily defensive checks on output format, type validity, and domain-specific boundary conditions.

## Takeaway (RQ2)

The virtual verifier uses no ground-truth tests, yet its proxy tests cover most human-authored test intents and track the true outcomes closely enough to gate skill generation on their own.

## 4.3 RQ3: Ablation Studies

We conduct two ablation studies on SocialMaze using Opus 4.6 to isolate the contributions of key design choices.

Iteration count. The virtual verifier loop refines each generated strategy through iterative critique– revision cycles. We vary the maximum number of iterations across {1, 3, 5, 10} to examine its effect on downstream task performance. Figure 4a reports the results. Performance peaks at 3 iterations (82.7%), matching the default configuration, and degrades slightly with additional iterations (79.9% at 5, 78.0% at 10), suggesting that excessive refinement introduces overfitting to virtual test feedback

![](images/d034443073aa482f66545aced055682f5b4188bdf5facf77616387bfb20fa55c.jpg)

![](images/b81e437ae6ca83dbb719082c743cfef839e440983581ff68cec369c566cc1170.jpg)  
Figure 4 Ablations on SocialMaze (Opus 4.6). (a) Reward peaks at a few refinement iterations and degrades with more, indicating overfitting to virtual feedback. (b) Open-world query (DR) and the virtual verifier (VV) each improve over the parametric-only baseline and are largely complementary, with the combination performing best.

Component contribution. We ablate two core components of the pipeline: (1) the open world query module, which retrieves external domain knowledge to inform strategy generation, and (2) the virtual verifier, which provides proxy test feedback for iterative refinement. Figure 4b reports the results under four configurations. Removing both components yields 74.5%, establishing a lower bound where the model relies solely on parametric knowledge for strategy generation. Adding either component individually recovers most of the performance: open world query alone contributes +6.1 percentage points (80.6%), while the virtual verifier alone contributes +6.3 percentage points (80.8%). Their contributions are largely complementary—combining both achieves 82.7%, a further +2.1 percentage points over the better individual component—though the marginal gain of each is smaller in the presence of the other, indicating partial overlap in the errors they correct.

## Takeaway (RQ3)

Open-world query and the virtual verifier each contribute substantially over a parametric-only baseline and are largely complementary; refinement helps only up to a point, after which additional iterations overfit to virtual feedback.

## 5 Related Work

## 5.1 Self-Evolving Agents and Agent Skills

LLM agents interleave reasoning and actions (Yao et al., 2022), teach themselves to call tools (Schick et al., 2023), and improve planning through structured deliberation (Yao et al., 2023; Zhou et al., 2023). To improve after deployment, self-evolving agents accumulate reusable knowledge through reflection over past attempts (Shinn et al., 2023), executable skills learned by exploration (Wang et al., 2023), subagents distilled from successful solutions (Zhang et al., 2026b), and cumulative skill creation (Huang et al., 2025a); recent work couples skill learning with verification, co-evolving skills and their verifiers (Zhang et al., 2026a). Reinforcement-learning variants instead internalize skills into model weights (Xia et al., 2026; Wang et al., 2025; Lu et al., 2026), yielding knowledge that is hard to inspect, edit, or transfer across

<table><tr><td>Method</td><td>OW retr. Refine</td><td></td><td></td><td>SF verif. Artifact</td></tr><tr><td>No Skill</td><td>×</td><td>×</td><td>×</td><td>×</td></tr><tr><td>Self-Gen</td><td>×</td><td>×</td><td>×</td><td>√</td></tr><tr><td>CoT</td><td>×</td><td>×</td><td>×</td><td></td></tr><tr><td>Skill Creator</td><td>×</td><td>√</td><td>×</td><td></td></tr><tr><td>AutoSkill</td><td>×</td><td></td><td>×</td><td></td></tr><tr><td>Memento</td><td>×</td><td></td><td>×</td><td></td></tr><tr><td> OPENSKILL (ours)</td><td></td><td></td><td></td><td></td></tr></table>

Table 4 Capability comparison of the automated methods. OW retr.: acquires open-world knowledge beyond parametric/experience memory; Refine: iteratively refines skills; SF verif.: builds a supervision-free verification signal (no target-task feedback); Artifact: produces an explicit, model-transferable skill. SkillNet (ScienceWorld only) is omitted.

models. A growing body treats skills as a managed resource: benchmarks show self-generated skills are unreliable (Li et al., 2026; Xu & Yan, 2026), while retrieval, compression, and multi-objective selection improve deployment (Li et al., 2025; Liu et al., 2026a; Gao et al., 2026; Wang et al., 2026; Gong et al., 2026); skills can further be viewed as structured prompts, linking to automatic prompt optimization (Zhou et al., 2022; Pryzant et al., 2023). Unlike these lines, OpenSkill makes open-world acquisition the primary source of skill content, keeps skills as explicit and transferable artifacts rather than model-bound behaviors, and refines them without target-task supervision (Table 4); better retrieval and compression remain complementary at deployment.

## 5.2 Open-World Knowledge Acquisition

Retrieval-augmented generation grounds outputs in external, non-parametric evidence (Lewis et al., 2020; Gao et al., 2023), and browser-assisted, deep-research, and environment-interactive agents search the web, repositories, and tools to complete knowledge-intensive, long-horizon tasks (Nakano et al., 2021; Huang et al., 2025b; Zhou et al., 2024; Yang et al., 2024). These methods retrieve knowledge to answer a query or complete a single task, whereas OpenSkill uses open-world retrieval as the substrate for synthesizing persistent, reusable skills and for grounding a self-verification signal.

## 5.3 Self-Verification and Self-Generated Evaluation

Without target-task supervision, an agent must judge its own outputs: prior work aggregates multiple reasoning paths (Wang et al., 2022b), iterates on self-feedback (Madaan et al., 2023), and uses LLMs as judges (Zheng et al., 2023; Gu et al., 2024), while in code domains self-generated tests filter or repair solutions through execution feedback (Chen et al., 2022, 2024; Chu et al., 2025). Closer to our setting, skill-centric methods verify synthesized skills before use, either by synthesizing skills with verification at inference time (Ma et al., 2026) or by co-evolving skills together with a learned verifier (Zhang et al., 2026a). These signals derive from the model’s own priors or the target task itself, which limits calibration and risks supervision leakage. OpenSkill’s virtual tasks differ by anchoring verification to independently verifiable facts retrieved from the open world, yielding a practice environment that remains isolated from hidden target-task supervision.

## 6 Conclusion

We studied open-world self-evolution, where an agent starts from only a task prompt and must build both its skills and its own verification signals from open-world resources, without target-task supervision during learning. OpenSkill realizes this by acquiring grounded knowledge and verification anchors from the open world, synthesizing them into transferable skills, and refining those skills against self-built virtual tasks rather than target answers. Across three benchmarks and two target agents it attains the best automated pass rate while honoring the no-supervision constraint, with skills that transfer to weaker models and a self-built verifier that aligns with ground-truth outcomes it never observes. This points to open-world acquisition and leakage-free verification as a path toward agents that keep improving after deployment, where curated skills and reliable feedback are unavailable.

## Limitations

Open-world self-evolution introduces new challenges. First, web and repository sources may be noisy, outdated, or contradictory. The framework therefore requires provenance tracking and source validation. Second, virtual tasks may fail to capture the full difficulty of real target tasks. If virtual tasks are too easy, they may overestimate skill quality; if they are derived from hidden answers or verifier behavior, they may reintroduce target-task supervision. Third, open-world research can increase cost and latency relative to closed-world skill generation.

## References

Anthropic. Skill creator: A skill for creating new skills and iteratively improving them. https://github.com/anthropics/ skills/tree/main/skills/skill-creator, 2025. Claude Code Agent Skill. Accessed 2026-05-25.

Bei Chen, Fengji Zhang, Anh Nguyen, Daoguang Zan, Zeqi Lin, Jian-Guang Lou, and Weizhu Chen. Codet: Code generation with generated tests. arXiv preprint arXiv:2207.10397, 2022.

Xinyun Chen, Maxwell Lin, Nathanael Schärli, and Denny Zhou. Teaching large language models to self-debug. In International Conference on Learning Representations, volume 2024, pp. 8746–8825, 2024.

Bei Chu, Yang Feng, Kui Liu, Zhaoqiang Guo, Yichi Zhang, Hange Shi, Zifan Nan, and Baowen Xu. Large language models for unit test generation: Achievements, challenges, and opportunities. arXiv preprint arXiv:2511.21382, 2025.

Yudong Gao, Zongjie Li, Yuanyuanyuan, Zimo Ji, Pingchuan Ma, and Shuai Wang. SkillReducer: Optimizing llm agent skills for token efficiency, 2026. URL https://arxiv.org/abs/2603.29919.

Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yixin Dai, Jiawei Sun, Haofen Wang, Haofen Wang, et al. Retrieval-augmented generation for large language models: A survey. arXiv preprint arXiv:2312.10997, 2(1):32, 2023.

Jingzhi Gong, Ruizhen Gu, Zhiwei Fei, Yazhuo Cao, Lukas Twist, Alina Geiger, Shuo Han, Dominik Sobania, Federica Sarro, and Jie M. Zhang. SkillMOO: Multi-objective optimization of agent skills for software engineering, 2026. URL https://arxiv.org/abs/2604.09297.

Jiawei Gu, Xuhui Jiang, Zhichao Shi, Hexiang Tan, Xuehao Zhai, Chengjin Xu, Wei Li, Yinghan Shen, Shengjie Ma, Honghao Liu, et al. A survey on llm-as-a-judge. The Innovation, 2024.

Xu Huang, Junwu Chen, Yuxing Fei, Zhuohan Li, Philippe Schwaller, and Gerbrand Ceder. CASCADE: Cumulative agentic skill creation through autonomous development and evolution, 2025a. URL https://arxiv.org/abs/2512. 23880.

Yuxuan Huang, Yihang Chen, Haozheng Zhang, Kang Li, Huichi Zhou, Meng Fang, Linyi Yang, Xiaoguang Li, Lifeng Shang, Songcen Xu, et al. Deep research agents: A systematic examination and roadmap. arXiv preprint arXiv:2506.18096, 2025b.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented generation for knowledge-intensive nlp tasks. Advances in neural information processing systems, 33:9459–9474, 2020.

Fangzhou Li, Pagkratios Tagkopoulos, and Ilias Tagkopoulos. SkillFlow: Scalable and efficient agent skill retrieval system, 2025. URL https://arxiv.org/abs/2504.06188.

Xiangyi Li, Wenbo Chen, Yimin Liu, Shenghan Zheng, Xiaokun Chen, Yifeng He, Yubo Li, Bingran You, Haotian Shen, Jiankai Sun, Shuyi Wang, Binxu Li, Qunhong Zeng, Di Wang, Xuandong Zhao, Yuanli Wang, Roey Ben Chaim, Zonglin Di, Yipeng Gao, Junwei He, Yizhuo He, Liqiang Jing, Luyang Kong, Xin Lan, Jiachen Li, Songlin Li, Yijiang Li, Yueqian Lin, Xinyi Liu, Xuanqing Liu, Xuanqing Liu, Haoran Lyu, Ze Ma, Bowei Wang, Runhui Wang, Tianyu Wang, Wengao Ye, Yue Zhang, Hanwen Xing, Yiqi Xue, Steven Dillmann, and Han chung Lee. SkillsBench: Benchmarking how well agent skills work across diverse tasks, 2026. URL https://arxiv.org/abs/2602.12670.

Yuan Liang, Ruobin Zhong, Haoming Xu, Chen Jiang, Yi Zhong, Runnan Fang, Jia-Chen Gu, Shumin Deng, Yunzhi Yao, Mengru Wang, et al. Skillnet: Create, evaluate, and connect ai skills. arXiv preprint arXiv:2603.04448, 2026.

Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024.

Dawei Liu, Zongxia Li, Hongyang Du, Xiyang Wu, Shihang Gui, Yongbei Kuang, and Lichao Sun. Graph of skills: Dependency-aware structural retrieval for massive agent skills, 2026a. URL https://arxiv.org/abs/2604.05333.

Yujian Liu, Jiabao Ji, Li An, Tommi Jaakkola, Yang Zhang, and Shiyu Chang. How well do agentic skills work in the wild: Benchmarking llm skill usage in realistic settings, 2026b. URL https://arxiv.org/abs/2604.04323.

Zhengxi Lu, Zhiyuan Yao, Jinyang Wu, Chengcheng Han, Qi Gu, Xunliang Cai, Weiming Lu, Jun Xiao, Yueting Zhuang, and Yongliang Shen. SKILL0: In-context agentic reinforcement learning for skill internalization, 2026. URL https://arxiv.org/abs/2604.02268.

Yuchen Ma, Yue Huang, Han Bao, Haomin Zhuang, Swadheen Shukla, Michel Galley, Xiangliang Zhang, and Stefan Feuerriegel. Skillgen: Verified inference-time agent skill synthesis. arXiv preprint arXiv:2605.10999, 2026.

Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, et al. Self-refine: Iterative refinement with self-feedback. Advances in neural information processing systems, 36:46534–46594, 2023.

Mike A. Merrill, Alexander G. Shaw, Nicholas Carlini, Boxuan Li, Harsh Raj, Ivan Bercovich, Lin Shi, Jeong Yeon Shin, Thomas Walshe, E. Kelly Buchanan, Junhong Shen, Guanghao Ye, Haowei Lin, Jason Poulos, Maoyu Wang, Marianna Nezhurina, Jenia Jitsev, Di Lu, Orfeas Menis Mastromichalakis, Zhiwei Xu, Zizhao Chen, Yue Liu, Robert Zhang, Leon Liangyu Chen, Anurag Kashyap, Jan-Lucas Uslu, Jeffrey Li, Jianbo Wu, Minghao Yan, Song Bian, Vedang Sharma, Ke Sun, Steven Dillmann, Akshay Anand, Andrew Lanpouthakoun, Bardia Koopah, Changran Hu, Etash Guha, Gabriel H. S. Dreiman, Jiacheng Zhu, Karl Krauth, Li Zhong, Niklas Muennighoff, Robert Amanfu, Shangyin Tan, Shreyas Pimpalgaonkar, Tushar Aggarwal, Xiangning Lin, Xin Lan, Xuandong Zhao, Yiqing Liang, Yuanli Wang, Zilong Wang, Changzhi Zhou, David Heineman, Hange Liu, Harsh Trivedi, John Yang, Junhong Lin, Manish Shetty, Michael Yang, Nabil Omi, Negin Raoof, Shanda Li, Terry Yue Zhuo, Wuwei Lin, Yiwei Dai, Yuxin Wang, Wenhao Chai, Shang Zhou, Dariush Wahdany, Ziyu She, Jiaming Hu, Zhikang Dong, Yuxuan Zhu, Sasha Cui, Ahson Saiyed, Arinbjörn Kolbeinsson, Jesse Hu, Christopher Michael Rytting, Ryan Marten, Yixin Wang, Alex Dimakis, Andy Konwinski, and Ludwig Schmidt. Terminal-bench: Benchmarking agents on hard, realistic tasks in command line interfaces, 2026. URL https://arxiv.org/abs/2601.11868.

Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu, Long Ouyang, Christina Kim, Christopher Hesse, Shantanu Jain, Vineet Kosaraju, William Saunders, et al. Webgpt: Browser-assisted question-answering with human feedback. arXiv preprint arXiv:2112.09332, 2021.

Reid Pryzant, Dan Iter, Jerry Li, Yin Tat Lee, Chenguang Zhu, and Michael Zeng. Automatic prompt optimization with “gradient descent” and beam search, 2023. URL https://arxiv.org/abs/2305.03495.

Timo Schick, Jane Dwivedi-Yu, Roberto Dessi, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools, 2023. URL https: //arxiv.org/abs/2302.04761.

Noah Shinn, Federico Cassano, Edward Berman, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning, 2023. URL https://arxiv.org/abs/2303.11366.

Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. Voyager: An open-ended embodied agent with large language models, 2023. URL https://arxiv.org/abs/2305.16291.

Jiongxiao Wang, Qiaojing Yan, Yawei Wang, Yijun Tian, Soumya Smruti Mishra, Zhichao Xu, Megha Gandhi, Panpan Xu, and Lin Lee Cheong. Reinforcement learning for self-improving agent with skill library, 2025. URL https://arxiv.org/abs/2512.17102.

Ruoyao Wang, Peter Jansen, Marc-Alexandre Côté, and Prithviraj Ammanabrolu. Scienceworld: Is your agent smarter than a 5th grader? In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pp. 11279–11298, 2022a.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. arXiv preprint arXiv:2203.11171, 2022b.

Zimu Wang, Yuling Shi, Mengfan Li, Zijun Liu, Jie M. Zhang, Chengcheng Wan, and Xiaodong Gu. EffiSkill: Agent skill based automated code efficiency optimization, 2026. URL https://arxiv.org/abs/2603.27850.

Peng Xia, Jianwen Chen, Hanyang Wang, Jiaqi Liu, Kaide Zeng, Yu Wang, Siwei Han, Yiyang Zhou, Xujiang Zhao, Haifeng Chen, Zeyu Zheng, Cihang Xie, and Huaxiu Yao. SkillRL: Evolving agents via recursive skill-augmented reinforcement learning, 2026. URL https://arxiv.org/abs/2602.08234.

Renjun Xu and Yang Yan. Agent skills for large language models: Architecture, acquisition, security, and the path forward, 2026. URL https://arxiv.org/abs/2602.12430.

Zixiang Xu, Yanbo Wang, Yue Huang, Jiayi Ye, Haomin Zhuang, Zirui Song, Lang Gao, Chenxi Wang, Zhaorun Chen, Yujun Zhou, et al. Socialmaze: A benchmark for evaluating social reasoning in large language models. arXiv preprint arXiv:2505.23713, 2025.

John Yang, Carlos E Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik Narasimhan, and Ofir Press. Swe-agent: Agent-computer interfaces enable automated software engineering. Advances in Neural Information Processing Systems, 37:50528–50652, 2024.

Yutao Yang, Junsong Li, Qianjun Pan, Bihao Zhan, Yuxuan Cai, Lin Du, Jie Zhou, Kai Chen, Qin Chen, Xin Li, et al. Autoskill: Experience-driven lifelong learning via skill self-evolution. arXiv preprint arXiv:2603.01145, 2026.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. ReAct: Synergizing reasoning and acting in language models, 2022. URL https://arxiv.org/abs/2210.03629.

Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: Deliberate problem solving with large language models, 2023. URL https://arxiv.org/abs/2305.10601.

Hanrong Zhang, Shicheng Fan, Henry Peng Zou, Yankai Chen, Zhenting Wang, Jiayu Zhou, Chengze Li, Wei-Chieh Huang, Yifei Yao, Kening Zheng, Xue Liu, Xiaoxiao Li, and Philip S. Yu. CoEvoSkills: Self-evolving agent skills via co-evolutionary verification, 2026a. URL https://arxiv.org/abs/2604.01687.

Zhang Zhang, Shuqi Lu, Hongjin Qian, Di He, and Zheng Liu. Agentfactory: A self-evolving framework through executable subagent accumulation and reuse, 2026b. URL https://arxiv.org/abs/2603.18000.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging llm-as-a-judge with mt-bench and chatbot arena. Advances in neural information processing systems, 36:46595–46623, 2023.

Andy Zhou, Kai Yan, Michal Shlapentokh-Rothman, Haohan Wang, and Yu-Xiong Wang. Language agent tree search unifies reasoning acting and planning in language models, 2023. URL https://arxiv.org/abs/2310.04406.

Huichi Zhou, Siyuan Guo, Anjie Liu, Zhongwei Yu, Ziqin Gong, Bowen Zhao, Zhixun Chen, Menglong Zhang, Yihang Chen, Jinsong Li, et al. Memento-skills: Let agents design agents. arXiv preprint arXiv:2603.18743, 2026.

Shuyan Zhou, Frank F Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Tianyue Ou, Yonatan Bisk, Daniel Fried, et al. Webarena: A realistic web environment for building autonomous agents. In International Conference on Learning Representations, volume 2024, pp. 15585–15606, 2024.

Yongchao Zhou, Andrei Ioan Muresanu, Ziwen Han, Keiran Paster, Silviu Pitis, Harris Chan, and Jimmy Ba. Large language models are human-level prompt engineers, 2022. URL https://arxiv.org/abs/2211.01910.

## A Experimental Details

## A.1 Model Details

The OpenSkill pipeline uses models from three families. All LLM roles in the pipeline are instantiated either with Anthropic Claude (claude-opus-4-6),4 or OpenAI GPT (gpt-5.2).5 The two open-world retrieval roles use Google Gemini:6 the main knowledge acquisition (D) calls the Gemini Deep Research agent (deepresearch-pro-preview-12-2025), while verification-knowledge retrieval (Dv) and diagnostic-driven targeted retrieval use the search-grounded gemini-3.1-flash-lite.7 Zero-shot evaluation uses the claude code8 for Opus 4.6, codex9 for GPT-5.2, and terminus-2 agent (Merrill et al., 2026) for other LLMs. Version strings denote the exact model snapshots used in our experiments.

## A.2 Dataset Details

We evaluate on three agentic benchmarks. SkillsBench (Li et al., 2026) is our primary benchmark and spans 11 task domains: Software, Office, Science, Media, Cybersecurity, Finance, Robotics, Energy, Manufacturing, Health, and Math. It is constructed so that skill quality, rather than base reasoning, is the limiting factor on task success. SocialMaze (Xu et al., 2025) is a social-reasoning suite of six subtasks—FTS, HRD, REFT, RDP, SGA, and UPI (broken out in Appendix D). ScienceWorld (Wang et al., 2022a) is an interactive science-experiment environment. On every benchmark the ground-truth test suite $\dot { T } _ { i } ^ { \mathrm { G T } }$ is hidden during skill construction and consulted only at final evaluation.

## A.3 Baselines

We compare OpenSkill against seven automated conditions and one reference upper bound, all sharing the same target agents and hidden-test evaluation protocol:

• No Skill runs the target agent on the task with no skill artifact provided. It isolates the agent’s parametric competence and serves as the zero-knowledge floor against which every skill-construction method is measured.

• Self-Gen (Self-Generated Skills) reproduces the single-pass self-generation condition of SkillsBench (Li et al., 2026). Drawing on its parametric knowledge alone, the agent authors one to five SKILL.md documents in a single forward pass and immediately uses them to solve the task. There is no open-world retrieval, no interaction-trace mining, and no evolution or verification loop, so the skills can only re-express knowledge the model already holds. This baseline directly tests the SkillsBench observation that models cannot reliably author the procedural knowledge they benefit from consuming (prompt in Appendix G.1).

• CoT (CoT-Guided Self-Generation) preserves the single-pass, parametric-only setting of Self-Gen but scaffolds the drafting with an explicit five-step chain-of-thought prompt that walks the agent through analyzing the task, recalling relevant procedures, structuring the skill, drafting it, and self-reviewing before solving. It isolates whether structured reasoning over existing knowledge—without any external information or feedback—is sufficient to produce useful skills (prompt in Appendix G.2).

• Skill Creator (Anthropic, 2025) is Anthropic’s official Claude Code agent skill for authoring, evaluating, and iteratively improving skills. It runs a Draft → Test → Review → Improve → Repeat loop: it captures intent, writes a SKILL.md with progressive disclosure (metadata, body, and bundled scripts/references), executes with-skill and baseline test cases, grades them with quantitative assertions, and revises on the resulting feedback. Crucially, this refinement loop draws on parametric knowledge and self-graded test cases rather than open-world sources or a supervision-free verifier, so it iterates but does not acquire genuinely new external knowledge.

• AutoSkill (Yang et al., 2026) is an experience-driven lifelong learning framework that abstracts, maintains, and reuses skills from dialogue and interaction traces, organizing them into a hierarchical skill bank (domain → family → leaf skill) as a model-agnostic plugin layer that injects relevant skills into future requests without retraining. In our protocol we invoke the released AutoSkill toolkit to construct a skill set per task from the task instruction and copy the generated skills into the agent’s skill directory. Because its self-evolution operates over prior knowledge and accumulated traces rather than open-world retrieval, it remains a closed-world refinement baseline.

• Memento (Zhou et al., 2026) is a memory-based reinforcement learning framework in which reusable skills are stored as markdown files and improved through a Read–Write Reflective Learning mechanism: a skill router selects relevant skills in the read phase, and the agent updates its skill library from new experience in the write phase, with a background consolidation (“dream”) step that compresses accumulated memory—all without updating model parameters. We run the released system per task, excluding its built-in utility skills and retaining only the newly created ones. Its updates are driven by self-generated experience rather than external knowledge or a supervision-free verifier.

• SkillNet (Liang et al., 2026) is an open infrastructure for creating, evaluating, and connecting AI skills at scale, scoring skills along five dimensions (safety, completeness, executability, maintainability, and cost-awareness). Following its reported evaluation setting, we include SkillNet on ScienceWorld only; the corresponding cells are left empty on the other benchmarks.

• Human Curated skills are the expert-authored reference skills shipped with each benchmark. They serve as a reference upper bound and are excluded from the best-automated-method comparison.

For the third-party methods that ship as executable systems—Skill Creator, AutoSkill, and Memento— we use the released implementations with claude-opus-4-7 as the underlying backbone, the same target agents, and the same hidden-test evaluation protocol as OpenSkill, so that differences in reward reflect the skill-construction mechanism rather than the model or the evaluation harness.

## A.4 Evaluation Metrics

For SkillsBench, we report the average reward (pass rate) over a benchmark’s tasks, computed from the hidden ground-truth test suite as in Eq. 6. Each constructed skill set is deployed under $n _ { \mathrm { e v a l } } = 5$ independent zero-shot evaluation runs (Appendix B.6), and per-domain scores are averaged into the overall figure reported in each table. For the virtual-verifier analysis (Section 4.2) we additionally report alignment statistics—precision, recall, agreement, and association tests (Fisher’s exact test and point-biserial correlation)—and the coverage of ground-truth test intents.

For SocialMaze, each of the six tasks is evaluated by task-specific accuracy (%): the fraction of test scenarios in which the model produces a correct prediction (e.g., identifying the spy, predicting the accept/reject decision, exact-matching the star rating). We report the macro-average across all six tasks.

For ScienceWorld, we use the environment completion score (0–100) returned by the simulator, which measures the degree to which the agent fulfills the task objective. We report the mean score across all task variations.

For brevity, we refer to all metrics uniformly as reward throughout the paper.

## A.5 OPENSKILL Hyperparameters

Table 5 lists the concrete configuration used in our experiments.

<table><tr><td>Component</td><td>Setting</td></tr><tr><td>Knowledge acquisition (D)</td><td></td></tr><tr><td>Deep-research agent</td><td>deep-research-pro</td></tr><tr><td>Poll interval /hard timeout</td><td>10s /3600s</td></tr><tr><td>Skills per plan</td><td>1-4</td></tr><tr><td>Verification knowledge (Du)</td><td></td></tr><tr><td>Search model</td><td>gemini-3.1-f</td></tr><tr><td>Temperature</td><td>0.3</td></tr><tr><td>“Already known” context</td><td>4000 chars</td></tr><tr><td>Virtual testing &amp; evolution (g,r)</td><td></td></tr><tr><td>Pass threshold to exit</td><td>r=1.0</td></tr><tr><td>Max refinement retries (J)</td><td>3</td></tr><tr><td>Max host interventions</td><td>1</td></tr><tr><td>Idle /stale episode limit</td><td>3/10</td></tr><tr><td>Token-budget warn / stop</td><td>70% / 90%</td></tr><tr><td>Verifier/creator max episodes</td><td>20/60</td></tr><tr><td>Diagnostic-driven retrieval (D on gaps)</td><td></td></tr><tr><td>Classifier model</td><td>gemini-3.1-f</td></tr><tr><td>Classifier temp./max tokens</td><td>0.1/200</td></tr><tr><td>Max searches per task</td><td>3</td></tr><tr><td>Zero-shot evaluation</td><td></td></tr><tr><td>Evaluation runs  $n _ { \mathrm { e v a l } }$ </td><td>5</td></tr></table>

Table 5 OpenSkill configuration used in experiments. Model names are abbreviated; see Appendix B for the exact identifiers and the role of each component.

## B Pipeline Implementation Details

This appendix specifies how the abstract functions in Section 2—the open-world retrieval D and $\mathcal { D } ^ { v } .$ , the virtual-test generator g, the refinement loop, and the gap/bug diagnosis—are realized, so that the pipeline is reproducible. The pipeline is orchestrated on the host, while skill creation and verification run inside a per-task Docker container.

## B.1 Open-World Retrieval D

D is decomposed into three host-side stages.

Query synthesis. An LLM Agent Planner reads the task instruction $\mathcal { T } _ { i } .$ the task metadata (category, tags, difficulty), and a preview of the environment files in $\mathcal { E } _ { i }$ (text files truncated to 2000 characters; PDFs extracted to 5000 characters; skills/ and doc/ directories excluded). It emits a single free-text research query that requests library APIs, function signatures, parameter defaults, working code examples, and common pitfalls, while being explicitly instructed not to include the solution approach.

Leakage filtering. Before any query is issued, a sanitize\_query step removes the benchmark name and its spelling variants (case-insensitive) from the query string, so that the retrieval engine cannot match the benchmark’s own pages and leak target answers. This filter is applied to every query issued by D and $\mathcal { D } ^ { v }$

Retrieval and planning. The sanitized query is submitted to a commercial deep-research agent (deepresearch-pro-preview-12-2025) through an asynchronous interactions API: a single query is submitted and polled every 10 s up to a 3600 s hard timeout; the multi-step web search and synthesis are performed server-side. The result is written as a background document $k _ { i }$ together with a deduplicated list of source URLs. A second LLM, the Skill Planner, then decomposes $k _ { i }$ into 1–4 skills (each with a name, responsibility, key functions, and the background sections it depends on) to form the plan $p _ { i } ,$ , and slices the background document into per-skill reference material by fuzzy-matching section headers (falling back to the full document when no section matches).

## B.2 Verification-Knowledge Retrieval $\mathcal { D } ^ { v }$

$\mathcal { D } ^ { v }$ is a second, orthogonal retrieval pass implemented as a single search-grounded generation call (gemini-3.1- flash-lite, temperature 0.3, Google-Search grounding). Unlike ${ \mathcal { D } } ,$ its prompt does not ask for API knowledge; it asks only for four classes of independently checkable anchors: (i) reference values that can be computed by hand for small, well-known inputs; (ii) dataset-level statistical invariants (row counts, sum-to-one constraints, monotonicity, value ranges); (iii) cross-validation procedures using alternative tools or libraries; and (iv) published benchmarks or reference implementations with known input–output pairs. To keep $k _ { i } ^ { v }$ disjoint from $k _ { i }$ , the first 4000 characters of the background document are inserted into the prompt under an “Already Known—do not repeat” block. The result $k _ { i } ^ { v }$ is stored and later injected into the virtual-test generator.

## B.3 Virtual-Test Generation g

The virtual test suite $\tilde { \mathcal { T } } _ { i }$ is produced by an Independent Verifier: a fresh LLM session that shares the container (and therefore the produced output files) with the skill creator but not its conversation, reasoning, or implementation code. This isolation is by design, to avoid confirmation bias—the verifier judges outputs without seeing how they were generated.

The verifier is instructed to derive expected values either by independently recomputing them from the environment inputs and the documented task rules, or directly from the verification knowledge $k _ { i } ^ { v } .$ , which is injected into its prompt as the “primary oracle” of externally verified values. It emits a pytest script of deterministic equality assertions (e.g., assert $\mathbf { x } = = \mathbf { y } )$ that is executed in the container. Each assertion $\tilde { t } _ { i , k } \in \{ 0 , 1 \}$ is thus an exact, reproducible check. We emphasize that the hidden test suite $\mathcal { T } _ { i } ^ { \mathrm { G T } }$ is never referenced: its file paths are not provided to the verifier, and the SkillWeaver loop never invokes the groundtruth oracle during construction. The isolation is enforced at the process and prompt level (a separate session with no GT paths) rather than by a filesystem sandbox.

A non-LLM parser reads the pytest output and computes the virtual pass rate $\tilde { r } ^ { ( j ) }$ as passed/total (skipped tests excluded), yielding the quality signal in Eq. 4. Across refinement rounds the verifier inherits its previous script and failure list, repairing broken test code before adding deeper assertions, and is capped at 60 tests to favor correctness over quantity.

## B.4 Iterative Refinement and Termination

Refinement is performed by the skill creator itself, in a single long-lived session: when $\tilde { r } ^ { ( j ) } < 1$ , the host returns a structured failure report—passed/total counts, the failed-assertion list, and an optional diagnosis—and instructs the creator to fix the logic in its skill scripts and regenerate outputs (never to edit outputs directly). The loop exits with status surrogate\_pass only when r˜ = 1.0 and auxiliary structural checks pass. The bound J in Section 2.2.2 is realized as a cap of 3 surrogate-failure retries (with at most one successful intervention); additional early-stopping guards trigger on idle/stale episodes (3/10) and on token-budget exhaustion (warning at 70%, stop at 90%).

## B.5 Gap-vs-Bug Diagnosis and Targeted Retrieval

When a refinement round fails, an LLM classifier (gemini-3.1-flash-lite, temperature 0.1, 200 max tokens) reads the failed assertions, the diagnosis, and the prior domain knowledge, and emits one line: SELF-FIXABLE (an implementation bug—wrong variable, off-by-one, format mismatch, type error, or a fix already implied by existing knowledge) or NEEDS-DR (a knowledge gap—an unknown correct value, an algorithm/parameter choice requiring domain expertise, or library usage not covered by prior knowledge). On SELF-FIXABLE, only the verifier feedback is returned and the creator fixes the code unaided. On NEEDS-DR, a targeted search $k _ { i } ^ { ( \mathrm { g a p } ) }$ is issued (sharing the search-grounding path of $\mathcal { D } ^ { v } )$ and its result is appended to the feedback. Targeted retrieval is capped at 3 searches per task, with already-searched topics listed in the query to discourage repetition; if the classifier call fails, the system conservatively defaults to SELF-FIXABLE (no retrieval).

## B.6 Final Skill Selection

Because the creator edits one skill set in place across rounds, the final skill set $\hat { S } _ { i } ^ { * }$ deployed in Stage 3 is the state of the evo-\* skills at loop termination—the most recently refined version—rather than a best-of-N snapshot selected by virtual pass rate. These skills are exported from the container and copied into the target workspace, where the agent uses them under $n _ { \mathrm { e v a l } } = 5$ independent zero-shot evaluation runs scored by $\mathcal { T } _ { i } ^ { \mathrm { { \breve { G } T } } }$

## C Failure Modes of Virtual Verifier

We analyze the 33 disagreement cases between the virtual verifier and GT evaluation (25 false positives and 8 false negatives) to characterize the failure modes of the virtual verifier.

False positives (proxy pass, GT fail). The 25 false positive tasks fall into three categories. (1) Highaccuracy near-misses (12 tasks, mean acc = 0.81): these tasks pass the majority of GT tests but receive zero reward due to the strict all-or-nothing evaluation criterion (reward > 0 requires every test to pass across all runs). The virtual verifier correctly identifies that the skill produces largely correct outputs, but cannot anticipate the few remaining edge-case failures that the GT suite catches. (2) Partial correctness (11 tasks, mean acc = 0.52): the surrogate tests are satisfied by outputs that are structurally valid but semantically incomplete—for instance, producing a well-formed JSON with incorrect numerical values. This reflects the virtual verifier’s reliance on format-level and boundary checks, which are insufficient for tasks requiring deep computational verification. (3) Genuine misalignment (2 tasks, acc = 0.0): the surrogate tests fail to capture any aspect of the correct behavior, approving entirely incorrect outputs. Both cases involve domain-specific chemical similarity search and code translation tasks where the virtual verifier lacks the prerequisite knowledge to generate meaningful test oracles.

False negatives (proxy fail, GT pass). The 8 false negative tasks cluster into two patterns. (1) Near-pass (3 tasks): the virtual verifier achieved 89–97% surrogate pass rate but fell short of 100%, typically failing on one or two overly strict surrogate assertions. These tasks succeed on GT evaluation because the unmet surrogate tests target edge cases that are not tested by the GT suite. (2) Verifier infrastructure failure (5 tasks): the surrogate test generation or execution pipeline failed entirely (0% pass rate), yet the agent still produced a correct solution. These tasks are predominantly non-standard (CVE patches, build system fixes, proof assistants) where the virtual verifier’s pytest-based testing framework is fundamentally unsuitable for validating the output.

## D SocialMaze Per-Subtask Results

Table 6 expands the SocialMaze averages of Table 2 into the six SocialMaze subtasks (FTS, HRD, REFT, RDP, SGA, and UPI) for both target agents. Per-subtask scores vary widely across methods; OpenSkill attains the best overall average on both target agents (82.7% on Opus, 70.7% on GPT), driven largely by the harder reasoning subtasks (REFT, UPI on Opus; REFT, RDP on GPT) rather than uniform gains across all subtasks.

## E Computational Cost

Table 7 reports the per-task computational cost of OpenSkill on SkillsBench (84 tasks, Opus 4.6). The pipeline consists of two phases: skill creation (Stages 1–4: deep research, skill planning, and skill synthesis with a virtual verifier loop) and evaluation (Stage 5: 5 independent agent runs using the generated skill).

Skill creation dominates the token budget (749K out of 1.14M tokens, 66%) but accounts for only 30% of the wall-clock time (39 out of 131 minutes). This discrepancy arises because the virtual verifier loop (Stage 4, averaging 3.0 iterations) runs as a single sequential LLM session, whereas ground-truth (GT) evaluation requires 5 independent, Docker-containerized agent runs. Across all 84 tasks, the total skill creation cost accumulates to 31.4 API-hours and ∼47M tokens. The full end-to-end pipeline (creation and evaluation)

<table><tr><td>Method</td><td>FTS</td><td>HRD</td><td>REFT</td><td>RDP</td><td>SGA</td><td>UPI</td><td>Avg</td></tr><tr><td colspan="8"> Opus 4.6</td></tr><tr><td> OPENSKILL</td><td>98.0</td><td>90.0</td><td>53.7</td><td>80.0</td><td>100.0</td><td>74.5</td><td>82.7</td></tr><tr><td>No Skill</td><td>96.0</td><td>91.2</td><td>50.7</td><td>80.0</td><td>100.0</td><td>71.5</td><td>81.6</td></tr><tr><td>Skill Creator</td><td>98.0</td><td>91.2</td><td>49.6</td><td>77.5</td><td>100.0</td><td>70.0</td><td>81.0</td></tr><tr><td>Self-Gen</td><td>92.0</td><td>92.5</td><td>51.1</td><td>75.0</td><td>100.0</td><td>72.0</td><td>80.4</td></tr><tr><td>Memento</td><td>100.0</td><td>91.2</td><td>46.7</td><td>75.0</td><td>98.8</td><td>70.0</td><td>80.3</td></tr><tr><td>CoT</td><td>94.0</td><td>88.8</td><td>52.3</td><td>70.0</td><td>100.0</td><td>71.5</td><td>79.4</td></tr><tr><td>AutoSkill</td><td>100.0</td><td>92.5</td><td>27.3</td><td>75.0</td><td>100.0</td><td>68.5</td><td>77.2</td></tr><tr><td colspan="8">GPT 5.2</td></tr><tr><td> OPENSKILL</td><td>98.0</td><td>60.0</td><td>54.6</td><td>62.5</td><td>80.0</td><td>69.0</td><td>70.7</td></tr><tr><td>Skill Creator 100.0</td><td></td><td>72.5</td><td>39.3</td><td>55.0</td><td>86.2</td><td>65.5</td><td>69.8</td></tr><tr><td>Self-Gen</td><td>98.0</td><td>67.5</td><td>43.7</td><td>55.0</td><td>68.8</td><td>72.5</td><td>67.6</td></tr><tr><td>Memento</td><td>100.0</td><td>71.2</td><td>46.3</td><td>57.5</td><td>63.7</td><td>65.5</td><td>67.4</td></tr><tr><td>CoT</td><td>94.0</td><td>58.8</td><td>51.9</td><td>55.0</td><td>70.0</td><td>70.5</td><td>66.7</td></tr><tr><td>No Skill</td><td>96.0</td><td>52.5</td><td>51.0</td><td>57.5</td><td>72.5</td><td>70.0</td><td>66.6</td></tr><tr><td>AutoSkill</td><td>98.0</td><td>67.5</td><td>29.7</td><td>55.0</td><td>72.5</td><td>72.5</td><td>65.9</td></tr></table>

Table 6 Per-subtask reward (%) on SocialMaze for the Opus 4.6 and GPT 5.2 target agents. OpenSkill rows are shaded; best average per target agent in bold.
<table><tr><td>Stage</td><td>Description</td><td>Tokens</td><td>Time</td></tr><tr><td>1</td><td>Deep Research (Gemini)</td><td>~10K</td><td>~8min</td></tr><tr><td>2</td><td>Skill Plan Generation</td><td>~10K</td><td>&lt;1 min</td></tr><tr><td>3</td><td>Surrogate DR (Gemini)</td><td>~2K</td><td>~2 min</td></tr><tr><td>4</td><td>Skill Creation+ VV Loop</td><td>727K</td><td>29.0min</td></tr><tr><td></td><td>Skill creation total</td><td>~749K</td><td>~39 min</td></tr><tr><td>5</td><td>GT Evaluation (5 runs)</td><td>~400K</td><td>91.5 min</td></tr><tr><td></td><td>End-to-end total</td><td>~1.14M</td><td>~131min</td></tr></table>

Table 7 Per-task computational cost on SkillsBench (Opus 4.6). Token counts for Stage 4 are measured from execution logs; others are estimated from artifact sizes. Stages 1 and 3 use Gemini 2.5 Flash; all other stages use Opus 4.6.

totals 140 hours and ∼97M tokens, corresponding to an estimated API cost of ∼\$1,800 at Opus 4.6 list pricing (\$15/M input, \$75/M output).

Importantly, skill creation is a one-time cost: once generated, skills are reused across models and runs without additional creation overhead. In the cross-model transfer experiments, skill-equipped evaluation takes 16–27 minutes per task depending on the target model (Haiku: 17.7 min, DeepSeek: 18.1 min, Qwen 3 Coder: 15.7 min, Mistral: 27.0 min), incurring zero additional skill creation cost.

Computational Cost Comparison. Table 8 compares the per-task, per-run evaluation cost between the No-Skill baseline and OpenSkill. Both configurations use Opus 4.6 as the backbone LLM under the same evaluation harness.

The No-Skill agent spends an average of 465.0 s per evaluation run. Considering the evaluation stage alone, OpenSkill requires a median of 368.2 s per run, which is comparable with the No-Skill baseline. The substantial gap between the eval-only mean (845.4 s) and median (368.2 s) is driven by a small number of long-running tasks.

## F Information Isolation Audit

A key design requirement of the Virtual Verifier is strict information isolation: it must generate surrogate tests without access to ground-truth (GT) tests, solutions, or the skill creator’s internal state. We verify this through a four-layer audit.

<table><tr><td>Result</td><td>No-Skill</td><td>OpenSkill</td></tr><tr><td>Per-run time,mean (s)</td><td>465.0</td><td>845.4</td></tr><tr><td>Per-run time, ,median (s)</td><td>347.6</td><td>368.2</td></tr><tr><td>Mean reward</td><td>25.5%</td><td>43.6%</td></tr></table>

Table 8 Per-run evaluation cost on SkillsBench (Opus 4.6 backbone). It reports GT evaluation time, excluding skill creation.

Code-level isolation. The base agent accepts exactly two inputs: the task instruction and environment files (input data, Dockerfile). Its function signature explicitly excludes solution and test paths. The independent verifier (independent\_verifier.py) instantiates a separate LLM session with no shared context from the skill creator, preventing any cross-agent information leakage.

Container-level isolation. Each task runs inside a Docker container built from the task’s environment/- Dockerfile. The GT test directory (tests/) and solution directory (solution/) reside on the host filesystem as siblings to environment/ and are never mounted or copied into the container. The verifier agent can only observe files under /app/environment/ and outputs written by the creator agent.

GT oracle bypass. The OpenSkill evolution loop overrides the parent class’s GT oracle, ensuring the GT oracle is never invoked during skill creation. The loop exits upon surrogate test passage, without ever reaching the GT evaluation code path. The only GT-derived signal is a single pass/fail bit used when the verifier is re-invoked to write deeper tests; no GT test content, assertion details, or expected values are exposed.

Log-level verification. We audited evolution\_run\_log.json files. Zero references to GT test files appear in agent execution trajectories. The only GT-related entries reside in post-hoc pipeline evaluation fields that are recorded after the agent has exited and are never fed back to any agent.

Prompt-level enforcement. Both the surrogate writer and independent verifier system prompts contain explicit instructions: “You must ONLY use information from the task instruction and environment files. You have NO access to the solution or ground-truth tests.”

## G Baseline Prompts

## G.1 Self-Generated Skills Prompt

This prompt replicates the self-generation condition of SkillsBench (Li et al., 2026) (Appendix C.6). It is appended to the task instruction; the agent generates skills in-session before solving the task, with no external verification.

## Self-Generated Skills Prompt

Important: Generate Skills First

Before attempting to solve this task:

1. Analyze the task requirements and identify what domain knowledge, APIs, or techniques are needed.

2. Write 1–5 modular skill documents.

3. Save each skill as a markdown file in environment/skills/.

4. Then solve the task using the skills you created as reference.

## G.2 CoT-Guided Self-Generation Prompt

This prompt extends the Self-Generated Skills baseline with a structured five-step chain-of-thought workflow. Despite the added structure, the agent still lacks external verification feedback, and this condition achieves only 30.7% pass rate (comparable to the no-skill baseline).

## CoT-Guided Self-Generation Prompt

Step 1: Task Analysis – identify domain, tools, output format, pitfalls

Step 2: Skill Architecture Design – plan 1–3 focused skills

Step 3: Write Skills with Progressive Disclosure

(a) YAML frontmatter: name and description

(b) Key constraints and rules

(c) Step-by-step workflow with decision points

(d) Common mistakes to avoid and edge cases

(e) If helpful, include scripts/ with reusable utility code

Step 4: Self-Verify – re-read instruction, check every requirement has coverage

Step 5: Execute