Title: ASPIRE: Agentic /Skills Discovery for Robotics

URL Source: https://arxiv.org/html/2607.00272

Markdown Content:
Yubo Wu{}^{1\;3\;*} Ethan Kou{}^{1\;4\;*}

Letian Fu 1 4 Wenli Xiao 1 5 Ajay Mandlekar 1 Yinzhen Xu 1

Guanya Shi 5 Ken Goldberg 4 Ang Chen 2 Mosharaf Chowdhury 2

Yuke Zhu{}^{1\;\dagger} Linxi “Jim” Fan{}^{1\;\dagger} Guanzhi Wang{}^{1\;\dagger}

1 NVIDIA, 2 UMich, 3 UIUC, 4 UC Berkeley, 5 CMU 

∗ Equal contribution, † Project leads 

[https://research.nvidia.com/labs/gear/aspire/](https://research.nvidia.com/labs/gear/aspire/)

###### Abstract

Abstract:

Traditional robot programming is notoriously challenging: it requires orchestrating multimodal perception, managing complex physical contact dynamics, and handling diverse environment configurations and execution failures. We introduce Aspire (A gentic S kill P rogramming through I terative R obot E xploration), a continual learning system for robotics that autonomously writes and refines robot control programs in a code-as-policy paradigm while compounding experience into a reusable skill library. Aspire enables automated discovery of reusable skills that persist across multiple tasks, simulation and real-world settings, and different embodiments. Rather than relying on fixed, human-engineered pipelines, Aspire operates in an open-ended learning loop, consisting of three key components: (1) a closed-loop robot execution engine that exposes fine-grained multimodal traces (e.g., perception overlays, grasp candidates, motion trajectories, and collision feedback), enabling the agent to autonomously diagnose failures, synthesize repairs, and validate outcomes; (2) a continually expanding skill library that distills validated fixes into reusable, transferable robotic knowledge; and (3) an evolutionary search procedure that generates diverse task sequences and control programs, systematically debugging them to explore beyond single-trajectory refinement. As Aspire encounters more tasks, its growing skill library enables increasingly rapid adaptation. Consequently, Aspire surpasses prior methods by up to 77% on manipulation tasks under perturbation (LIBERO-Pro), 72% on Robosuite’s bimanual handover task, and up to 32% on long-horizon household tasks (BEHAVIOR-1K). The accumulated skill library further enables strong zero-shot generalization: on representative unseen long-horizon tasks (LIBERO-Pro Long), Aspire achieves 31% success, substantially outperforming the 4% success rate of prior methods despite their heavy reliance on test-time reasoning and retries. Finally, skills discovered in simulation provide initial evidence of sim-to-real transfer, substantially reducing real-robot programming effort despite different embodiments and robot APIs.

## 1 Introduction

Recent progress in software engineering agents demonstrates that language models can autonomously inspect execution traces, localize failures, revise implementations, and improve through repeated interaction with execution environments (claudecode2025; openai2025codex; opencode2025; wang2025openhands; yang2024sweagent). In robotics, this paradigm has inspired _code-as-policy_ systems that compose perception modules, planning APIs, and control primitives into executable robot programs (liang2023code; singh2023progprompt; ahn2022saycan; huang2023voxposer; mu2024robocodex; capx2026). Because robot behaviors are represented explicitly as programs, they can in principle be inspected, edited, debugged, and refined through interaction feedback.

However, existing robotic coding agents remain fundamentally limited by naive execution environments that provide only coarse task-level feedback. Debugging robot programs is intrinsically challenging because failures can arise from many interacting components, including multimodal perception, motion planning, grasp generation, contact dynamics, and long-horizon task coordination. A failed rollout may indicate that the task did not succeed, but not whether the root cause is incorrect perception, an unstable grasp, a planning error, or a downstream recovery failure. Without fine-grained diagnostic traces, agents have limited ability to determine what evidence to inspect, how to localize failures, or what repair strategy to attempt.

Moreover, existing systems do not accumulate experience across tasks. Once a task is completed, discovered fixes and recovery strategies are discarded rather than consolidated into reusable skills. As a result, the agent solving its hundredth task is effectively no more experienced than the agent solving its first.

Human robotics engineers take a fundamentally different approach. When a robot program fails, they replay executions, inspect perception outputs and motion trajectories, localize the failing subsystem, revise the implementation, and internalize reusable recovery strategies. Over time, debugging experience compounds into transferable knowledge, including grasp recovery heuristics, navigation strategies, prompting recipes, and procedural fixes that generalize across tasks. This accumulation of reusable knowledge is a key reason why human robot programmers become progressively more effective over time.

![Image 1: Refer to caption](https://arxiv.org/html/2607.00272v1/x1.png)

Figure 1: Aspire system overview. A _coordinator_ spawns an _actor agent_ (coding agent) per task, enabling parallel learning across tasks. Each actor refines and validates robot programs through _iterative debugging_ with the _robot execution engine_, which exposes per-primitive multimodal traces for failure attribution and repair. _Evolutionary search_ samples diverse candidate programs (\pi_{0},\ldots,\pi_{k}), sends each through the engine, and conditions the next generation (\pi^{\prime}_{0},\ldots,\pi^{\prime}_{k}) on surviving programs and residual failure traces. The _coordinator_ writes validated repairs into a shared _skill library_, which future actors retrieve as in-context guidance; skills discovered in sim can also be adapted as cross-embodiment guidance for real-robot programming. 

In this work, we introduce Aspire, a self-improving continual learning robotic system that autonomously writes and refines robot control programs in a code-as-policy paradigm while accumulating experience into a reusable skill library. Rather than operating within a fixed perception-plan-execute pipeline, Aspire runs in an open-ended learning loop, in which the agent self-determines how to inspect execution traces, diagnose failures, synthesize repairs, validate corrected behaviors, and consolidate successful recovery patterns into persistent skills that transfer across tasks.

An overview of Aspire is shown in Fig. [1](https://arxiv.org/html/2607.00272#S1.F1 "Figure 1 ‣ 1 Introduction ‣ ASPIRE: Agentic /Skills Discovery for Robotics"). Aspire is built from three components. First, a closed-loop robot execution engine replaces coarse rollout-level feedback with per-primitive execution traces. For each perception, planning, grasping, and control call, the execution engine records the observations, inputs, outputs, and visual evidence if possible. These rich multimodal traces allow the agent to selectively inspect salient primitive logs, progressively localize failures, and validate repairs through re-execution. Second, Aspire maintains a growing skill library that distills validated fixes into reusable, transferable robotic knowledge retrievable as in-context guidance for future tasks. Third, Aspire employs an evolutionary search procedure that generates diverse task sequences and control programs, exploring beyond single-trajectory self-improvement through iterative debugging and parallel refinement. Together, these components establish a self-improving robotic system whose performance scales with experience: the more tasks Aspire sees, the larger its skill library grows, and the more it transfers to novel tasks, longer-horizon behaviors, and real-world robotic settings where similar failure-recovery patterns emerge under different embodiments.

Empirically, Aspire demonstrates strong self-improvement across diverse short- and long-horizon robot benchmarks. Against prior coding agents (capx2026), Aspire improves success rate by up to 77 points on LIBERO-Pro (zhou2025liberopro) perturbation suites, by up to 72 points on Robosuite (zhu2020robosuite) contact-rich manipulation tasks, and by up to 32 points on BEHAVIOR-1K (behavior1k2023) long-horizon household tasks with procedurally generated layouts. Based on the skill library accumulated on LIBERO-90 (libero2024), Aspire transfers zero-shot to LIBERO-Pro Long (zhou2025liberopro), and reaches 31% success, while prior methods saturate at 4% despite their reliance on test-time reasoning and retries. We further evaluate three simulation-discovered skills on a real bimanual robot, showing that retrieving skills discovered in sim as in-context guidance reduces real-world reasoning tokens and enables successful programs where naive debugging without skill transfer fails entirely.

## 2 Method

![Image 2: Refer to caption](https://arxiv.org/html/2607.00272v1/x2.png)

Figure 2: Robot execution engine. Trace-guided debugging on a BEHAVIOR-1K navigate-and-pick-up-radio task. (a) Ego-view keyframes and overlays show the robot locating the radio but failing to approach it. (b) The primitive trace localizes the failure to repeated PLANNING_ERROR s: candidate navigation goals fall inside the table’s collision-avoidance buffer. (c) The agent patches the program with a multi-angle approach routine, re-perceives the radio from a reachable side, and completes the grasp. The validated repair is admitted as a reusable _Multi-Angle Approach_ skill. 

![Image 3: Refer to caption](https://arxiv.org/html/2607.00272v1/x3.png)

Figure 3: Skill library.Aspire stores validated, agent-discovered repair knowledge as reusable in-context skills rather than a fixed set of human-written primitives. Top: representative entries show learned skills about localization disambiguation, motion-primitive construction, and navigation recovery. Middle: the library grows across heterogeneous categories, including localization, navigation, motion primitives, object-level grasping, scene understanding, and debugging workflows. Bottom: selected skills discovered in sim are used as in-context guidance for real-robot programming, providing evidence that skills can transfer across embodiments. Additional skill examples and prompts are shown in Appendix [A](https://arxiv.org/html/2607.00272#A1 "Appendix A Skill Library Details ‣ ASPIRE: Agentic /Skills Discovery for Robotics"). 

Aspire consists of three components that together form an open-ended learning loop: (1) a _robot execution engine_ (§[2.1](https://arxiv.org/html/2607.00272#S2.SS1 "2.1 Robot Execution Engine ‣ 2 Method ‣ ASPIRE: Agentic /Skills Discovery for Robotics")) that exposes per-primitive multimodal traces for failure attribution and executes agent-written repairs for closed-loop validation; (2) a _skill library_ (§[2.2](https://arxiv.org/html/2607.00272#S2.SS2 "2.2 Skill Library ‣ 2 Method ‣ ASPIRE: Agentic /Skills Discovery for Robotics")) that accumulates validated repairs as reusable skills; and (3) an _evolutionary search_ procedure (§[2.3](https://arxiv.org/html/2607.00272#S2.SS3 "2.3 Evolutionary Search ‣ 2 Method ‣ ASPIRE: Agentic /Skills Discovery for Robotics")) that broadens exploration beyond single-trajectory repair. As Aspire encounters more tasks, its skill library grows, allowing future tasks to inherit accumulated repairs and reusable strategies.

As shown in Fig. [1](https://arxiv.org/html/2607.00272#S1.F1 "Figure 1 ‣ 1 Introduction ‣ ASPIRE: Agentic /Skills Discovery for Robotics"), Aspire adopts a coordinator–actor architecture. A central coordinator manages the shared skill library and dispatches actor coding agents to individual tasks, while each actor writes, executes, diagnoses, and repairs robot programs within the robot execution engine. Actors do not exchange full chat histories or raw rollout trajectories. Instead, transferable experience is distilled into the skill library, allowing each actor’s context window to remain focused on the task specification, current program, and structured execution traces associated with the current failure.

### 2.1 Robot Execution Engine

Embodied coding agents need execution evidence to debug robot programs. Prior methods expose this evidence through fixed, human-designed interfaces, typically as manually curated scene-level summaries or a set of pre-defined observations. This creates a trade-off: too little evidence can hide the failing primitive, while too much raw visual context can distract the agent from the causal chain that produced the failure.

Aspire turns this fixed feedback channel into an open-ended debugging environment. The robot execution engine records per-primitive multimodal traces for perception, planning, and control calls, exposes the trace to the coding agent, and executes agent-written repairs for closed-loop validation. For each primitive call, the trace stores the invoked API, inputs and outputs, return status, and relevant multimodal evidence such as RGB keyframes, overlays, grasp candidates, object poses, and motion-planning results. The agent does not receive full video frames; the engine keeps frames immediately before and after each primitive call together with the corresponding overlays and return values, so the agent can focus on evidence around calls implicated by the failure.

Fig. [2](https://arxiv.org/html/2607.00272#S2.F2 "Figure 2 ‣ 2 Method ‣ ASPIRE: Agentic /Skills Discovery for Robotics") zooms in on one BEHAVIOR-1K navigate-and-pick-up-radio debugging episode. The ego-view keyframes show that the robot finds the radio, repeatedly fails to approach it, and then succeeds after changing its approach direction. The primitive trace localizes the failure: perception succeeds and returns a radio pose, but repeated navigate_to_pose calls return PLANNING_ERROR. By checking the navigation return values and associated logs, the agent finds that the generated navigation target lies too close to the table boundary, approximately within 20 centimeters of the table edge. This triggers collision avoidance and causes the planner to fail. Thus, the failure is not due to detecting or grasping the radio, but rather to the infeasibility of the target pose under the table’s collision constraints.

The repair follows directly from this diagnosis. Rather than changing the perception prompt or grasp primitive, the agent writes a multi-angle approach routine that samples alternative navigation targets around the radio and executes an approach direction that clears the collision buffer. The execution engine exposes the evidence, validates the patched program, and enables the agent to analyze the resulting logs, form a hypothesis, and make a targeted repair decision.

### 2.2 Skill Library

Program failures recur across tasks, but the reusable knowledge is rarely an entire task program. Aspire’s skill library stores heterogeneous repair knowledge: localization heuristics, perception prompts, grasping constraints, navigation recovery strategies, motion primitives, scene-understanding routines, and debugging workflows. We do not prescribe this taxonomy in advance. Skills are induced from validated repairs: the coding agent diagnoses a failure from execution traces, patches the program, validates the fix on debugging configurations, and the coordinator admits only reusable patterns into the shared library.

Each skill is stored as compact in-context guidance, including the failure signature, when-to-apply condition, repair strategy, and, when useful, a representative code sketch. Fig. [3](https://arxiv.org/html/2607.00272#S2.F3 "Figure 3 ‣ 2 Method ‣ ASPIRE: Agentic /Skills Discovery for Robotics") shows representative entries and the resulting library breadth. For the radio task in Fig. [2](https://arxiv.org/html/2607.00272#S2.F2 "Figure 2 ‣ 2 Method ‣ ASPIRE: Agentic /Skills Discovery for Robotics"), the admitted skill is a navigation recovery pattern rather than a complete radio-pickup program: when planner errors recur near an obstacle boundary because sampled target poses fall inside the collision buffer, sample alternative approach directions around the object before retrying perception and grasping. This representation lets future actors reuse validated repairs instead of rediscovering them through test-time reasoning, supports zero-shot transfer to harder simulated tasks, and provides the mechanism for selected simulation-discovered skills to generalize across embodiments and transfer to real robots.

Actors report structured findings that summarize the failure mode, validated fix, and potentially transferable repair pattern. The coordinator audits these findings, verifies compliance with the allowed API policy, and promotes only reusable repairs that have passed debug validation into the shared skill library. Appendix [E.1](https://arxiv.org/html/2607.00272#A5.SS1 "E.1 Skill Library Prompts ‣ Appendix E Agent Pipeline Skills ‣ ASPIRE: Agentic /Skills Discovery for Robotics") provides the prompts used for actor reporting and coordinator-guided skill admission.

### 2.3 Evolutionary Search

Trace-guided debugging alone can collapse into local repair loops, where the agent repeatedly patches the same failed strategy instead of exploring fundamentally different ways to solve the task. Aspire uses evolutionary search to broaden exploration of executable robot programs, encouraging diverse repair hypotheses and task strategies.

In each round, based on the skill library, the coding agent proposes a population of K candidate programs conditioned on the top-performing previous programs and failure traces from previous evaluations. Each candidate is executed in the robot execution engine, producing task outcomes together with new diagnostic traces. The next round is then conditioned on the best-performing programs together with their remaining failure modes, allowing the search to explore distinct strategies rather than repeatedly refining the same solution.

The search target is the robot program itself. Candidates are selected through closed-loop execution, and validated repairs are admitted into the skill library after search concludes, provided they generalize across environment variations and tasks. Search terminates when a candidate solves the debugging configurations or when the search budget is exhausted. Algorithm [1](https://arxiv.org/html/2607.00272#alg1 "Algorithm 1 ‣ 2.3 Evolutionary Search ‣ 2 Method ‣ ASPIRE: Agentic /Skills Discovery for Robotics") summarizes the evolutionary search procedure. See Appendix [E.4](https://arxiv.org/html/2607.00272#A5.SS4 "E.4 ASPIRE Evolutionary Search Skills ‣ E.3 ASPIRE Fix Loop Skills ‣ E.2 Agent System Prompt and Persistent Memory ‣ Appendix E Agent Pipeline Skills ‣ ASPIRE: Agentic /Skills Discovery for Robotics") for pipeline details.

Algorithm 1 Evolutionary search over programs

1:task

\tau
, program

P^{0}
, sets

S_{\rm dbg},S_{\rm val}
, skill library

\mathcal{L}
, agent

M
, budget

(K,T)
, threshold

\theta

2:Notation:

\textsc{Execute}(P,S)=(r,Z)
returns score

r
and trace bundle

Z
.

3:

(r^{\star},Z^{0})\leftarrow\textsc{Execute}(P^{0},S_{\rm dbg})
;

P^{\star}\leftarrow P^{0}

4:

\mathcal{H}\leftarrow\{(P^{0},r^{\star},Z^{0})\}

5:for

i=1,\ldots,T
do

6:

\{P_{i}^{k}\}_{k=1}^{K}\leftarrow{}

7:

\textsc{ProposeRepairs}(M,\tau,\mathrm{Top3}(\mathcal{H}),\mathcal{L},\mathcal{H})

8:for

k=1,\ldots,K
do

9:

(r_{i}^{k},Z_{i}^{k})\leftarrow\textsc{Execute}(P_{i}^{k},S_{\rm dbg})

10:end for

11:

\mathcal{H}\leftarrow\mathcal{H}\cup\{(P_{i}^{k},r_{i}^{k},Z_{i}^{k})\}_{k=1}^{K}
;

k^{\star}\leftarrow\arg\max_{k}r_{i}^{k}

12:if

r_{i}^{k^{\star}}>r^{\star}
then

13:

(P^{\star},r^{\star})\leftarrow(P_{i}^{k^{\star}},r_{i}^{k^{\star}})

14:end if

15:if

r^{\star}\geq\theta
then

16:break

17:end if

18:end for

19:

(r_{\rm val},Z_{\rm val})\leftarrow\textsc{Execute}(P^{\star},S_{\rm val})

20:

\mathcal{G}\leftarrow{}

21:

\textsc{ExtractValidatedPatterns}(\mathcal{H},P^{\star},r_{\rm val},Z_{\rm val})

22:return

(P^{\star},r_{\rm val},\mathcal{G})

## 3 Experiments

### 3.1 Experimental Setup

For all the simulation benchmarks, we use Claude Code (claudecode2025) with Claude Opus 4.6 and a 1M-token context window as the coding agent. The agent writes executable Python robot programs in CaP-X (capx2026), an open-source code-as-policy framework built on MuJoCo Playground (zakka2025mujocoplayground), with robot programming APIs for perception, geometry, and motion planning. The agent, environment, and API set are fixed across all experiments.

For the real-robot skill transfer study (§[3.6](https://arxiv.org/html/2607.00272#S3.SS6 "3.6 Real-Robot Skill Transfer Across Embodiments ‣ 3 Experiments ‣ ASPIRE: Agentic /Skills Discovery for Robotics")), we use OpenAI Codex GPT-5.5 in reasoning-xhigh mode on a bimanual YAM manipulation station. We select three skills compiled by Aspire in Franka-based simulation: soda-can pickup, bowl-on-plate placement, and drawer push/pull. We provide these as in-context guidance to the real-robot coding agent. The real-robot setting uses a different embodiment and API from simulation, but exposes Aspire multimodal execution traces so the agent can autonomously run, inspect, and debug programs without task-specific human guidance during the debugging loop. We compare runs with and without the corresponding simulation-discovered skill, measuring tokens to first success and held-out real-robot success rate.

![Image 4: Refer to caption](https://arxiv.org/html/2607.00272v1/x4.png)

![Image 5: Refer to caption](https://arxiv.org/html/2607.00272v1/x5.png)

(a)LIBERO-Pro(zhou2025liberopro): macro-averaged success over 10 tasks \times 50 held-out environment seeds per suite/perturbation. For each task, Aspire learns and collects skills on seeds 51–65 and evaluates one generated program on seeds 1–50.

![Image 6: Refer to caption](https://arxiv.org/html/2607.00272v1/x6.png)

(b)Robosuite(zhu2020robosuite): success over 100 held-out trials per task. Aspire learns on seeds 101–125 and evaluates one generated program per task on seeds 1–100; 2A tasks are bimanual.

![Image 7: Refer to caption](https://arxiv.org/html/2607.00272v1/x7.png)

(c)BEHAVIOR-1K(behavior1k2023): long-horizon mobile manipulation on two household tasks. Aspire learns on seeds 26–35 and evaluates on seeds 1–25 with incremental block execution; Navigation and Task success are reported separately.

Figure 4: Aspire improves over prior coding agents and end-to-end VLAs across three benchmark families. (a) Short-horizon manipulation on LIBERO-Pro; (b) contact-rich manipulation on Robosuite; (c) long-horizon mobile manipulation on BEHAVIOR-1K. Aspire evaluates one generated program per task across held-out seeds, while CaP-Agent0 regenerates a separate program per seed with test-time reasoning and retries. Aspire improves performance across all benchmarks, with several results surpassing programs written by human experts. Detailed per-task results are in Appendix [B](https://arxiv.org/html/2607.00272#A2 "Appendix B Main Benchmark Result Tables ‣ ASPIRE: Agentic /Skills Discovery for Robotics") and [D.1](https://arxiv.org/html/2607.00272#A4.SS1 "D.1 LIBERO-Pro Position Perturbation (Swap) Per-Task Breakdown ‣ Appendix D Ablation Details ‣ ASPIRE: Agentic /Skills Discovery for Robotics"). 

### 3.2 Benchmarks and Baselines

We evaluate Aspire on three benchmark families: LIBERO-Pro (zhou2025liberopro) for short-horizon robustness under object, goal, and spatial perturbations; Robosuite (zhu2020robosuite) for contact-rich single- and dual-arm manipulation; and BEHAVIOR-1K (behavior1k2023) for long-horizon household mobile manipulation on navigate-and-pick-up-soda-can and navigate-and-pick-up-radio. Our primary coding-agent baseline is CaP-Agent0 (capx2026), which uses visual differencing, a predefined skill library, and per-episode test-time retries. We also compare with end-to-end VLA policies, including OpenVLA (kim2024openvlaopensourcevisionlanguageactionmodel), \pi_{0}(black2024pi0visionlanguageactionflowmodel), and \pi_{0.5}(intelligence2025pi05). For zero-shot transfer, we evaluate held-out LIBERO-Pro Long tasks using the skill library accumulated on LIBERO-90 (libero2024).

### 3.3 Evaluation Protocol

Across all benchmarks, an environment seed fixes each task instance, including object poses, distractors, and initial robot/object states. We use disjoint debug and evaluation seeds: Aspire learns on a small debug split, then reports success on larger held-out evaluation seeds with one generated program per LIBERO-Pro/Robosuite task, while CaP-Agent0 regenerates a separate program for each seed with test-time reasoning and retries. For BEHAVIOR-1K evaluation, Aspire uses incremental block execution, generating each next code block from the current multimodal trace.

### 3.4 Main Evaluation Results

Figure [4](https://arxiv.org/html/2607.00272#S3.F4 "Figure 4 ‣ 3.1 Experimental Setup ‣ 3 Experiments ‣ ASPIRE: Agentic /Skills Discovery for Robotics") summarizes the main evaluation. “Human” means the programs are written by human experts. On LIBERO-Pro, Aspire improves over all the baselines on all three suites: averaging the Pos and Task perturbation axes, Aspire gains 77% on Object, 41.5% on Goal, and 42.5% on Spatial over the strongest baseline in each suite. \pi_{0.5} is stronger than OpenVLA and \pi_{0} on some position perturbations, but remains far below Aspire and largely collapses under task paraphrases. On Robosuite, Aspire preserves near-saturated performance on easier contact-rich tasks and substantially improves bimanual handover, from 20% to 92%. On BEHAVIOR-1K, Aspire outperforms both human and CaP-Agent0 on navigation and task success, with the largest task-level gain over CaP-Agent0 on navigate-and-pick-up-radio, from 56% to 88%.

### 3.5 Zero-Shot Transfer to Unseen Tasks

We evaluate whether repair skills accumulated on LIBERO-90 transfer zero-shot to held-out LIBERO-Pro Long tasks. Aspire builds library snapshots from N\in\{0,25,50,90\} source tasks, where N{=}0 is an empty-library setting and N{=}90 is the full suite. For each held-out task, Aspire generates one program and evaluates it across seeds with no additional debugging, retries, or task-specific library updates.

Figure [5](https://arxiv.org/html/2607.00272#S3.F5 "Figure 5 ‣ 3.5 Zero-Shot Transfer to Unseen Tasks ‣ 3 Experiments ‣ ASPIRE: Agentic /Skills Discovery for Robotics")(a) compares the full N{=}90 skill library with prior baselines on the two LIBERO-Pro Long transfer axes: Aspire reaches 23% success on position perturbations and 38% on task perturbations, outperforming CaP-Agent0 and \pi_{0.5} on both axes. Figure [5](https://arxiv.org/html/2607.00272#S3.F5 "Figure 5 ‣ 3.5 Zero-Shot Transfer to Unseen Tasks ‣ 3 Experiments ‣ ASPIRE: Agentic /Skills Discovery for Robotics")(b) shows that success increases consistently as the size of the skill library grows, indicating that validated repairs from short-horizon tasks provide reusable robotic knowledge for longer-horizon compositions.

![Image 8: Refer to caption](https://arxiv.org/html/2607.00272v1/x8.png)

(a)Full N{=}90 library zero-shot results versus baselines.

![Image 9: Refer to caption](https://arxiv.org/html/2607.00272v1/x9.png)

(b)Zero-shot success improves as skill-library size increases.

Figure 5: Cross-task zero-shot transfer on LIBERO-Pro Long. Skills accumulated on LIBERO-90 improve zero-shot performance on held-out long-horizon tasks. Figure (a) compares the full N{=}90 library with baselines. Figure (b) shows Pos/Task success as the size of the skill library increases. All success rates are macro-averaged over 10 tasks per axis. Per-task results are in Appendix [C](https://arxiv.org/html/2607.00272#A3 "Appendix C LIBERO-Pro Long Zero-Shot Transfer ‣ ASPIRE: Agentic /Skills Discovery for Robotics").

### 3.6 Real-Robot Skill Transfer Across Embodiments

We evaluate whether skills learned in simulation can reduce debugging effort on a real robot with a different embodiment. The transfer is not a direct policy deployment: the real robot uses its own perception, calibration, and control stack, and the coding agent must still adapt programs through real-world execution feedback. The question is whether retrieving simulation-discovered skills provides useful in-context guidance that reduces the amount of real-world debugging needed to reach a successful program.

Table [1](https://arxiv.org/html/2607.00272#S3.T1 "Table 1 ‣ 3.6 Real-Robot Skill Transfer Across Embodiments ‣ 3 Experiments ‣ ASPIRE: Agentic /Skills Discovery for Robotics") shows that transferred skills consistently reduce debugging cost, while their effect on final success is task-dependent. Bowl placement succeeds in both settings but uses fewer tokens with skill retrieval; soda-can lifting improves from 13/20 to 19/20 while reducing total tokens by nearly an order of magnitude; and drawer manipulation reaches 11/20 success with skill guidance, while the no-skill baseline exhausts a larger token budget without producing a successful evaluation program. These results indicate that selected failure-derived skills can guide real-robot program synthesis across embodiment and API changes rather than merely memorizing simulator-specific code.

Table 1: Real-robot cross-embodiment skill transfer. For each task, we compare real-robot debugging with and without retrieving a corresponding simulation-discovered skill from Aspire’s library. Token counts are measured until the first successful real-robot program, or until the debugging budget is exhausted for runs that do not reach success. Success rate reports evaluation trials for the generated programs. 

### 3.7 Ablation Studies

#### Robot execution engine and evolutionary search.

We ablate two key components of Aspire on LIBERO-Pro: the robot execution engine, which exposes dense execution traces and validates task-level repairs, and evolutionary search, which explores additional repair candidates after robot-execution-engine debugging. Figure [6](https://arxiv.org/html/2607.00272#S4.F6 "Figure 6 ‣ Self-improving agents and skill libraries. ‣ 4 Related Work ‣ ASPIRE: Agentic /Skills Discovery for Robotics")(a) and Figure [6](https://arxiv.org/html/2607.00272#S4.F6 "Figure 6 ‣ Self-improving agents and skill libraries. ‣ 4 Related Work ‣ ASPIRE: Agentic /Skills Discovery for Robotics")(b) decompose success into a base system without either component, the gain from adding the robot execution engine, and the additional gain from evolutionary search. The robot execution engine provides the largest average improvement, raising macro-average success from 14% to 62%; evolutionary search further improves the remaining hard tasks, reaching 72% with both components.

#### Evolutionary search iterations.

Figure [6](https://arxiv.org/html/2607.00272#S4.F6 "Figure 6 ‣ Self-improving agents and skill libraries. ‣ 4 Related Work ‣ ASPIRE: Agentic /Skills Discovery for Robotics")(c) tracks performance on low-performing tasks as the evolutionary-search budget increases. Success improves steadily in the first few rounds, suggesting that sampling multiple repair hypotheses quickly recovers alternatives missed by single-iteration debugging. The curve continues to increase more gradually afterward, indicating that additional rounds still help on residual hard cases, but with diminishing returns.

## 4 Related Work

#### Agentic robot control.

Robot control has been studied through end-to-end vision-language-action policies and executable programs that compose perception, planning, and control APIs (ahn2022saycan; brohan2023rt2; octo2024; kim2024openvlaopensourcevisionlanguageactionmodel; black2024pi0visionlanguageactionflowmodel; mu2024robocodex; intelligence2025pi05; bjorck2025gr00t; atomvla2025; li2026roboclaw; capx2026). Software-engineering agents provide a related write-execute-debug loop for code (jimenez2024swebench; wang2025openhands; yang2024sweagent; wang2024codeact; chen2024selfdebug; claudecode2025; openai2025codex; opencode2025). Aspire builds on the executable-program paradigm, but focuses on persistent embodied improvement: the agent freely selects primitive-level multimodal traces, writes repairs, validates, and preserves successful experiences across tasks.

#### Self-improving agents and skill libraries.

LLM agents have been improved through open-ended memory, skill libraries, and self-evolving skill repositories (voyager2023; tziafas2024lrll; peng2024hyvin; vistawise2025; skillflow2026; uniskill2026). Other systems use LLMs to generate rewards, curricula, environments, or search candidates (eureka2023; yu2023l2r; xie2024text2reward; wang2024eurekaverse; ma2024dreureka; du2023ellm; wang2024robogen; romeraparedes2024funsearch; novikov2025alphaevolve; guo2025evoengineer; guo2026codeevolution; cao2026ksearch). Unlike success-only memories, textual reflections, or reward functions for downstream policy training, Aspire stores validated repair knowledge extracted from attributed embodied failures. Its skills are open-ended in both content and admission: they emerge from the agent’s own debugging experience, span heterogeneous categories (§[2.2](https://arxiv.org/html/2607.00272#S2.SS2 "2.2 Skill Library ‣ 2 Method ‣ ASPIRE: Agentic /Skills Discovery for Robotics")), and are reused as in-context guidance for future program repair. Evolutionary search (§[2.3](https://arxiv.org/html/2607.00272#S2.SS3 "2.3 Evolutionary Search ‣ 2 Method ‣ ASPIRE: Agentic /Skills Discovery for Robotics")) further broadens the space of executable program repairs before reusable patterns are admitted to the library.

![Image 10: Refer to caption](https://arxiv.org/html/2607.00272v1/x10.png)

(a)Pos perturbation.

![Image 11: Refer to caption](https://arxiv.org/html/2607.00272v1/x11.png)

(b)Task perturbation.

![Image 12: Refer to caption](https://arxiv.org/html/2607.00272v1/x12.png)

(c)Per-iteration progress.

Figure 6: Robot execution engine and evolutionary search ablations on LIBERO-Pro. Figures (a) and (b) show stacked bars for position and task perturbations: the base system without the robot execution engine or evolutionary search, the gain from adding the robot execution engine, and the additional gain from evolutionary search. On average, the robot execution engine provides the largest gain, raising macro-average success from 14% to 62%; evolutionary search further improves remaining hard tasks. Figure (c) shows average progress over low-performing tasks across evolutionary-search iterations. Per-task results are in Appendix [D](https://arxiv.org/html/2607.00272#A4 "Appendix D Ablation Details ‣ ASPIRE: Agentic /Skills Discovery for Robotics").

## 5 Limitations

Aspire has several important limitations. First, while selected simulation-discovered skills support autonomous real-robot debugging on a different embodiment, our system is not yet a fully autonomous real-world lifelong learner. Unlike simulation, where success checking and scene resets are cheap and programmatic, real-world deployment still requires robust success detection, safe reset, safety monitoring, and calibration maintenance. Future work should close this evaluation-and-reset loop to scale sim-to-real skill transfer across broader real-world task suites. Second, our method relies on a frozen frontier LLM (Claude Opus 4.6 with a 1M-token context window) to interpret multimodal traces, write program repairs, and propose evolutionary-search candidates; we have not verified that smaller or weaker LLMs can sustain the same debugging loop. Third, Aspire writes programs using a predefined API of perception, planning, and control primitives. This API makes debugging tractable and safe, but also bounds the behaviors the agent can express: if a task requires sensing, control, or interaction capabilities outside the exposed primitives, the agent must either approximate them inefficiently or rely on humans to extend the API. Future systems should study how agents can safely propose, validate, and incorporate new robot primitives. Fourth, the skill library currently prioritizes validated reusable repairs but does not fully solve long-term memory management. As the library grows, some entries may become stale, overly specific, redundant, or misleading for a new task, which can explain non-monotonic trends in zero-shot transfer. More robust retrieval, pruning, ranking, and re-validation mechanisms are needed to keep the library useful at scale. Finally, the debug and evolutionary-search loop is compute-intensive, consuming many LLM calls and simulator or robot rollouts per task; scaling to very large task suites will require either cheaper LLM inference, more sample-efficient search, or stronger mechanisms for reusing prior repairs.

## 6 Conclusion

We present Aspire, a continual learning robotic system that autonomously writes and refines robot control programs while compounding experience into a reusable skill library. Aspire operates in an open-ended learning loop with three components: a closed-loop robot execution engine that exposes fine-grained multimodal traces, a continually expanding skill library that distills validated fixes into transferable knowledge, and an evolutionary search procedure that explores diverse task sequences and control programs. Across diverse benchmarks, Aspire substantially outperforms existing VLA and coding-agent baselines, demonstrates strong zero-shot transfer to unseen long-horizon tasks, and provides initial evidence that the skills discovered in sim can transfer across embodiment to significantly reduce real-robot programming token cost despite different robot embodiments and APIs.

## Acknowledgments

We thank Nadun Ranawaka, Jimmy Wu, Matin Furutan, Haotian Lin, Abhi Maddukuri, Yulu Gan, Matin Nikoui, and Yuqi Xie for their help with open-source support, real robot infrastructure, advice and guidance on the paper, website release, and experimental equipment. We are also grateful to the members of NVIDIA GEAR, UMich SymbioticLab, UsesysLab, UC Berkeley AUTOLab, CMU LeCAR Lab for their kind support.

## References

## Appendix A Skill Library Details

This appendix expands the representative entries shown in Figure [3](https://arxiv.org/html/2607.00272#S2.F3 "Figure 3 ‣ 2 Method ‣ ASPIRE: Agentic /Skills Discovery for Robotics") into the skill-library taxonomy. Each entry will list: (i) the _problem_ extracted from the triggering failure trace; (ii) the _when-to-apply_ guard encoding the situational retrieval condition; (iii) the validated _repair snippet_; and (iv) the origin task(s) that produced the entry.

![Image 13: Refer to caption](https://arxiv.org/html/2607.00272v1/x13.png)

Figure 7: Debugging skills. Representative skill-library entries that encode reusable debugging strategies, including failure signatures, when-to-apply guards, and validated repair sketches.

![Image 14: Refer to caption](https://arxiv.org/html/2607.00272v1/x14.png)

Figure 8: Localization skills. Representative entries for grounding ambiguous language and object references into robust perception and localization routines.

![Image 15: Refer to caption](https://arxiv.org/html/2607.00272v1/x15.png)

Figure 9: Navigation skills. Representative entries for recovering from motion-planning failures and selecting collision-aware approach poses.

![Image 16: Refer to caption](https://arxiv.org/html/2607.00272v1/x16.png)

Figure 10: Strategic grasping skills. Representative entries for choosing task-appropriate grasp points and adapting grasp strategy to object geometry and scene context.

![Image 17: Refer to caption](https://arxiv.org/html/2607.00272v1/x17.png)

Figure 11: Motion-primitive skills. Representative entries for reusable low-level motion patterns, contact-rich alignment, and execution-time recovery.

![Image 18: Refer to caption](https://arxiv.org/html/2607.00272v1/x18.png)

Figure 12: Scene-reasoning skills. Representative entries for reasoning over spatial relations, support surfaces, occlusions, and scene-level task constraints.

## Appendix B Main Benchmark Result Tables

### B.1 Macro-Averaged Main Results

This section reports the exact numerical values visualized in Figure [4](https://arxiv.org/html/2607.00272#S3.F4 "Figure 4 ‣ 3.1 Experimental Setup ‣ 3 Experiments ‣ ASPIRE: Agentic /Skills Discovery for Robotics") of the main paper.

Table 2: LIBERO-Pro [zhou2025liberopro] performance of OpenVLA [kim2024openvlaopensourcevisionlanguageactionmodel], \pi_{0}[black2024pi0visionlanguageactionflowmodel], \pi_{0.5}[intelligence2025pi05], CaP-Agent0 [capx2026], and Aspire on the libero-object, libero-goal, and libero-spatial benchmarks under initial position perturbations (Pos) and instruction perturbations (Task), macro-averaged across 10 tasks per suite. Overall columns are macro-averaged across the three suites. All Aspire numbers are on the 50-seed held-out evaluation (seeds 1–50). See the per-task breakdown in Appendix [D.1](https://arxiv.org/html/2607.00272#A4.SS1 "D.1 LIBERO-Pro Position Perturbation (Swap) Per-Task Breakdown ‣ Appendix D Ablation Details ‣ ASPIRE: Agentic /Skills Discovery for Robotics").

Table 3: Robosuite [zhu2020robosuite] performance of CaP-Agent0 [capx2026] and Aspire on 7 manipulation tasks (100-trial held-out evaluation). Values are task success rates in [0,1]. Average is the unweighted mean across all 7 tasks.

Table 4: BEHAVIOR-1K [behavior1k2023] performance on two representative household tasks: Soda Can pick-up and Radio pick-up. Navigation and Task Success rates in [0,1] are reported separately; all numbers are on the 25-seed held-out evaluation (seeds 1–25); Aspire runs interactive block-by-block generation with the skill library accumulated on seeds 26–35.

## Appendix C LIBERO-Pro Long Zero-Shot Transfer

### C.1 Macro-Averaged Main Results

Table 5: Macro-averaged zero-shot transfer of the Aspire skill library (accumulated on LIBERO-90) to LIBERO-Pro Long [zhou2025liberopro]. N denotes the number of LIBERO-90 tasks whose repair skills seed the library; no additional debugging is performed at test time. Columns report macro-averaged success rate on Pos (position perturbation) and Task (instruction perturbation) axes, 10 tasks per axis.

### C.2 Per-Task Breakdown

Table 6: Zero-shot transfer of the Aspire skill library (accumulated on LIBERO-90) to LIBERO-Pro Long, per task and snapshot size N (seeds 1–50, n{=}50). Pos: positional-perturbation variant. Task: semantic-perturbation variant.

## Appendix D Ablation Details

### D.1 LIBERO-Pro Position Perturbation (Swap) Per-Task Breakdown

Table 7: LIBERO-Pro per-task ablation, Position perturbation (seeds 1–50, n{=}50). Aspire w/o Robot execution engine and Evolutionary search: zero-shot Claude Opus 4.6 with 15 example programs. Aspire w/o Evolutionary search: Robot Execution Engine repaired program (Execution engine + skill library). Evo. search: raw evolutionary-search result (best validated candidate), or “–” if Evolutionary Search was not run for that task. Aspire: per-task winner selected on a held-out validation set (seeds 66–80) between the repaired program and Evolutionary Search.

### D.2 LIBERO-Pro Task Perturbation (Goal) Per-Task Breakdown

Table 8: LIBERO-Pro per-task ablation, Task perturbation (seeds 1–50, n{=}50). Aspire w/o Robot execution engine and Evolutionary search: zero-shot Claude Opus 4.6 with 15 example programs. Aspire w/o Evolutionary search: Robot Execution Engine repaired program (Execution engine + skill library). Evo. search: raw evolutionary-search result (best validated candidate), or “–” if Evolutionary Search was not run for that task. Aspire: per-task winner selected on a held-out validation set (seeds 66–80) between the repaired program and Evolutionary Search.

### D.3 Evolutionary Search Progress Per-Task Breakdown

Table 9: Evolutionary search progress on selected LIBERO-Pro tasks. Each column shows the held-out success rate (seeds 1–50) of the best candidate at that iteration. Blank cells indicate search terminated early.

## Appendix E Agent Pipeline Skills

### E.1 Skill Library Prompts

Actor agents report candidate repairs using a structured findings schema: the observed failure mode, the validated repair, transferable patterns, task-specific quirks, and the validation success rate on debug configurations. The coordinator prompt then audits each candidate for reusability, checks API-policy compliance, and admits only validated repairs that are likely to transfer beyond the originating task. Parallel actor agents are instructed to write task-level repairs and findings, while the coordinator serializes skill admission to avoid conflicting library writes. See subagent skills in [E.3](https://arxiv.org/html/2607.00272#A5.SS3 "E.3 ASPIRE Fix Loop Skills ‣ E.2 Agent System Prompt and Persistent Memory ‣ Appendix E Agent Pipeline Skills ‣ ASPIRE: Agentic /Skills Discovery for Robotics") and [E.4](https://arxiv.org/html/2607.00272#A5.SS4 "E.4 ASPIRE Evolutionary Search Skills ‣ E.3 ASPIRE Fix Loop Skills ‣ E.2 Agent System Prompt and Persistent Memory ‣ Appendix E Agent Pipeline Skills ‣ ASPIRE: Agentic /Skills Discovery for Robotics") for exact formats. See code release for all agent skill files.

### E.2 Agent System Prompt and Persistent Memory

```
CLAUDE.md

 

MEMORY.md

 

api-reference.md

E.3 ASPIRE Fix Loop Skills

 

coordinator.md

 

subagent.md

E.4 ASPIRE Evolutionary Search Skills

Evolutionary search (Section 2.3) maintains a persistent task-analysis document AiA_{i} that is carried across rounds. AiA_{i} consists of (i) a per-task scene description populated once from an initial snapshot (object shapes, goal geometry, obstacles, blocked approach directions), (ii) running hypotheses and the candidate metadata that tests them, and (iii) a ledger of eliminated directions (through testing) and blocked, untested directions (failed due to a workspace constraint). UpdateAnalysis rewrites AiA_{i} each round from the new traces, keyframes, and library retrievals. This document is what lets a later round avoid re-testing an eliminated branch while still retrying blocked branches once a new technique (e.g. a wrist rotation) becomes available.
 

coordinator.md

 

subagent.md

E.5 Initial Skill Library Templates

 

grasp initial SKILL.md

 

localize initial SKILL.md

 

manipulation initial SKILL.md

 

transport initial SKILL.md
```

