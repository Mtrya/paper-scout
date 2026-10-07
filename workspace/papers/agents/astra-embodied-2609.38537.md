# 2609.38537 (from arXiv HTML; MinerU OOM fallback)

##### Report GitHub Issue

×

Title:

Content selection saved. Describe the issue below:

Description:

Submit without GitHub Submit in GitHub

![](/static/base/1.0.1/images/icons/smileybones-small.svg) arXiv is now an independent nonprofit! [Learn more](https://info.arxiv.org/about) ×

[![arXiv logo](/static/base/1.0.1/images/arxiv-logo-primary-light.svg) Back to arXiv ](/)

[Why HTML?](https://info.arxiv.org/about/accessible_HTML.html) Report Issue [ Back to Abstract ](/abs/2609.38537v1 "Back to abstract page") [ Download PDF](/pdf/2609.38537v1 "Download PDF") [ ](javascript:toggleNavTOC\(\); "Toggle navigation") [ ](javascript:toggleReadingMode\(\); "Disable reading mode, show header and footer")

  1. Abstract
  2. 1 Introduction
  3. 2 Evaluation Framework
     1. 2.1 How Astra Produces Robot Actions
     2. 2.2 What the Comparisons Establish
     3. 2.3 Measuring Progress and Timing
  4. 3 Gripper Manipulation
     1. 3.1 Control and evaluation setup
     2. 3.2 RoboDojo: paired bimanual evaluation
        1. Setting.
        2. Results.
     3. 3.3 RoboLab: zero-shot evaluation
        1. Setting.
        2. Results.
     4. 3.4 Discussion: why the benchmark rankings differ
  5. 4 Dexterous Manipulation
     1. 4.1 Policy Assistance Across Ten Tasks
     2. 4.2 Feedback-Guided Bimanual Manipulation in DexJoCo
     3. 4.3 Stable Grasping, Limited Rotation
     4. 4.4 Contact Changes and Object Support
  6. 5 Mobile Manipulation
     1. 5.1 Control Conditions and Shared Starts
     2. 5.2 Uneven Gains from Policy Assistance
     3. 5.3 Recovery and Lost Policy Successes
  7. 6 Navigation
     1. 6.1 RGB Navigation on Four Fixed Subsets
     2. 6.2 Success and Search Efficiency
     3. 6.3 Local Comparisons and Failure Modes
  8. 7 Locomotion
     1. 7.1 Dense Motion on a Matched Course
     2. 7.2 Persistent Locomotion Failures
     3. 7.3 Compact-Reference Diagnostics without Astra
  9. 8 Humanoid Loco-Manipulation
     1. 8.1 HumanoidBench Evaluation Setup
     2. 8.2 Command Selection with Pretrained Whole-Body Controllers
     3. 8.3 Contact Failures Between Task Stages
     4. 8.4 Refining Task Guidance through Experience in SIMPLE
     5. 8.5 Why Handover Remains Fragile
     6. 8.6 Scene and Motion from Egocentric Video
  10. 9 Discussion
     1. 9.1 Capability Depends on the Complete Control Path
     2. 9.2 Intervention and Verified Progress
     3. 9.3 Retaining Success across Trials
     4. 9.4 Resource Costs and Capability Limits
  11. 10 Conclusion
  12. References
  13. A Additional Interface Details
     1. A.1 Gripper Interfaces
     2. A.2 Humanoid Interfaces
     3. A.3 Experience and Evaluation Scope
  14. B Detailed Manipulation Results
     1. B.1 RoboDojo
     2. B.2 RoboLab
  15. C Focused Experiments, Resource Use, and Reproducibility
     1. C.1 Focused Push and Walk Records
     2. C.2 Independent Reach Records
     3. C.3 RoboDojo Execution and Token Accounting
     4. C.4 Source Snapshots and Verification
  16. D Manipulation Tasks and Behavioral Cases
     1. D.1 Task Coverage
     2. D.2 Behavioral Case Index
  17. E Humanoid Loco-Manipulation: Supporting Results
     1. E.1 Additional Physical Diagnostics
        1. Walk.
        2. Run and Crawl.
        3. Stair.
        4. Push.
     2. E.2 SIMPLE Development and Retention Checks
        1. Control and observations.
        2. Experience representation.
        3. Single-scene development.
        4. Separate Handover studies.
        5. L2 evaluation protocol.
  18. F Harness Development and Repeatability
     1. F.1 Contact-Conditioned Guidance
     2. F.2 A Saved Diagnosis and a Failed Repeat
     3. F.3 Finger Control Does Not Solve Door
     4. F.4 Provenance and Evidence Availability
  19. G Dexterous Manipulation: Data, Interfaces, and Evaluation
     1. G.1 Task Suite and Demonstration Data
        1. Initial configurations.
     2. G.2 Multitask Policy Finetuning
     3. G.3 Observations, Numerical Actions, and Execution
        1. Policy observations.
        2. Policy action representation.
        3. Execution and model budgets.
     4. G.4 Evaluation Protocol and Scoring
        1. Fixed evaluation cases.
        2. Continuous execution and verification.
        3. Task subgoals.
        4. Partial-credit scores.
        5. Safety and termination.
        6. Aggregation.
  20. H In-Hand Manipulation: Controllers and Complete Results
     1. H.1 Benchmark Definition
        1. Case selection and pairing.
     2. H.2 Task-Specific RL Controllers and Training
        1. Sharpa continuous-rotation policies.
        2. Allegro translation policy.
        3. Allegro translation-and-rotation policy.
     3. H.3 Astra Inference Protocol
        1. Model and prompt.
        2. Observation and temporal context.
        3. Action execution.
     4. H.4 Pairing and Evaluation Protocol
        1. RL inference and paired starts.
        2. Execution endpoints.
     5. H.5 Evaluation Metrics
        1. In-hand rotation.
        2. In-hand translation and reorientation.
     6. H.6 Complete Per-Case Results
  21. I Navigation and Mobile Manipulation: Protocols and Cases
     1. I.1 Navigation Coverage and Termination
     2. I.2 Released Navigation Policies: Interfaces and Complete Results
        1. Fixed tasks and checkpoint selection.
        2. Observations and history.
        3. Action conversion and stopping.
        4. Evaluation configuration.
        5. Published reference sources and coverage.
     3. I.3 RoboCasa Task Results and Paired Outcomes
        1. Additional policy evaluation.
     4. I.4 RoboCasa Control Configurations, Information Access, and Resources
        1. Runtime and action semantics.
        2. Harness configurations.
        3. Information-boundary audit.
        4. Reproducibility.



[ License: CC BY 4.0 ](https://info.arxiv.org/help/license/index.html#licenses-available)

arXiv:2609.38537v1 [cs.RO] 29 Sep 2026

\code

[GPT-as-Policy](https://github.com/anonymous-report-421/GPT-as-Policy) \page[Project page](https://galaxygeneralrobotics.github.io/astra-policy/)

# Systematically Exploring the Capabilities of   
GPT-6 Astra as Embodied Policies

Galbot Team 

September 2026

###### Abstract

GPT-6 Astra exhibits a remarkable ability to generate numerical robot actions, extending its role beyond high-level planning. To assess Astra’s capabilities as general-purpose embodied policies, we conduct comprehensive evaluations across six domains, examining direct control, cooperation with learned policies, and feedback-driven adaptation. In gripper manipulation, Astra can correct task targets and prepare contact conditions for subsequent policy execution; hybrid control with π0.5\pi_{0.5} achieves 48% success on the evaluated RoboDojo subset. In dexterous manipulation, hybrid control achieves 50% success in ten experience-guided DexJoCo trials, while direct in-hand control struggles to coordinate finger contacts. In mobile manipulation, hybrid control reaches 38.7% success on the evaluated RoboCasa365. In navigation, Astra leads our local comparisons, reaching 92% success on RxR instruction following and 82% on HM3D object search, although search incurs substantial detours. In locomotion, dense motion-reference generation remains unreliable: none of five sequential attempts on a single obstacle course reaches the goal, despite improvements in stability and forward progress. In humanoid loco-manipulation, Astra exceeds baseline methods on 13 of 30 HumanoidBench tasks with pretrained whole-body controllers. These findings reveal a gap between useful task decisions and reliable physical control. Inference latency further constrains practical control: across 50 RoboDojo instances per condition, policy-assisted and direct control consume 624.8 million and 1.132 billion tokens. A 30-second locomotion run requires 250 model calls averaging 39.86 seconds each, with physics paused during inference.

## Introduction

Community experiments with Astra reveal possibilities ranging from 3D scene understanding and real-to-sim workflows [14] to robot manipulation [64]. These demonstrations motivate a systematic assessment of Astra as an embodied policy: _Which tasks can it perform, where does it remain unreliable, and what resources do its decisions require?_ The assessment must account for both successes and failures.

We evaluate GPT-6 Astra [32] across six domains: gripper manipulation, dexterous manipulation, mobile manipulation, navigation, locomotion, and humanoid loco-manipulation. Our primary aim is to characterize performance and capability boundaries through task outcomes, comparisons with existing methods, failure analysis, and inference demands. We then analyze how interfaces, whole-body controllers, and feedback shape these outcomes.

Existing work spans spatial understanding [35, 16] and reasoning for robot actions [36]. Vision–language–action models such as π0.5\pi_{0.5} map observations and instructions to actions learned from robot experience [34]. SayCan selects grounded skills [1], Code as Policies composes perception and control through programs [21], and Prompt a Robot to Walk and Natural Language as Policies explore numerical feedback control [50, 28]. Recent Astra evaluations report uneven manipulation performance across tasks [64]. Our evaluation covers both Astra-generated robot commands, termed _Direct_ , and cooperation with learned action policies or whole-body controllers, termed _Hybrid_. Figure 1 summarizes these control paths.

CONTROL MODECOMMAND INTERFACELOW-LEVEL CONTROLSTUDIES Direct or   
Hybrid End-effector or joint targets;   
gripper and base commands Inverse kinematics, joint or   
operational-space control Gripper, dexterous,   
mobile manipulation Direct Forward steps, turns,   
and stopping decisions Discrete navigation   
primitives Navigation Direct or   
Hybrid Joint PD targets or   
dense motion references PD controller or   
frozen whole-body controller Locomotion Hybrid Velocity commands or   
sparse body targets Frozen whole-body controller Humanoid   
loco-manipulation Feedback loop: observe →\rightarrow Astra generates or reviews →\rightarrow robot moves →\rightarrow observe

Figure 1: Control modes, interfaces, and low-level control. Direct uses Astra-generated commands with analytic control. Hybrid combines Astra with a learned task policy or whole-body controller: Astra reviews policy proposals or supplies commands or motion references to frozen controllers. Observations of the resulting motion inform subsequent decisions.

The domains expose complementary demands. Gripper and dexterous manipulation test grasping, local correction, and changing contact. Mobile manipulation adds base–arm coordination and multistage goals. Navigation tests spatial decisions and stopping, while locomotion requires coordinated body motion. Humanoid loco-manipulation combines task progression with pretrained whole-body controllers. We retain each study’s metrics and protocols to make the scope of its conclusions explicit.

The results reveal substantial but uneven capabilities. Astra leads our local navigation comparisons on success and path efficiency, and Hybrid systems achieve higher aggregate performance in several manipulation studies. Yet direct in-hand control trails task-specific RL, dense-reference locomotion remains unreliable despite repeated attempts, and assistance can fail on cases completed by a standalone policy. Token use and inference latency qualify these outcomes and must be measured separately from motion duration.

To interpret these boundaries, we examine the division of control. Once a reasoning model can generate numerical actions, task success alone does not reveal which responsibilities it assumes, which capabilities interfaces and whole-body controllers supply, or how feedback connects them. In successful manipulation cases, Astra can prepare conditions for a policy’s next action without generating the full trajectory. Conversely, preserving a grasp can impede required contact changes, and delegating motion can retain substantial proposal-review costs. This analysis helps explain the measured strengths and limitations while distinguishing Astra’s decisions from the capabilities of the complete system.

## Evaluation Framework

### How Astra Produces Robot Actions

An _action_ is a command submitted to the robot’s control interface, such as a target pose, a joint increment, or a navigation move. Astra receives the observations allowed by the study, selects an action, and uses the resulting feedback to choose the next one. The interface checks command bounds, and the environment determines task success. Tools support calculations, image inspection, and notes.

_Direct_ uses Astra-generated robot commands without a learned action policy or whole-body controller. Inverse kinematics and proportional–derivative control can implement these commands. _Hybrid_ adds a learned task policy or whole-body controller.

Hybrid takes two forms in this report. In manipulation, Astra accepts, modifies, or replaces a task policy’s action proposals; RoboCasa also permits subgoal rewriting. In humanoid control, Astra supplies dense motion references, velocity commands, or sparse body targets to frozen whole-body controllers. Locomotion includes both direct joint PD targets and references executed by ScaleTrack. Figure 1 shows these control paths. Each study specifies what Astra decides, how commands become motion, and what feedback informs the next action.

Within a trial, Astra uses feedback to generate or review the next action. Some studies retain notes or refine task guidance between trials with model weights fixed. Separate development studies involve researcher changes to prompts, interfaces, or whole-body controller settings. These procedures are described with their results and detailed in Appendix A.

### What the Comparisons Establish

Table 1 identifies the tasks, sample sizes, and comparison conditions. Where methods start from the same physical state, their outcomes can be compared on the same task instance. Each study specifies its observations, controllers, and inference budgets.

Table 1: Evaluation scope and comparison conditions. Direct uses Astra-generated commands with analytic control; Hybrid adds a learned task policy or whole-body controller. Sample sizes are reported separately for each study. Domain |  Tasks and sample sizes |  What is compared  
---|---|---  
Gripper manipulation |  RoboDojo and RoboLab each cover 10 tasks with 5 results per task and method. |  RoboDojo compares paired Direct and Hybrid runs with published policy scores. RoboLab compares Direct and Hybrid with three zero-shot policies. Control settings and evaluation procedures appear in Appendix B.  
Dexterous manipulation |  10 manipulation tasks with 5 cases each; 4 in-hand tasks with 5 initial states each. |  Manipulation compares Direct, Hybrid, and a standalone policy on the same cases. In-hand control compares Astra and task-specific RL from identical physical states, with different observations and action rates.  
Mobile manipulation |  15 RoboCasa tasks with 5 episodes per task and method. |  Direct, Hybrid, and a locally evaluated standalone policy share initial states and 20-step action segments. The Astra conditions differ in policy access, context management, and feedback guards.  
Navigation |  4 dataset subsets with 50 episodes per system in each. |  Astra and released navigation policies use the same episode lists, scoring rules, and action budgets. Camera views and the conversion of outputs to navigation actions differ between systems.  
Locomotion |  5 sequential Astra attempts on one course. Two reference interfaces, each tested on 6 courses in 2 physics backends: 12 rollouts per interface. |  Astra and PASSAGE [26] share the initial state, goal, and frozen tracker. Separate analytic tests compare five-point and whole-body reference interfaces without Astra.  
Humanoid loco-manipulation |  HumanoidBench covers 30 tasks. SIMPLE covers 6 L2 tasks with 10 scenes each. |  HumanoidBench compares Astra returns with published DreamerV3, TD-MPC2, and SAC results. SIMPLE uses the official codebase and protocol for comparison.  
  
We report fixed-configuration evaluations and development trials separately. The former measure performance under specified conditions; the latter track changes in behavior as prompts, interfaces, or whole-body controller settings are refined. Transfer trials assess the resulting behavior on additional states.

Performance depends on the complete system, including the available commands, written guidance, and whole-body controller, even when Astra’s weights remain fixed. For example, the grasp interface determines which actions are possible, while gait calibration changes how commands produce motion.

### Measuring Progress and Timing

We distinguish completing a task from making partial progress or merely avoiding failure. The metrics must capture the intended change in physical state. We interpret rotation error together with angular speed, navigation success with path efficiency, and humanoid reward with observed progress and termination.

Controller frequency and reasoning speed are separate quantities. A fast controller can apply actions while Astra takes much longer to choose the next command. When physics pauses during inference, the robot-motion clock excludes decision latency. Resource summaries distinguish robot control steps, action segments, model requests, and cached input, uncached input, and output tokens. Astra can review many policy proposals while changing few actions. Section 9.4 summarizes inference demands.

## Gripper Manipulation

We compare two closed-loop control architectures for manipulation. _Direct_ uses GPT 6 Astra alone: given images, proprioception, the task instruction, and execution history, it generates bimanual end-effector (EEF) targets and gripper commands, executes one to five control steps, and then observes the new state. _Hybrid_ uses a learned policy, π0.5\pi_{0.5}, to propose a 50-step joint-space action sequence. GPT 6 Astra reviews the proposal and either executes its first one to fifteen steps or replaces it with a one-to-five-step EEF correction; the two branches are mutually exclusive. Both architectures use the same task descriptions, success criteria, GPT 6 Astra reasoning setting, and EEF execution interface.

Figure 2: Gripper manipulation results. (a,b) RoboDojo Score and final success on ten tasks. Public references are reweighted to the same task and scene mixture, whereas Direct and Hybrid use paired starts. Direct has 48 valid Score entries; success is computed over all 50 task instances. (c) RoboLab success rates. (d) Of the Hybrid control steps, 85.6% follow π0.5\pi_{0.5} and 14.4% are generated or corrected by GPT 6 Astra.

### Control and evaluation setup

The two architectures test complementary uses of GPT 6 Astra. Direct asks the model to construct the complete EEF action from the current state. Hybrid gives the model a candidate trajectory from π0.5\pi_{0.5} and asks it to preserve the trajectory or make a local correction. In Hybrid, GPT 6 Astra can therefore change the target, prepare the contact geometry, or continue after the simulator reports that the task is not yet complete, while π0.5\pi_{0.5} supplies most of the low-level motion. These architectures are evaluated separately on RoboDojo and RoboLab. Qualitative trajectories and the complete set of video cases are available in the accompanying [web report](https://anonymous-report-421.github.io/public-website/?view=1).

### RoboDojo: paired bimanual evaluation

#### Setting.

RoboDojo [8] is a bimanual manipulation benchmark covering semantic classification, sequence imitation, packing, construction, and deformable-object manipulation. We select ten tasks by stratifying the published π0.5\pi_{0.5} success rates: six tasks from the lowest interval, two from the next, and one from each of the two higher intervals. This selection emphasizes tasks with room for improvement while retaining diverse interaction requirements. Each task is evaluated five times for each architecture. Direct and Hybrid use the same task instances, scene configurations, and random seeds. Number arrangement, packing, and clothes folding use two standard and three randomized scenes; the other tasks use five standard layouts. We report RoboDojo’s native final success rate and partial-completion Score. Hybrid uses the task-finetuned π0.5\pi_{0.5} weights released for RoboDojo.

#### Results.

Hybrid succeeds on 24 of 50 instances (48%), compared with 13 of 50 (26%) for Direct. Its mean Score is 62.60, compared with 37.81 for Direct. The improvement is concentrated in tasks that require long-horizon coordination or contact-sensitive interaction: Hybrid scores 53 versus 0 on sequence imitation, 64 versus 12 on tower construction, 100 versus 40 on clothes folding, and 100 versus 36 on bottle disposal. Direct is higher on language-based classification (60 versus 38) and object classification (100 versus 71), showing that the Hybrid advantage is not uniform across tasks. In the 50 Hybrid trajectories, 36,576 of 42,750 executed control steps (85.6%) follow π0.5\pi_{0.5}, while 6,174 (14.4%) are generated or corrected by GPT 6 Astra. The model therefore intervenes selectively rather than replacing the learned policy.

Reweighting the published RoboDojo results to the evaluated ten-task and scene mixture gives π0.5\pi_{0.5} a success rate of 15.67% and a Score of 24.43. Table 2 lists the published references alongside our paired Direct and Hybrid runs.

Table 2: RoboDojo task-level Score. Public references are reweighted to the evaluation task and scene mixture. Direct and Hybrid are our paired evaluations; the Direct Score uses 48 valid entries, while success uses all 50 task instances. | Reweighted public references | Our evaluation  
---|---|---  
Task | DM0.5 | Galaxea | Xiaomi R1 | OpenWAM | π0.5\pi_{0.5} | Hybrid | Direct  
Organize the table | 44.00 | 46.33 | 57.67 | 62.50 | 23.33 | 60.00 | 30.00  
Classify by language | 0.47 | 1.07 | 2.00 | 1.33 | 0.60 | 38.00 | 60.00  
Imitate a sorting sequence | 1.80 | 1.67 | 2.50 | 2.90 | 1.60 | 53.00 | 0.00  
Arrange the largest number | 7.85 | 4.11 | 8.56 | 4.36 | 2.29 | 50.00 | 57.00  
Pack objects into a box | 14.72 | 17.12 | 18.69 | 20.83 | 18.36 | 50.00 | 50.00  
Classify objects | 26.83 | 10.33 | 17.20 | 5.53 | 24.67 | 71.00 | 100.00  
Build a tower | 55.20 | 82.93 | 52.60 | 52.53 | 37.73 | 64.00 | 12.00  
Make a Kong in Mahjong | 56.67 | 90.00 | 41.33 | 32.00 | 26.67 | 40.00 | 0.00  
Fold clothes | 28.96 | 32.75 | 41.49 | 51.31 | 29.12 | 100.00 | 40.00  
Put bottles in a bin | 81.70 | 96.30 | 97.70 | 94.03 | 79.93 | 100.00 | 36.00  
Aggregate | 31.82 | 38.26 | 33.97 | 32.73 | 24.43 | 62.60 | 37.81  
  
### RoboLab: zero-shot evaluation

#### Setting.

RoboLab [55] evaluates a single-arm Franka on ten tasks involving semantic pick-and-place, ordered block stacking, and mug reorientation. We report five trials per task and compare Direct, Hybrid, π0.5\pi_{0.5}, Cosmos3-Nano-Policy [30], and DreamZero [56]. This benchmark is evaluated with final task success only; RoboDojo’s partial-completion Score is not used. RoboLab provides no training data for the test tasks. The policy baselines therefore use DROID-trained weights for zero-shot transfer, unlike RoboDojo, where Hybrid uses task-finetuned π0.5\pi_{0.5}.

#### Results.

Direct succeeds on 49/50 trials (98%), with five successes on nine of the ten tasks. Hybrid succeeds on 46/50 (92%), while π0.5\pi_{0.5}, Cosmos3-Nano-Policy, and DreamZero succeed on 18/50 (36%), 18/50 (36%), and 17/50 (34%), respectively. Hybrid therefore improves over π0.5\pi_{0.5} by 56 percentage points, but is three successful trajectories below Direct on this task subset.

### Discussion: why the benchmark rankings differ

RoboDojo and RoboLab favor different capabilities. RoboDojo contains bimanual, long-horizon, deformable, and contact-sensitive tasks. On these tasks, the learned policy supplies interaction patterns and coordinated motion, while GPT 6 Astra can correct the target, local geometry, or task-progress interpretation. This division of labor is consistent with Hybrid outperforming Direct on RoboDojo.

The RoboLab tasks are predominantly single-arm semantic pick-and-place problems, with additional ordered stacking and mug reorientation. Direct can already plan these operations from the current observation and reaches 98% success. In this setting, a candidate action from a zero-shot policy is not necessarily a useful prior: π0.5\pi_{0.5} alone succeeds on only 36% of trials, so Hybrid may spend decisions correcting an action that is poorly adapted to the task. This suggests that the value of policy assistance depends on how well the learned motion prior fits the task.

## Dexterous Manipulation

Dexterous manipulation requires both task-level planning, such as selecting a suitable grasp and deciding how to use it, and precise coordination of finger contacts during execution. To assess these capabilities, we evaluate Astra on end-to-end manipulation tasks, with optional policy assistance, and on in-hand control from an established grasp. The former tests acquiring and using a grasp; the latter tests changing finger contacts while maintaining object support.

### Policy Assistance Across Ten Tasks

The first benchmark covers grasping, retrieval, placement, insertion, stacking, and rearrangement with simulated Sharpa hands [45]. One multitask π0.5\pi_{0.5} policy is finetuned on 100 demonstrations per task (1,000 in total). For evaluation, Direct, standalone π0.5\pi_{0.5}, and Hybrid are each tested on the same five unseen cases per task, with simulation horizons of 20–60 s. Direct outputs wrist and finger targets; Hybrid reviews or corrects policy actions, executing prefixes of 1–16 steps or corrections of 1–5 steps at approximately 30 Hz. When the review or token budget is exhausted, the system falls back to π0.5\pi_{0.5}-only control.

We evaluate performance using task completion scores on a 0–100 scale. The scores give partial credit for progress rather than measuring binary success, and we average them equally across tasks. Full credit requires task-specific verification of a maintained hold or a stable release. Appendix G provides details on observations, training, actions, and scoring.

As shown in Table 3, Hybrid achieves a mean Score of 61.6, compared with 44.2 for π0.5\pi_{0.5} and 16.6 for Direct. Hybrid achieves a higher mean score than Direct on all ten tasks. Compared with π0.5\pi_{0.5}, it scores higher on eight tasks and ties on the remaining two. Gains over the policy are largest for mahjong storage at 40 points, upright egg placement at 32, and bottle/can sorting at 28. On bread insertion and nesting-doll ordering, Hybrid improves only slightly over π0.5\pi_{0.5}, with scores remaining at 28 and 24, respectively.

Table 3: Dexterous manipulation scores on a custom simulation benchmark. Scores measure subgoal completion on a 0–100 scale. The evaluation protocol and scoring details are provided in Appendix G.4. Task |  π0.5\pi_{0.5} |  Direct |  Hybrid  
---|---|---|---  
Pot lift and hold  |  40.0  |  44.0  |  62.0  
Headphones in box  |  42.0 |  22.0  |  42.0  
Toy retrieval  |  100.0 |  22.0  |  100.0  
Mug hanging  |  62.0  |  4.0  |  82.0  
Mahjong tile storage  |  48.0  |  18.0  |  88.0  
Bottles/cans sorting  |  72.0  |  14.0  |  100.0  
Upright egg placement  |  6.0  |  12.0  |  38.0  
Two-bowl stacking  |  46.0  |  20.0  |  52.0  
Bread-slot insertion  |  10.0  |  8.0  |  28.0  
Nesting-doll ordering  |  16.0  |  2.0  |  24.0  
Overall mean |  44.2  |  16.6  |  61.6  
  
Figure 3(a) categorizes Astra’s interventions by failure type. The most common address missed grasps or dropped objects and failed placements, followed by placement alignment and grip stabilization. In individual trajectories, Astra redirects wrists toward displaced objects, clears occlusions, refines placements, and resumes unfinished subgoals. These interventions account for only 11.98% of executed Hybrid steps (Figure 3(b)). Together with the task scores, this suggests that targeted changes to a small fraction of steps can yield substantial performance gains.

Even with these gains, grasp acquisition and completion checking remain the primary sources of failure. Direct can fail to establish a grasp when simultaneous finger closure pushes the object away or a weak two-finger lift leaves it poorly supported. Hybrid faces a similar difficulty when an object drops into an unfamiliar configuration: if the policy can no longer provide useful actions, Astra must generate a new grasp pose but may still fail to secure the object. These grasp failures suggest that Astra still relies heavily on a capable policy to generate viable action candidates for dexterous manipulation. In addition, Astra can misjudge task completion, as in an egg-placement case where it declares success while the egg remains on its side.

Figure 3: Astra corrections in dexterous manipulation. (a) Primary reasons for Astra correction decisions across 50 Hybrid cases. (b) Executed steps comprise 88.02% unmodified π0.5\pi_{0.5} actions and 11.98% Astra corrections.

### Feedback-Guided Bimanual Manipulation in DexJoCo

The Hanoi disk-stacking task in DexJoCo [48] tests two ordered transfers with dual Franka Panda arms and Allegro hands: the right hand moves the medium disk to the destination peg, then the left hand places the small disk above it. Astra reviews a fixed π0.5\pi_{0.5} policy trained for this task, accepting action prefixes or correcting wrist and finger targets using visual feedback. Each trial allows up to 1,500 control steps at 50 Hz, with at most 30 steps per action segment. Ten trials run in five successive pairs, starting from prior task experience. Written guidance is shared within each pair and refined between pairs, with model weights fixed. Image-history limits and compact references to older proposals are introduced during the run.

The system completes 5/10 trials under the native success criterion. Figure 4 places this result alongside published policy success rates on the same task. Of 11,320 executed steps, 73.6% use unmodified policy actions and 26.4% use Astra corrections. These results summarize the full sequence of experience-guided trials. The trajectories in Figure 5 show useful local interventions alongside persistent failures in grasp retention and final placement.

Figure 4: Success on bimanual disk stacking. Published baselines average three sets of 50 rand-obj episodes [48]; Astra + π0.5\pi_{0.5} succeeds in 5/10 experience-guided trials.

A successful placement requires more than reaching the correct peg. In one successful trial, Astra delays a proposed release because the small disk remains above its support. Short downward corrections bring the disk into contact with the stack, after which Astra returns control to π0.5\pi_{0.5} for release and withdrawal. The resulting hand clearance allows a separate check that the disk remains supported. This sequence illustrates Astra’s role in deciding when a physical precondition has been met, while the policy supplies the subsequent motion.

The same strategy is not consistently effective. In a failed trial, repeated seating corrections leave the small disk tilted high on the destination peg at the step limit. Other failures involve losing a grasp during transport or failing to lift the small disk from its source. Astra’s own corrections can also hinder progress: in another successful trial, two wrist adjustments move the held disk away from the destination, whereas returning control to the policy restores transport. Effective cooperation therefore requires both timely intervention and recognition that the policy may offer the better recovery. The trials and five reviews consume 63.60 million recorded tokens, including cached input; only 0.136 million are used for review. Most token use arises during online control, even though the policy supplies most physical actions.

![Refer to caption](2609.38537v1/hanoi-cases.png) Figure 5: Bimanual disk stacking: successful placement and failed seating. Cropped wrist-camera observations show (a) Astra corrections followed by policy release and (b) persistent placement failure despite further corrections. Labels give control steps; outcomes follow the native success criterion.

### Stable Grasping, Limited Rotation

Once a grasp is established, success requires moving the object while maintaining support. We compare direct Astra joint commands with four frozen, task-specific RL policies to examine this demand. Sharpa tasks require continuous cylinder or cuboid rotation about world-frame +Z+Z; Allegro tasks require cylinder translation, with or without an additional −40∘-40^{\circ} long-axis rotation. Rotation tasks draw on ConTrack [22] and mjlab_hand [38]; translation follows tactile in-hand manipulation [58].

Each task uses five paired saved hand/object states and targets. Astra receives images and state and commands 22 Sharpa or 16 Allegro joints for one to five 20-Hz steps. RL uses its training observations and acts every step. Thus physical starts are paired, but observations and decision rates differ. Four cuboid starts overlap the RL initialization bank. Training and per-case records appear in Appendix H.

An established grasp does not remove Astra’s difficulty with sustained object motion. RL tracks the rotation target within tolerance for 76.90% of cylinder steps and 63.50% of cuboid steps, versus 0.51% and 4.40% for Astra (Table 4). Astra reaches the horizon in three cylinder runs, drops the object once, and exhausts its decision budget once. The cuboid results isolate the distinction between retention and progress more clearly: both controllers retain the object through all five horizons, yet Astra rotates it too slowly. The gap therefore persists even when the grasp remains secure.

Figure 6: Continuous rotation over five paired initial states. Rows show cylinder results over 20 s and cuboid results over 10 s. Columns report orientation error, axial speed over the preceding second, cumulative at-goal rate, and cumulative reward. Curves and shading show means and one sample SD among active runs. Dashed lines mark target speeds and the 0.10.1-rad goal threshold. Periodic orientation error requires interpretation alongside axial speed.

The motion traces show how this gap develops. Astra’s cylinder speed stays near zero, while the cuboid turns gradually but falls behind the target (Figure 6). Representative cuboid sequences show little contact reconfiguration under Astra, whereas RL continues reorientation (Appendix H). This behavior suggests that preserving support is easier for Astra than reorganizing finger contacts to sustain motion. Periodic decreases in orientation error do not contradict this interpretation: the rotating target returns to an equivalent orientation after each revolution, so error must be read alongside axial speed.

Table 4: Paired in-hand rotation results. Full, Budget, and Drop denote runs reaching the horizon, exhausting Astra’s decision limit, or losing the object, respectively. At-goal is the fraction of steps with quaternion geodesic error below 0.1​rad0.1\,\mathrm{rad}. Error and Speed MAE are means over executed steps; terminated trajectories are not padded. Task | Controller | Steps | Full | Budget | Drop | At-goal ↑\uparrow | Error (rad) ↓\downarrow | Speed MAE ↓\downarrow  
---|---|---|---|---|---|---|---|---  
Cylinder, 20 s | Astra Direct | 1,749 | 3/5 | 1/5 | 1/5 | 0.51% | 1.626 | 0.960  
| RL | 2,000 | 5/5 | – | 0/5 | 76.90% | 0.173 | 0.276  
Cuboid, 10 s | Astra Direct | 1,000 | 5/5 | 0/5 | 0/5 | 4.40% | 0.742 | 0.143  
| RL | 1,000 | 5/5 | – | 0/5 | 63.50% | 0.096 | 0.063  
Table 5: Paired in-hand translation and reorientation results. Errors are measured at the stated terminal endpoint. Joint success requires both position error below 22.4​mm22.4\,\mathrm{mm} and long-axis rotation error below 10∘10^{\circ} at the same endpoint. Task | Controller | Endpoint | Position (mm) ↓\downarrow | Rotation (deg) ↓\downarrow | Success  
---|---|---|---|---|---  
Translation | Astra Direct | 15 s | 59.05 | – | 1/5  
| RL | 15 s | 17.29 | – | 4/5  
Translation + rotation | Astra Direct | 120 decisions | 47.35 | 32.72 | 0/5  
| RL | matched steps | 16.22 | 8.31 | 4/5  
| RL | 15 s | 16.08 | 3.49 | 5/5  
  
Translation reveals some useful direct control, but limited consistency across starts. Astra succeeds in 1/5 cases and RL in 4/5, with mean terminal errors of 59.05 and 17.29 mm (Table 5). Astra achieves lower position error in one paired case, showing that it can produce an effective displacement. Yet it retains the object in every translation run and still misses most targets. As in rotation, failure to reach the goal cannot be explained by object drops alone.

Figure 7: Per-case terminal errors for Allegro tasks. (a) Translation at 15 s. (b,c) Translation and rotation: Astra at its 120-decision endpoint, RL at matched control steps, and RL at 15 s. Shading and dashed lines mark the 22.422.4-mm position and 10∘10^{\circ} rotation success thresholds.

For combined translation and rotation, Astra succeeds in 0/5 cases, compared with 4/5 for RL at matched endpoints and 5/5 after 15 s. Success requires position error below 22.4 mm and rotation error below 10∘10^{\circ} at the same endpoint. Astra’s 120-decision budget ends after 6.00–7.65 s, but RL already performs better over those same durations (Figure 7). A longer action horizon alone therefore does not explain the gap. The results favor task-specific learned control for reliable object motion.

### Contact Changes and Object Support

Together, the two studies distinguish useful task correction from reliable contact coordination. Astra can redirect a learned policy when its actions no longer match the scene or instruction, but direct control remains unreliable both when acquiring a grasp and when moving within one. A stable grasp is only an intermediate condition: continued manipulation requires some fingers to release and reposition while others preserve support. The central limitation is sustaining this coordination as the object moves. Hybrid benefits from retaining learned motion generation while allowing Astra to guide recovery, yet recovery and completion still require verification.

## Mobile Manipulation

Mobile manipulation couples base placement, arm reachability, and object interaction. Using RoboCasa365 [29], we evaluate Astra Direct, Astra Hybrid with optional π0.5\pi_{0.5}, and standalone π0.5\pi_{0.5}.

Figure 8: Cooperation changes the RoboCasa capability profile. (a) Success on matched physical starts. Error bars show descriptive episode-level 95% Wilson intervals [52]. (b) Step shares from accepted proposals and Astra actions.

### Control Conditions and Shared Starts

The fixed subset contains five atomic seen, five composite seen, and five composite unseen tasks, each with five seeds: 75 episodes per condition. Selection uses a seeded ordering without model outcomes. Conditions share initial-state fingerprints, the PandaOmron robot, controller, task horizon, and native success check. Seen/unseen refers to the policy’s task-training split. All layouts use split=pretrain.

Astra receives three RGB views and end-effector, base, and gripper state, without object poses, contact truth, reference trajectories, or intermediate rewards. Direct supplies bounded arm, gripper, base, and torso commands through an operational-space controller. Segments normally execute 20 steps at 20 Hz; horizons range from 450 to 4,350 steps.

Hybrid adds a frozen multitask π0.5\pi_{0.5} checkpoint without separate finetuning for these tasks. Astra can accept a 20-step prefix, construct actions, or rewrite the policy instruction as a subgoal. The standalone baseline runs the same checkpoint locally on the same initial states, using the task instruction and the same 20-step cadence. Direct and Hybrid also differ in context management and feedback guards, including context compaction and required observations between action calls, as detailed in Appendix I.4.

![Refer to caption](2609.38537v1/robocasa-cases.png) Figure 9: RoboCasa outcomes from matched physical starts. Rows identify four task/seed pairs. Initial images come from the independent-policy run; subsequent columns show each condition's terminal view. Labels report native outcomes and executed steps. All images are saved observations; the coffee case succeeds only with the standalone policy. Table 6: RoboCasa completion on matched initial states. Each group contains five tasks with five episodes each; entries give successes/episodes and percentage. All conditions use the same 20-step feedback cadence and task horizons. Task groups describe the policy’s training coverage. Task group |  π0.5\pi_{0.5} alone |  Astra Direct |  Astra Hybrid  
---|---|---|---  
Atomic seen  |  13/25 (52.0%)  |  7/25 (28.0%)  |  13/25 (52.0%)   
Composite seen  |  3/25 (12.0%)  |  4/25 (16.0%)  |  7/25 (28.0%)   
Composite unseen  |  1/25 (4.0%)  |  14/25 (56.0%)  |  9/25 (36.0%)   
All tasks  |  17/75 (22.7%)  |  25/75 (33.3%)  |  29/75 (38.7%)   
  
### Uneven Gains from Policy Assistance

Astra Hybrid completes 29/75 episodes, compared with 25/75 for Direct and 17/75 for standalone π0.5\pi_{0.5} in Table 6. Hybrid exceeds Direct on atomic seen tasks, 52% versus 28%, and composite seen tasks, 28% versus 16%. Direct is stronger on unseen compositions, 56% versus 36%. Hybrid and standalone π0.5\pi_{0.5} both reach 13/25 on atomic tasks, but succeed on different instances.

Hybrid exceeds Direct on tasks seen during policy training but trails it on unseen compositions. These averages also conceal complementary successes. In Table 35, Hybrid alone succeeds on 13 paired starts and Direct alone on nine, with 16 shared successes. An exploratory exact McNemar test [10] gives p=0.5235p=0.5235.

Astra organizes grasping, transport, placement, and device operation through language decomposition and numerical control. In Hybrid, 66/75 episodes use rewritten instructions, which condition 72.6% of proposals. Figure 8 attributes 55.2% of executed steps to accepted policy actions and 44.8% to Astra. Rewriting changes what the policy is asked to achieve; numerical correction changes the robot commands, while context and feedback guards structure proposal review. The reported results evaluate these components jointly. Delegation still requires proposal review; Section 9.4 examines its inference demands.

### Recovery and Lost Policy Successes

Figure 9 illustrates these differences on matched starts. Only Hybrid succeeds on KettleBoiling/00, and only Direct succeeds on RecycleBottlesByType/01. On PickPlaceDrawerToCounter/04, both Hybrid and standalone π0.5\pi_{0.5} succeed.

Conversely, only standalone π0.5\pi_{0.5} succeeds on CoffeeSetupMug/03. Across all five seeds, both Astra conditions fail CoffeeSetupMug and WashLettuce, while the policy completes three and two cases, respectively. Policy access need not preserve success.

Visual feedback lets Astra detect some action errors, but recognizing a problem does not ensure a successful correction. Failure trajectories expose weaknesses in precise grasping and sustained contact, base–arm coordination while carrying objects, and tracking preconditions and object states across task stages. π0.5\pi_{0.5} helps on some tasks but does not ensure reliable state verification or physical control.

## Navigation

We evaluate Astra on instruction following in continuous environments (VLN-CE) and category-directed object search (ObjectNav) [19, 5], measuring both goal reaching and path efficiency. Astra chooses where to move and when to stop using RGB observations without pose or map input; the action interface applies discrete forward steps and turns.

### RGB Navigation on Four Fixed Subsets

We select 50 validation-unseen episodes each from R2R and English-guide RxR, and 50 validation episodes each from MP3D ObjectNav v1 and HM3D ObjectNav v2 [3, 20, 7, 54]. Sampling uses seed 0 and balances scenes and ObjectNav categories.

Astra receives one forward-facing 512×384512\times 384 RGB image, the instruction or object category, a step counter, and action feedback. GPS, compass, depth, semantic labels, goal coordinates, reference routes, and evaluation distances are withheld. A persistent episode session selects discrete moves or local waypoints. The adapter converts these to 0.25 m forward steps and 30∘30^{\circ} turns without path planning, under a 500-primitive budget. Habitat [40] executes these primitives; VLN-CE enables simulator sliding, whereas ObjectNav disables it.

Success requires STOP within 3 m of the VLN-CE goal or strictly within 1 m of the ObjectNav annotated goal-viewpoint set. The latter is not distance to the object’s surface and does not require final-image visibility; HM3D records are rescored at this tolerance. We report success rate (SR), success weighted by path length (SPL), and VLN-CE route agreement through nDTW and sDTW [2, 15].

Table 7: Navigation on four fixed subsets, 50 episodes each. Task, dataset, and split are listed separately. All metric values are percentages; RxR uses English guide instructions. ObjectNav has no instruction-specific reference trajectory. Task | Dataset | Split | Successes | SR ↑\uparrow | SPL ↑\uparrow | nDTW ↑\uparrow | sDTW ↑\uparrow  
---|---|---|---|---|---|---|---  
VLN-CE | R2R | Val-Unseen | 39/50 | 78 | 65.27 | 72.20 | 59.35  
VLN-CE | RxR (English) | Val-Unseen | 46/50 | 92 | 77.25 | 84.73 | 80.42  
ObjectNav | MP3D | Val | 28/50 | 56 | 22.43 | – | –  
ObjectNav | HM3D | Val | 41/50 | 82 | 43.69 | – | –  
Figure 10: Navigation success, efficiency, and termination. (a) SR and SPL. (b) Successful STOP, failed STOP, or exhaustion of the 500-step budget. Each subset contains 50 episodes.

### Success and Search Efficiency

Astra completes 39/50 R2R and 46/50 RxR episodes, with nDTW of 72.20% and 84.73%, showing goal reaching and route agreement in Table 7.

ObjectNav yields 28/50 MP3D and 41/50 HM3D successes, but SPL is only 22.43% and 43.69%. SPL discounts excess travel even when the final endpoint succeeds; the gap therefore reveals inefficient search that SR alone conceals. Figure 10 shows ten budget-limited MP3D episodes and five for HM3D; all evaluated VLN-CE episodes end with STOP.

A route instruction supplies spatial cues; an object category leaves the route to be discovered. Search thus requires deciding where to explore and when revisiting an area is no longer useful, in addition to grounding local observations. The subsets use different scenes, goals, instructions, and sliding settings.

Table 8: Local evaluation of released navigation policies on the same episode lists. Each dataset contains 50 episodes per system; values are percentages. The first three systems share the front-camera specification. OmniNav Flow uses three wider-angle views and is separated accordingly. All systems share task-specific scoring and primitive-action budgets, while their history processing, prompts, and action adapters differ. |  | VLN-CE | ObjectNav  
---|---|---|---  
|  | R2R | RxR | MP3D | HM3D  
System | Views | SR | SPL | SR | SPL | SR | SPL | SR | SPL  
Astra | 1 | 78.00 | 65.27 | 92.00 | 77.25 | 56.00 | 22.43 | 82.00 | 43.69  
LightNav-0 | 1 | 58.00 | 54.14 | 72.00 | 64.08 | 20.00 | 8.39 | 66.00 | 37.75  
Uni-NaVid 7B | 1 | 30.00 | 27.76 | 34.00 | 25.67 | 8.00 | 5.46 | 30.00 | 20.48  
_Multi-view methods (separate observation regime)_  
OmniNav Flow | 3 | 56.00 | 53.89 | 72.00 | 56.29 | 4.00 | 2.96 | 6.00 | 2.90  
NavFoM | 3 | 44.00 | 40.04 | 58.00 | 51.79 | 18.00 | 13.01 | 42.00 | 30.38  
SPAN-Nav | 3 | 70.00 | 61.71 | 72.00 | 56.19 | – | – | – | –  
  
### Local Comparisons and Failure Modes

Released LightNav-0, Uni-NaVid 7B, OmniNav Flow, NavFoM and SPAN-Nav policies are evaluated on the same 200 episodes with shared scoring and primitive budgets [49, 61, 53, 62, 23]. LightNav-0 and Uni-NaVid use Astra’s front-camera specification in place of their published camera settings; OmniNav Flow receives three wider-angle views. Local adapters convert LightNav-0 and OmniNav Flow outputs to the shared discrete actions. The OmniNav condition evaluates the released Flow policy without the full system’s slow exploration planner.

Under this local protocol, Astra has higher SR and SPL than the evaluated systems on all subsets with reported results in Table 8. Relative to LightNav-0, SR gains are 20, 20, 36, and 16 percentage points. LightNav-0 has slightly higher R2R nDTW, 72.77% versus 72.20%, while Astra has higher sDTW, 59.35% versus 48.89%. Appendix I.2 details the camera settings and action adapters.

![Refer to caption](2609.38537v1/navigation-cases.png) Figure 11: Two navigation successes and one budget-limited failure. Rows show initial, midpoint, and final front views beside the executed route on the Habitat navigation mesh. These evaluator-only maps were not shown to Astra.

Successful trajectories show Astra interpreting instructions, tracking landmark sequences, and recognizing target categories through multiple indoor stages. Figure 11 illustrates instruction following, search with detours, and budget exhaustion. Failure trajectories expose difficulties linking visual observations to specific locations, verifying goal completion, and using collision feedback or search history to change course. A plausible rationale for moving does not ensure correct goal judgment or spatial progress.

See Appendix I for protocols and termination counts.

## Locomotion

Navigation primitives implement the motion Astra chooses. Locomotion exposes an additional demand: constructing the body motion itself. Here Astra generates dense motion references for a frozen tracker over repeated attempts on one obstacle course. Analytic tests examine compact interfaces without Astra.

### Dense Motion on a Matched Course

We replace PASSAGE’s learned motion generator with Astra while retaining its scene-aligned 25×6525\times 65 representation and frozen G1 ScaleTrack controller [26, 60]. Each proposal specifies 0.5 s at 50 Hz: root height, projected gravity, planar velocity, yaw rate, and positions and velocities for 29 joints. Each call requires 1,625 consistent values.

On one open-clutter course, Astra and PASSAGE share the initial state, goal, elevation observation, local route guidance, tracker, and termination criteria. Success requires final planar error at most 0.5 m. Falls trigger resets; eight seconds with less than 0.10 m root displacement triggers no-progress termination. The supplied route leaves motion construction to Astra.

Astra makes five sequential attempts. The first is zero-shot; later attempts retain textual summaries of progress, falls, and collisions. Before the fifth, Astra also receives a retargeted walking clip from another course. Experience and motion context therefore change without weight updates. This is one adaptation trajectory, not five independent trials.

Table 9: PASSAGE and sequential Astra attempts on one matched open-clutter course. Progress is the reduction in planar distance to the goal, so a negative value denotes motion away from it. The Astra rows are successive stages of one adaptation trajectory, not independent trials. Planner / Attempt | Outcome | Time (s) | Progress (m) | Final error (m) | Fall | Reached  
---|---|---|---|---|---|---  
PASSAGE | Goal reached | 13.18 | 8.622 | 0.495 | No | Yes  
Astra Attempt 1 | Fall | 9.80 | 1.561 | 7.556 | Yes | No  
Astra Attempt 2 | No progress | 20.28 | -0.024 | 9.141 | No | No  
Astra Attempt 3 | No progress | 8.00 | 0.012 | 9.104 | No | No  
Astra Attempt 4 | No progress | 8.48 | 0.159 | 8.958 | No | No  
Astra Attempt 5 | Time horizon | 30.00 | 2.598 | 6.519 | No | No  
  
### Persistent Locomotion Failures

PASSAGE reaches the goal in 13.18 s. Astra initially advances 1.561 m before falling; the next three attempts remain upright with little useful progress. Table 9 shows that the fifth attempt stays upright for 30 s, travels 3.833 m, and reduces goal distance by 2.598 m. It still ends 6.519 m from the goal, before the principal obstacle field.

After five rounds of trial and error, Astra still does not achieve reliable locomotion through this dense-reference interface. Later attempts improve stability and produce some forward motion, but body and limb coordination is inconsistent. Across attempts, Astra uses motion feedback while the walking clip, gait timing, and amplitude are adjusted together.

The interface helps explain the difficulty. Joint limits alone do not make a reference coherent: root velocity, limb posture, and their evolution must describe compatible motion. A tracker can follow a reference yet produce in-place stepping, unwanted contacts, or a fall. A supplied route simplifies planning but does not supply coordinated body motion.

Timing is also limiting. The final 30 s episode uses 250 synchronous calls averaging 39.86 s of recorded latency, versus approximately 0.08 s for PASSAGE planning. Physics pauses during inference. The inference budget permits five attempts.

### Compact-Reference Diagnostics without Astra

We additionally test two compact-reference interfaces with analytic generators, without Astra, to identify failures in converting references into robot motion. Both use six-frame link-pose sequences for the same 29-DoF robot. Five-point specifies the pelvis, wrists, and ankles; WholeBody-14 adds the torso and bilateral hips, knees, shoulders, and elbows.

The deterministic generators run on three open-clutter and three low-ceiling courses in two physics backends: 12 rollouts per interface. Strict success requires an upright robot, final error at most 0.2 m, a five-second hold, tracked points within the course, and no obstacle force above 5 N. Goal + stop requires only arrival and a hold.

Table 10: Paired analytic interface feasibility study over six courses and two physics backends. Each course–backend cell contains one rollout. Contact magnitudes are useful failure diagnostics, but are not treated as calibrated measurements across simulators. These rollouts do not use Astra. Interface | NN |  Strict success |  Goal \+ stop | Falls |  Final error (m) |  Travel (m) |  Peak contact (N) |  Non-foot contact (N) |  Five-point RMSE (m)  
---|---|---|---|---|---|---|---|---|---  
Five-point | 12 | 0 | 4 (33.3%) | 1 (8.3%) | 3.441 | 5.745 | 1853 | 644 | 0.161  
WholeBody-14 | 12 | 0 | 3 (25.0%) | 4 (33.3%) | 5.334 | 3.866 | 2293 | 1562 | 0.197  
  
Neither interface achieves strict success in Table 10. Five-point reaches and holds the endpoint in 4/12 rollouts with one fall; WholeBody-14 does so in 3/12 with four falls. Five-point has lower mean endpoint and tracking errors. Low ceilings remain difficult, with only 1/6 and 0/6 rollouts reaching the endpoint.

The comparison favors the tested Five-point generator–interface pair descriptively, without isolating dimensionality because the generators differ. Large obstacle contacts also show that trackable motion is not necessarily safe motion.

## Humanoid Loco-Manipulation

We study Hybrid humanoid control, with Astra directing pretrained whole-body controllers. In HumanoidBench [44], Astra commands frozen Humanoid-GPT [37] policies across a broad task suite. In SIMPLE [51], it uses a frozen ScaleBFM [60] whole-body controller with task instructions refined through experience, examining manipulation and the retention of successful behavior. Astra selects task commands; whole-body controllers coordinate motion.

### HumanoidBench Evaluation Setup

We evaluate the 30 executable HumanoidBench tasks in the pinned registry, excluding the two Highbar variants. Evaluation uses seeds 1 and 2 for each task. Torque replays independently verify episodes.

Astra receives RGB, privileged state, task geometry, and guidance. It selects walking velocities and yaw rate, or sparse pelvis and wrist references. Frozen whole-body controllers run at 50 Hz over 500-Hz physics and joint control. Wrist targets couple to the body and depend on the control mode.

Table 11 compares Astra with published methods on eight tasks; Figure 12 covers all 30. Appendix E gives complete Astra results and evaluation details.

Table 11: HumanoidBench returns against published methods. Entries are mean returns; bold marks the highest value per task. DreamerV3, TD-MPC2, and SAC results follow HumanoidBench [44]. Threshold denotes the native task criterion. Appendix E documents result sources and evaluation protocols. Task | DreamerV3 | TD-MPC2 | SAC | Astra | Threshold  
---|---|---|---|---|---  
Maze | 272.3 | 244.3 | 144.8 | 1358.8 | 1200  
Reach | 7580.9 | 7316.1 | 4565.1 | 11430.2 | 12000  
Walk | 800.2 | 782.0 | 31.7 | 848.7 | 700  
Run | 633.8 | 93.3 | 5.0 | 642.1 | 700  
Crawl | 878.8 | 957.4 | 330.0 | 971.9 | 700  
Stair | 131.1 | 70.4 | 14.1 | 272.5 | 700  
Push | -1251.9 | -258.7 | -97.9 | 877.3 | 700  
Door | 213.0 | 274.7 | 39.4 | 142.4 | 600  
Figure 12: Results across 30 HumanoidBench tasks. Green bars show Astra mean returns and sample SDs. Left bars reproduce published method returns [44]. Axes are task-specific; dashed lines mark native thresholds. Appendix E gives exact values and reference protocols.

### Command Selection with Pretrained Whole-Body Controllers

Maze provides a clear example of adapting commands to state. Astra slows near intersections, turns, and resumes forward motion using supplied geometry. Maze achieves a mean return of 1358.8, with the evaluated trajectories reaching native stage 4. Reach achieves 11430.2, below the 12000 threshold, with individual returns straddling it.

Walk combines posture calibration with startup and speed guidance, achieving a mean return of 848.7. Run uses a five-point gait and reaches 642.1; both runs stay upright but below threshold. Crawl uses a low, forward-pitched gait and reaches 971.9, with about 27.39 m displacement per run. Astra selects speed and heading, while calibrated generators supply posture and limb coordination. Performance reflects Astra’s decisions and whole-body controllers.

![Refer to caption](2609.38537v1/assets/figures/push-seed7-000.png)

Push: initial state (0)

![Refer to caption](2609.38537v1/assets/figures/push-seed7-005.png)

Base staged (175)

![Refer to caption](2609.38537v1/assets/figures/push-seed7-009.png)

Contact approach (267)

![Refer to caption](2609.38537v1/assets/figures/push-seed7-013.png)

Native success (323)

Figure 13: Decisions and physical response in three separate experiments. Top left: Maze trajectories and supplied checkpoints; both runs reach stage 4. Top right: a scripted Walk frequency sweep with fixed weights and command speed; the dashed line marks the native threshold. Bottom: experience-conditioned Push, seed 7, ending at 4.81 cm box error. Parentheses give control steps.

Astra achieves the highest mean native return among the compared methods: 581.9, versus 338.2 for TD-MPC2, 42.7 for SAC, and −41.9-41.9 for DreamerV3, averaging task means equally across all 30 tasks in Figure 12. Astra also exceeds the published returns of DreamerV3 on 16 tasks, TD-MPC2 on 19, and SAC on 25. They exceed the best of these three references on 13 tasks and tie on Kitchen [44]. Six Astra task means reach native thresholds: Stand, Walk, Maze, Crawl, Push, and Sit Simple.

Calibration also affects what a command accomplishes. In the scripted Walk sweep, changing gait-clock frequency from 1.2 to 1.8 Hz raises mean return from 705.72 to 746.60 across eight paired seeds without changing weights, command speed, or feedback rule. Actual forward speed is about 1.48 m/s for a 1.25 m/s command. Return and measured speed together describe the effect of controller calibration.

### Contact Failures Between Task Stages

Push illustrates useful preparation and limited transfer. A demonstration-guided configuration succeeds on three development seeds and one transfer seed fixed in advance with box errors below 5 cm; two development trajectories reproduce supplied demonstrations. On the evaluation cases, staging and contact guidance support successful pushing in 277 and 357 controls, at 4.89 and 4.97 cm error.

A separate scripted diagnostic clarifies the interaction: changing entry alone or contact references alone fails, while changing both reaches 4.97 cm error. A hand offset depends on the body’s configuration. The records in Figure 13 and Appendix C show why base preparation and contact must be coordinated: an effective offset can fail from a different entry state.

Door exposes the next difficulty. Episodes rotate the hatch without opening a usable passage; scripted finger and body variants achieve partial opening without passage, and Astra also fails to complete the task. Reaching a mechanism is insufficient if contact fails under the loading and body movement needed to continue. Stair achieves a mean return of 272.5 with terrain-aware references, but one run falls after limited progress and the other retreats. Cube and Window lose objects early; Kitchen returns zero.

These failures occur between task stages, when changes in posture or contact disrupt the next action. Whole-body controllers alone do not resolve them. SIMPLE examines whether guidance drawn from experience helps Astra manage these transitions during grasping, transport, and handover.

### Refining Task Guidance through Experience in SIMPLE

The SIMPLE study examines whether experience can help Astra coordinate grasping, transport, and handover. Using visual and proprioceptive feedback, Astra selects actions for a G1 humanoid, while ScaleBFM supplies whole-body motion. After each trial, Astra reviews the outcome and revises the instructions and notes used in subsequent attempts. This adaptation changes the written guidance while keeping both models’ weights fixed.

We evaluate Astra using the official SIMPLE codebase and L2 evaluation protocol, covering six tasks with ten scenes each. Astra succeeds in 50 of 60 trials, achieving an overall success rate of 83.3% (Table 12). Performance is strongest in tabletop and picking tasks, while mobile pick/place remains more difficult. Appendix E.2 details the protocol and separate retention checks.

Table 12: Astra with ScaleBFM on SIMPLE L2. Success under the official evaluation protocol, with ten scenes per task. Task | Successes | Rate  
---|---|---  
Handover | 8/10 | 80%  
Mobile pick/place | 7/10 | 70%  
Tabletop | 9/10 | 90%  
XMove bend pick | 9/10 | 90%  
XMove pick | 9/10 | 90%  
Bend | 8/10 | 80%  
Total | 50/60 | 83.3%  
  
### Why Handover Remains Fragile

Handover reveals a limitation hidden by the aggregate score: native success does not require the carton to remain stable after complete release. In the observed failures, finger closure or carton tilt can be mistaken for a secure grasp while the table or the other hand still carries the load. Moving the receiving hand can then shift the supplying wrist as the body adjusts. A grasp that appears adequate at one stage may therefore fail during the next.

The grasp procedure addresses this problem by checking support before advancing. It instructs Astra to separate grasping from lifting, withdraw the assisting hand, and check that the entire carton rises and moves with the supplying hand. If it merely rotates or the body keeps leaning forward, the procedure calls for restoring support and trying again. These instructions specify physical checks; their reliability still depends on Astra’s observations and actions.

The handover guidance does not consistently preserve successful behavior across repeated attempts. In a sequence of instruction-adaptation trials, a successful handover is followed by a failed attempt after further reflection. A fixed procedure tested on three previously successful scenes passes native success and an additional hold check on only two. The recovery trials in Appendix E.2 also fail on all three valid attempts despite longer budgets. A diagnosis need not yield correction or repeatable success.

![Refer to caption](2609.38537v1/scene-reconstruction.png) Figure 14: Interactive scenes from egocentric video. Source frames and reconstructed geometry for two apple-transfer cases. Object and support layout is recovered, while table edges and furniture remain approximate.

### Scene and Motion from Egocentric Video

Astra also builds and refines a pipeline that converts egocentric videos into interactive scenes and humanoid motion references. It assembles perception and retargeting tools, inspects their outputs, and corrects geometric and temporal inconsistencies. Camera estimates, object roles, and support relations define the scene; declared camera or support heights provide metric scale. Motion references specify pelvis, wrist, and ankle poses together with hand closure. They are generated from video-derived geometry or through EgoAllo body estimation [57], SMPL-X fitting [33], and motion retargeting [4].

In a kitchen with motion-capture-calibrated cameras, image-based refinement reduces scene Chamfer-L1 error from 35.93 to 24.73 mm; incorporating CAD dimensions reduces it further to 12.14 mm. Transferring to other recordings exposes the importance of object placement: reusing a bottle position displaced by a few centimeters causes grasp failures. Figure 14 shows reconstructions from separate everyday videos. Across three apple-transfer cases, complete-video references support one complete interaction: opening a bin by its pedal, releasing the apple inside, and allowing the lid to close. The other two cases still lose the apple during lifting despite local scene adjustments.

The central challenge is preserving interaction structure through reconstruction and retargeting. Support surfaces, container interiors, and movable mechanisms must remain usable, while motion references must connect approach, grasp, transport, and release in the correct order. The successful bin interaction preserves the approach and pedal contact before release; local grasp corrections alone leave the task incomplete. Astra can identify and revise these dependencies, but geometric plausibility does not ensure a successful physical interaction. The in-the-wild study consumed 217 million tokens across development, evaluation, and associated documentation, highlighting the cost of per-video refinement.

## Discussion

Across the six domains, Astra shows strong navigation results and useful manipulation capabilities, alongside persistent difficulties in contact coordination, locomotion, and completion verification. To interpret this uneven performance, we examine how control is divided and why feedback does not consistently turn useful local decisions into successful, efficient task behavior.

Table 13: Inference demands across studies. M denotes million tokens; rounded totals may not sum exactly. Units and recording scope are listed for each study. Full manipulation accounting appears in Tables 17 and 37. Setting |  Recorded resource use |  Scope and interpretation  
---|---|---  
RoboDojo, 50 task instances per condition |  Hybrid / Direct: 624.8M / 1,132.3M total tokens; 607.6M / 1,107.3M cached input; 15.90M / 22.91M uncached input; 1.31M / 2.09M output |  Earlier attempts excluded. Cached input dominates the totals.  
RoboCasa, 75 episodes per condition |  Direct / Hybrid: 7,910 / 8,941 recorded model requests; median 85 / 108 per episode |  Counts cover saved request logs, including recovery history, rather than all inference calls.  
Dense locomotion, final attempt |  250 synchronous calls for 30 s of simulated robot motion; mean model latency 39.86 s per call |  Physics pauses during inference; PASSAGE planning takes approximately 0.08 s per call.  
  
### Capability Depends on the Complete Control Path

An action acquires physical meaning through its interface and controller. A navigation move invokes a discrete motion primitive; a dense motion reference requires compatible body and joint targets; a whole-body controller command invokes learned coordination. Evaluating Astra therefore requires specifying both the commands it constructs and the motion other components supply. The varied tasks and controllers provide complementary evidence about the complete control paths.

The in-hand results reveal a tension between stability and progress: retaining the current grasp can prevent the release and repositioning needed for continued rotation. Navigation presents a different problem, since a recognizable landmark need not establish the correct location or justify stopping. Control responsibility thus includes deciding which state changes matter and checking that they occur. These failures require different diagnostics and explanations.

### Intervention and Verified Progress

Useful cooperation requires both repairing failures and preserving behavior that already works. In the manipulation cases, Astra can improve the conditions under which a policy acts by redirecting a wrist or preparing contact, without constructing the subsequent trajectory. This division succeeds unevenly: RoboCasa Hybrid completes some instances that defeat the standalone policy and fails on others the policy completes. The trajectories illustrate useful roles for reasoning, while the aggregate comparisons measure the combined effects of policy access, action timing, and feedback management.

Detecting an error, proposing a correction, and verifying its effect are separate demands. A new grasp still needs clearance and support; an adjusted placement still needs a correct completion check. Action-source fractions describe who supplies commands, but neither the number of corrections nor the plausibility of their explanations measures their value. That value rests on the ensuing physical outcome.

### Retaining Success across Trials

Feedback can improve a procedure on a known case without making it reliable across attempts. The Push development records include a successful trial followed by a failed repeat. Diagnosis, written guidance, and sustained control thus need separate assessment: a sound explanation can identify what went wrong while leaving the next action inadequate.

This distinction matters when interpreting improvement with fixed model weights. Changes to guidance, interfaces, or whole-body controller calibration can change system performance. Repeated trials reveal whether the resulting behavior persists; transfer to new states tests whether the guidance extends beyond the cases that produced it.

### Resource Costs and Capability Limits

A capability is demonstrated with a particular budget. Long input histories and repeated review increase resource use, while paused physics separates simulated motion time from inference time. Table 13 reports recorded resource use.

The two manipulation studies illustrate different effects of cooperation. RoboDojo Hybrid reduces recorded tokens by 44.8%, yet retains substantial uncached input and output. RoboCasa Hybrid reduces Astra-authored action segments but records more model requests than Direct. Delegating motion therefore does not necessarily delegate the work of reviewing it. Inference efficiency depends on review frequency as well as action generation.

Dense locomotion exposes a more immediate limit: a decision takes far longer than the motion executed before the next observation. Pausing physics makes that delay tolerable in the experiment; a physical scene cannot be assumed to wait. Practical control therefore requires assessing inference delays alongside the rate at which physical feedback changes.

## Conclusion

We systematically evaluate GPT-6 Astra as an embodied policy across six domains. Astra shows strengths in navigation and task-oriented manipulation, correcting targets and contact conditions while cooperating with learned policies; pretrained whole-body controllers also support humanoid loco-manipulation. However, direct in-hand control and dense-reference locomotion remain unreliable, and policy assistance can fail on tasks that the standalone policy solves. These findings expose a gap between generating useful action decisions and reliably producing and verifying their physical effects. Learned policies and whole-body controllers help bridge this gap, but effective cooperation still requires Astra to recognize when to intervene and when to return control to the policy. Substantial token use and inference latency further constrain practical control, even when learned policies supply most physical actions.

## Contributors

Each group lists names alphabetically by surname.

Manipulation
    

Xiaoqian Cheng, Jiayi Su, Mi Yan, Yixin Zheng

Navigation
    

Jiahang Liu, Ruochen Xu, Tianyu Xu

Dexterous Manipulation
    

Yu Deng, Lihe Ding, Shaocong Dong, Xiangjun Gao, Sikai Liang, Qingtao Liu, Zhe Xu, Siming Yan

Locomotion
    

Haozhe Jia, Zhoujian Li, Yunrui Lian, Chenghuai Lin, Yuxuan Ma, Xudong Xu, Zhikai Zhang, Weiyi Zhu

Humanoid Loco-Manipulation
    

Xuchuan Chen, Zekai Li, Dairu Liu, Zekun Qi, Ruixi Yu, Jinlu Zhang, Yintianrun Zhang

Report Organization
    

Yu Deng, Jiahang Liu, Qingtao Liu, Yuxuan Ma, Zekun Qi, Jiayi Su, Ruochen Xu, Siming Yan

Supervision
    

He Wang, Li Yi, Zhizheng Zhang

## References

  * [1] Michael Ahn, Anthony Brohan, Noah Brown, Yevgen Chebotar, Omar Cortes, Byron David, Chelsea Finn, et al.  Do as i can, not as i say: Grounding language in robotic affordances.  _arXiv preprint arXiv:2204.01691_ , 2022.  <https://arxiv.org/abs/2204.01691>. 
  * [2] Peter Anderson, Angel Chang, Devendra Singh Chaplot, Alexey Dosovitskiy, Saurabh Gupta, Vladlen Koltun, Jana Kosecka, Jitendra Malik, Roozbeh Mottaghi, Manolis Savva, and Amir R. Zamir.  On evaluation of embodied navigation agents.  _arXiv preprint arXiv:1807.06757_ , 2018a.  <https://arxiv.org/abs/1807.06757>. 
  * [3] Peter Anderson, Qi Wu, Damien Teney, Jake Bruce, Mark Johnson, Niko Sünderhauf, Ian Reid, Stephen Gould, and Anton van den Hengel.  Vision-and-language navigation: Interpreting visually-grounded navigation instructions in real environments.  In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ , 2018b.  <https://arxiv.org/abs/1711.07280>. 
  * [4] Joao Pedro Araujo, Yanjie Ze, Pei Xu, Jiajun Wu, and C. Karen Liu.  Retargeting Matters: General Motion Retargeting for Humanoid Motion Tracking.  _arXiv preprint arXiv:2510.02252_ , 2025.  <https://arxiv.org/abs/2510.02252>. 
  * [5] Dhruv Batra, Aaron Gokaslan, Aniruddha Kembhavi, Oleksandr Maksymets, Roozbeh Mottaghi, Manolis Savva, Alexander Toshev, and Erik Wijmans.  ObjectNav revisited: On evaluation of embodied agents navigating to objects.  _arXiv preprint arXiv:2006.13171_ , 2020.  <https://arxiv.org/abs/2006.13171>. 
  * [6] Yihan Cao, Jiazhao Zhang, Zhinan Yu, Shuzhen Liu, Zheng Qin, Qin Zou, Bo Du, and Kai Xu.  CogNav: Cognitive process modeling for object goal navigation with LLMs.  _arXiv preprint arXiv:2412.10439_ , 2024.  <https://arxiv.org/abs/2412.10439v2>. 
  * [7] Angel Chang, Angela Dai, Thomas Funkhouser, Maciej Halber, Matthias Nießner, Manolis Savva, Shuran Song, Andy Zeng, and Yinda Zhang.  Matterport3D: Learning from RGB-D Data in Indoor Environments.  _arXiv preprint arXiv:1709.06158_ , 2017.  <https://arxiv.org/abs/1709.06158>. 
  * [8] Tianxing Chen, Yue Chen, Zixuan Li, Junyuan Tang, Kailun Su, Haoran Lu, Weijie Wan, Baijun Chen, et al.  RoboDojo: A Unified Sim-and-Real Benchmark for Comprehensive Evaluation of Generalist Robot Manipulation Policies.  _arXiv preprint arXiv:2607.04434_ , 2026.  <https://arxiv.org/abs/2607.04434>. 
  * [9] Olive Jean Dunn.  Multiple Comparisons among Means.  _Journal of the American Statistical Association_ , 56(293):52–64, 1961.  [10.1080/01621459.1961.10482090](https://doi.org/10.1080/01621459.1961.10482090).  <https://doi.org/10.1080/01621459.1961.10482090>. 
  * [10] Morten W Fagerland, Stian Lydersen, and Petter Laake.  The McNemar test for binary matched-pairs data: mid-p and asymptotic are better than exact conditional.  _BMC Medical Research Methodology_ , 13(1), 2013.  [10.1186/1471-2288-13-91](https://doi.org/10.1186/1471-2288-13-91).  <https://doi.org/10.1186/1471-2288-13-91>. 
  * [11] Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine.  Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor.  _arXiv preprint arXiv:1801.01290_ , 2018.  <https://arxiv.org/abs/1801.01290>. 
  * [12] Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, and Timothy Lillicrap.  Mastering Diverse Domains through World Models.  _arXiv preprint arXiv:2301.04104_ , 2023.  <https://arxiv.org/abs/2301.04104>. 
  * [13] Nicklas Hansen, Hao Su, and Xiaolong Wang.  TD-MPC2: Scalable, Robust World Models for Continuous Control.  _arXiv preprint arXiv:2310.16828_ , 2023.  <https://arxiv.org/abs/2310.16828>. 
  * [14] HKU SAIL.  Real2Sim_GPT6_ASTRA: Three-view robot RGB scene reconstruction and action replay.  Community implementation, 2026.  <https://github.com/hku-sail/Real2Sim_GPT6_ASTRA>.  Accessed September 25, 2026. 
  * [15] Gabriel Ilharco, Vihan Jain, Alexander Ku, Eugene Ie, and Jason Baldridge.  General evaluation for instruction conditioned navigation using dynamic time warping.  _arXiv preprint arXiv:1907.05446_ , 2019.  <https://arxiv.org/abs/1907.05446>. 
  * [16] Mengdi Jia, Zekun Qi, Shaochen Zhang, Wenyao Zhang, Xinqiang Yu, Jiawei He, He Wang, and Li Yi.  OmniSpatial: Towards comprehensive spatial reasoning benchmark for vision language models.  In _International Conference on Learning Representations (ICLR)_ , 2026.  <https://openreview.net/forum?id=6nZKT2rL0H>. 
  * [17] O. Khatib.  A unified approach for motion and force control of robot manipulators: The operational space formulation.  _IEEE Journal on Robotics and Automation_ , 3(1):43–53, 1987.  [10.1109/JRA.1987.1087068](https://doi.org/10.1109/JRA.1987.1087068).  <https://doi.org/10.1109/JRA.1987.1087068>. 
  * [18] Alexander Khazatsky, Karl Pertsch, Suraj Nair, Ashwin Balakrishna, Sudeep Dasari, Siddharth Karamcheti, Soroush Nasiriany, Mohan Kumar Srirama, et al.  DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset.  _arXiv preprint arXiv:2403.12945_ , 2024.  <https://arxiv.org/abs/2403.12945>. 
  * [19] Jacob Krantz, Erik Wijmans, Arjun Majumdar, Dhruv Batra, and Stefan Lee.  Beyond the nav-graph: Vision-and-language navigation in continuous environments.  _arXiv preprint arXiv:2004.02857_ , 2020.  <https://arxiv.org/abs/2004.02857>. 
  * [20] Alexander Ku, Peter Anderson, Roma Patel, Eugene Ie, and Jason Baldridge.  Room-across-room: Multilingual vision-and-language navigation with dense spatiotemporal grounding.  _arXiv preprint arXiv:2010.07954_ , 2020.  <https://arxiv.org/abs/2010.07954>. 
  * [21] Jacky Liang, Wenlong Huang, Fei Xia, Peng Xu, Karol Hausman, Brian Ichter, Pete Florence, and Andy Zeng.  Code as policies: Language model programs for embodied control.  In _2023 IEEE International Conference on Robotics and Automation (ICRA)_ , 2023.  <https://arxiv.org/abs/2209.07753>. 
  * [22] Yutong Liang, Quanquan Peng, Ri-Zhao Qiu, and Xiaolong Wang.  ConTrack: Constrained hand motion tracking with adaptive trade-off control.  In _European Conference on Computer Vision (ECCV)_ , 2026.  <https://arxiv.org/abs/2606.03177>. 
  * [23] Jiahang Liu, Tianyu Xu, Jiawei Chen, Lu Yue, Jiazhao Zhang, Zhiyong Wang, Minghan Li, Qisheng Zhao, Anqi Li, Qi Su, et al.  Span-nav: Generalized spatial awareness for versatile vision-language navigation.  _arXiv preprint arXiv:2603.09163_ , 2026a. 
  * [24] Zixuan Liu, Ruoyi Qiao, Chenrui Tie, Xuanwei Liu, Yunfan Lou, Chongkai Gao, Zhixuan Xu, and Lin Shao.  ContactExplorer: Contact Coverage-Guided Exploration for General-Purpose Dexterous Manipulation.  _arXiv preprint arXiv:2603.10971_ , 2026b.  <https://arxiv.org/abs/2603.10971>. 
  * [25] Ilya Loshchilov and Frank Hutter.  Decoupled Weight Decay Regularization.  _arXiv preprint arXiv:1711.05101_ , 2017.  <https://arxiv.org/abs/1711.05101>. 
  * [26] Yuxuan Ma, Zicheng Zeng, Chunlin Peng, Zhoujian Li, Zetong Zhao, Zhikai Zhang, Yunrui Lian, Han Xue, Sikai Liang, Weiyi Zhu, Mulin Chen, Chenghuai Lin, Jiayu Zeng, Yanwei An, Songan Zhang, Jiayuan Gu, Jilong Wang, Jingbo Wang, He Wang, and Li Yi.  Passage: Scaling scene-aligned motion learning for perceptive humanoid traversal in cluttered environments, 2026.  <https://arxiv.org/abs/2609.18732>. 
  * [27] Viktor Makoviychuk, Lukasz Wawrzyniak, Yunrong Guo, Michelle Lu, Kier Storey, Miles Macklin, David Hoeller, Nikita Rudin, et al.  Isaac Gym: High Performance GPU-Based Physics Simulation For Robot Learning.  _arXiv preprint arXiv:2108.10470_ , 2021.  <https://arxiv.org/abs/2108.10470>. 
  * [28] Yusuke Mikami, Andrew Melnik, Jun Miura, and Ville Hautamäki.  Natural language as policies: Reasoning for coordinate-level embodied control with LLMs.  _arXiv preprint arXiv:2403.13801_ , 2024.  <https://arxiv.org/abs/2403.13801>. 
  * [29] Soroush Nasiriany, Sepehr Nasiriany, Abhiram Maddukuri, and Yuke Zhu.  RoboCasa365: A large-scale simulation framework for training and benchmarking generalist robots.  _arXiv preprint arXiv:2603.04356_ , 2026.  <https://arxiv.org/abs/2603.04356>. 
  * [30] NVIDIA.  Cosmos3-Nano-Policy-DROID.  Official model card, 2026.  <https://huggingface.co/nvidia/Cosmos3-Nano-Policy-DROID>.  Accessed September 25, 2026. 
  * [31] NVIDIA, Mayank Mittal, Pascal Roth, James Tigue, Antoine Richard, Octi Zhang, Peter Du, Antonio Serrano-Muñoz, et al.  Isaac Lab: A GPU-Accelerated Simulation Framework for Multi-Modal Robot Learning.  _arXiv preprint arXiv:2511.04831_ , 2025.  <https://arxiv.org/abs/2511.04831>. 
  * [32] OpenAI.  GPT-6 Astra: A new generation of intelligence.  Model release, 2026.  <https://openai.com/index/gpt-6-astra/>.  Accessed September 25, 2026. 
  * [33] Georgios Pavlakos, Vasileios Choutas, Nima Ghorbani, Timo Bolkart, Ahmed A. A. Osman, Dimitrios Tzionas, and Michael J. Black.  Expressive Body Capture: 3D Hands, Face, and Body from a Single Image.  _arXiv preprint arXiv:1904.05866_ , 2019.  <https://arxiv.org/abs/1904.05866>. 
  * [34] Physical Intelligence, Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Equi, et al.  π0.5\pi_{0.5}: a Vision-Language-Action Model with Open-World Generalization.  _arXiv preprint arXiv:2504.16054_ , 2025.  <https://arxiv.org/abs/2504.16054>. 
  * [35] Zekun Qi, Runpei Dong, Shaochen Zhang, Haoran Geng, Chunrui Han, Zheng Ge, Li Yi, and Kaisheng Ma.  ShapeLLM: Universal 3D object understanding for embodied interaction.  In _European Conference on Computer Vision (ECCV)_ , 2024.  <https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05974.pdf>. 
  * [36] Zekun Qi, Wenyao Zhang, Yufei Ding, Runpei Dong, XinQiang Yu, Jingwen Li, Lingyun Xu, Baoyu Li, Xialin He, Guofan Fan, Jiazhao Zhang, Jiawei He, Jiayuan Gu, Xin Jin, Kaisheng Ma, Zhizheng Zhang, He Wang, and Li Yi.  SoFar: Language-grounded orientation bridges spatial reasoning and object manipulation.  In _Advances in Neural Information Processing Systems_ , volume 38, pages 76367–76412, 2025.  [10.52202/085713-2565](https://doi.org/10.52202/085713-2565). 
  * [37] Zekun Qi, Xuchuan Chen, Dairu Liu, Chenghuai Lin, Yunrui Lian, Sikai Liang, Zhikai Zhang, Yu Guan, et al.  Humanoid-GPT: Scaling Data and Structure for Zero-Shot Motion Tracking.  _arXiv preprint arXiv:2606.03985_ , 2026.  <https://arxiv.org/abs/2606.03985>. 
  * [38] Ruoyi Qiao and contributors.  mjlab_hand: Dexterous manipulation tasks built on mjlab.  Code repository, 2026.  <https://github.com/ruoyiqiao/mjlab_hand>.  Accessed September 25, 2026. 
  * [39] Stan Salvador and Philip Chan.  Toward accurate dynamic time warping in linear time and space.  _Intelligent Data Analysis_ , 11(5):561–580, 2007.  [10.3233/IDA-2007-11508](https://doi.org/10.3233/IDA-2007-11508).  <https://doi.org/10.3233/IDA-2007-11508>. 
  * [40] Manolis Savva, Abhishek Kadian, Oleksandr Maksymets, Yili Zhao, Erik Wijmans, Bhavana Jain, Julian Straub, Jia Liu, et al.  Habitat: A Platform for Embodied AI Research.  _arXiv preprint arXiv:1904.01201_ , 2019.  <https://arxiv.org/abs/1904.01201>. 
  * [41] John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, and Pieter Abbeel.  High-Dimensional Continuous Control Using Generalized Advantage Estimation.  _arXiv preprint arXiv:1506.02438_ , 2015.  <https://arxiv.org/abs/1506.02438>. 
  * [42] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov.  Proximal Policy Optimization Algorithms.  _arXiv preprint arXiv:1707.06347_ , 2017.  <https://arxiv.org/abs/1707.06347>. 
  * [43] Clemens Schwarke, Mayank Mittal, Nikita Rudin, David Hoeller, and Marco Hutter.  RSL-RL: A Learning Library for Robotics Research.  _arXiv preprint arXiv:2509.10771_ , 2025.  <https://arxiv.org/abs/2509.10771>. 
  * [44] Carmelo Sferrazza, Dun-Ming Huang, Xingyu Lin, Youngwoon Lee, and Pieter Abbeel.  HumanoidBench: Simulated humanoid benchmark for whole-body locomotion and manipulation.  _arXiv preprint arXiv:2403.10506_ , 2024.  <https://arxiv.org/abs/2403.10506>. 
  * [45] Sharpa Robotics.  Sharpa Wave dexterous hand.  [Online], n.d.  <https://www.sharpa.com/pages/wave>.  Accessed: September 20, 2026. 
  * [46] Emanuel Todorov, Tom Erez, and Yuval Tassa.  MuJoCo: A physics engine for model-based control.  In _2012 IEEE/RSJ International Conference on Intelligent Robots and Systems_ , pages 5026–5033, 2012.  [10.1109/IROS.2012.6386109](https://doi.org/10.1109/IROS.2012.6386109).  <https://doi.org/10.1109/IROS.2012.6386109>. 
  * [47] Charles Wampler.  Manipulator Inverse Kinematic Solutions Based on Vector Formulations and Damped Least-Squares Methods.  _IEEE Transactions on Systems, Man, and Cybernetics_ , 16(1):93–101, 1986.  [10.1109/TSMC.1986.289285](https://doi.org/10.1109/TSMC.1986.289285).  <https://doi.org/10.1109/TSMC.1986.289285>. 
  * [48] Hanwen Wang, Weizhi Zhao, Xiangyu Wang, Siyuan Huang, He Lin, Boyuan Zheng, Rongtao Xu, Gang Wang, Yao Mu, He Wang, Lue Fan, Hongsheng Li, Zhaoxiang Zhang, and Tieniu Tan.  DexJoCo: A benchmark and toolkit for task-oriented dexterous manipulation on MuJoCo, 2026a.  <https://arxiv.org/abs/2605.16257v2>. 
  * [49] Shaoan Wang, Aocheng Luo, Fei Huang, Jingyi Xu, Xiaoyang Wang, Yueyu Wang, Qianli Ma, Fan Yang, Ran Mei, Jia Wei, et al.  LightNav-0: Eliciting VLM spatial intelligence for generalist embodied navigation.  _arXiv preprint arXiv:2608.30935_ , 2026b.  <https://arxiv.org/abs/2608.30935>. 
  * [50] Yen-Jen Wang, Bike Zhang, Jianyu Chen, and Koushil Sreenath.  Prompt a robot to walk with large language models.  In _2024 IEEE 63rd Conference on Decision and Control (CDC)_ , 2024.  <https://arxiv.org/abs/2309.09969>. 
  * [51] Songlin Wei, Zhenhao Ni, Jie Liu, Zhenyu Zhao, Junjie Ye, Hongyi Jing, Junkai Xia, Xiawei Liu, Michael Leong, Liang Heng, Di Huang, and Yue Wang.  SIMPLE: Simulation-based policy learning and evaluation for humanoid loco-manipulation.  _arXiv preprint arXiv:2606.08278_ , 2026.  <https://arxiv.org/abs/2606.08278>. 
  * [52] Edwin B. Wilson.  Probable Inference, the Law of Succession, and Statistical Inference.  _Journal of the American Statistical Association_ , 22(158):209–212, 1927.  [10.1080/01621459.1927.10502953](https://doi.org/10.1080/01621459.1927.10502953).  <https://doi.org/10.1080/01621459.1927.10502953>. 
  * [53] Xinda Xue, Junjun Hu, Minghua Luo, Shichao Xie, Jintao Chen, Zixun Xie, Kuichen Quan, Wei Guo, Mu Xu, and Zedong Chu.  OmniNav: A unified framework for prospective exploration and visual-language navigation.  _arXiv preprint arXiv:2509.25687_ , 2025.  <https://arxiv.org/abs/2509.25687>. 
  * [54] Karmesh Yadav, Ram Ramrakhya, Santhosh Kumar Ramakrishnan, Theo Gervet, John Turner, Aaron Gokaslan, Noah Maestre, Angel Xuan Chang, Dhruv Batra, Manolis Savva, et al.  Habitat-matterport 3d semantics dataset.  _arXiv preprint arXiv:2210.05633_ , 2022.  <https://arxiv.org/abs/2210.05633>. 
  * [55] Jenai Xuning Yang, Rishit Dagli, Alex Zook, Hugo Hadfield, Ankit Goyal, Stan Birchfield, Fabio Ramos, and Jonathan Tremblay.  RoboLab: A high-fidelity simulation benchmark for analysis of task generalist policies.  _arXiv preprint arXiv:2604.09860_ , 2026.  <https://arxiv.org/abs/2604.09860>. 
  * [56] Seonghyeon Ye, Yunhao Ge, Kaiyuan Zheng, Shenyuan Gao, Sihyun Yu, George Kurian, Suneel Indupuru, You Liang Tan, et al.  World Action Models are Zero-shot Policies.  _arXiv preprint arXiv:2602.15922_ , 2026.  <https://arxiv.org/abs/2602.15922>. 
  * [57] Brent Yi, Vickie Ye, Maya Zheng, Yunqi Li, Lea Müller, Georgios Pavlakos, Yi Ma, Jitendra Malik, and Angjoo Kanazawa.  Estimating Body and Hand Motion in an Ego-sensed World.  _arXiv preprint arXiv:2410.03665_ , 2024.  <https://arxiv.org/abs/2410.03665>. 
  * [58] Jessica Yin, Haozhi Qi, Jitendra Malik, James Pikul, Mark Yim, and Tess Hellebrekers.  Learning in-hand translation using tactile skin with shear and normal force sensing.  In _2025 IEEE International Conference on Robotics and Automation (ICRA)_ , pages 5850–5856. IEEE, 2025.  <https://arxiv.org/abs/2407.07885>. 
  * [59] Kevin Zakka, Qiayuan Liao, Brent Yi, Louis Le Lay, Koushil Sreenath, and Pieter Abbeel.  mjlab: A lightweight framework for gpu-accelerated robot learning.  _arXiv preprint arXiv:2601.22074_ , 2026.  <https://arxiv.org/abs/2601.22074>. 
  * [60] Weishuai Zeng, Kangning Yin, Xiaojie Niu, Shunlin Lu, Weixiang Zhong, Jiahe Chen, Feiyu Jia, Xiao Chen, et al.  Scaling Behavior Foundation Model for Humanoid Robots.  _arXiv preprint arXiv:2607.15163_ , 2026.  <https://arxiv.org/abs/2607.15163>. 
  * [61] Jiazhao Zhang, Kunyu Wang, Shaoan Wang, Minghan Li, Haoran Liu, Songlin Wei, Zhongyuan Wang, Zhizheng Zhang, and He Wang.  Uni-NaVid: A video-based vision-language-action model for unifying embodied navigation tasks.  _Robotics: Science and Systems_ , 2025.  <https://arxiv.org/abs/2412.06224>. 
  * [62] Jiazhao Zhang, Anqi Li, Yunpeng Qi, Minghan Li, Jiahang Liu, Shaoan Wang, Haoran Liu, Gengze Zhou, Yuze Wu, Xingxing Li, et al.  Embodied navigation foundation model.  In _International Conference on Learning Representations_ , volume 2026, pages 127293–127322, 2026a.  <https://arxiv.org/abs/2509.12129>. 
  * [63] Jiazhao Zhang, Gengze Zhou, Hale Yin, Yiyang Huang, Zixing Lei, Qihang Peng, Haoqi Yuan, Jie Zhang, et al.  Qwen-RobotNav Technical Report: A Scalable Navigation Model Designed for an Agentic Navigation System.  _arXiv preprint arXiv:2606.18112_ , 2026b.  <https://arxiv.org/abs/2606.18112>. 
  * [64] Wenbo Zhang, Kaixuan Wang, Yutao Ouyang, Xiaoyu Huang, Liyang Li, Kailun Su, Weiyang Jin, Wenhao Chai, et al.  An Unexpected Robot Policy: Early Evaluations of GPT-6 Astra on RoboDojo and Beyond.  _arXiv preprint arXiv:2609.24170_ , 2026c.  <https://arxiv.org/abs/2609.24170>. 
  * [65] Yi Zhou, Connelly Barnes, Jingwan Lu, Jimei Yang, and Hao Li.  On the Continuity of Rotation Representations in Neural Networks.  _arXiv preprint arXiv:1812.07035_ , 2018.  <https://arxiv.org/abs/1812.07035>. 
  * [66] Yuke Zhu, Josiah Wong, Ajay Mandlekar, Roberto Martín-Martín, Abhishek Joshi, Kevin Lin, Abhiram Maddukuri, Soroush Nasiriany, and Yifeng Zhu.  robosuite: A Modular Simulation Framework and Benchmark for Robot Learning.  _arXiv preprint arXiv:2009.12293_ , 2020.  <https://arxiv.org/abs/2009.12293>. 



## Appendix A Additional Interface Details

This appendix specifies the gripper and humanoid interfaces summarized in Section 2. Figure 15 illustrates how these systems connect action proposals, whole-body controllers, and permitted feedback.

Figure 15: Action and development loops. Path 01 illustrates Direct control; paths 02 and 03 illustrate Hybrid with a task policy or whole-body controller. The action loop returns only study-permitted observations and feedback; task evaluation is a separate branch. The outer loop represents researcher and development-agent changes to guidance, tools, or interfaces before evaluation.

### Gripper Interfaces

RoboDojo supplies head and bilateral wrist images, a 14-dimensional proprioceptive state, measured end-effector poses, the instruction, execution history, progress, step budget, and completion rules. Hidden object states, future trajectories, rollback, and hidden planners are excluded. Policy images remain 640×480640\times 480; Astra generally views previews with a 480-pixel long side, with full-resolution images accessible.

Direct specifies bimanual position, orientation, and gripper targets. Bounded damped least-squares inverse kinematics [47] converts them to joint commands, with 5 cm translation and 0.35 rad rotation safeguards. Segments last one to five 25-Hz control steps. Hybrid exposes a 50×1450\times 14 task-finetuned π0.5\pi_{0.5} proposal through forward kinematics. Astra accepts a prefix of one to fifteen steps or replaces it with a one-to-five-step correction. Both use xhigh reasoning and the same native goals and safeguards. Many tasks require both arms home.

RoboLab uses a single-arm Franka and the same division between direct targets and policy assistance. Its π0.5\pi_{0.5} weights are DROID-trained [18], without finetuning on demonstrations from the test tasks. Appendix B gives the evaluation procedure and decision budgets.

### Humanoid Interfaces

The HumanoidBench interface uses Humanoid-GPT whole-body controllers [37]. It supplies body-frame forward/lateral velocity and yaw rate to a frozen walking policy, or pelvis-height and wrist offsets to a frozen sparse five-point tracker. Feet and keypoint orientations keep nominal references. Walking and tracking policies operate at 50 Hz over 500-Hz native physics and joint proportional–derivative control. IK and PD implement the commanded motion.

Sparse targets are coupled whole-body references, re-entering sparse mode resets the anchor, and walking changes arm posture. Observations include privileged state and task geometry. The task suite also includes task-specific access such as bounded finger closure for Door.

Run, Crawl, and Stair use continuous five-point gait generators. Astra chooses speed, heading, and observation interval; calibrated generators supply posture, phase, and limb references. Stair additionally uses measured support heights and lateral foot-placement feedback. Walk combines cadence and leg-extension calibration with startup guidance. Push uses staging, contact guidance, and bounded hand control. The dense ScaleTrack locomotion study uses a separate robot–tracker protocol.

### Experience and Evaluation Scope

Completion and timing conventions follow Section 2; replay verification is documented in Appendix C. The interfaces explicitly supply the state variables listed above to Astra.

Astra’s weights remain fixed; guidance may include calibrated examples, feedback rules, or motion clips.

## Appendix B Detailed Manipulation Results

### RoboDojo

Table 14 expands the aggregate results in Section 3.2. Direct Score averages the 48 episodes with valid Scores; its success rate uses all 50 instances. Published policy scores are reweighted to the same task and scene mixture. Direct and Hybrid share scene configurations and seeds.

Table 14: RoboDojo results. Each architecture has five evaluation instances per task. Direct and Hybrid are paired on scene configuration and recorded seeds. |  Native Score |  Success rate (%)  
---|---|---  
Task |  Direct |  Hybrid |  Direct |  Hybrid  
Organize Table |  30.0 |  60.0 |  0 |  0  
Classify Objects By Language |  60.0 |  38.0 |  40 |  20  
Imitate Sorting Sequence |  0.0 |  53.0 |  0 |  40  
Arrange Largest Number |  57.0 |  50.0 |  40 |  40  
Pack Objects Into Box |  50.0 |  50.0 |  20 |  20  
Classify Objects |  100.0 |  71.0 |  100 |  60  
Build Tower |  12.0 |  64.0 |  0 |  60  
Make Kong |  0.0 |  40.0 |  0 |  40  
Fold Clothes |  40.0 |  100.0 |  40 |  100  
Put Bottles Into Dustbin |  36.0 |  100.0 |  20 |  100  
Aggregate |  37.81 |  62.60 |  26 |  48  
  
### RoboLab

Table 15: RoboLab task success. Each method contributes five outcomes per task. Cosmos denotes Cosmos3-Nano-Policy [30]; DreamZero follows Ye et al. [56]. Task | Direct | Hybrid | π0.5\pi_{0.5} | Cosmos | DreamZero  
---|---|---|---|---|---  
Blocks into bin | 5/5 | 5/5 | 0/5 | 2/5 | 0/5  
Pumpkins in clutter | 5/5 | 4/5 | 0/5 | 0/5 | 0/5  
Butter on raisin box | 5/5 | 5/5 | 0/5 | 1/5 | 2/5  
Stack blocks in order | 5/5 | 4/5 | 0/5 | 0/5 | 0/5  
Reorient red mug | 5/5 | 4/5 | 2/5 | 1/5 | 1/5  
Larger raisin box into bin | 4/5 | 4/5 | 3/5 | 0/5 | 0/5  
Sauce bottle into crate | 5/5 | 5/5 | 2/5 | 5/5 | 5/5  
Canned food into bin | 5/5 | 5/5 | 4/5 | 4/5 | 4/5  
Yogurt into bowl | 5/5 | 5/5 | 2/5 | 0/5 | 0/5  
Rubik’s cube into bowl | 5/5 | 5/5 | 5/5 | 5/5 | 5/5  
Total | 49/50 | 46/50 | 18/50 | 18/50 | 17/50  
  
RoboLab permits repeat attempts for Astra. Two Direct Blocks-in-Bin attempts use a 500-decision budget; the standard budget is 180. Simulation horizons and native success criteria remain fixed. Each baseline contributes its first five runs per task, ordered by run, episode, and environment ID, with the initial states, task versions, and control settings of those runs.

## Appendix C Focused Experiments, Resource Use, and Reproducibility

### Focused Push and Walk Records

These focused studies use separately frozen configurations. The table distinguishes development seeds from transfer seeds fixed in advance; these runs are separate from the task-suite results in Section 8.

Table 16: Focused online Astra runs under separately frozen configurations. Push uses calibrated examples, with seeds 3 and 5 reproducing supplied physical trajectories; its physical outcome gives the final box-to-goal error and native success. Walk uses a gait frequency calibrated offline and a supplied heading-feedback rule. These outcomes are not substituted into the 30-task suite. Configuration | Seed role | Seed | Native return | Controls | Physical outcome  
---|---|---|---|---|---  
Focused Push | Development | 0 | 872.58 | 396 | 4.66 cm; success  
Focused Push | Development | 3 | 866.75 | 423 | 4.78 cm; success  
Focused Push | Development | 5 | 848.01 | 457 | 4.85 cm; success  
Focused Push | Transfer | 7 | 900.37 | 323 | 4.81 cm; success  
Walk, 1.8 Hz | Development | 1 | 743.54 | 1000 | No fall  
Walk, 1.8 Hz | Development | 2 | 748.92 | 1000 | No fall  
Walk, 1.8 Hz | Transfer | 8 | 743.18 | 1000 | No fall  
Walk, 1.8 Hz | Transfer | 9 | 748.54 | 1000 | No fall  
  
A separate scripted Push diagnostic tests whether contact must be calibrated jointly with the state at which the robot enters the contact phase. On seed 5, the final box-to-goal error is 27.02 cm with unchanged entry and references, 36.81 cm with changed entry only, 23.11 cm with changed contact references only, and 4.97 cm when both are changed. Heading and velocity also vary slightly between conditions.

### Independent Reach Records

Two Reach runs from a separate Walk/Reach/Door batch measure target dwell time. Seeds 0 and 2 finish all 1000 controls with native returns of 15675.19 and 15380.91, respectively, both above the 12000 threshold. The hand is within the 5 cm target radius for 11.36 s and 12.78 s. These records use separate archived instructions and execution settings; they are not part of the task suite’s Reach sample.

### RoboDojo Execution and Token Accounting

Table 17 provides exact counts for Section 9.4. Simulated duration excludes inference latency and can be shortened by early failure.

Table 17: RoboDojo action and token accounting. Mean simulated duration is the total control-step count multiplied by 0.04 s and divided by 50 slots. Token totals include cached input and exclude earlier attempts. Metric |  Hybrid |  Direct  
---|---|---  
Executed control steps  |  42,750  |  38,221   
Executed action segments  |  3,776  |  7,729   
Mean simulated duration per slot (s)  |  34.20  |  30.58   
Total tokens, including cached input  |  624,762,828  |  1,132,343,772   
Cached input tokens  |  607,555,840  |  1,107,349,760   
Uncached input tokens  |  15,901,963  |  22,907,448   
Output tokens  |  1,305,025  |  2,086,564   
  
### Source Snapshots and Verification

The audit reports 270 checked hashes across 90 online HumanoidBench episodes and checks on ten further runs. Reproduction requires the archived manifests, prompts, execution traces, and robot runtime. Replay verifies recorded dynamics and rewards.

## Appendix D Manipulation Tasks and Behavioral Cases

### Task Coverage

![Refer to caption](2609.38537v1/assets/figures/manipulation/organize_table.jpg)

Table organization

![Refer to caption](2609.38537v1/assets/figures/manipulation/classify_objects_by_language.jpg)

Language classification

![Refer to caption](2609.38537v1/assets/figures/manipulation/imitate_sorting_sequence.jpg)

Sequence imitation

![Refer to caption](2609.38537v1/assets/figures/manipulation/arrange_largest_number.jpg)

Number arrangement

![Refer to caption](2609.38537v1/assets/figures/manipulation/pack_objects_into_box.jpg)

Packing

![Refer to caption](2609.38537v1/assets/figures/manipulation/classify_objects.jpg)

Object classification

![Refer to caption](2609.38537v1/assets/figures/manipulation/build_tower.jpg)

Tower construction

![Refer to caption](2609.38537v1/assets/figures/manipulation/make_kong.jpg)

Mahjong

![Refer to caption](2609.38537v1/assets/figures/manipulation/fold_clothes.jpg)

Clothes folding

![Refer to caption](2609.38537v1/assets/figures/manipulation/put_bottles_into_dustbin.jpg)

Bottle disposal

Figure 16: The ten RoboDojo manipulation tasks. Task images from our evaluation website are reproduced without alteration. They illustrate the environments rather than evaluation outcomes; the Direct and Hybrid results in Table 2 use the paired episodes.

### Behavioral Case Index

The 21 recorded cases cover five aspects of manipulation: cases 1–4 concern goal alignment; 5–8 contact preparation; 9–12 terminal-state checking; 13–16 Hybrid contact and feedback failures; and 17–21 Direct strategies and local recovery. The cases illustrate behaviors observed in the recorded trajectories. Both Direct and Hybrid receive robot state, and the interpretations refer to complete trajectories rather than still images.

## Appendix E Humanoid Loco-Manipulation: Supporting Results

Table 18 expands the HumanoidBench study in Section 8, reporting Astra returns across all 30 tasks. Task-specific return thresholds and observed physical outcomes are reported separately. The separate SIMPLE development records follow below.

Figure 12 reproduces published returns from HumanoidBench Table V [44]: DreamerV3 [12] and SAC [11] use 10M training steps, and TD-MPC2 [13] uses 2M. We use the average returns in Table V, reported over three seeds, rather than the maximum returns in Table VI. The comparison counts a task only when Astra’s reported mean is strictly higher than the reference value; Kitchen is a tie at zero.

Table 18: All 30 evaluated HumanoidBench tasks. Entries are means ±\pm sample SD over two seeds. Bold values meet or exceed the listed return threshold. Task |  Threshold |  Astra  
---|---|---  
Stand  |  800  |  954.2 ±\pm 0.1   
Walk  |  700  |  848.7 ±\pm 5.9   
Run  |  700  |  642.1 ±\pm 2.8   
Kitchen  |  4  |  0.0 ±\pm 0.0   
Maze  |  1200  |  1358.8 ±\pm 7.4   
Hurdle  |  700  |  183.0 ±\pm 0.3   
Cube  |  370  |  6.3 ±\pm 0.3   
Bookshelf Simple  |  2000  |  666.3 ±\pm 7.6   
Bookshelf Hard  |  2000  |  572.8 ±\pm 201.0   
Crawl  |  700  |  971.9 ±\pm 0.2   
Window  |  650  |  6.1 ±\pm 0.7   
Spoon  |  650  |  319.1 ±\pm 8.6   
Door  |  600  |  142.4 ±\pm 7.8   
Push  |  700  |  877.3 ±\pm 59.6   
Reach  |  12000  |  11430.2 ±\pm 2089.0   
Basketball  |  1200  |  24.0 ±\pm 2.9   
Truck  |  3000  |  787.3 ±\pm 87.1   
Package  |  1500  |  -5148.0 ±\pm 1288.1   
Cabinet  |  2500  |  274.1 ±\pm 110.0   
Sit Simple  |  750  |  793.0 ±\pm 19.1   
Sit Hard  |  750  |  277.5 ±\pm 283.5   
Balance Simple  |  800  |  53.1 ±\pm 6.7   
Balance Hard  |  800  |  48.3 ±\pm 1.8   
Stair  |  700  |  272.5 ±\pm 44.0   
Slide  |  700  |  141.1 ±\pm 45.9   
Pole  |  700  |  628.7 ±\pm 10.5   
Room  |  400  |  143.9 ±\pm 27.6   
Insert Normal  |  350  |  50.6 ±\pm 1.5   
Insert Small  |  350  |  46.7 ±\pm 8.8   
Powerlift  |  800  |  84.6 ±\pm 53.7   
  
### Additional Physical Diagnostics

#### Walk.

The four 1.8 Hz runs are listed in Appendix C. The task-suite results use posture calibration together with startup and speed guidance.

#### Run and Crawl.

Run finishes 1,000 controls without falls, with mean forward center-of-mass speeds of 3.85 and 3.83 m/s and displacements of 77.05 and 76.50 m. Crawl maintains mean forward speed near 1.37 m/s. Both Crawl runs stay within the lateral band, and neither records head height below 0.2 m. A standing-based diagnostic flags the intended pitched posture; native task terms and progress are more informative.

#### Stair.

Seed 1 reaches root X of 1.55 m before falling at control 970; seed 2 reaches 1.06 m, retreats, and finishes behind its starting X position.

#### Push.

The Push runs are described in Section 8.3; Appendix C gives the separate focused runs and exact entry/contact diagnostic outcomes.

### SIMPLE Development and Retention Checks

#### Control and observations.

The SIMPLE implementation uses G1 with Dex3 hands, MuJoCo physics and native scoring, and Isaac Sim rendering. Astra runs with xhigh reasoning and commands both wrists and both ankles, together with independent finger actions. Frozen ScaleBFM executes the whole-body targets. Astra receives external, head, and wrist RGB, joint encoders, forward kinematics, and a base-state estimate. The depth-assisted variants also use relative depth inferred from RGB. Object ground truth, simulator contact diagnostics, and scoring signals are excluded from both action selection and reflection.

#### Experience representation.

Two text documents support adaptation: _Skill_ contains task instructions, and _Memory_ contains cross-episode notes. Both are separate from shared control instructions and the robot interface. After each episode, Astra uses its observations and actions to revise these documents. Astra and ScaleBFM remain frozen.

#### Single-scene development.

Across ten tasks, 58 attempts are recorded, including interruptions. The task-development endpoints meet native success on nine tasks; eight also meet execution, audit, and additional hold requirements. Handover triggers native success but times out before the hold check. Basket carrying receives a favorable qualitative assessment of its action sequence but fails native scoring. Neither is counted as a complete success. These outcomes summarize development rather than 58 independent trials.

#### Separate Handover studies.

Table 19 separates grasp stabilization, fallen-carton recovery, and instruction adaptation. The stabilization variants pass native success and the additional hold check in 2/5 and 3/5 trials, respectively. The latter changes both instructions and relative-depth input. Recovery starts with the carton already fallen and allows a longer budget; its results are not pooled with upright-carton trials. In the retention check, one fixed procedure fails on scene00 and passes both criteria on scene02 and scene08. The instruction-adaptation trials use changing guidance.

Table 19: SIMPLE Handover development records. Native success and success with the additional hold check have identical counts in these trials and are reported together. Rows describe separate development conditions and are excluded from the L2 total in Table 12. Stage |  Condition |  Valid trials |  Success + hold  
---|---|---|---  
Grasp stabilization |  Grasp entry, enclosure, and support transfer |  5 |  2/5  
Depth-assisted grasp |  Relative depth and side-entry grasp guidance |  5 |  3/5  
Fallen-carton recovery |  Table-edge support, regrasping, and handover |  1 |  0/1  
Depth-assisted recovery |  Relative depth and recovery notes |  2 |  0/2  
Trajectory-informed guidance |  Scene 1; instructions derived from contrasting trajectories |  1 |  1/1  
Post-trial reflection |  Scene 1; guidance informed by the preceding trial |  1 |  0/1  
Difficult-scene check |  Scene 9; guidance informed by trial feedback |  1 |  0/1  
Frozen retention check |  Scenes 00, 02, 08; one fixed candidate |  3 |  2/3  
  
#### L2 evaluation protocol.

We follow the official SIMPLE codebase and evaluation rules, using the same L2 protocol as the benchmark baselines. The evaluation comprises six tasks with ten scenes each; 50 of 60 trials meet the official success criteria, yielding an overall success rate of 83.3%. The separate Handover development and retention studies are excluded from this total.

## Appendix F Harness Development and Repeatability

These development cases supplement Section 8.3. Each retains its own configuration and reasoning setting; none belongs to the frozen evaluation suite.

### Contact-Conditioned Guidance

Contact-conditioned Push guidance reduces the prompt from 129 to 90 lines, replacing unconditional replay of a calibrated example with checks on contact, measured box displacement, and remaining error. Controller code, observations, action limits, native rules, and the 1.0 s walking stop delay remain unchanged. On development seed 3, return changes from −148.56-148.56 to 863.90 and final box error from 29.45 to 4.88 cm, with success after 428 controls. Seed 0 also succeeds; the additional seed 5 fails at 500 controls with 27.02 cm error.

The archived replay reports record no state, reward, or native-information discrepancies. Each condition has one online attempt on known development seeds, with the prompt changes described above.

A separate Push request selects ten controls while noting rising lateral box speed and the need to observe before overshoot. Its explicit body and wrist references show state-informed action timing. The record captures local anticipation from observed state changes.

### A Saved Diagnosis and a Failed Repeat

After a seed-2 Push failure, an online tool event records Astra writing NOTES.md: repeated lifting leaves the hand below the tabletop, without useful pushing contact. The subsequent guidance calls for outward lifting on the robot side and checks clearance before extension.

This procedure succeeds once in 381 controls with return 803.86, then fails on a repeat after 500 controls with return −259.83-259.83. Task and composed-skill hashes match; full transmitted-prompt hashes differ because the workspace path changes. All three workers use high reasoning. The archive attributes the written diagnosis to Astra; it does not identify the author of the subsequent guidance change.

### Finger Control Does Not Solve Door

Developers extend Door with bounded finger closure and observations of finger positions and contact geometry. One online Astra episode issues five full-closure commands and completes 1,000 controls, but returns 144.38 against a threshold of 600, with no passage and maximum door angle approximately 0.0195 rad. Guidance, observations, and action access change together in this condition.

### Provenance and Evidence Availability

The development archive is organized under sources/harness-study/, with provenance, hashes, source excerpts, prompt versions, and replay reports. Its collection-time commit, db82157bf9a64143433ec75e824a8cb7f89e2b8f, identifies the inspected checkout. Verification requirements are given in Appendix C.

## Appendix G Dexterous Manipulation: Data, Interfaces, and Evaluation

This appendix specifies the task suite, demonstration data, control interfaces, and evaluation protocol for Section 4. Aggregate scores are reported in Table 3.

### Task Suite and Demonstration Data

Table 20 lists instructions and simulation budgets for the ten tasks; Figure 17 shows their initial configurations.

![Refer to caption](2609.38537v1/assets/figures/dexterous/initial-20260920-02_pack_headset.png)

(a) Headphones in box

![Refer to caption](2609.38537v1/assets/figures/dexterous/initial-20260920-03_grasp_from_bin_toy.png)

(b) Toy retrieval

![Refer to caption](2609.38537v1/assets/figures/dexterous/initial-20260920-05_mahjong_into_basket.png)

(c) Mahjong placement

![Refer to caption](2609.38537v1/assets/figures/dexterous/initial-20260920-06_sort_bottles_cans.png)

(d) Bottle/can sorting

![Refer to caption](2609.38537v1/assets/figures/dexterous/initial-20260920-07_egg_into_tray.png)

(e) Upright egg placement

![Refer to caption](2609.38537v1/assets/figures/dexterous/initial-20260920-08_stack_bowls_on_coaster.png)

(f) Two-bowl stacking

![Refer to caption](2609.38537v1/assets/figures/dexterous/initial-20260920-09_bread_into_slots.png)

(g) Bread insertion

![Refer to caption](2609.38537v1/assets/figures/dexterous/initial-20260920-10_sort_nesting_dolls.png)

(h) Nesting-doll ordering

Figure 17: Example initial states for dexterous manipulation. Panels show eight benchmark tasks, with one initial state each. Table 20: Dexterous manipulation tasks and evaluation horizons. Task instructions reproduce the prompts used for training and evaluation. KK is the number of subgoals used for partial-credit scoring. tmaxt_{\max} and the continuous verification window are measured in simulation time. Subgoal predicates are specified in Table 22. Task |  Task instruction |  KK |  tmaxt_{\max} (s) |  Verification (s)  
---|---|---|---|---  
Pot lift and hold |  Lift the pot at least 10 centimeters above the table and hold it steadily without dropping it. |  1 |  30 |  3  
Headphones in box |  Pick the headset and place it into the box. |  1 |  20 |  2  
Toy retrieval |  Take the toy out of the bin and place it on the table. |  2 |  20 |  2  
Mug hanging |  Hang the mug on the mug rack   
without requiring handle-specific alignment. |  1 |  25 |  2  
Two mahjong tiles |  Place both mahjongs into the basket. |  2 |  30 |  2  
Bottles/cans sorting |  Place bottles into the front-left basket   
and cans into the front-right basket. |  4 |  60 |  2  
Upright egg placement |  Move the egg from the blue platform to the orange platform and leave it upright in the holder. |  2 |  40 |  2  
Two-bowl stacking |  Stack the two bowls nested upright on the black mousemat. |  2 |  30 |  2  
Bread-slot insertion |  Place both bread slices into the toaster,   
one slice in each slot. |  2 |  45 |  2  
Nesting-doll ordering |  Arrange the four nesting dolls in a row from left to right, from smallest to largest. |  2 |  60 |  2  
  
Demonstrations use the instructions in Table 20; their counts and training configuration appear in Table 21.

#### Initial configurations.

The task configurations vary object locations and, where applicable, object yaw and initial hand posture. Initial hand postures are drawn from the first frames of the corresponding task’s demonstrations. The evaluation consequently measures execution across specified scene configurations with the same task objectives and assets. The fixed evaluation subset is described in Section G.4.

### Multitask Policy Finetuning

Standalone π0.5\pi_{0.5} and Hybrid share one policy finetuned on the pooled demonstrations from all ten tasks. Table 21 specifies initialization, AdamW optimization [25], and execution settings for this shared policy.

### Observations, Numerical Actions, and Execution

#### Policy observations.

We use Sharpa dexterous hands. The simulation uses MuJoCo 3.11.0 [46]. The π0.5\pi_{0.5} policy receives RGB images from the head and both wrist cameras together with the task instruction and a 62-dimensional state vector comprising the two wrist poses and finger-joint angles. Each image is resized from 640×360640\times 360 to 224×224224\times 224 with aspect-ratio-preserving padding.

#### Policy action representation.

Each π0.5\pi_{0.5} action comprises two wrist poses, each encoded by 3 position values and a 6D rotation representation [65], and 22 joint angles per hand. The resulting action dimension is 2×(3+6)+2×22=622\times(3+6)+2\times 22=62.

#### Execution and model budgets.

The baseline discards each unexecuted prediction suffix before observing and replanning. Table 21 gives execution cadences, Hybrid corrections, and budget-exhaustion behavior. Task limits in Table 20 count simulated time; model calls and inference wall time are separate.

Table 21: Dexterous manipulation implementation details. Configuration |  Specification  
---|---  
Demonstration budget |  100 trajectories per task; ten tasks; 1,000 trajectories in total.  
S1 adaptation |  One multi-task π0.5\pi_{0.5} policy, shared by the standalone baseline and Hybrid.  
Direct action responsibility |  Astra specifies wrist/arm targets and finger-joint targets.  
Hybrid action responsibility |  π0.5\pi_{0.5} supplies manipulation actions; Astra can edit wrist and finger targets.  
S1 initialization and training |  π0.5\pi_{0.5} base initialization; full-parameter finetuning; 50,000 optimization steps; global batch size 256.  
S1 optimizer |  AdamW with β=(0.9,0.95)\beta=(0.9,0.95), ϵ=10−8\epsilon=10^{-8}, weight decay 10−1010^{-10}, and gradient-norm clipping at 1.0.  
S1 learning rate |  Peak 2.5×10−52.5\times 10^{-5}; cosine decay to 2.5×10−62.5\times 10^{-6}.  
S1 numerical actions |  50×6250\times 62 output; world-frame wrist poses with 6D rotations and absolute finger-joint targets; wrist increments restored to absolute poses before execution.  
π0.5\pi_{0.5} baseline execution |  30 Hz; execute the first 16 steps of each 50-step prediction before replanning; physics paused during inference.  
Hybrid observations and actions |  Astra receives the task instruction, synchronized head and wrist RGB views, measured wrist poses, 22 joint angles per hand, hand keypoints, and the remaining simulation budget. A persistent episode conversation retains observation and action history; each decision also includes the previous execution record, a 50-step π0.5\pi_{0.5} wrist preview, and hand-motion summaries. Astra selects an unchanged π0.5\pi_{0.5} action prefix or specifies bounded wrist and finger corrections, which are converted into 62-dimensional control targets.  
Hybrid execution and prompting |  Control runs at approximately 30 Hz, with one reset-settling step; π0.5\pi_{0.5} action prefixes execute for 1–16 steps and Astra corrections for 1–5 steps before re-observation. Astra uses xhigh reasoning, with prompts and model configuration archived per run. Takeover requires observed execution failure or misaligned π0.5\pi_{0.5} intent; control returns to π0.5\pi_{0.5} once recovery permits. Each run can set a maximum number of reviewed action chunks and a token budget. Under the default fallback, either limit ends Astra’s involvement, and π0.5\pi_{0.5} continues alone until the simulator terminates the episode. Simulation time pauses while Astra deliberates.  
  
### Evaluation Protocol and Scoring

#### Fixed evaluation cases.

We evaluate each task on five fixed cases, yielding 50 evaluation cases across the ten tasks. All three methods are evaluated on the same case set.

#### Continuous execution and verification.

Each task has a simulation-time limit tmaxt_{\max}, chosen to cover at least 1.5 times the longest demonstration and twice the median demonstration duration. Verification is performed during continuous execution. For the nine placement and rearrangement tasks, full completion requires all task subgoals to hold, the manipulated objects to be released, and their positions to remain stable for 2 s. Release is verified from contact data showing zero contacts between an object and the hand collision geometries. Pot lifting instead requires the pot to remain at least 10 cm above its initial height, in contact with a hand and out of contact with the table, for a 3 s stable-holding window.

The positional stability threshold is 1 cm over the verification window. The documented window definition measures each object’s maximum root-position displacement from the start of that window; it is not a sum of frame-to-frame displacements. An interrupted verification condition restarts the window. Successful verification permits early completion; otherwise the episode is scored at termination. There is no separate stage in which the robot is frozen to validate an earlier placement.

#### Task subgoals.

Table 22 defines each predicate, including geometric tolerances, release requirements, and dependencies between subgoals.

Table 22: Task-specific completion criteria. Positions refer to object roots unless stated otherwise. Full credit additionally requires continuous stability within 1 cm and the verification windows in Table 20; all tasks require release except pot holding. Headset and mahjong release conditions also apply to their subgoal predicates. Task |  Subgoal predicates and task-specific verification conditions  
---|---  
Pot lift and hold |  The pot root is at least 10 cm above its initial height. The 0.8 tier requires this lift condition at termination. Full-score verification requires continued hand contact, no pot–table contact, and stability for 3 s.  
Headphones in box |  The root projects inside the box-bottom polygon; at least 50% of the vertically constrained minimum bounding-box volume lies in the vertical prism extending upward from that polygon; and hand contact is absent. The prism has no upper height limit, so a released headset resting across the rim can qualify.  
Toy retrieval |  Two independent subgoals: (i) the root projects outside the box-bottom polygon; (ii) the toy contacts the table and its root height is at most the table height plus its reference resting height plus 3 cm. Reference resting height is the initial root height above the box-bottom reference plane.  
Mug hanging |  Conjunctive subgoal: lift from the initial root height >4.5>4.5 cm, root height above the table >13.5>13.5 cm, and root-to-rack horizontal distance <9<9 cm. Handle alignment is unnecessary.  
Two mahjong tiles |  One subgoal per tile: the root projects inside the basket interior (half-widths 10.85 and 6.67 cm), and hand contact is absent. No constraint is imposed on the tiles' vertical positions. Full credit requires both released tiles to remain stable for 2 s.  
Bottles/cans sorting |  One subgoal per object: the root projects inside its assigned basket interior and its height relative to the basket reference plane is in [−6,10][-6,10] cm. Bottles belong in the front-left basket and cans in the front-right basket; placement in the other basket receives no credit.  
Upright egg placement |  Two nested subgoals: (i) the root is within 2 cm horizontally and 1.5 cm vertically of the target seat; (ii) condition (i) holds and the egg’s local zz axis is within 30∘30^{\circ} of world vertical. Upright orientation at the source provides no subgoal credit.  
Two-bowl stacking |  Two independent subgoals: (i) the bowls’ horizontal separation is at most 2 cm and vertical separation at least 0.5 cm, each bowl tilts at most 45∘45^{\circ}, and the lower bowl at most 7∘7^{\circ}; (ii) the lower bowl’s center lies inside the coaster-local rectangle x∈[−12.005,12.005]x\in[-12.005,12.005] cm, y∈[−9.601,9.601]y\in[-9.601,9.601] cm, with no edge tolerance.  
Bread-slot insertion |  One subgoal per distinct occupied slot: a bread root projects within the slot polygon, its height above the slot reference plane is in [2,8][2,8] cm, and at least one of its local ±x\pm x or ±y\pm y axes lies within 30∘30^{\circ} of world upward. Two slices in one slot count once; either end may face upward. Full-score verification requires both slices to be released.  
Nesting-doll ordering |  Two subgoals: (i) the four roots span at most 6 cm along world xx; (ii) in increasing size order, every adjacent pair has a signed world-yy separation of at least 4 cm.  
  
#### Partial-credit scores.

Let KtK_{t} be the number of subgoals for task tt, and let kk be the number satisfied at termination. The documented score tiers use an internal scale s∈[0,1]s\in[0,1]:

  * •

s=1s=1 when all subgoals and the full continuous verification window are satisfied;

  * •

s=0.8s=0.8 when all subgoals hold at termination but full verification has not been completed;

  * •

s=0.8​k/Kts=0.8k/K_{t} for partial completion, where 0<k<Kt0<k<K_{t};

  * •

s=0.1s=0.1 when no subgoal holds at termination but a manipulated object was lifted by 5 cm during execution;

  * •

s=0s=0 for no effective progress or an applicable invalidating termination.




Thus, completing one of two subgoals ordinarily gives 40 points on the reported scale, whereas satisfying all subgoals without verification gives 80 points. Partial credit reflects the terminal configuration, not accumulated subgoals. The 5 cm lift event separately records earlier progress.

#### Safety and termination.

The task configurations specify a 200 N hand–table threshold, a 60 N wrist-camera contact threshold, and self-collision checks. Model-call exhaustion must be distinguished from task timeout and physical termination.

#### Aggregation.

For method mm, let st,i,ms_{t,i,m} denote the internal score on case ii of task tt. Task and overall scores are

| St,m=1005​∑i=15st,i,m,Sm=110​∑t=110St,m.S_{t,m}=\frac{100}{5}\sum_{i=1}^{5}s_{t,i,m},\qquad S_{m}=\frac{1}{10}\sum_{t=1}^{10}S_{t,m}. |  | (1)  
---|---|---|---  
  
Each task contributes equally to the 0–100 partial-credit score reported in Table 3; this is not a success rate.

## Appendix H In-Hand Manipulation: Controllers and Complete Results

This appendix details the in-hand experiments in Section 4.

### Benchmark Definition

Sharpa Wave [45] rotation runs in Isaac Lab [31], with task formulations drawn from ConTrack [22] and the mjlab_hand codebase [38], built on mjlab [59]. Allegro tasks use the Isaac Gym [27] tactile-IHT implementation: translation follows Yin et al. [58], and translation with long-axis rotation is our extension. Both simulators run at 20 control steps per simulated second. Table 23 and Figure 18 define the goals and endpoints.

Table 23: In-hand manipulation benchmark. Each task contains five paired initial states evaluated by Astra Direct and the corresponding frozen RL policy. Task |  Platform |  Goal |  Endpoint |  Reported measures  
---|---|---|---|---  
Cylinder rotation |  Sharpa / Isaac Lab |  Rotate about world +Z+Z at 1.0​rad​s−11.0\,\mathrm{rad\,s^{-1}} |  20 s (400 steps) |  Orientation error, axial-speed MAE, At-goal, completion, drop  
Cuboid rotation |  Sharpa / Isaac Lab |  Rotate about world +Z+Z at 0.2​rad​s−10.2\,\mathrm{rad\,s^{-1}} |  10 s (200 steps) |  Orientation error, axial-speed MAE, At-goal, completion, drop  
Cylinder translation |  Allegro / Isaac Gym |  Reach a Cartesian target |  15 s (300 steps) |  Terminal position error, success, drop  
Translation + rotation |  Allegro / Isaac Gym |  Reach the Cartesian target and rotate the long axis by −40∘-40^{\circ} |  Astra at 120 decisions; RL at matched steps and 15 s |  Terminal position and rotation errors, joint success, drop  
![Refer to caption](2609.38537v1/dexterous_benchmark_overview.png) Figure 18: The four dexterous-manipulation tasks. (a–b) The Sharpa hand rotates a cylinder or cuboid continuously about world-frame +Z+Z at the stated target speed. (c) The Allegro Hand translates the cylinder center to a Cartesian target. (d) The second Allegro task additionally requires a −40∘-40^{\circ} long-axis rotation. The images and lower schematics define the tasks rather than illustrate performance.

#### Case selection and pairing.

For every reported pair, the two controllers load the same saved hand joint state, object pose and velocity, and task target before control begins. Simulator randomization is disabled after loading. The five cylinder-rotation cases are drawn from the held-out grasp bank. The cuboid grasp source yielded only six validated stable starts; we fixed the first five case identifiers before running the formal Astra evaluation. Four of these starts overlap the RL initialization bank and one is held out. The two Allegro tasks use five distinct saved physical initial states each. All translation-and-rotation states come from the stabilized rerun; the earlier unstable pilot is excluded from every reported value.

### Task-Specific RL Controllers and Training

Table 24 summarizes the four frozen task-specific policies. Evaluation uses no adaptation or action tuning.

Table 24: Training provenance of the evaluated RL policies. NN is the number of parallel training environments and HH is the rollout length per PPO update. The translation baseline uses the released checkpoint without retraining; training counts are listed for the three policies trained in this study. Task |  Policy construction |  Training setup  
---|---|---  
Cylinder rotation |  PPO from scratch; four distributed ranks; speed curriculum |  N=×8192N=4\\!\times\\!8192, H=16H=16, 1,000 updates; ×1085.24\\!\times\\!10^{8} transitions  
Cuboid rotation |  PPO initialized from the cylinder actor and actor normalizer |  N=8192N=8192, H=16H=16, 4,000 updates; ×1085.24\\!\times\\!10^{8} transitions  
Cylinder translation |  Released privileged-teacher/tactile-student policy |  Released checkpoint; no retraining  
Translation + rotation |  Privileged PPO teacher initialized from the released oracle |  N=4096N=4096, H=16H=16, 20,000 updates; ×1091.31\\!\times\\!10^{9} transitions  
  
#### Sharpa continuous-rotation policies.

Both Sharpa policies use PPO [42] in RSL-RL [43] and a 22-dimensional action at 20 Hz. Their actor and critic are three-layer ELU MLPs with widths (512,256,128)(512,256,128) and separate running observation normalization. PPO uses discount γ=0.99\gamma=0.99, generalized advantage estimation [41] with λ=0.95\lambda=0.95, clipping parameter 0.20.2, five optimization epochs, an adaptive learning rate initialized at 5×10−45\times 10^{-4}, target KL 0.010.01, value coefficient 2.02.0, zero entropy coefficient, and gradient-norm clipping at 1.01.0. The policy observation contains finger joint positions and velocities, object pose and velocity relative to the palm, the moving rotation target, goal-relative orientation, and the previous action. Both tasks share the reward

| rt=Δ​t​[0.02et+0.1+𝕀⁡(et≤0.1)−10−6​∥𝐚t−𝐚t−1∥22+0.02​rtcontact],r_{t}=\Delta t\left[\frac{0.02}{e_{t}+0.1}+\mathbb{I}(e_{t}\leq 0.1)-10^{-6}\lVert\mathbf{a}_{t}-\mathbf{a}_{t-1}\rVert_{2}^{2}+0.02r_{t}^{\mathrm{contact}}\right], |  | (2)  
---|---|---|---  
  
where Δ​t=0.05​s\Delta t=0.05\,\mathrm{s}, et=2​arccos⁡(|⟨qt,qt⋆⟩|)e_{t}=2\arccos\\!\left(\left|\langle q_{t},q_{t}^{\star}\rangle\right|\right) is the geodesic error between the object orientation and the moving target, and rtcontactr_{t}^{\mathrm{contact}} is the ContactExplorer novelty reward [24] over fingertip proximity and contacted surface regions. Thus angular-speed tracking is encouraged indirectly by following the continuously advancing orientation target rather than by a separate velocity-reward term.

The cylinder policy is trained with seed 42 on a group-disjoint 80% grasp-bank split. Training advances through target-speed stages of 0.250.25, 0.50.5, 0.750.75, and 1.0​rad​s−11.0\,\mathrm{rad\,s^{-1}}, samples the training grasp bank with probability 0.5, and randomizes wrist position by ±5​mm\pm 5\,\mathrm{mm} and wrist orientation by ±5∘\pm 5^{\circ}. The policy is frozen after 1,000 updates and evaluated on five starts from the held-out 20% grasp-bank split.

The cuboid policy is trained only on the cuboid training split. Its actor and actor-observation normalizer are initialized from the cylinder policy; the critic, critic normalizer, action variance, optimizer, update counter, and curriculum state are reinitialized. We evaluate the final seed-45 policy deterministically at 0.2​rad​s−10.2\,\mathrm{rad\,s^{-1}}. Table 26 marks each start’s training-set overlap.

#### Allegro translation policy.

The translation baseline is the unmodified released signed-three-axis tactile policy of Yin et al. [58], obtained from their public tactile-IHT codebase.11 1 <https://github.com/jessicayin/tactile_skin_model> Its source method first trains a privileged PPO teacher with simulator object and contact information. The teacher’s environment reward is

| r~ttrans=\displaystyle\tilde{r}_{t}^{\mathrm{trans}}={} | −100​ep,t2+10​Ip,t−0.3​Dpose,t−0.1​Dτ,t−2​Dwork,t\displaystyle-100e_{p,t}^{2}+10I_{p,t}-0.3D_{\mathrm{pose},t}-0.1D_{\tau,t}-2D_{\mathrm{work},t} |  | (3)  
---|---|---|---|---  
|  | −1000​Dz,t−170​Dtilt,t,\displaystyle-1000D_{z,t}-170D_{\mathrm{tilt},t}, |   
  
where ep,t=∥𝐩t−𝐩⋆∥2e_{p,t}=\lVert\mathbf{p}_{t}-\mathbf{p}^{\star}\rVert_{2} and Ip,t=𝕀⁡(ep,t2<5×10−4)I_{p,t}=\mathbb{I}(e_{p,t}^{2}<5\times 10^{-4}). The regularizers are Dpose,t=∥𝐪t−𝐪0∥22D_{\mathrm{pose},t}=\lVert\mathbf{q}_{t}-\mathbf{q}_{0}\rVert_{2}^{2}, Dτ,t=∥𝝉t∥22D_{\tau,t}=\lVert\bm{\tau}_{t}\rVert_{2}^{2}, Dwork,t=(∑i|τt,i|​|q˙t,i|)2D_{\mathrm{work},t}=(\sum_{i}|\tau_{t,i}||\dot{q}_{t,i}|)^{2}, Dz,t=clip⁡(0.48−pz,t,0,1)D_{z,t}=\operatorname{clip}(0.48-p_{z,t},0,1), and Dtilt,t=(c1​y−c2​y)2+(c1​z−c2​z)2D_{\mathrm{tilt},t}=(c_{1y}-c_{2y})^{2}+(c_{1z}-c_{2z})^{2}, where 𝐜1\mathbf{c}_{1} and 𝐜2\mathbf{c}_{2} are the cylinder endpoints. Terms present in the released implementation with zero weight are omitted from Eq. 3. The student is then trained by action imitation rather than by optimizing this reward directly. It uses a 30-step temporal Transformer to infer an eight-dimensional latent from joint history and signed three-axis tactile measurements, followed by an MLP with widths (512,256,128)(512,256,128) that outputs 16 joint actions. We evaluate the released policy without retraining.

#### Allegro translation-and-rotation policy.

For the translation-and-rotation task, we train a privileged PPO teacher initialized from the released oracle policy. The actor MLP has widths (512,256,128)(512,256,128) and receives the 99-dimensional task observation together with an eight-dimensional embedding of privileged simulator state produced by a (256,128,8)(256,128,8) MLP. PPO uses γ=0.99\gamma=0.99, λ=0.95\lambda=0.95, learning rate 10−410^{-4}, minibatches of 4,096 transitions, three epochs, clipping parameter 0.20.2, critic coefficient 44, zero entropy coefficient, and gradient-norm clipping at 1.01.0. The target is fixed to +0.045​m+0.045\,\mathrm{m} translation and −40∘-40^{\circ} long-axis rotation. Starting from Eq. 3, the task replaces the translation-only completion bonus and adds rotation terms:

| r~ttrans+rot=\displaystyle\tilde{r}_{t}^{\mathrm{trans+rot}}={} | r~ttrans−10​Ip,t+10​Ip,t​Iθ,t−2​eθ,t2\displaystyle\tilde{r}_{t}^{\mathrm{trans}}-10I_{p,t}+10I_{p,t}I_{\theta,t}-2e_{\theta,t}^{2} |  | (4)  
---|---|---|---|---  
|  | +15​vtoward,t+10​Iθ,t.\displaystyle+15v_{\mathrm{toward},t}+10I_{\theta,t}. |   
  
Here eθ,te_{\theta,t} is the wrapped long-axis angle error, Iθ,t=𝕀⁡(eθ,t<10∘)I_{\theta,t}=\mathbb{I}(e_{\theta,t}<10^{\circ}), and vtoward,tv_{\mathrm{toward},t} is the signed angular speed toward the target, clipped to [−1,1]​rad​s−1[-1,1]\,\mathrm{rad\,s^{-1}} and set to zero inside the rotation tolerance. The joint-success reward therefore applies only when both Ip,tI_{p,t} and Iθ,tI_{\theta,t} are one. For both Allegro teachers, the PPO implementation multiplies the complete environment reward by 0.010.01 before storing it for optimization. We evaluate the saved policy with the highest recorded training return. The reference policy uses privileged observations as specified above.

### Astra Inference Protocol

#### Model and prompt.

Every direct-control rollout uses gpt-6-astra with xhigh reasoning through the Responses API. Temperature is not explicitly configured, provider fallback is disabled, and each model request has a 900-s controller timeout. The fixed developer prompt defines the robot, task, coordinate conventions, observation fields, action contract, and the requirement to preserve the grasp while making progress toward the object-level target. It contains no demonstrations or RL actions. The per-step packet explicitly states the numerical target: world-frame axis and angular speed for Sharpa, Cartesian goal for translation, and Cartesian plus signed long-axis goal for the translation-and-rotation task.

#### Observation and temporal context.

At every decision, Astra receives three synchronized RGB views together with named joint positions and velocities, the current joint target, object pose and velocity, target state, and the host’s contact representation. Sharpa supplies per-fingertip contact validity and target angular velocity in both palm and world coordinates. Allegro supplies 16 signed three-axis tactile readings, or 48 scalar values, as well as task-native position and rotation errors and threshold indicators. Images are resized to at most 480 pixels on the longest edge, except translation case 000, which uses the recorded 384-pixel setting. Reward terms and cumulative reward are logged by the host but never placed in the model observation. The runs maintain one model thread for the physical episode, so earlier decisions remain in the conversation context. Changing the model thread never resets or replays the simulator state.

#### Action execution.

The model must answer with a schema-validated tool call containing the current request identifier, one action vector, an integer repeat count, and a short action rationale. On Sharpa, the vector is a 22-dimensional normalized joint-target increment in [−0.1,0.1][-0.1,0.1]; increments accumulate and the absolute normalized target is clipped to [−1,1][-1,1] before the unchanged EMA/PD controller is applied. On Allegro, the vector has 16 entries in [−1,1][-1,1], and the native controller applies qtarget←qtarget+0.04167​aq_{\mathrm{target}}\leftarrow q_{\mathrm{target}}+0.04167a followed by joint-limit clipping. The repeat count k∈{1,…,5}k\in\\{1,\ldots,5\\} holds the requested command for kk consecutive 0.05-s control steps. The next observation is generated only after those steps have executed.

Each rollout ends when the object is dropped, the task horizon is reached, or Astra reaches its limit of 100 decisions on Sharpa and 120 decisions on Allegro. Because an Astra command may span one to five control steps, this decision limit can be reached before the task horizon; RL instead produces one action at every 20-Hz control step.

### Pairing and Evaluation Protocol

#### RL inference and paired starts.

Each frozen RL policy receives its native normalized observation and produces a deterministic action at every control step. Evaluation disables observation corruption and does not update weights, normalization statistics, or curriculum state. The RL controller never receives Astra’s action history. Pairing holds the saved physical start and goal fixed while allowing the two controllers to retain their intended observations, architectures, and inference schedules.

#### Execution endpoints.

Horizons are specified in Table 23. Early-ending trajectories contribute only executed steps, without padding. Translation-and-rotation results use three endpoints: Astra at its decision limit, RL after the same number of control steps, and RL at the full 300-step horizon. These separate performance over matched simulated durations from eventual completion; per-case lengths appear in Section H.6.

### Evaluation Metrics

#### In-hand rotation.

Let qo​(t)q_{o}(t) and qg​(t)q_{g}(t) denote the object and moving target quaternions. The benchmark’s orientation error is their geodesic distance,

| eq​(t)=2​arccos⁡(|⟨qo​(t),qg​(t)⟩|).e_{q}(t)=2\arccos\\!\left(\left|\langle q_{o}(t),q_{g}(t)\rangle\right|\right). |  | (5)  
---|---|---|---  
  
At-goal is the fraction of evaluated control steps for which eq​(t)<0.1​rade_{q}(t)<0.1\,\mathrm{rad}. To measure sustained motion, consecutive object-quaternion increments are unwrapped, projected onto world-frame +Z+Z, and averaged over a trailing one-second window. Speed MAE is the mean absolute difference between that estimate and the commanded angular speed. Orientation error, At-goal, and Speed MAE are arithmetic means over executed control steps. Full records whether the specified analysis horizon is reached, and Drop records object loss within that window.

As Section 4 explains, periodic orientation agreement need not imply sustained rotation; eq​(t)e_{q}(t) must be interpreted with axial speed. Raw cumulative reward is diagnostic and is not compared across tasks.

#### In-hand translation and reorientation.

Cylinder translation is evaluated by the Euclidean distance between the cylinder center and its Cartesian target at the final control step. Terminal success requires a position error below 22.4​mm22.4\,\mathrm{mm}. The translation-and-rotation task uses the same position threshold together with an absolute wrapped long-axis rotation error below 10∘10^{\circ} relative to the requested −40∘-40^{\circ} change. Joint success requires both conditions at the same terminal endpoint.

### Complete Per-Case Results

The tables contain all 40 rollouts from five paired starts per task. Rotation means pool executed steps; Allegro means average terminal values over cases. _Decisions_ counts accepted outputs and equals _Steps_ for RL.

“Budget” in Table 25 denotes exhaustion of Astra’s 100 decisions.

Table 25: Cylinder rotation over the 20-s analysis window. All five saved initial states are held out from the RL grasp bank. Case | Controller | End | Steps | Decisions | At-goal ↑\uparrow | Error (rad) ↓\downarrow | Speed MAE ↓\downarrow  
---|---|---|---|---|---|---|---  
000 | Astra Direct | Budget | 223 | 100 | 0.45% | 1.756 | 0.977  
| RL | Horizon | 400 | 400 | 73.25% | 0.193 | 0.208  
004 | Astra Direct | Drop | 326 | 84 | 0.31% | 1.532 | 0.906  
| RL | Horizon | 400 | 400 | 77.00% | 0.182 | 0.335  
008 | Astra Direct | Horizon | 400 | 96 | 0.75% | 1.610 | 0.957  
| RL | Horizon | 400 | 400 | 75.50% | 0.201 | 0.297  
012 | Astra Direct | Horizon | 400 | 100 | 0.50% | 1.632 | 0.940  
| RL | Horizon | 400 | 400 | 79.50% | 0.138 | 0.349  
016 | Astra Direct | Horizon | 400 | 100 | 0.50% | 1.638 | 1.017  
| RL | Horizon | 400 | 400 | 79.25% | 0.153 | 0.190  
Table 26: Cuboid rotation over the 10-s analysis window. Cases 000–003 overlap the RL initialization set; 004 is held out. Case | Controller | Split | Steps | Decisions | At-goal ↑\uparrow | Error (rad) ↓\downarrow | Speed MAE ↓\downarrow  
---|---|---|---|---|---|---|---  
000 | Astra Direct | overlap | 200 | 50 | 0.50% | 0.798 | 0.143  
| RL | overlap | 200 | 200 | 81.00% | 0.071 | 0.053  
001 | Astra Direct | overlap | 200 | 56 | 6.50% | 0.917 | 0.168  
| RL | overlap | 200 | 200 | 51.00% | 0.123 | 0.067  
002 | Astra Direct | overlap | 200 | 54 | 5.50% | 0.690 | 0.139  
| RL | overlap | 200 | 200 | 65.00% | 0.111 | 0.063  
003 | Astra Direct | overlap | 200 | 53 | 7.00% | 0.714 | 0.147  
| RL | overlap | 200 | 200 | 51.00% | 0.104 | 0.072  
004 | Astra Direct | held out | 200 | 59 | 2.50% | 0.592 | 0.117  
| RL | held out | 200 | 200 | 69.50% | 0.073 | 0.061  
Figure 19: Per-case in-hand rotation performance. Columns report mean orientation error, axial-speed MAE, At-goal rate, and cumulative environment reward. Cylinder bars use all executed steps within the 20-s horizon; cuboid bars use the complete 10-s window. Reward is comparable only within each task row. Early termination and recording lengths are reported in Tables 25 and 26. ![Refer to caption](2609.38537v1/dexterous_rollout_sequences.png) Figure 20: Paired rotation sequences from identical saved starts. (a) Cylinder case 004: Astra drops the object at 16.3 s; RL completes 20 s. (b) Cuboid case 004: Astra rotates gradually with little contact reconfiguration, while RL continues reorientation through 10 s. Axes above the hand and on the object indicate world and object frames, respectively.

Figures 19 and 20 support the contact analysis in Section 4. Contact transitions are assessed qualitatively from the trajectories.

Table 27 records no object drops during translation.

Table 27: Cylinder translation at the complete 15-s endpoint. Pass denotes terminal position error below 22.4​mm22.4\,\mathrm{mm}. Sample |  Controller |  Steps |  Decisions |  Final error (mm) ↓\downarrow |  Outcome  
---|---|---|---|---|---  
000  |  Astra Direct  |  300  |  102  |  74.73  |  Fail   
|  RL  |  300  |  300  |  10.02 |  Pass  
001  |  Astra Direct  |  300  |  79  |  19.13 |  Pass  
|  RL  |  300  |  300  |  31.12  |  Fail   
002  |  Astra Direct  |  300  |  101  |  61.67  |  Fail   
|  RL  |  300  |  300  |  9.84 |  Pass  
003  |  Astra Direct  |  300  |  100  |  37.68  |  Fail   
|  RL  |  300  |  300  |  14.99 |  Pass  
004  |  Astra Direct  |  300  |  104  |  102.06  |  Fail   
|  RL  |  300  |  300  |  20.50 |  Pass  
Table 28: Translation-and-rotation terminal results. All Astra endpoints occur after 120 decisions. Error pairs are position in millimeters / long-axis rotation in degrees. Outcomes are listed for Astra / matched RL / full-horizon RL. Sample | Astra steps | Astra endpoint | RL matched | RL full, 15 s | Outcome (A / M / F)  
---|---|---|---|---|---  
000 | 153 | 51.22 / 38.77 | 18.84 / 3.22 | 19.37 / 3.39 | Fail / Pass / Pass  
001 | 129 | 38.12 / 28.56 | 15.53 / 3.57 | 15.38 / 3.55 | Fail / Pass / Pass  
002 | 134 | 49.68 / 35.16 | 21.07 / 21.70 | 20.94 / 5.71 | Fail / Fail / Pass  
003 | 120 | 50.60 / 27.41 | 10.42 / 9.56 | 11.26 / 1.87 | Fail / Pass / Pass  
004 | 132 | 47.15 / 33.67 | 15.22 / 3.53 | 13.45 / 2.91 | Fail / Pass / Pass  
  
The terminal-error plots appear in Figure 7; Figure 21 adds representative rollout sequences.

![Refer to caption](2609.38537v1/dexterous_allegro_rollout_sequences.png) Figure 21: Qualitative Allegro rollouts from paired initial states. (a) Translation case 000. (b) Translation-and-rotation case 003, with a −40∘-40^{\circ} long-axis target. Each row shows the same front view at 0, 5, 10, and 15 s. Astra reaches its 120-decision endpoint at 6.0 s in (b), so its terminal frame remains displayed while RL continues to 15 s.

## Appendix I Navigation and Mobile Manipulation: Protocols and Cases

### Navigation Coverage and Termination

The four navigation subsets each contain 50 episodes. The VLN-CE subsets of R2R and RxR distribute four or five episodes per scene across 11 scenes. The ObjectNav subsets balance scenes and goal categories: MP3D has two or three episodes per category across 21 categories; HM3D has eight or nine across six categories. The available source pools contain 1,839 R2R, 3,669 English-guide RxR, 2,195 MP3D, and 1,000 HM3D episodes. Balancing changes the source episode distribution, so the reported means characterize these evaluation subsets.

Table 29: Navigation termination and travel. Each row has 50 episodes. Steps count primitive turns, forward moves, and STOP, rather than model requests or waypoint submissions. Mean path measures translation; turning consumes steps without adding equivalent path length. Task / dataset | Successful STOP | Unsuccessful STOP | Step limit | Mean steps | Mean path (m)  
---|---|---|---|---|---  
VLN-CE / R2R | 39 | 11 | 0 | 85.2 | 12.03  
VLN-CE / RxR | 46 | 4 | 0 | 98.1 | 12.95  
ObjectNav / MP3D | 28 | 12 | 10 | 220.8 | 26.58  
ObjectNav / HM3D | 41 | 4 | 5 | 163.2 | 16.80  
  
SPL [2] combines success SS and path efficiency as S​d/max⁡(d,ℓ)Sd/\max(d,\ell), where dd is initial shortest-path distance and ℓ\ell is traveled distance. Unsuccessful episodes contribute zero; averages use all 50 episodes in the subset. VLN-CE route agreement follows nDTW and sDTW [15], computed using FastDTW [39]; sDTW additionally requires success. ObjectNav uses annotated viewpoints, and HM3D’s 1 m results rescore saved trajectories rather than rerunning the policy.

### Released Navigation Policies: Interfaces and Complete Results

#### Fixed tasks and checkpoint selection.

The released LightNav-0 [49], Uni-NaVid [61], and OmniNav Flow [53] policies each complete the same 200 episodes as Astra, without resampling or outcome-based exclusion. Source commits and checkpoints are fixed before evaluation. An eight-episode interface check tests valid execution, not navigation success; its outcomes are excluded. The archives retain code, manifests, and episode outputs.

Table 30: Source commits and released checkpoints. LightNav-0 and Uni-NaVid use their official Hugging Face releases; OmniNav uses the official ModelScope Flow release. Uni-NaVid is the uninavid-7b-full-224-video-fps-1-grid-2 checkpoint. The open models are evaluated without local finetuning. Released policy |  Source commit |  Checkpoint commit  
---|---|---  
LightNav-0 |  3015508b70fb |  826dc5fbfa37  
Uni-NaVid 7B |  79ef5ea3fea1 |  0437222534b2  
OmniNav Flow |  e8b485953c65 |  f73a8d094f53  
  
#### Observations and history.

LightNav-0 and Uni-NaVid use the same 512×384512\times 384, 90∘90^{\circ} front camera at 0.88 m as Astra. LightNav-0 uses its official trajectory prompt and aspect-preserving image preprocessing. Uni-NaVid uses its official visual processor and online feature cache, with intermediate RGB frames retained between two-action executions. All three released policies receive the simulator’s route instruction or fixed goal-category instruction. Their input-history processing differs from Astra’s persistent agent session.

OmniNav Flow receives real, simultaneous left, right, and front images; the side cameras are oriented +90∘+90^{\circ} and −90∘-90^{\circ} relative to the front camera. Each image is rendered at 720×640720\times 640 with a 110∘110^{\circ} horizontal field of view and camera height 0.88 m, then processed by the released model’s image and history pipeline. Side views do not require additional turns or consume primitive steps. No black-image substitutes are used in these evaluation episodes. The policy receives neither simulator pose nor evaluation distance.

#### Action conversion and stopping.

All policies use the primitives, 500-step budget, sliding settings, and STOP criteria in Section 6. Collisions interrupt the proposed sequence; budget exhaustion does not force STOP.

LightNav-0’s residual-vector-quantized output is decoded into local future poses. The adapter selects the first pose with a representable forward or yaw component, rounds it to primitive actions, and executes turns before forward motion, bounded by 2 m and 180∘180^{\circ}. The lateral component is not executed directly. A decoded zero trajectory yields STOP. A nonzero trajectory below primitive resolution produces one attainable primitive along its dominant component.

Uni-NaVid retains its VLN action template and sampling temperature 0.2, requests four actions, and executes the first two, stopping the sequence if STOP occurs. Forward, left, and right map directly to the benchmark primitives. Its visual cache resets at each episode. The released implementation notes that the L3MVN-derived ObjectNav imitation-learning component is omitted; our category-goal evaluation uses this public checkpoint.22 2 <https://github.com/jzhzhang/Uni-NaVid>

OmniNav Flow predicts five local waypoints, yaw angles, and arrival scores. The adapter uses the first waypoint’s planar length and predicted yaw, quantizes them to the common primitives, and executes turns before forward motion with a 2 m limit. A non-arrival output below action resolution produces one minimum primitive. STOP requires all five arrival scores to exceed 0.5. The evaluation omits the upstream loop’s goal-distance-based stagnation rule and forced stopping. This condition uses the released vision-only Flow policy; it does not instantiate the paper’s slow exploration planner. Invalid outputs raise execution errors rather than being converted to STOP; no such errors occur in the 600 completed episodes.

#### Evaluation configuration.

The policies retain their released prompts and history processing. We replace the published LightNav-0 and Uni-NaVid camera settings with the shared front camera and discretize continuous outputs under the protocol in Section 6.

#### Published reference sources and coverage.

Tables 32 and 33 use Qwen-RobotNav Tables 1 and 4, LightNav-0 Tables III and IV, NavFoM Table 1, Uni-NaVid Tables II and III, OmniNav Table 1, and CogNav Table 1 [63, 49, 62, 61, 53, 6]. These are reported benchmark-split results under the published protocols. The source papers do not enumerate exact episode counts for every row. CogNav, for example, describes an MP3D validation set of approximately 1,800 episodes and 20 categories, whereas our 50 episodes are drawn from a 2,195-episode, 21-category pool. We therefore preserve the reported coverage rather than assign a common denominator. Qwen-RobotNav’s closed-vocabulary ObjectNav results are marked RGB without assuming a camera count from its separate OVON experiment. The literature table excludes HM3D v1 and HM3D-OVON values from the HM3D v2 column.

Table 31: Route agreement and termination on every fixed subset. Each system–dataset row contains 50 episodes. nDTW and sDTW are percentages and apply only to VLN-CE. Successes, failed STOP, and step-limit terminations sum to 50 in every row. OmniNav Flow uses the separate three-view observation condition. Task / dataset | System | Successes | nDTW | sDTW | Failed STOP | Step limit  
---|---|---|---|---|---|---  
VLN-CE / R2R | Astra | 39/50 | 72.20 | 59.35 | 11 | 0  
VLN-CE / R2R | LightNav-0 | 29/50 | 72.77 | 48.89 | 21 | 0  
VLN-CE / R2R | Uni-NaVid 7B | 15/50 | 61.76 | 26.38 | 14 | 21  
VLN-CE / R2R | OmniNav Flow (3 views) | 28/50 | 71.18 | 49.23 | 22 | 0  
VLN-CE / RxR | Astra | 46/50 | 84.73 | 80.42 | 4 | 0  
VLN-CE / RxR | LightNav-0 | 36/50 | 77.44 | 64.89 | 14 | 0  
VLN-CE / RxR | Uni-NaVid 7B | 17/50 | 56.34 | 27.73 | 19 | 14  
VLN-CE / RxR | OmniNav Flow (3 views) | 36/50 | 68.51 | 57.34 | 13 | 1  
ObjectNav / MP3D | Astra | 28/50 | – | – | 12 | 10  
ObjectNav / MP3D | LightNav-0 | 10/50 | – | – | 6 | 34  
ObjectNav / MP3D | Uni-NaVid 7B | 4/50 | – | – | 17 | 29  
ObjectNav / MP3D | OmniNav Flow (3 views) | 2/50 | – | – | 33 | 15  
ObjectNav / HM3D | Astra | 41/50 | – | – | 4 | 5  
ObjectNav / HM3D | LightNav-0 | 33/50 | – | – | 2 | 15  
ObjectNav / HM3D | Uni-NaVid 7B | 15/50 | – | – | 5 | 30  
ObjectNav / HM3D | OmniNav Flow (3 views) | 3/50 | – | – | 17 | 30  
Figure 22: Navigation observation and execution adapters. Astra, LightNav-0, and Uni-NaVid share the front-camera specification; OmniNav Flow uses three views. Habitat executes adapted outputs and returns RGB observations. Goal distances and metrics remain evaluator-only. Table 32: VLN-CE: subset-to-full-benchmark reference comparison. Published rows report R2R and RxR Val-Unseen results under the published protocols; the Astra row uses only 50 fixed episodes per dataset. Values are percentages. Each row retains its source protocol for observations, actions, and scoring. Full-benchmark scope follows each paper; exact episode counts are not supplied for every reference. |  |  R2R |  RxR  
---|---|---|---  
Method |  Observations |  SR |  SPL |  SR |  SPL  
 _Published full-benchmark references (published protocols)_  
Uni-NaVid [61] |  1 RGB |  47.0 |  42.7 |  48.7 |  40.9  
NavFoM [62] |  1 RGB |  56.2 |  51.2 |  57.4 |  49.4  
LightNav-0 [49] |  1 RGB |  68.5 |  62.8 |  73.6 |  64.5  
Qwen-RobotNav-4B [63] |  1 RGB |  66.9 |  60.5 |  71.3 |  61.5  
Qwen-RobotNav-8B [63] |  1 RGB |  65.7 |  59.6 |  73.4 |  63.5  
NavFoM [62] |  4 RGB |  61.7 |  55.3 |  64.4 |  56.2  
OmniNav [53] |  Multi-view RGB |  69.5 |  66.1 |  73.6 |  62.0  
Qwen-RobotNav-8B [63] |  Panoramic RGB |  72.1 |  66.6 |  76.5 |  65.7  
 _This work: fixed subsets, 50 episodes per dataset_  
Astra (this work) |  1 RGB |  78.00 |  65.27 |  92.00 |  77.25  
Table 33: Object-goal navigation: subset-to-full-benchmark reference comparison. Published rows report full validation-benchmark results; the Astra row uses only 50 fixed episodes per dataset. Values are percentages. Coverage and protocols differ, as in Table 32. RGB without a view count preserves the source's unspecified camera configuration. A dash denotes no result for that dataset/version; HM3D v1 and HM3D-OVON scores are excluded from HM3D v2. |  |  MP3D |  HM3D v2  
---|---|---|---  
Method |  Observations |  SR |  SPL |  SR |  SPL  
 _Published full-benchmark references (published protocols)_  
CogNav [6] |  RGB-D + pose  |  46.6  |  16.1  |  –  |  –   
LightNav-0 [49] |  1 RGB  |  53.3  |  21.2  |  79.5  |  43.7   
Qwen-RobotNav-4B [63] |  RGB  |  52.2  |  16.0  |  75.6  |  30.6   
Qwen-RobotNav-8B [63] |  RGB  |  48.8  |  17.7  |  71.2  |  33.0   
 _This work: fixed subsets, 50 episodes per dataset_  
Astra (this work) |  1 RGB  |  56.00  |  22.43  |  82.00  |  43.69   
  
### RoboCasa Task Results and Paired Outcomes

Table 34: All RoboCasa tasks and control horizons. Every method uses five matched initial states per task. Horizons count 20 Hz simulator steps. A single success changes a task’s observed rate by 20 percentage points. Group | Task | Horizon | π0.5\pi_{0.5} alone | Direct | Hybrid  
---|---|---|---|---|---  
Atomic seen | CoffeeSetupMug | 600 | 3/5 | 0/5 | 0/5  
| OpenDrawer | 750 | 3/5 | 2/5 | 3/5  
| OpenStandMixerHead | 450 | 1/5 | 0/5 | 2/5  
| PickPlaceDrawerToCounter | 750 | 2/5 | 0/5 | 3/5  
| PickPlaceSinkToCounter | 900 | 4/5 | 5/5 | 5/5  
Composite seen | DeliverStraw | 2550 | 0/5 | 1/5 | 1/5  
| KettleBoiling | 1500 | 0/5 | 1/5 | 3/5  
| PackIdenticalLunches | 3900 | 1/5 | 1/5 | 2/5  
| SearingMeat | 4350 | 0/5 | 1/5 | 1/5  
| WashLettuce | 1650 | 2/5 | 0/5 | 0/5  
Composite unseen | BreadSelection | 1950 | 0/5 | 4/5 | 2/5  
| GatherTableware | 2250 | 0/5 | 0/5 | 1/5  
| HeatKebabSandwich | 2700 | 0/5 | 2/5 | 1/5  
| MakeIceLemonade | 3000 | 0/5 | 3/5 | 2/5  
| RecycleBottlesByType | 2850 | 1/5 | 5/5 | 3/5  
All | 15 tasks | – | 17/75 | 25/75 | 29/75  
  
The task sampler sorts each frozen candidate group by sha256(20260915|group|task), selects the first five tasks, and retains episodes 0–4 and their seeds. The “seen” labels refer to the local policy-training partition. All environments use split=pretrain; task composition and kitchen-layout split are different dimensions. Horizons are frozen local reference durations multiplied by 1.5.

Table 35: Paired outcomes on 75 shared starts. Hybrid adds optional π0.5\pi_{0.5}. Two-sided exact McNemar tests [10] use discordant pairs. Exploratory pp-values are unadjusted for multiple comparisons and account for neither within-task dependence nor harness differences. Comparison (AA vs. BB) |  Both succeed |  AA only |  BB only |  Both fail |  Exact pp  
---|---|---|---|---|---  
Hybrid vs. Direct  |  16  |  13  |  9  |  37  |  0.5235   
Hybrid vs. π0.5\pi_{0.5} |  11  |  18  |  6  |  40  |  0.0227   
Direct vs. π0.5\pi_{0.5} |  7  |  18  |  10  |  40  |  0.1849   
  
The 16 pairs where both Astra conditions succeed use an average of 1,110.4 steps with Hybrid and 1,256.6 with Direct; Hybrid uses fewer steps in 11 of these pairs. This duration comparison covers the jointly successful pairs. For the atomic-group comparison between Hybrid and Direct, a Bonferroni correction [9] over the three task groups raises the nominal pp-value from 0.03125 to 0.09375.

#### Additional policy evaluation.

An additional evaluation of the multitask checkpoint reports 403/2,500 successes across 50 tasks, or 16.12%, and 132/750 on the matching 15-task subset, or 17.6%. Table 6 uses the local baseline with shared initial states and a 20-step replanning cadence.

### RoboCasa Control Configurations, Information Access, and Resources

#### Runtime and action semantics.

The recorded environment uses RoboCasa 1.0.0 [29], robosuite 1.5.2 [66], MuJoCo 3.3.1 [46], and Codex CLI 0.153.4. The policy is pi05_pretrain_human300/multitask_learning/75000, with prediction horizon 50 and zero history images. The controller is the native OperationalSpaceController, based on operational-space control [17]. Pose quaternions use x​y​z​wxyzw order. Base and torso controls are normalized to [−1,1][-1,1]; selecting arm/base mode does not automatically zero the other channels. Policy acceptance preserves its native continuous action values, whereas Direct exposes open/close gripper commands.

#### Harness configurations.

Astra uses xhigh reasoning and 20-step prefixes. Resumed trajectories can span harness configurations. The conservative version compacts context at 96,000 tokens, summarizes policy proposals, and prefetches them. The guarded version also checks rendering and permits at most one action call per response, requiring intervening feedback. Runs use different hardware and service channels.

Table 36: Harness configuration coverage for the two Astra conditions. Counts identify the configuration at the end of each trajectory; resumed trajectories may span multiple configurations. The independent policy does not use Astra context or proposal-review tools. Harness configuration |  Direct |  Hybrid  
---|---|---  
Standard, xhigh, 20 steps  |  2  |  19   
Conservative optimization  |  0  |  8   
Conservative optimization with guards  |  73  |  48   
  
#### Information-boundary audit.

An audit of 150 Astra sessions found no tool access to hidden object truth, complete physics state, reference trajectories, or action-guiding scores. It verifies mode-specific tools and zero policy queries or acceptances in Direct. Proprioception is supplied; pretraining is outside the audit. All 75 policy-only records retain proposals, actions, observations, outcomes, and matching initial-state fingerprints.

Table 37: Execution and request accounting. Policy-only prefixes execute automatically, without Astra acceptance. Retained-chain requests differ from upstream requests and action segments. Counts cover saved request logs, including recovery history, and exclude discarded attempts; they are not totals over all inference calls. Quantity |  π0.5\pi_{0.5} alone |  Direct |  Hybrid  
---|---|---|---  
Episodes |  75 |  75 |  75  
Executed simulator steps |  140,958 |  130,665 |  130,549  
Policy proposals |  7,072 |  0 |  6,185  
Astra-authored action segments |  0 |  6,560 |  2,941  
Accepted policy segments |  – |  0 |  3,621  
Recorded model requests, retained chains |  0 |  7,910 |  8,941  
Median recorded model requests per episode |  0 |  85 |  108  
  
Policy-only execution takes a median 121.6 s after loading and scene reconstruction, excluding queueing, initialization, and infrastructure time. Astra timing includes inference and recovery waits. Both Astra conditions pause physics during reasoning; see Section 9.4.

#### Reproducibility.

Archives retain outcomes, initial-state IDs, audits, images, and checks on all 75 state triples.

Experimental support, please [view the build logs](./2609.38537v1/__stdout.txt) for errors. Generated by [ L A T E xml ![\[LOGO\]](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAOCAYAAAD5YeaVAAAAAXNSR0IArs4c6QAAAAZiS0dEAP8A/wD/oL2nkwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAAd0SU1FB9wKExQZLWTEaOUAAAAddEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIFRoZSBHSU1Q72QlbgAAAdpJREFUKM9tkL+L2nAARz9fPZNCKFapUn8kyI0e4iRHSR1Kb8ng0lJw6FYHFwv2LwhOpcWxTjeUunYqOmqd6hEoRDhtDWdA8ApRYsSUCDHNt5ul13vz4w0vWCgUnnEc975arX6ORqN3VqtVZbfbTQC4uEHANM3jSqXymFI6yWazP2KxWAXAL9zCUa1Wy2tXVxheKA9YNoR8Pt+aTqe4FVVVvz05O6MBhqUIBGk8Hn8HAOVy+T+XLJfLS4ZhTiRJgqIoVBRFIoric47jPnmeB1mW/9rr9ZpSSn3Lsmir1fJZlqWlUonKsvwWwD8ymc/nXwVBeLjf7xEKhdBut9Hr9WgmkyGEkJwsy5eHG5vN5g0AKIoCAEgkEkin0wQAfN9/cXPdheu6P33fBwB4ngcAcByHJpPJl+fn54mD3Gg0NrquXxeLRQAAwzAYj8cwTZPwPH9/sVg8PXweDAauqqr2cDjEer1GJBLBZDJBs9mE4zjwfZ85lAGg2+06hmGgXq+j3+/DsixYlgVN03a9Xu8jgCNCyIegIAgx13Vfd7vdu+FweG8YRkjXdWy329+dTgeSJD3ieZ7RNO0VAXAPwDEAO5VKndi2fWrb9jWl9Esul6PZbDY9Go1OZ7PZ9z/lyuD3OozU2wAAAABJRU5ErkJggg==) ](https://math.nist.gov/~BMiller/LaTeXML/). 

## Instructions for reporting errors

We are continuing to improve HTML versions of papers, and your feedback helps enhance accessibility and mobile support. To report errors in the HTML that will help us improve conversion and rendering, choose any of the methods listed below:

  * Click the "Report Issue" ( ) button, located in the page header.



**Tip:** You can select the relevant text first, to include it in your report.

Our team has already identified [the following issues](https://github.com/arXiv/html_feedback/issues). We appreciate your time reviewing and reporting rendering errors we may not have found yet. Your efforts will help us improve the HTML versions for all readers, because disability should not be a barrier to accessing research. Thank you for your continued support in championing open access for all.

Have a free development cycle? Help support accessibility at arXiv! Our collaborators at LaTeXML maintain a [list of packages that need conversion](https://github.com/brucemiller/LaTeXML/wiki/Porting-LaTeX-packages-for-LaTeXML), and welcome [developer contributions](https://github.com/brucemiller/LaTeXML/issues).

We gratefully acknowledge support from our **major funders** , [**member institutions**](https://info.arxiv.org/about/ourmembers.html) , ****, and all contributors.

[About](https://info.arxiv.org/about) * [Help](https://info.arxiv.org/help) * [Contact](https://info.arxiv.org/help/contact.html) * [Subscribe](https://info.arxiv.org/help/subscribe) * [Copyright](https://info.arxiv.org/help/license/index.html) * [Privacy](https://info.arxiv.org/help/policies/privacy_policy.html) * [Accessibility](https://info.arxiv.org/help/web_accessibility.html) * [Operational Status (opens in new tab)](https://status.arxiv.org)

Major funding support from

[ ![Simons Foundation](/static/base/1.0.1/images/funders/simons-foundation.png) ](https://www.simonsfoundation.org/) [ ![Simons Foundation International](/static/base/1.0.1/images/funders/simons-foundation-international.png) ](https://www.sfi.org.bm/) [ ![Schmidt Sciences](/static/base/1.0.1/images/funders/schmidt-sciences.png) ](https://www.schmidtsciences.org/)

[ ](javascript:toggleReadingMode\(\); "Disable reading mode, show header and footer")
