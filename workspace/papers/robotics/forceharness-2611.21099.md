<!-- page 1 of 31 -->

arXiv:2611.21099v1 [cs.RO] 16 Nov 2026

# ForceHarness: Turning Force into Action and Physics for Frozen Frontier Models

Ke Sun Kevin Yuchen Ma Yike Zhang Yanzhe Chen Mike Zheng Shou†

Show Lab, National University of Singapore

Project page: [showlab.github.io/ForceHarness](https://showlab.github.io/ForceHarness)

![Image block](doc:537a3fa/tier:basic/page:1/block:6)

Figure 1. ForceHarness lets a frozen frontier model use wrist force. Left: the model only issues tool calls; the harness calibrates hand force, runs self-terminating force primitives beneath the model, and returns physical quantities. Middle: four real-robot tasks; each column shows one primitive call, the force logged during that call, the quantity its tool result reported, and the model’s next call with a summary of its reason. Right: success rate of GPT-6 Astra with and without force on the real robot, ForceBench, and HiddenPhys, with the best trained policy where one exists.

## Abstract

Frontier multimodal models can now control robots zeroshot from images, yet many manipulation tasks depend on contact forces that images cannot show. Bringing force to an API-only frozen model raises two challenges. First, contact evolves much faster than an API model can decide, so the system must close the loop on force beneath the model. Second, a frozen model cannot interpret raw force streams, so force must be returned in a form the model can reason about. Therefore, we present ForceHarness, a harness that lets frozen models use force without demonstrations or finetuning. After calibrating hand force from a wrist sensor or joint torques alone, ForceHarness addresses the first challenge with self-terminating force primitives that regulate force on their own and return at a contact event. To address the second challenge, a physics translation layer turns each force trace into physical quantities with uncertainty as the

tool result. To evaluate ForceHarness against expert force controllers and learned policies under identical conditions, we also build HiddenPhys, a seven-task MuJoCo benchmark with hidden physical variables. On a real arm without a wrist force sensor, ForceHarness raises the zero-shot success rate of GPT-6 Astra from 3.3% to 90.0%. In simulation, it adds 17.5 percentage points on the public ForceBench suite and 34.3 percentage points on HiddenPhys, surpassing force-aware policies trained on demonstrations on both benchmarks and matching hand-written expert controllers on HiddenPhys. Ablations further show that smaller models make significantly better decisions when given physical quantities rather than raw force values.

## 1. Introduction

Frontier multimodal models can now control robots directly without task-specific training. For example, GPT-6 Astra runs behind an API with frozen weights and still con-

<small><span class="docvortex-page-footnote" data-block-type="page_footnote" style="color:#6b7280">†Corresponding author.</span></small>

1

<!-- page 2 of 31 -->

trols a robot zero-shot from images through a pose interface [27, 43]. Much of this progress comes from the harness that wraps the frozen model and provides semantic action units [5], virtual camera views [32], agent-written tools and code [13, 17], or low-level VLA skills [44]. However, existing harnesses mainly help the model see and move.

Beyond seeing and moving, many manipulation tasks in the physical world depend on contact forces that images cannot show. For example, driving a screw requires pressing down while turning until the thread engages, and sorting identical-looking bottles requires comparing their weight. Existing work brings force into such tasks mainly through learning. Specifically, force-aware policies [15, 18, 39] and benchmarks [14] add wrist force as a network input to visuomotor policies [6] and vision-language-action (VLA) models [24]. However, this approach requires demonstrations for every task and updates model weights, so it cannot be applied to an API-only frozen model. Meanwhile, existing harnesses for frozen models that expose contact typically pass raw wrench and tactile values [7, 29] or ask the model to generate force-control code [4, 28]. Yet these designs still leave the model to parse raw sensor numbers or write low-level control without calibrated physical feedback. We therefore ask: can a frozen frontier model understand force and use it to complete manipulation tasks that cannot be done without it?

Answering this question raises two challenges. First, contact evolves much faster than an API model can decide, so the system must close the loop on force beneath the model through general skills that regulate force on their own under the model’s direction. Second, a frozen model cannot interpret raw force streams or choose actions from raw numbers alone, so the system must return force in a form the model can reason about.

Therefore, we present ForceHarness, a harness that lets a frozen frontier model use wrist force without demonstrations or training. As shown in Figure 1, ForceHarness places three layers between the model and the robot. At the bottom, a force sensing and calibration layer estimates hand force from a wrist sensor or joint torques alone and removes sensor offsets. In the middle, self-terminating force primitives address the first challenge by turning common contact skills such as guarded moves, sliding, and insertion into callable tools. Once called, each primitive regulates force on its own and returns at a contact event. At the top, a physics translation layer addresses the second challenge by converting each primitive’s force trace into physical quantities with uncertainty estimates and status flags, such as payload mass, contact height, and friction. At each turn, the model calls a primitive, reads the returned physical quantities, and decides what to do next. Together, these three layers form a general set of building blocks that bridges a frozen frontier model and low-level force control.

We evaluate ForceHarness on a real robot arm without a wrist force sensor and across two simulation benchmarks, as summarized in Figure 1. On the real robot, we design six contact-rich tasks and find that ForceHarness raises the zeroshot success rate of GPT-6 Astra from 3.3% without force to 90.0%. In simulation, the same model with ForceHarness operates from RGB images on the public ForceBench suite [14] and improves from 70.0% to 87.5%, surpassing the force-aware policies trained on that benchmark. To further compare frozen models with classical controllers and learned force policies under identical conditions, we build HiddenPhys, a seven-task MuJoCo benchmark with hidden physical variables. On HiddenPhys, ForceHarness raises the success rate of GPT-6 Astra from 62.1% to 96.4%, matching hand-written expert force controllers and far exceeding Diffusion Policy [6], π<sub>0</sub>.5 [24], and ForceVLA [39] trained on about 200 demonstrations per task.

Our contributions are:

• We propose **ForceHarness**, a three-layer harness of force calibration, self-terminating force primitives, and physics translation that bridges an API-only frozen model and low-level force control without demonstrations or training.

• We build **HiddenPhys**, a seven-task MuJoCo benchmark with hidden physical variables that includes expert force controllers and learned force policies evaluated on identical seeds.

• Experiments on a real robot and two simulated benchmarks show that ForceHarness lets frozen frontier models make full use of force for contact-rich manipulation, outperforming both training-based methods and the same harness without force.

## 2. Related work

**Harness for embodied AI.** GPT-6 Astra showed that a frozen frontier model, given images and a pose interface, can act as a zero-shot robot policy [43], and systematic evaluations have mapped what such models can and cannot do as embodied policies [11, 27]. Following Code as Policies [20], in which a language model writes robot programs over control primitives, the harness around the frozen model has become a design variable of its own. Some harnesses change how the model acts, through semantic action units [5], a task-agnostic agent harness [31], agent-written tools [13], or code that runs with no model in the loop [17]. Others change what it perceives, through calibrated perception tools [19] or rendered virtual views [32]. Others add memory of demonstrations and past interaction [21], orchestrate learned policies as skills [10, 44], or let a second agent revise the harness itself [30, 36]. However, few of these harnesses consider force: most have no force channel, and the rest mainly pass raw readings or a force cap to the agent [7–9, 37]. ForceHarness is instead built around force: it regulates force beneath the model and reports it to the model as physical quantities.

2

<!-- page 3 of 31 -->

**Force for embodied AI.** Force has been central to manipulation since hybrid position/force control [26] and impedance control [16], and force-torque measurements can identify a payload [1] or locate a contact [2], but such methods are engineered task by task. Learned policies instead take force as a network input [12, 15, 18, 23, 33, 39], and ForceBench [14] compares integration designs on π<sub>0</sub>.5; all of them require demonstrations or simulated training, and weight updates. Language models have also used force at inference time by reading sensed properties as sentences [45], generating code with force feedback [4, 22], tuning the impedance of a base policy [40, 41], or choosing recovery actions above a trained skill [25]. Recent studies give GPT-6 Astra tactile or force readings directly [29, 34, 35, 38, 46], and FORCA [28] has a coding agent write force-feedback programs and revise them between rollouts. ForceHarness differs in bringing control and perception of force together with no training: primitives regulate force beneath the model, and the model reads calibrated physical quantities after each one.

## 3. Method

## 3.1. Overview

ForceHarness sits between a frozen model and a robot and has three layers, shown in Figure 2. The bottom layer, force sensing and calibration, turns the robot’s raw signals into a calibrated estimate of the force that the environment applies to the hand. The middle layer is a menu of self-terminating force primitives, contact skills that regulate force on their own and stop at a contact event. The top layer, the physics translation layer, converts the force trace of each primitive into physical quantities with uncertainties.

Because the model is reached only through an API and needs seconds for each decision, force enters the loop twice. Inside a primitive it controls contact at the rate of the controller, and in the tool result it informs the model’s next decision. None of the layers assumes a particular arm, and Sec. A of the supplementary material lists the constants of each robot and simulator.

## 3.2. Force sensing and calibration

The bottom layer provides one signal to the layers above, the force F that the environment applies to the hand. A wrist force/torque sensor measures it directly. Many arms have no such sensor, so the layer can also recover the force from the external joint torques $\tau _ { \mathrm { e x t } }$ that a torque-controlled arm estimates. It solves for the wrench w at the flange by least squares, then removes a zero point b and flips the sign,

$$
w = ( J ^ { \top } ) ^ { + }   \tau _ { \mathrm { e x t } } , \qquad F = - ( w _ { F } - b ) ,\tag{1}
$$

where J is the flange Jacobian and $w _ { F }$ is the force part of $w .$

Such an estimate has little noise, but its zero point drifts as the arm moves and changes pose, and no filter removes a bias

Table 1. The twelve force primitives. Rules separate the five groups, which make contact, hold force, find and seat a part, turn under load, and measure. Sec. A.4 of the supplementary material specifies them and the other tools.

| Tool | What it does | Quantities returned |
| --- | --- | --- |
| move until contact, | Approach until contact | Contact height, impact, |
| probe |  | stiffness |
| push | Hold a normal force | Mean normal force |
| slide | Slide while pressing | Normal force, friction |
| search | Spiral while pressing until the part drops | Advance, lateral force |
| insert | Push in, slowing as force rises | Advance, lateral force |
| wiggle | Press while rocking | Outcome |
| screw | Press and turn with the thread | Advance per turn, side load |
| press turn | Press while turning | Resisting torque, depth |
| lift, hold | Hold the load still | Mass, center-of-mass offset |
| pull | Pull along the free path | Pull force, path radius |

that changes this slowly. The layer therefore controls when it reads rather than how it filters. It tares at rest, re-zeroing b before each empty-hand primitive so that the zero is taken close to the pose of the coming contact. After each primitive it lets the arm settle, then reads, relaxing the controller onto the measured pose and averaging the force over a short still window. Finally, it reports changes, giving each still reading relative to a reference from the same episode, such as the empty hand or the moment of the grasp, so that the model compares readings instead of trusting absolute values. In simulation, the wrist sensor is instead zeroed at reset and after reorienting the tool.

## 3.3. Self-terminating force primitives

A primitive packages one contact skill as a tool that the model calls with a few parameters, such as the direction to move and the force to hold. Table 1 lists the twelve force primitives in five groups, and ten further tools move the hand, grip, perceive, or end the episode. Three design choices let a model that decides only every few seconds use these tools.

**Force through the impedance controller.** Most arms accept position targets rather than forces, and their impedance controller acts as a spring between the hand and the target. A primitive therefore applies force $- F$ by letting a target pose $p ^ { * }$ lead the measured pose $p ,$

$$
- F \approx K \left( p ^ { * } - p \right) , \qquad \left\| p ^ { * } - p \right\| \leq L ,\tag{2}
$$

where K is the stiffness of the controller and the clamp L on this lead bounds the force that any primitive can build. Every primitive thus runs on the stock controller and leaves the low-level control of the robot unchanged.

**One shared loop.** All primitives run in the same loop and differ only in a small control law. At every tick the law

3

<!-- page 4 of 31 -->

![Image block](doc:537a3fa/tier:basic/page:4/block:1)

Figure 2. Overview of ForceHarness, illustrated with one logged call. Blue marks the command path and teal the force path. The frozen model reads its inputs and returns one tool call, here a search during a USB-plug insertion. The harness runs this call as a self-terminating force primitive that updates the target pose by Eqs. (2) and (3) at every tick and stops on a contact event. The inset sketches how search presses with $F _ { d }$ and follows a spiral until the plug drops into the hole, while the photo and the force trace come from this call. Beneath the primitive, the robot’s stock impedance controller turns the lead into force, and the bottom layer of ForceHarness estimates and re-zeros the hand force with Eq. (1). The same harness drives a real arm and two simulators, and a fixed guard bounds what any primitive can do. Fixed estimators turn the trace into physical quantities with uncertainties and flags, which return to the model as the tool result.

reads the tared force and the pose and returns an increment to $p ^ { * }$ , and the loop clamps the lead, sends the new target, and checks for contact events to decide whether to stop. To hold a force $F _ { d }$ along a direction u, the law turns the force error into a target velocity, which is admittance control,

$$
v = \mathrm { c l i p } \big ( k _ { f }   ( F _ { d } - F _ { u } ) ,   \pm v _ { \mathrm { m a x } } \big ) , \quad \Delta p ^ { * } = u   v   \Delta t ,\tag{3}
$$

where ${ \boldsymbol { F } } _ { u } = - { \boldsymbol { F } } { \boldsymbol { \cdot } } { \boldsymbol { \imath } }$ u is the reaction against u. The target keeps moving until the error vanishes, so the outer law integrates the force error while the inner spring supplies the force.

**Self-termination.** A primitive ends on an event rather than on a clock. It stops as soon as one of the contact events listed in Figure 2 occurs or its own goal is met, and a force cap and a time limit stop it if nothing else does. It then unloads the spring and holds the hand still while the model thinks. This division of work is what makes a slow decision rate workable. The model decides what to do and how hard to press, the primitive decides when to stop, and a fixed guard beneath both enforces limits that the model cannot change.

## 3.4. A representative primitive: search

We illustrate the design with search, the primitive in Figure 2, which finds a hole whose position the model knows only

roughly. The model chooses how hard to press, how wide to search, and how deep the part must sink. The primitive moves toward the surface until it touches it and then keeps pressing with the target force $F _ { d }$ through Eq. (3), while the target travels outward on an Archimedean spiral in the plane of the surface, as the inset of Figure 2 shows. When the part passes over the hole, the reaction falls, the spiral pauses, and the same law drives the part down. The primitive reports that the part dropped in once it has sunk past the set depth, or that no hole was found if the spiral leaves the search range first, and in both cases it also reports how far the part advanced and the lateral force at the end. Most other primitives follow the same pattern of one force law along the contact normal plus their own motion and stop conditions, and Sec. A.4 of the supplementary material specifies them one by one.

## 3.5. Physics translation layer

A frozen model has not learned to read raw force, and the hundreds of samples that a primitive logs give it little basis for its next decision. The top layer therefore translates each force trace into a few physical quantities, the numbers a person would read off the trace, and returns them as the tool result. Its estimators are deliberately simple and fixed, built only from averages and least-squares fits with propagated

4

<!-- page 5 of 31 -->

errors. Every quantity follows the recipe shown as a strip in Figure 2. The layer removes the zero and any held load, keeps only the window in which the quantity has a single meaning, and reports the value with its uncertainty σ and with flags that warn, for instance, of lost contact. It contains no task thresholds, so it states how heavy a load is or how far a plug advanced but never whether that is enough, which is left to the model.

For example, payload mass comes from a still hold and friction from the part of a slide that stays in contact,

$$
m = | \bar { F } _ { z } | / g , \quad \sigma _ { m } = \sqrt { s _ { z } ^ { 2 } / n + \sigma _ { b } ^ { 2 } } \: / \: g ,\tag{4}
$$

$$
\mu = \overline { { | F _ { t } | } } / \bar { F } _ { n } , \quad ( \sigma _ { \mu } / \mu ) ^ { 2 } = \left( \sigma _ { t } / \overline { { | F _ { t } | } } \right) ^ { 2 } + ( \sigma _ { n } / \bar { F } _ { n } ) ^ { 2 } ,\tag{5}
$$

where $s _ { z }$ is the sample standard deviation of the vertical force $F _ { z }$ over the n samples, $\sigma _ { b }$ is the residual zero bias, and $\sigma _ { t }$ and $\sigma _ { n }$ are the standard errors of the tangential force $F _ { t }$ and the normal force $F _ { n }$ . Contact height is corrected for the lag of the contact detector, stiffness is the least-squares slope of normal force over penetration, and insertion advance carries a flag that says whether the part dropped in or jammed. Sec. B.2 of the supplementary material gives every quantity.

## 3.6. Model in the loop

Figure 2 shows one turn of this loop from a real USB-plug insertion. At every turn the model reads the instruction, the tool menu, the call history, and the latest observations, and it answers with one tool call. Instructions are written by hand for each task and never reveal the hidden variables of an episode. A task usually starts without force, as the model points at a target in the image, locate turns that pixel into a 3D point from depth, and position moves bring the hand close. The model then switches to force primitives and sets their parameters from what it has seen and read so far. Each primitive stops on its own, and its force comes back translated into physical quantities as the tool result. From these quantities the model judges what happened, for example whether a part has seated or needs another attempt, and it chooses the next call until it ends the episode. Calibration and safety stay beneath the model, because the bottom layer keeps the force zeroed by the rules of Sec. 3.2 and the fixed guard limits force, speed, and workspace for every primitive whatever the model asks.

## 4. The HiddenPhys benchmark

Public force benchmarks such as ForceBench [14] are built to train and compare learned policies. We complement them with HiddenPhys, a benchmark of seven manipulation tasks in robosuite on MuJoCo. Its tasks come from real-world contact scenarios and add fine manipulation that ForceBench does not cover.

Table 2. The seven tasks of HiddenPhys. Success is checked in code on the final state, and Sec. C.2 of the supplementary material gives the full task cards.

| Task | Hidden variable | What force must do |
| --- | --- | --- |
| Weight CoM | Cube mass, 0.15 or 0.60 kg Bar's center of mass, centered or Measure the held moment | Weigh the held cube |
| Wipe | 4 cm off Table height, up to about 6 cm low Press 5 to 15 N on 12 spots |  |
| Stack Peg | Lower cube's top, known to ±1 cm Feel contact, place gently Hole position, 15 mm error per axis Search until the peg drops |  |
| Latch |  |  |
| Lock | Unlock angle, 15 to 35° Locked, or stiff and freed at 6–11 N Pull below 15 N | Press to release, then pull |

**Tasks.** Each task hides one physical variable that decides success, as listed in Table 2. In Weight and CoM the robot must judge a load that it is holding, and in Wipe and Stack it must find a surface whose height is uncertain. Peg asks it to find a hole by touch, while Latch and Lock ask it to operate a mechanism without damaging it. The instruction states the range of the hidden variable but never its value, which is drawn from the episode seed, so every method faces the same episodes.

**Setting and references.** A Panda arm runs operationalspace control, and force is the wrist sensor reading with added noise and a per-episode bias. To isolate force from visual perception, a frozen model reads object poses as noisy text instead of images, while the real robot and ForceBench cover camera input. HiddenPhys ships with three kinds of reference. The expert force controller is a hand-written state machine for each task, and the position-only controller makes the same motions without using force. The learned policies are Diffusion Policy [6] and $\pi _ { 0 . 5 }$ [24], each trained with and without force input, together with a controlled multi-task variant of ForceVLA [39], and all of them learn from about 200 expert demonstrations per task. Besides success, HiddenPhys counts damage, an episode in which the force on a handle passes its damage threshold. Sec. C of the supplementary material gives the task cards, success checks, and seeds, and Sec. G gives the training recipes.

## 5. Experiments

We evaluate ForceHarness on a real robot, on the public ForceBench suite, and on HiddenPhys, shown in Figure 3. In each setting, the same frozen model runs with ForceHarness and with a no-force control, which keeps the harness and its safety limits but removes force primitives and force readings.

## 5.1. Real-robot experiments

**Experimental setting.** The robot is a Franka FR3 with a parallel gripper and no wrist force/torque sensor, so force is estimated from joint torques with Eq. (1). Its stock joint-

5

<!-- page 6 of 31 -->

![Image block](doc:537a3fa/tier:basic/page:6/block:1)

Figure 3. The three experimental settings, with one frame per task, cover six tasks on a real arm without a wrist force sensor, the eight tasks of ForceBench in Isaac Lab, and the seven tasks of HiddenPhys in MuJoCo. The colored zones in Weight and CoM are drawn for display only.

Table 3. Real-robot results: successes out of 10 episodes per condition, with the same instructions. For Astra, every task differs at p < 0.01 by Fisher’s exact test.

<table><tr><td rowspan="2">Task</td><td colspan="2">GPT-6 Astra</td><td rowspan="2">GPT-6.1 Sol ForceHarness</td></tr><tr><td>No force</td><td>ForceHarness</td></tr><tr><td>Wipe a whiteboard</td><td>1/10</td><td>8/10</td><td>6/10</td></tr><tr><td>Drive a screw</td><td>0/10</td><td>9/10</td><td>7/10</td></tr><tr><td>Peel an eggplant</td><td>1/10</td><td>9/10</td><td>9/10</td></tr><tr><td>Insert a USB plug</td><td>0/10</td><td>8/10</td><td>7/10</td></tr><tr><td>Sort three bottles by weight</td><td>0/10</td><td>10/10</td><td>8/10</td></tr><tr><td>Tell soft from hard</td><td>0/10</td><td>10/10</td><td>10/10</td></tr><tr><td>Total</td><td>2/60</td><td>54/60</td><td>47/60</td></tr></table>

impedance controller runs unchanged, and a wrist RGB-D camera and an exterior RGB camera give the observations. We test GPT-6 Astra in both conditions and check transfer to GPT-6.1 Sol with ForceHarness, both through the Codex CLI at medium reasoning effort. Each model receives the text history and the images of the last two turns.

We design six real-world tasks listed in Table 3. Four contact tasks require regulating force during motion, either to wipe all marker writing off a whiteboard and peel a strip of skin from an eggplant, or to seat a screw in a threaded hole and fully insert a USB plug. The other two tasks test properties that cameras cannot see. Bottle sorting requires placing the heaviest of three look-alike bottles on a white plate and the lightest on a green plate while leaving the middle one in place, and telling soft from hard requires probing two identically wrapped objects to name each one correctly. Across these tasks, ForceHarness offers 20 to 21 tools, whereas the no-force control keeps only the ten other tools and hold with kinematics-only results, plus a guarded move in peeling that stops at a fixed force without reporting it. Instructions include procedural hints except in telling soft from hard, where no method is given. Both conditions share the same safety limits with an operator at the emergency stop. Each task runs 10 episodes per condition with the same model and instruction after a scene reset, and the operator labels success from the final state. An

external web monitor replays every episode, as Sec. E of the supplementary material shows.

**Results.** In Table 3, ForceHarness raises GPT-6 Astra from 2/60 without force to 54/60 across all six tasks, while GPT-6.1 Sol reaches 47/60 under the same prompts. On bottle sorting and telling soft from hard, RGB-D images give no cue to mass or stiffness, leaving the no-force control at 0/10 on both. On the sensorless FR3, joint-torque estimates drift with arm pose and misstate absolute grams, yet taring at rest, settling before reading, and comparing within-episode changes keep relative order reliable. This lets Astra score 10/10 on both tasks and Sol score 8/10 and 10/10, even when no probing method is given for soft versus hard.

On the four contact tasks, wrist depth localizes targets only to about 1 to 2 cm, so position-only steps either miss the surface or hit the collision guard without knowing the contact load, succeeding in only 2 of 40 episodes. With ForceHarness, force acts both inside and between primitives, as Figure 1 illustrates. Inside each call, guarded moves and admittance control absorb height and pose error, holding steady contact across the flat board or curved eggplant and spiraling at 1 N until a resistance drop guides the USB plug into its port. Between calls, tool results and images guide adaptation: remaining ink after an 8 N wipe prompts Astra to retry at 10 N, rising side load during screwing triggers an inplace release before binding, and a 2.77 N mean force over a 51 mm peel tells Astra to keep its 6 N target. Calibrated jointtorque sensing, self-terminating primitives, and physical quantities thus make zero-shot contact manipulation reliable without a wrist sensor.

## 5.2. ForceBench

**Experimental setting.** ForceBench [14] is a public Isaac Lab suite of eight contact-rich tasks, where we keep its evaluator and run its first ten seeds. Our harness serves as the policy with the same primitives, estimators, and a few suite-specific tools, reading two RGB views, object sizes, and key heights without object coordinates. Instructions state task facts that demonstrations expose, such as the two mass

6

<!-- page 7 of 31 -->

Table 4. ForceBench success in percent. The trained rows are selected from the ForceBench paper and not reproduced, with 200 demonstrations and 200 evaluation episodes per task, and Sec. H of the supplementary material lists all ten. The zero-shot rows are ours, with 10 episodes per task.

| Method | Weight | CoM | Faucet | Faucet dir. | Stove | Stove press | Drawer | Nut | Mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| π0.5, no force | 54.0 | 45.5 | 9.5 | 31.5 | 8.5 | 1.0 | 4.5 | 31.5 | 23.25 |
| π<sub>0.5</sub> + Prompt Fusion | 88.0 | 56.5 | 21.0 | 41.5 | 16.0 | 2.5 | 23.0 | 34.0 | 35.31 |
| π<sub>0.5</sub> + Dual-Path Fusion | 99.5 | 72.0 | 22.0 | 30.0 | 17.5 | 19.0 | 4.5 | 16.5 | 35.13 |
| Trained |  |  |  |  |  |  |  |  |  |
| π<sub>0</sub> [3] | 61.5 | 39.0 | 3.5 | 27.0 | 7.0 | 5.0 | 7.5 | 36.0 | 23.31 |
| ForceVLA | 35.5 | 45.5 | 12.5 | 14.0 | 10.0 | 12.0 | 20.0 | 26.0 | 21.94 |
| ForceFlow [42] | 51.5 | 3.0 | 0.0 | 0.5 | 0.0 | 0.0 | 0.0 | 0.0 | 6.88 |
| GPT-6 Astra, no force | 80 | 50 | 100 | 50 | 100 | 80 | 0 | 100 | 70.00 |
| GPT-6 Astra + ForceHarness | 100 | 70 | 100 | 50 | 100 | 90 | 100 | 90 | 87.50 |
| Zero-shot GPT-6.1 Sol, no force | 90 | 50 | 90 | 30 | 100 | 60 | 0 | 100 | 65.00 |
| GPT-6.1 Sol + ForceHarness | 100 | 60 | 100 | 40 | 100 | 100 | 100 | 100 | 87.50 |
| Claude Fable 5.2 + ForceHarness | 100 | 80 | 100 | 20 | 100 | 50 | 100 | 50 | 75.00 |

classes, and the no-force control adds a height-following turn tool so it lacks force rather than motion. GPT-6 Astra and GPT-6.1 Sol run at low reasoning effort, and Claude Fable 5.2 runs with ForceHarness at high effort.

**Results.** In Table 4, Astra and Sol reach 87.5% with Force-Harness and Fable reaches 75.0%, above 35.31% for the best trained policy. Since Astra and Sol already score 70.0% and 65.0% without force through the model, harness, and instructions, that gap is not from force alone. Within our harness, force brings paired gains at McNemar’s p of 0.009 and 0.001 mainly on Drawer Pull from 0/10 to 10/10, while other tasks do not separate at 10 episodes, showing clean transfer when subtle resistance decides success.

Two protocol notes apply. First, tools and perception thresholds were tuned on eight of the ten seeds, and Stove Press and Nut Thread were revised after initial runs. Second, on Faucet Turn, Stove Turn, and Drawer Pull the public evaluator omits checks such as release and clearance.

## 5.3. HiddenPhys

**Experimental setting.** GPT-6 Astra runs at low reasoning effort and Claude Fable 5.2 runs at high effort, each with at most 40 turns, 80 calls, and 300 simulated seconds per episode. All rows of Table 5 share the 20 test seeds per task, the hidden variables, and the success checks of Sec. 4. Only the learned policies also see two camera views, which show them the true hole on Peg. Because learned policies have no force cap, we report damage for every row. The classical rows confirm that the tasks depend on force: over 50 seeds in Sec. C.4, the position-only controller falls from 49 expert successes to none on Wipe and from 50 to 25 on Weight.

**Main results.** In Table 5, both Astra and Fable with Force-Harness reach 135/140 with zero handle damage, matching the hand-written expert at 136/140. All trained policies score only 65/140 to 83/140 with frequent damage, and adding

wrist force to Diffusion Policy, $\pi _ { 0 . 5 }$ , or ForceVLA brings no consistent gain because passive conditioning neither teaches active probing nor caps force.

Within our harness, force lifts Astra from 87/140 to 135/140 and Fable from 96/140 to 135/140, with consistent gains across all eight OpenAI models in Sec. D.4 of the supplementary material. Without force, Astra partially copes on Wipe and Weight by small position steps, while Fable takes six times as many calls and Astra overloads half of its Lock handles. ForceHarness eliminates handle damage and cuts Astra’s cost per episode by 45%, matching task-specific force controllers safely.

**Distribution shift.** We next shift the hidden ranges on four tasks without retraining while updating instructions only for the frozen models, as detailed in Sec. D.7 of the sup plementary material. With ForceHarness, Astra and Fable show relative drops of only 8% and 5% to 69/80 and 71/80, matching the fixed-threshold expert at 68/80. Without force they lose 16% to 19% and learned policies lose 17% to 25%, aside from image-only Diffusion Policy losing 5% from a low 44/80 start as its wrist camera sees the Peg hole.

## 5.4. Ablation studies

To see where the gains of ForceHarness come from, we ablate its force primitives, feedback form, and decision loop on HiddenPhys. Table 6 compares seven feedback forms on identical primitives and seeds, Figure 4 splits the gains by task group and loop structure, and Sec. D of the supplementary material gives per-task results.

**Force primitives.** The event-words row in Table 6 keeps our force primitives but reports only events with |F|. Even without force traces, these primitives raise success from 0.63 to 0.77 for Sol and from 0.18 to 0.54 for smaller GPT-5.x models, mainly on the contact tasks in Figure 4a, while cutting Lock handle damage for smaller models from 22

7

<!-- page 8 of 31 -->

Table 5. Results on HiddenPhys as successes out of 20 test seeds per task. Damage counts episodes in which the force on a handle passes its damage threshold. Demos is the number of demonstrations per task, and Code marks task-specific code. DP trains one model per task, while π0.5 and ForceVLA train one multi-task model, each with one training seed. Shift gives successes out of 80 on four tasks, in distribution and under shift.

| Method | Weight | CoM | Wipe | Stack | Peg | Latch | Lock | Total | Damage | Demos | Code | Shift |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Expert force controller | 20 | 19 | 20 | 20 | 18 | 19 | 20 | 136 | 1 | 0 | ✓ | 77 → 68 |
| Classical |  |  |  |  |  |  |  |  |  |  |  |  |
| Position only | 10 | 10 | 0 | 17 | 5 | 13 | 16 | 71 | 6 | 0 | ✓ | 25 → 22 |
| DP | 16 | 17 | 4 | 11 | 15 | 9 | 11 | 83 | 20 | 200 | - | 52 → 40 |
| DP + force | 18 | 17 | 7 | 6 | 15 | 1 | 10 | 74 | 13 | 200 | - | 57 → 43 |
| DP, images only | 9 | 17 | 4 | 12 | 14 | 0 | 9 | 65 | 25 | 200 | - | 44 → 42 |
| Learned DP, images only + force | 16 | 17 | 5 | 10 | 14 | 3 | 9 | 74 | 16 | 200 | - | 52 → 43 |
| π<sub>0</sub>.5 | 9 | 9 | 14 | 12 | 12 | 2 | 11 | 69 | 24 | 200 | - | 44 → 35 |
| π<sub>0.5</sub> + force, Prompt Fusion | 15 | 10 | 7 | 14 | 10 | 0 | 11 | 67 | 18 | 200 | - | 42 → 35 |
| ForceVLA | 11 | 10 | 12 | 12 | 15 | 2 | 8 | 70 | 29 | 200 | - | 48 → 36 |
| GPT-6 Astra, no force | 17 | 10 | 16 | 20 | 1 | 13 | 10 | 87 | 17 | 0 | - | 44 → 37 |
| GPT-6 Astra + ForceHarness | 20 | 20 | 20 | 20 | 15 | 20 | 20 | 135 | 0 | 0 | - | 75 → 69 |
| Ours |  |  |  |  |  |  |  |  |  |  |  |  |
| Claude Fable 5.2, no force | 10 | 18 | 17 | 20 | 3 | 18 | 10 | 96 | 11 | 0 | - | 48 → 39 |
| Claude Fable 5.2 + ForceHarness | 20 | 20 | 20 | 20 | 15 | 20 | 20 | 135 | 0 | 0 | - | 75 → 71 |

Table 6. Form of force. L2, our default, adds ± σ, window, and flags to L1. Success is the share of episodes passing the task check, decision the share of correct verdicts on Weight, CoM, and Lock. Sol runs 70 episodes per row, the five smaller models 125.

<table><tr><td rowspan="2">Form</td><td colspan="2">GPT-6.1 Sol</td><td colspan="2">Smaller models</td></tr><tr><td>Success</td><td>Decision</td><td>Success</td><td>Decision</td></tr><tr><td>No force</td><td>0.63</td><td>0.73</td><td>0.18</td><td>0.57</td></tr><tr><td>Event words</td><td>0.77</td><td>0.70</td><td>0.54</td><td>0.64</td></tr><tr><td>Force plot</td><td>0.90</td><td>1.00</td><td>0.56</td><td>0.71</td></tr><tr><td>Raw numbers</td><td>0.87</td><td>1.00</td><td>0.63</td><td>0.69</td></tr><tr><td>Derived signals</td><td>0.87</td><td>0.97</td><td>0.69</td><td>0.85</td></tr><tr><td>Quantities, L1</td><td>0.91</td><td>1.00</td><td>0.64</td><td>0.84</td></tr><tr><td>+ uncertainty, L2</td><td>0.90</td><td>1.00</td><td>0.70</td><td>0.85</td></tr></table>

of 25 episodes to 3. Self-terminating primitives alone thus secure safe contact execution beneath the model.

**Physics translation layer.** Judging a held load in Figure 4b gives no stop event, so event words fail and force values are needed. Table 6 compares returning force plots, tared raw numbers, or harness-computed signals and quantities. Sol reaches 0.87 to 0.91 across all three. For smaller models, force plots reach only 0.56 success, and raw numbers lift decision accuracy insignificantly from 0.57 to 0.69. L2 quantities raise decision accuracy to 0.85 at McNemar’s p of 0.023 over raw numbers and lift success to 0.70, beating plots at p of 0.005 and gaining modestly over 0.63 for raw numbers since primitives drive contact tasks. Derived signals, L1, and L2 match closely, showing that smaller models need harnesscomputed quantities rather than plots or raw samples.

**Model in the loop.** In Figure 4c, we compare turn-by-turn calls with one-shot Code-as-Policies [20] programs written

![Chart block](doc:537a3fa/tier:basic/page:8/block:8)

Figure 4. Ablations on HiddenPhys. (a) Contact tasks, without Peg and Latch for GPT-5.x. (b) Weight and CoM. (c) GPT-6 Astra, five seeds per task.

by Astra. Our force interface lifts one-shot programs from 23% to 77% though only our harness was tuned, and turnby-turn feedback reaches 91% by adapting after failures on Stack and Peg.

## 6. Conclusion

We presented ForceHarness, a three-layer harness that lets an API-only frozen frontier model use wrist force with no demonstrations or weight updates. Across a real arm without a wrist sensor, ForceBench, and HiddenPhys, keeping force calibration, admittance control, and event termination beneath the model while returning calibrated physical quantities lets frozen models match hand-written expert controllers and surpass policies trained on demonstrations. Ablations show that self-terminating primitives keep contact safe without force values, physical quantities help smaller models judge a held load, and calling the model after each primitive lets it recover from failures.

**Limitations.** Model queries add seconds between primitives and preclude reactive coordination, a fixed menu limits new motion laws, and our ablations run only in simulation.

8

<!-- page 9 of 31 -->

## References

[1] Christopher G. Atkeson, Chae H. An, and John M. Hollerbach. Estimation of inertial parameters of manipulator loads and links. Int. J. Robot. Res., 5(3):101–119, 1986.

[2] Antonio Bicchi, J. Kenneth Salisbury, and David L. Brock. Contact sensing from force measurements. Int. J. Robot. Res., 12(3):249–262, 1993.

[3] Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, et al. π0: A vision-languageaction flow model for general robot control. arXiv preprint arXiv:2410.24164, 2024.

[4] Kaylee Burns, Ajinkya Jain, Keegan Go, Fei Xia, Michael Stark, Stefan Schaal, and Karol Hausman. GenCHiP: Generating robot policy code for high-precision and contact-rich manipulation tasks. In IEEE/RSJ Int. Conf. Intell. Robots Syst., pages 9596–9603, 2024.

[5] Yanzhe Chen, Zechen Bai, Zhijun Cao, Wenzheng Zeng, Kevin Qinghong Lin, Yiqi Lin, Guoqiang Liang, Kevin Yuchen Ma, Qiming Huang, and Mike Zheng Shou. Show-Harness: Just a VLM agent can play robots. arXiv preprint arXiv:2609.10522, 2026.

[6] Cheng Chi, Siyuan Feng, Yilun Du, Zhenjia Xu, Eric Cousineau, Benjamin Burchfiel, and Shuran Song. Diffusion policy: Visuomotor policy learning via action diffusion. In Robot.: Sci. Syst., 2023.

[7] Zhaoyang Chu, Earl T. Barr, Claire Le Goues, Peter O’Hearn, Mark Harman, Federica Sarro, and He Ye. OpenRUA: Robotuse agents are zero-shot visuomotor policies. arXiv preprint arXiv:2610.02459, 2026.

[8] Zijie Diao, Yitong Chen, Sicheng Xie, Tianyi Lu, Wujian Peng, Guojin Zhong, Houze Xu, Ziyi Ye, Zuxuan Wu, and Yu-Gang Jiang. LIBERO-Agent: Evaluating general-purpose agents for direct embodied manipulation. arXiv preprint arXiv:2609.39507, 2026.

[9] Letian Fu, Justin Yu, Karim El-Refai, Ethan Kou, Haoru Xue, Huang Huang, et al. CaP-X: A framework for benchmarking and improving coding agents for robot manipulation. In Int. Conf. Mach. Learn., 2026.

[10] Liane Galanti, Dhruv Shah, and Tri Dao. Addressing the orchestration gap in generalist robots via physical agency. arXiv preprint arXiv:2607.21725, 2026.

[11] Galbot Team, Xuchuan Chen, Xiaoqian Cheng, Yu Deng, Lihe Ding, Shaocong Dong, et al. Systematically exploring the capabilities of GPT-6 Astra as embodied policies. arXiv preprint arXiv:2609.38537, 2026.

[12] Bohan Gan, Xuanzhang Wen, Yongsheng Zhao, Baoping Cheng, Wenhe Jia, Ye Wang, et al. TAO-Force: Unifying force-aware perception and fast-slow control for contact-rich manipulation. arXiv preprint arXiv:2609.18497, 2026.

[13] Shijia Ge, Alex Zhou, Jianshu Zeng, Yexing Wan, Di Wu, Zelin Zheng, Yazhe Wang, Zhiqi Jia, Xuan Shangguan, Jay Zhu, Yijun Liu, Lingyu He, Sihang Wu, Xiao He, and Hongcheng Gao. Make code as policy great again: Frontier agents write, call, and evolve robot tools. arXiv preprint arXiv:2609.39018, 2026.

[14] Xuan Han, Zinuo Jin, Tianlin Li, Yunfan Lou, Xin Liu, Yuchen Xue, Yuanpeng He, Boxian Yao, Shuhan Zhan, Rui-

hai Wu, and Zhibo Pang. ForceBench: Benchmarking force integration for robotic manipulation. arXiv preprint arXiv:2610.19604, 2026.

[15] Zihao He, Hongjie Fang, Jingjing Chen, Hao-Shu Fang, and Cewu Lu. FoAR: Force-aware reactive policy for contactrich robotic manipulation. IEEE Robot. Autom. Lett., 10(6): 5625–5632, 2025.

[16] Neville Hogan. Impedance control: An approach to manipulation: Part I—Theory. J. Dyn. Syst. Meas. Control, 107(1): 1–7, 1985.

[17] Kairui Hu, Siyuan Hu, Fangzhou Hong, Zhaoxi Chen, and Ziwei Liu. Embodied Turing machines: Stateful code for robot recursive self-improvement. arXiv preprint arXiv:2610.12369, 2026.

[18] Yang Li, Zhaxizhuoma, Hongru Jiang, Junjie Xia, Hongquan Zhang, Jinda Du, et al. ForceVLA2: Unleashing hybrid force-position control with force awareness for contact-rich manipulation. In IEEE Conf. Comput. Vis. Pattern Recog., 2026.

[19] Zexi Li, Yehang Zhang, Wenqian Li, Haojian Huang, Chenxu Wang, Shiyuan Deng, et al. Robo-Harness K1: Harnessing robot-use agents via perception augmentation. arXiv preprint arXiv:2609.29389, 2026.

[20] Jacky Liang, Wenlong Huang, Fei Xia, Peng Xu, Karol Hausman, Brian Ichter, Pete Florence, and Andy Zeng. Code as policies: Language model programs for embodied control. In IEEE Int. Conf. Robot. Autom., pages 9493–9500, 2023.

[21] Fangcheng Liu, Yeqing Shen, Anda Cheng, Weishi Mi, Chao Tang, Chenyuan Liu, Yushun Xiang, Tingguang Li, Yong-Lu Li, and Yehui Tang. RoboICL: Embodied in-context learning with GPT-6 Astra. arXiv preprint arXiv:2609.34261, 2026.

[22] Ruaridh Mon-Williams, Gen Li, Ran Long, Wenqian Du, and Christopher G. Lucas. Embodied large language models enable robots to complete complex tasks in unpredictable environments. Nat. Mach. Intell., 7(4):592–601, 2025.

[23] Michael Noseworthy, Bingjie Tang, Bowen Wen, Ankur Handa, Chad Kessens, Nicholas Roy, Dieter Fox, Fabio Ramos, Yashraj Narang, and Iretiayo Akinola. FORGE: Force-guided exploration for robust contact-rich manipulation under uncertainty. IEEE Robot. Autom. Lett., 10(5): 4436–4443, 2025.

[24] Physical Intelligence, Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, et al. π0.5: A vision-language-action model with open-world generalization. In Conf. Robot Learn., 2025.

[25] Kyupaeck Jeff Rah and Midum Oh. FORGE-plus: Forcebudgeted recovery for contact-rich assembly with a frozen LLM supervisor. arXiv preprint arXiv:2607.21227, 2026.

[26] Marc H. Raibert and John J. Craig. Hybrid position/force control of manipulators. J. Dyn. Syst. Meas. Control, 103(2): 126–133, 1981.

[27] Robocurve. GPT-6 Astra on robot arms. Third-party evaluation report, [https://openai.robocurve.org/gpt-6-astra/](https://openai.robocurve.org/gpt-6-astra/), 2026.

[28] Xiao Tan, Yixuan Wang, Zhexi Wang, and Wenzhen Yuan. FORCA: A force-aware coding agent for contact-rich robot manipulation. arXiv preprint arXiv:2610.31841, 2026.

9

<!-- page 10 of 31 -->

[29] Kazutoshi Tanaka. TacZero: Training-free peg insertion using a general-purpose vision-language model with tactile feedback. arXiv preprint arXiv:2610.07621, 2026.

[30] Wenqing Tian, Zeyu Zhang, Zhaocheng Liu, Fengwei Liu, Qiang Liu, and Liang Wang. PhysEvo: Astra can act, let it. arXiv preprint arXiv:2610.08995, 2026.

[31] Chengcheng Wang, Yuheng Shi, Siyu Xu, Yutong Wang, Jinjing Zhao, Chenghao Xia, et al. Auriga technical report: A task-agnostic agent harness for zero-shot embodied manipulation. arXiv preprint arXiv:2611.01112, 2026.

[32] Jiayu Wang, Yue Yu, Bin Zhu, Zhiyao Yang, and Jingjing Chen. SpatialHarness: Test-time spatial scaffolding for fine robotic manipulation. arXiv preprint arXiv:2610.12457, 2026.

[33] Yi Wang, Wendi Chen, Zimo Wen, Han Xue, Xueqi Li, Wenye Yu, et al. Never too late for force: Accelerating VLA post-training with reactive force injection. In Conf. Robot Learn., 2026.

[34] Yubang Wang, Zezheng Huai, Yifei Zhang, Jinhao Wu, Yitong Chen, Haozhe Zhang, et al. Embodied task agents can perform fine manipulation through tactile in-context learning. arXiv preprint arXiv:2610.27613, 2026.

[35] Ziyong Wang, Yupei Lin, Runxuan Huang, Liang Lin, Dongyu Zhang, and Guangrun Wang. GPT-6-Astra understands touch and outperform vision–tactile–language–action models. arXiv preprint arXiv:2610.28457, 2026.

[36] Zhijie Wei, Ferris Tan, Jinghui Wang, and Hao Zhou. Scale and selection: What makes automatic harness evolution work for visual-interface robot agents. arXiv preprint arXiv:2609.39304, 2026.

[37] Sicheng Xie, Yitong Chen, Haidong Cao, Shunlin Lu, Zuxuan Wu, and Yu-Gang Jiang. Explore, execute, evolve: A skill acquisition and reuse loop for embodied agents. arXiv preprint arXiv:2609.37810, 2026.

[38] Chen Yang, Linzhe Shi, Hang Zhang, Mu Xu, Jiansheng Fan, and Chen Wang. GPT-6 Astra is a native visuotactile policy for contact-rich manipulation. arXiv preprint arXiv:2610.26846, 2026.

[39] Jiawen Yu, Hairuo Liu, Qiaojun Yu, Jieji Ren, Ce Hao, Haitong Ding, Guangyu Huang, Guofan Huang, Yan Song, Panpan Cai, Cewu Lu, and Wenqiang Zhang. ForceVLA: Enhancing VLA models with a force-aware MoE for contactrich manipulation. In Adv. Neural Inform. Process. Syst., 2025.

[40] Heng Zhang, Wei-Hsing Huang, Gokhan Solak, and Arash Ajoudani. OmniVIC: A self-improving variable impedance controller with vision-language in-context learning for safe robotic manipulation. In IEEE-RAS Int. Conf. Humanoid Robots, 2026.

[41] Heng Zhang, Wei-Hsing Huang, Qiyi Tong, Gokhan Solak, Puze Liu, Kaidi Zhang, Sheng Liu, Jan Peters, Yu She, and Arash Ajoudani. CompliantVLA-adaptor: VLM-guided variable impedance action for safe contact-rich manipulation. arXiv preprint arXiv:2601.15541, 2026.

[42] Shuoheng Zhang, Yifu Yuan, Hongyao Tang, Yan Zheng, Qiaojun Yu, et al. ForceFlow: Learning to feel and act via contact-driven flow matching. arXiv preprint arXiv:2605.11048, 2026.

[43] Wenbo Zhang, Kaixuan Wang, Yutao Ouyang, Xiaoyu Huang, Liyang Li, Kailun Su, Weiyang Jin, Wenhao Chai, Haotian Liang, Zhiyang Dou, Yue Chen, and Tianxing Chen. An unexpected robot policy: Early evaluations of GPT-6 Astra on RoboDojo and beyond. arXiv preprint arXiv:2609.24170, 2026.

[44] Yixian Zhang, Huanming Zhang, Feng Gao, Xiao Li, Zhihao Liu, Yi Nie, Chunyang Zhu, Jiaxing Qiu, Yuchen Yan, Jiyuan Liu, Wenhao Tang, Jiaji Rao, Zhengru Fang, Changxu Wei, Yu Wang, Wenbo Ding, and Chao Yu. Harness VLA: Steering frozen VLAs into reliable manipulation primitives via memory-guided agents. arXiv preprint arXiv:2607.08448, 2026.

[45] Xufeng Zhao, Mengdi Li, Cornelius Weber, Muhammad Burhan Hafez, and Stefan Wermter. Chat with the environment: Interactive multimodal perception using large language models. In IEEE/RSJ Int. Conf. Intell. Robots Syst., pages 3590–3596, 2023.

[46] Youwei Zhou, Bohan Hou, Yankai Fu, Yaoxu Lyu, Yifan Ye, Shilin Shan, et al. Large multimodal models for direct robot control: GPT-6 as a beginning. arXiv preprint arXiv:2610.28259, 2026.

10

<!-- page 11 of 31 -->

# ForceHarness: Turning Force into Action and Physics for Frozen Frontier Models

Supplementary Material

Real-robot traces in this supplement illustrate the system, and the evaluation is Table 3 of the main paper.

## Contents

- **A. Self-terminating force primitives 11**
- **B. Physics translation layer 18**
- **C. The HiddenPhys benchmark 20**
- **D. Extended results 22**
- **E. Web monitor 26**
- **F. What the model sees 26**
- **G. Baselines 28**
- **H. ForceBench port 29**
- **I . Real-robot system and tasks 30**
- **J. Limitations and reproducibility 31**

## A. Self-terminating force primitives

This section specifies the force primitives of Sec. 3.3 and Table 1 of the main paper and every other tool of the menu. All of them share one control loop, so we describe the loop first, then how a position-controlled arm produces force inside it, then the constants that differ between backends, and finally each primitive with a sketch, pseudocode, and one logged call. The backends are MuJoCo, the robosuite Panda of HiddenPhys, ForceBench, the Isaac Lab arm behind that benchmark’s evaluator, and the real arm, the FR3 of Sec. 5.1 of the main paper. The constants are taken from the code of each backend.

Throughout, p is the measured tool center point and $p ^ { * }$ the target, so $p ^ { * } - p$ is the lead. F is the force that the environment applies to the hand, in the base frame, with the zero and the held-load tare $b _ { c }$ removed, and $F _ { \mathrm { r a w } }$ is the same force before $b _ { c }$ is removed. A unit normal n points into the surface, so that $F _ { n } = - \boldsymbol { F } \cdot \boldsymbol { n }$ is positive while the hand presses. The slow and fast target speeds are $v _ { \mathrm { n e a r } }$ and $v _ { \mathrm { f r e e } } ,$ and one tick lasts $\Delta t .$ . In the pseudocode, teal marks force terms and navy marks targets and motion, as in Figure 2 of the main paper.

## A.1. The shared loop

Every tool that moves the robot is a small control law plus one call of the loop in Figure A1. The loop starts from $p ^ { * } = p ,$ so no lead carries over from the previous call, and with a closed gripper it first re-estimates the held-load tare from a still, load-like reading. At every tick the law reads the tared force and the pose and returns an increment of the target. The loop clamps the lead to the limit L of the tool, which bounds the force that any primitive can build, sends the target to the impedance controller, and runs the event detector. Force and torque caps stop every call, while the other events stop a call only if the tool asked for them. After a stop on an event, the target returns to within a tenth of its lead, which unloads the spring, and the hand then holds still while the model thinks. The loop keeps at most 160 evenly spaced rows of the trace for the estimators of Sec. B, and the peak force is tracked before thinning.

A reading f counts as load-like when $f _ { z } ~ < ~ - 1   \mathrm { N } ,$ $\| f \|   <   1 5   \mathrm { N } ,$ and $\left\| f _ { x y } \right\|   \leq   \operatorname* { m a x } ( 1   \mathrm { N } ,   0 . 5   | f _ { z } | )$ The tare $b _ { c }$ is cleared by grasp and release, set by lift from its still window, and replaced at the end of any still, completed call that holds a load-like reading. On the real arm, the runner also reads the force still for a nominal 0.5 s after each call, which gives the still reading of Sec. 3.2 of the main paper.

## A.2. Force from lead, and contact events

No backend offers a force command at this level. Each accepts a position target and behaves as a spring between hand and target. MuJoCo runs an operational-space law with pose-dependent stiffness, ForceBench uses inverse kinematics with fixed joint gains, and the real arm runs its stock joint-impedance controller with fixed gains, so all compliance comes from the outer loop and the lead. With the hand blocked along a direction u,

$$
F _ { u } \approx K \left( p _ { u } ^ { * } - p _ { u } \right) , \qquad \left\| p ^ { * } - p \right\| \leq L ,\tag{A1}
$$

so the lead limit bounds the force at KL. On the real arm the measured K is 0.75 to 1 N/mm, because pressing with a 15 mm lead read 10 to 11 N and free-air steps gave about 1 N/mm, so 8 mm of lead holds at most about 8 N and 25 mm about 18 N.

To hold $F _ { d }$ along u, the primitives run admittance control on top of this spring,

$$
p _ { k + 1 } ^ { * } = p _ { k } ^ { * } + u \operatorname { c l i p } \bigl ( k _ { f } ( F _ { d } - F _ { u , k } ) , \thinspace \pm v _ { \operatorname* { m a x } } \bigr ) \thinspace \Delta t ,\tag{A2}
$$

with $F _ { u }   =   - \boldsymbol { F }   \cdot   \boldsymbol { u } .$ . The target integrates the force error, so the steady error is zero while the lead stays below L.

11

<!-- page 12 of 31 -->

![Image block](doc:537a3fa/tier:basic/page:12/block:1)

Figure A1. The loop behind every primitive. (a) Each tick reads the force and pose, applies the law of the tool, clamps the lead, and checks for events. Only an event or the law’s own goal ends the call, and only the resulting tool result reaches the model. (b) Pseudocode. Each card of Sec. A.4 gives the law and the stops of one tool.

Unsaturated and blocked, the error shrinks by a factor 1 − λ per tick with $\lambda = K k _ { f } \Delta t$ . Computed from the constants rather than measured, λ for push is 0.17 to 0.23 on the real arm, 0.09 to 0.14 in MuJoCo, and 0.15 to 0.37 in ForceBench, so the force approaches its target without oscillation. On the real arm each tick therefore corrects about a fifth of the error and moves the target at most 0.2 mm, and the force rises at 3 to 4 N/s.

Two real-arm primitives, screw and wiggle, press with a fixed lead instead of a force loop. They keep the target a fixed lead $\ell = F _ { d } / K _ { z }$ below the hand, with $K _ { z } = 7 5 0   \mathrm { N / m }$ . Both turn the wrist, which shifts the zero of the force estimate, and a fixed lead presses with about $K _ { z } \ell$ independently of that shift.

The events that stop a primitive come from fixed thresholds with hysteresis on $\tilde { F } ,$ , the mean of the last $N _ { f }$ values of ∥F∥, and nothing is learned. Table A1 gives the rules. Contact is suppressed in press turn, and with a closed gripper in lift, hold, pull, and turn, because a held load or handle is not a new contact. On the real arm, the position moves, the turns, lift, and hold are guarded against collisions, so the detector there uses $f _ { c }   =   8   \mathrm { N }$ and the runner also watches $\| F _ { \mathrm { r a w } } \|$ , which a held load cannot hide.

## A.3. Constants of each backend

The design of the primitives is the same on every backend, while speeds, gains, and limits follow from each robot’s controller and sensing. Table A2 collects these constants, and each card in Sec. A.4 gives the lead limit of its tool. The real arm moves its target five to twelve times more slowly than simulation and allows 8 mm of lead in free motion instead of 30 mm.

## A.4. The primitives one by one

The cards below follow the five groups of Table 1 of the main paper and then the tools that turn the hand or move it by position. Each shows a sketch of the mechanism, the control law and stops as pseudocode that runs inside RUN,

Table A1. Contact events. Thresholds act on the filtered force $\tilde { F }$ except for the caps, which act on every raw tick. Table A2 gives $f _ { c } ,$ $f _ { r } , m ,$ and the caps of each backend.

| Event | Rule | Stops |
| --- | --- | --- |
| contact_start | $\tilde { F } \geq f _ { c }$ on m ticks | if the tool asks |
| contact_end | $\tilde { F } < f _ { r }$ on m ticks | reported |
| force_cap | $\begin{array} { l } { \left\\\| F \right\\\| \ge f _ { \operatorname { c a p } } } \\ { \left\\\| \tau \right\\\| \ge t _ { \operatorname { c a p } } } \\ \end{array}$ | always |
| torque_cap |  | always |
| stall | in contact, the target moved 2 mm reported, tools in six ticks and the hand under test stalls their |  |
| slip | 0.35 of that $\tilde { F }$ gripper closed, fell below half reported its six-tick peak, which exceeded | own way |
| resistance_drop | $f _ { c }$ $\tilde { F }$ in contact, fell below 0.65 of if the tool asks a peak of at least 4N while the hand moves |  |

and one logged call whose tool-result excerpt is quoted as printed. Every moving tool also stops on a cap or a timeout. On the real arm the runner adds two stops, path deviation when the hand leaves its path by more than 15 mm, or 20 mm for turn and 10 mm for screw, and stalled when the hand gains under 0.5 mm in 3.5 s against more than 2 mm of lead, and a force cap or a deviation above 30 mm ends the episode.

**Making contact.** move until contact walks the target along u at $v _ { \mathrm { n e a r } }$ until the detector reports contact, and its law never reads force, as Figure A2 shows. A small lead limit keeps the impact low, and the call reports the contact height corrected for the lag of the detector, the impact peak, and the stiffness when the hand pressed on. On the real arm with a closed gripper it first re-tares from a still reading, so that the tare is taken at the pose of the contact. probe runs the same motion for at most an amplitude and returns to its start with move $\_ t o ,$ reporting whether contact was found.

12

<!-- page 13 of 31 -->

Table A2. Constants of each backend. All real-arm speeds and ticks are nominal and its forces use the torque-based estimate. Its lead limits are those of the workstation, and the real-time computer clamps again.

| Constant | MuJoCo | ForceBench | Real arm |
| --- | --- | --- | --- |
| Tick ∆t | 0.05 s | 1/15 s | 0.05 s nominal, about 66 ms measured |
| vnear, vfree | 20, 100 mm/s | 20, 100 mm/s | 4, 8 mm/s |
| push, pull, press_turn / slide, search $k _ { f } ;$ | 0.0045 / 0.0015 (m/s)/N | same | same, no press_turn |
| Stiffness command | 150, 40 on a soft axis $k _ { p }$ | joint gains 80 and 12, fixed | 750 N/m and 15 N·m/rad, fixed |
| Effective K when blocked | about 3N/mm at $k _ { p }$ 150, 0.4 to 0.6 | 1.23 N/mm near, about 0.5 far | 0.75 to 1 N/mm |
| Lead limit L: free, guarded | at 40 30, 12 mm | 30,12 mm | 8,8mm |
| Lead limit L while pressing | by tool, in each card | same | by tool, at most 25 mm once the raw force changed by 3N |
| Filter $N _ { f } ,$ samples m | 4 ticks, 1 | 4 ticks, 1 | 1 tick, 3 after a 0.3 s grace |
| $f _ { c } , f _ { r }$ | 3N,1N | 2N,1N | $0 . 5 f _ { c }$ 5N or 8N if guarded, |
| Caps fcap, tcap | 25N tared, 3N·m | same | 25 N, raw on the real-time computer, 3 N·m |
| Force target ceiling | 20N | 20N | 20 N, lower per tool |
| Rates of rotate _z, turn, point_tool | 30, 10, 30°/s | same | 8,2,2°/s |
| Position tolerance | 3 mm, 0.1 mm on Lock | 3 mm | 3 mm reported, 1 mm internal |
| Added force noise | 0.5 N per tick, 0.3N per episode | none | none, about 0.03 N static |

![Image block](doc:537a3fa/tier:basic/page:13/block:3)

$$
\leftarrow p ,
$$

$$
\mathbf { i f } \left( p - p _ { 0 } \right) \cdot u \geq
$$

$$
\Delta p ^ { * } \gets \mathbf { \mathit { u } } \: v _ { \mathrm { n e a r } } \Delta t
$$

$$
f_{c}L = 6mm, \; 15mm
$$

![Chart block](doc:537a3fa/tier:basic/page:13/block:8)

Figure A2. move until contact and probe. (a) The target leads the hand toward the surface until the filtered force passes $f _ { c } .$ (b) Pseudocode. (c) The approach before the search of Figure 2 of the main paper on the real arm. The estimate reads about 2 N of drift while the arm moves, and the call stops when $F _ { n }$ passes $f _ { c }$

**Holding force.** push regulates the reaction along u to $F _ { d }$ with the admittance law of Eq. (A2), as in Figure A3. The target advances at most max dist, and on the real arm the 25 mm lead is requested once $F _ { u }$ reaches 5 N. The call ends after its duration, or, when the model asks for it, on a resistance drop, which push detects itself on the raw force along u because the generic detector smooths too much to catch a click. It reports the mean normal force over the last 80% of the rows in contact.

slide presses along n by admittance while it moves at constant speed along u projected onto the surface, so friction is measured rather than controlled, as Figure A4 shows. The full stiffness and a 20 mm lead let the tangential spring overcome friction, and on the real arm at most 20 mm of the 25 mm lead goes to the press so that some is left for the motion. It stops after distance of measured travel or after backing 2 cm off the surface. It reports the mean normal force and the friction coefficient over the same window as push, with flags lost contact and stick slip.

**Finding and seating.** search presses lightly and moves the target on an Archimedean spiral of pitch a about its start until the part sinks, which Sec. 3.4 of the main paper walks through and Figure A5 details. It first descends and arms on surface contact, ending with no surface after 3 cm without one. When $F _ { n }$ falls below $0 . 3 F _ { d } ,$ the part is over a hole, so the spiral pauses and the target falls, and a part that is already deeper than drop and blocked sideways also counts as dropped. In simulation a lateral force above 6 N for 0.3 s means a corner caught on the rim, so the part lifts, moves 6 mm along the wall’s reaction, and restarts the spiral there. On the real arm, once the advance gains over 0.25 mm on its recent median, the target thrusts down with the 25 mm lead toward max( $[ F _ { d } ,$ 15 N), limited so that no force estimate would reach 23 N within 0.3 s, and 1.5 mm of gain after 2 s means dropped. Besides dropped and completed, the call can end stuck, out of range, or lost contact. A hop mode, the default in simulation, probes the spiral points one at a time instead of dragging the part, and a probe deeper than earlier ones by drop means dropped.

13

<!-- page 14 of 31 -->

![Image block](doc:537a3fa/tier:basic/page:14/block:1)

<div class="docvortex-algorithm" style="white-space: pre-wrap; font-family:monospace;">
(b) Pseudocode
1 input direction u, force $F _ { d } ,$ duration T, max dist, stop on a
drop or not L = 30 mm
2 p0 ← p, RUN with, every tick,
3 Fu ← −F · u
4 if asked and t ≥ 0.6 s and the two-tick mean of $F _ { u }$ fell
by max(2 N, peak/2) from a peak of at least 3 N then
stop resistance drop
5 v ← clipkf (Fd − Fu), ±vnear Eq. (A2)
6 if (p∗ − p0) · u ≥ max dist and v &gt; 0 then v ← 0
7 ∆p∗ ← u v ∆t, done ← t ≥ T
8 return mean Fn ± σ over the last 80% of rows in contact
</div>

![Chart block](doc:537a3fa/tier:basic/page:14/block:3)

Figure A3. push. (a) The target leads into the surface until the reaction equals $F _ { d } ,$ which turns force error into target velocity. (b) Pseudocode. (c) Pressing a soft object on the real arm with $F _ { d } = 7   \mathrm { N }$ for 6 s. The force settles at the target while the hand keeps sinking, and the tool result reports the mean force and the stiffness.

![Image block](doc:537a3fa/tier:basic/page:14/block:5)

<div class="docvortex-algorithm" style="white-space: pre-wrap; font-family:monospace;">
(b) Pseudocode
1 input direction u, distance, force $F _ { d } ,$ normal n L = 20 mm
2 u ← unitu − (u · n) n, p0 ← p, RUN with, every tick,
3 if (p − p0) · n &lt; −2 cm then done left the surface
4 vn ← clipkf (Fd − Fn), ±vnear
5 done ← (p − p0) · u ≥ distance
6 ∆p∗ ←u vnear [¬done] + n vn∆t
7 return mean Fn ± σ, µ = |Ft|/F¯ n ± σµ, flags
</div>

![Chart block](doc:537a3fa/tier:basic/page:14/block:7)

Figure A4. slide. (a) The tool presses with $F _ { n }$ regulated to $F _ { d }$ while it travels along u, and the tangential force $F _ { t }$ gives the friction coefficient. (b) Pseudocode. (c) One stroke of wiping the whiteboard on the real arm. The normal force holds at $F _ { d }$ over the 10 cm stroke and the tangential force is almost as large, which gives a friction coefficient near one.

![Image block](doc:537a3fa/tier:basic/page:14/block:9)

![Chart block](doc:537a3fa/tier:basic/page:14/block:10)

Figure A5. search. (a) The part presses with $F _ { d }$ and follows a spiral, and over the hole the reaction falls, the spiral pauses, and the part sinks. (b) Pseudocode of the slide mode, with $e _ { 1 } , e _ { 2 } \perp n . ~ ( \mathsf { c } )$ The call of Figure 2 of the main paper on the real arm. $F _ { n }$ holds near $F _ { d } = 1   \mathrm { N }$ for half a minute while the logged path spirals, then the plug drops and the thrust drives it in.

insert advances along n with a speed that falls linearly with the reaction while it traces a 2 mm circle at 1 Hz, as in Figure A6. In MuJoCo it is stiff along n and soft sideways, so the part can center itself. It reports seated at the requested depth, and after 1 s without 1 mm of progress it reports jammed if $F _ { n } \: \geq \: 0 . 5 F _ { \operatorname* { m a x } }$ and stalled otherwise, together with the advance, the lateral force, and a seating

flag. In the episode of Figure 2 of the main paper, the model followed a stalled insert with wiggle.

wiggle, which runs only on the real arm, presses the held part down with the fixed lead $\ell = \operatorname* { m i n } ( F / K _ { z } , 2 5$ mm) and rocks it so that a chamfer can guide it in, as Figure A7 shows. The press ramps from min(8 N, F) to F over $2   \mathrm { s } ,$ the yaw rocks with amplitude A, and the target circles sideways with

14

<!-- page 15 of 31 -->

![Image block](doc:537a3fa/tier:basic/page:15/block:1)

![Chart block](doc:537a3fa/tier:basic/page:15/block:2)

Figure A6. insert. (a) The part advances more slowly as the reaction rises and circles slightly so that a caught corner can find the opening. (b) Pseudocode. (c) A call on Peg in MuJoCo. The peg advances 10 mm while $F _ { n }$ hovers near $0 . 5 F _ { \mathrm { m a x } }$ , then progress stops and the call ends jammed.

radius w and a period of 3.9 s, so the two motions do not synchronize. Once the hand stops following the circle, the part is held sideways, so the rocking freezes and the tool only presses until the part stops sinking. It ends seated at sufficient depth below the support surface and jammed otherwise, and it reports the depth.

**Turning under load.** screw, which runs only on the real arm, first presses with the fixed lead $\ell \quad =$ min $( F / K _ { z } , 2 5   \mathrm { m m } )$ and then turns about the vertical at $5 ^ { \circ } / \mathrm { s }$ while it keeps pressing and follows the thread down, as Figure A8 shows. The turn starts once the tip stops sinking under a full lead, or after 6 s. While it turns, the lateral target becomes $p _ { 0 }   +   0 . 3 \left( p   -   p _ { 0 } \right)$ , and a call is split into segments of at most $3 0 ^ { \circ }$ . It ends completed at the angle or sink limit after 10 mm per segment, and it reports the advance while pressing and while turning, millimeters per turn, and advancing, rising, or not advancing.

press turn, which exists only in ForceBench, holds a press along −z by Eq. (A2) and, after press time, also turns about the vertical, on Stove Press and Nut Thread. It exists because each call resets $p ^ { * } = p ,$ which drops a push press before the next rotate z. Besides the press force and depth and the resisting torque over the second half of the turn, it reports on Stove Press the median resistance on each side of the start, which Figure A9 shows. The no-force control uses move turn instead, which Nut Thread also offers with force, and which moves the target down by a set amount and turns while it follows a commanded height, and whose description tells the model that the arm acts as a spring of about 1 N per mm, so that the control lacks force rather than a motion.

**Measuring.** lift rises by height at $v _ { \mathrm { n e a r } }$ and then stands still for hold time, so that the force on the hand is the load, as Figure A10 shows, and hold only stands still. In simulation both report the payload mass of Eq. 4 of the main paper and the center-of-mass offset over the still rows, and $l i \hbar$ sets the held-load tare from them. On the real arm the runner

instead gives the still reading, its change since the grasp, and a weight in grams, which serves to compare loads within an episode.

pull moves a gripped handle along the path that its mechanism allows, on Latch, Lock, and Drawer Pull, using force only, as Figure A11 shows. It regulates the pull force along d by admittance and lets the hand follow any sideways force, so no spring acts sideways, and after 5 mm of travel it turns d toward the free side by at most $3 ^ { \circ }$ per tick. It ends completed at the path length, stalled when the hand stops while $f _ { p }   \geq   0 . 9 5 F _ { d } ,$ or joint limit. It reports the pull and lateral forces, the turn of the path with the radius and pivot of a circle fit, the free direction, and mechanism stopped. On the real arm a direction with a vertical part runs a straight pull with the wrist fixed.

**Turning.** turn rotates the hand rigidly about an axis a through a pivot $c _ { z }$ so the tool center moves on a circle and the tool orientation turns with it, on Latch, the Faucet tasks, and the real arm, as Figure A12 shows. With force, the model can ask it to stop on a resistance drop, which it detects on the resisting moment about the turn axis in simulation and, without a verified torque sign, on the force along the arc tangent on the real arm. ForceBench can also print the moment, its peak, and its end value, which was off in all reported runs. rotate z turns the tool about the vertical with the position target fixed, and the wrist joint is kept within ±2.6 rad in MuJoCo and −0.8 to 2.4 rad on the real arm. It reports the yaw change and whether the angle was clipped.

**Moving by position.** move to moves the tool center on a straight line to a goal g with the orientation unchanged, and move by does the same toward $p _ { 0 }   +   [ d x , d y , d z ]$ , as Figure A13 shows. In simulation the speed drops to $v _ { \mathrm { n e a r } }$ in contact. On the real arm joint friction can stop the hand short of $g ,$ so a stuck hand near $g$ gets a target bias that grows up to the 8 mm lead and retracts within 1 mm of $g .$ Both stop on contact at $f _ { c } ,$ or at the collision limits on the real arm, where

15

<!-- page 16 of 31 -->

(a) Mechanism

![Image block](doc:537a3fa/tier:basic/page:16/block:2)

<div class="docvortex-algorithm" style="white-space: pre-wrap; font-family:monospace;">
(b) Pseudocode
1 input depth, force F , yaw amplitude A, shift w, duration T
real arm only
2 p0 ← p, phase ← rock, RUN with, every tick,
3 Fe ← ramp from min(8 N, F ) to F over 2 s,
$p _ { z } ^ { * } \leftarrow p _ { z } \stackrel { \mathbf { r } } { - } \operatorname* { m i n } ( F _ { e } / K _ { z } , 2 5 \mathrm { m m } )$
4 if phase is rock then
5 $\begin{array} { r } { \overset { \ast } { \underset { \cdot } { } } \psi _ { \cdot } ^ { \ast } \leftarrow \rho A \sin ( 2 \pi t / T _ { \psi } ) , } \end{array}$
$\vec { p } _ { x y } ^ { * } \gets \vec { p } _ { 0 , x y } + \rho \vec { w } \left[ \operatorname { c o s } , \operatorname { s i n } \right] ( 2 \pi t / 3 . 9 \operatorname { s } )$
6 if the hand stopped following the circle or the depth is
reached then phase ← press, freeze $\psi ^ { * } , p _ { x y } ^ { * }$
else if the part sank under 0.2 mm in 1 s at full force, or
pressed 4 s then stop
8 $\mathbf { i f }   t \geq T$ then stop
9 label seated if the depth below the surface reached depth, else
jammed or not seated
</div>

![Chart block](doc:537a3fa/tier:basic/page:16/block:4)

Figure A7. wiggle. (a) A fixed lead presses the part down while the yaw rocks and the target circles, and once the part is held sideways the rocking stops. (b) Pseudocode, with $\rho = \operatorname* { m i n } ( 1 , t / 1   \mathrm { s } )$ and $T _ { \psi } = \operatorname* { m a x } ( 3   \mathrm { s } , 2 \pi A / 0 . 1 2 6$ rad/s). (c) The call after the stalled insert in the episode of Figure 2 of the main paper. The press ramps up, the plug sinks 3 mm in 3 s, and the call ends seated.

![Image block](doc:537a3fa/tier:basic/page:16/block:6)

<div class="docvortex-algorithm" style="white-space: pre-wrap; font-family:monospace;">
(b) Pseudocode
1 input angle ϕ, press F real arm only, segments of at most 30◦
2 ℓ ← min $( \dot { F } / \dot { K } _ { z } ,$ 25 mm), $p _ { 0 } \leftarrow p ,$ RUN with, every tick,
3 if the tip sank more than 10 mm then stop sink limit
4 $p _ { z } ^ { * } \gets p _ { \bar { z } } - \ell ,$ at most 8 mm/s pressing and vnear
following press, follow
5 if pressing then $p _ { x y } ^ { * } \leftarrow p _ { 0 , x y } ,$ and turn once the tip stops
sinking or after 6 s
6 else $\psi ^ { * } \gets \psi ^ { * } + \mathrm { s g n } ( \phi )   5 ^ { \circ } / \mathrm { s } \cdot \Delta t ,$
$p _ { x y } ^ { * } \leftarrow p _ { 0 , x y } + 0 . 3 \widehat { ( p - p _ { 0 } ) _ { x y } }$
done once ϕ is turned
8 return advance while pressing and turning, mm per turn, trend
label
</div>

![Chart block](doc:537a3fa/tier:basic/page:16/block:8)

Figure A8. screw. (a) A fixed lead presses the screw into the thread while the hand turns about the vertical and follows the thread down. (b) Pseudocode. $\mathrm { ( c )   A - } 1 5 0 ^ { \circ }$ call on the real arm, run as five segments. Each segment re-presses, the force dips between them, and the advance climbs in steps as the thread pulls the screw in.

![Image block](doc:537a3fa/tier:basic/page:16/block:10)

<div class="docvortex-algorithm" style="white-space: pre-wrap; font-family:monospace;">
(b) Pseudocode
1 input angle ϕ, force $F _ { d } ,$ rate ω, press time $T _ { p }$ ForceBench
only, L = 30 mm
2 p0 $\leftarrow p ,$ RUN with, every tick,
3 $v \leftarrow \mathrm { c l i p } \big ( k _ { f } ( F _ { d } - F _ { z } ) , \pm v _ { \mathrm { n e a r } } \big ) ,$
$\Delta p ^ { * } \gets - \hat { z }   v   \Delta t$ hold the press
4 $\mathbf { i f }   t \geq T _ { p }$ then
5 $\bar { \psi ^ { * } } \xleftarrow { \tau } \psi ^ { * } +$ sgn $\mathbf { \nabla } _ { \mathbf { \nabla } ^ { \mathbf { \nabla } } } ( \phi )   \omega \Delta t$ until |ϕ| is turned
6 $\tau _ { \mathrm { r e s } } \leftarrow \operatorname { s g n } ( \phi )   \tau _ { z }$ , added to the side’s samples while
moving 8 to $2 \dot { 5 } ^ { \circ }$ out
7 done once ϕ is turned or the optional stop depth
8 return press force, depth, mean $\mathcal { T } _ { \mathrm { T e s } }$ over the second half,
peak, side medians
</div>

![Chart block](doc:537a3fa/tier:basic/page:16/block:12)

Figure A9. press turn. (a) The hand holds the knob down with $F _ { d }$ while it turns, and the resisting torque is read on each side of the start. (b) Pseudocode. (c) The two calls of one successful Stove Press episode of GPT-6.1 Sol in ForceBench, $+ 3 0 ^ { \circ }$ and then −60◦. The knob resists less on the clockwise side, and the tool result reports the median resistance of each side.

they can also end stalled short. They report kinematics and a note if the call ended over 5 mm short. point tool turns the fingers toward a direction with the tool center fixed, on Latch, Lock, Drawer Pull, and the real arm. In simulation the force zero is converted every tick as the tool turns.

## A.5. The remaining tools, guard, and validation

The gripper and perception tools do not regulate force, and Table A3 summarizes them. locate maps a pixel to a 3D point in ForceBench and on the real arm, while MuJoCo gives object poses as text, and locate object is an extra tool added for Nut Thread beyond the 22 base tools, as Sec. H notes. On the real arm a gripper width outside the expected

16

<!-- page 17 of 31 -->

![Image block](doc:537a3fa/tier:basic/page:17/block:1)

<div class="docvortex-algorithm" style="white-space: pre-wrap; font-family:monospace;">
(b) Pseudocode
1 input height, hold time $T _ { h }$ no contact events while holding
2 p0 ← p, RUN with, every tick,
if pz − p0,z &lt; height and not yet still then
$\Delta p ^ { * } \gets \hat { z }   \dot { v } _ { \mathrm { n e a r } } \Delta t$
4 else stand still, done after $T _ { h }$
5 W ← the last max(5, n/4) rows the still window
6 m ← |F¯ z|/g ± σm, bc ← F¯ over W Eq. 4 of the main
paper
7 return m and the center-of-mass offset from $\bar { \tau } _ { x } , \bar { \tau } _ { y }$
8 HOLD(T): RUN with $\Delta p ^ { * } = 0$ until $t \geq T$
</div>

![Chart block](doc:537a3fa/tier:basic/page:17/block:3)

Figure A10. lift and hold. (a) The hand lifts the grasped load and holds still, so the vertical force is the weight. (b) Pseudocode. (c) Two lifts on Weight in MuJoCo, for a light and a heavy cube on two seeds. The shaded still windows give masses within about one σ of 0.15 and 0.60 kg.

![Image block](doc:537a3fa/tier:basic/page:17/block:5)

<div class="docvortex-algorithm" style="white-space: pre-wrap; font-family:monospace;">
(b) Pseudocode
1 input direction d, force $F _ { d } ,$ distance L = 30 mm, soft in the
horizontal plane
2 d ← horizontal unit vector, RUN with, every tick,
3 $l \leftarrow \hat { z } \times d , \; f _ { \mathcal { P } } \leftarrow - F \cdot d , \; f _ { l } \leftarrow \dot { F \cdot l }$
4 if path ≥ 5 mm and $f _ { p } \geq$ max(1 N, Fd/4) then turn d
by 0.1 atan $2 ( f _ { l } , f _ { p } )$ , at most min(3◦, step/0.15 m)
5 ∆p∗ ←
d c(kf (Fd − fp)) + l c(kf fl)∆t + l(p − p∗) · l
c clips to the speed
6 done at the path length, stalled if still for 1 s with
$\bar { f } _ { p } \geq 0 . 9 5 \bar { F } _ { d }$
7 return $f _ { p } \pm \sigma ,$ lateral force, path turn, radius and pivot of a
circle fit
</div>

![Chart block](doc:537a3fa/tier:basic/page:17/block:7)

Figure A11. pull. (a) Seen from above, the hand pulls along d with no sideways spring, so the door’s constraint bends the path into an arc and d follows it. (b) Pseudocode. (c) The pull that opened the door on Latch in MuJoCo. The logged hand path turns by $1 8 ^ { \overline { { \mathsf { o } } } }$ and lies on the fitted circle.

![Image block](doc:537a3fa/tier:basic/page:17/block:9)

![Chart block](doc:537a3fa/tier:basic/page:17/block:10)

Figure A12. turn and rotate z. (a) The hand turns a lever about its pivot, and a peak of the resisting moment followed by a drop marks the latch giving way. (b) Pseudocode. (c) Pressing the latch handle on Latch in MuJoCo, recomputed from the logged rows. The moment rises to about 1 N·m and falls at $2 1 ^ { \circ }$ , where the call stops on resistance drop and the model goes on to pull.

range after grasp or release latches a fault, after which the gripper is never opened.

Beneath all tools, a fixed guard on the real-time computer enforces limits that the model cannot change. It caps the raw reading at 25 N and 3 N·m and keeps the hand inside a workspace box, and it limits the speed to 8 mm/s, the tracking

error to 15 mm, and the age of a command to 300 ms. The guard clips the target speed, and a breach of any other limit ends the episode.

A call that fails validation does not move the robot, and its tool result gives the reason. Simulation checks tool and argument names, clips force targets at 20 N inside the laws,

17

<!-- page 18 of 31 -->

![Image block](doc:537a3fa/tier:basic/page:18/block:1)

![Chart block](doc:537a3fa/tier:basic/page:18/block:2)

Figure A13. move to and move by. (a) The target runs ahead of the hand on a straight line, and the guard stops the move on contact. (b) Pseudocode. (c) Setting a held bottle down on the real arm. The hand stops 4.5 mm short of its goal when the force passes the guard threshold of 8 N.

Table A3. Tools that neither move by force nor hold a force.

| Tool | What it does | What it returns |
| --- | --- | --- |
| grasp, release | close or open the gripper, clear opening, holding or the tare | not |
| locate | map a pixel to a 3D point, no mo- point, valid pixels |  |
| locate_object | tion fit a known part's axis to a clicked axis, overlap score |  |
| scene | mask describe the scene, no motion | poses, or photos |
| done | end the episode | logged claim only |

and allows several calls per turn. The real arm also checks argument types, the gripper state, and reach along the path on a kinematic model, and it caps force targets at 10 to 20 N depending on the tool and task. It allows one call per turn, and a rejected call still uses up an action.

## B. Physics translation layer

This section specifies the physics translation layer of Sec. 3.5 of the main paper, which HiddenPhys, ForceBench, and the FR3 share, so that each number in a tool result can be recomputed from the primitive’s log.

## B.1. Common recipe

Five estimators, whose constants per backend are in Table B1, turn the log of one primitive into the quantities of Sec. B.2 by the recipe of Sec. 3.5 of the main paper. The log holds the force, torque, and tool position of every tick and is downsampled uniformly beyond 160 rows. Each estimator removes the zero and any held load, $F _ { i } = F _ { i } ^ { \mathrm { r a w } } - c ,$ where $F ^ { \mathrm { r a w } }$ is the zeroed force and the tare c is set by lift and cleared by grasp and release. Payload mass and the FR3 pull skip this tare, since the load or the resistance is their signal. Within the window where it has one meaning, a quantity is by default a mean with its standard error, while payload mass and pull force add the residual zero bias $\sigma _ { b } ,$ a ratio propagates relative errors, and a slope takes its least-squares

error,

$$
\begin{array} { r } { \bar { x } = \frac { 1 } { n } \sum _ { i \in W } x _ { i } , \qquad \sigma = s / \sqrt { n } , } \end{array}\tag{B1}
$$

$$
\begin{array} { r } { \sigma _ { m } = \frac { 1 } { g } \sqrt { s _ { z } ^ { 2 } / n + \sigma _ { b } ^ { 2 } } , \quad \sigma _ { \mathrm { p u l l } } = s / \sqrt { n } + \sigma _ { b } , } \end{array}\tag{B2}
$$

$$
\sigma _ { \mu } ^ { 2 } / \mu ^ { 2 } = \sigma _ { t } ^ { 2 } / \overline { { | F _ { t } | } } ^ { 2 } + \sigma _ { n } ^ { 2 } / \bar { F } _ { n } ^ { 2 } ,\tag{B3}
$$

$$
\sigma _ { k } = \sqrt { \textstyle \sum _ { i } e _ { i } ^ { 2 } / ( n - 2 ) } \left/ \sqrt { \textstyle \sum _ { i } ( \delta _ { i } - \bar { \delta } ) ^ { 2 } } , \right.\tag{B4}
$$

where W is the window of n rows and s the sample standard deviation. $\sigma _ { t }$ and $\sigma _ { n }$ are the standard errors of the two means, $e _ { i }$ the fit residuals, and $\delta _ { i }$ the penetrations. The flags, given with each quantity in Sec. B.2, are class bands fixed across tasks, warnings, and outcomes translated from how the primitive ended, and no step holds a task threshold.

## B.2. Every quantity

This subsection gives every quantity that a tool result can carry and its flags, grouped by the primitives that report it. Each paragraph states the window and the σ of a quantity. Here u is the call’s direction, or else its start-to-end direction, and nˆ its normal. The tool position $p$ starts at $p _ { 0 } , t _ { c }$ is the time of contact, and F is tared unless marked raw. Endings that no rule below translates, such as stuck or timeout, pass through by name.

**Held load.** lift and hold report the payload mass $m =$ $| \bar { F } _ { z } ^ { \mathrm { r a w } } | / g$ with the gripper closed and $g   =   9 . 8 1   \mathrm { m } / \mathrm { s } ^ { 2 }$ , av-

Table B1. Estimator constants per backend. MuJoCo adds this noise and bias to its sensor, and ForceBench uses the same defaults, with its lag and contact σ computed by formula. The other constants are in Table A2.

| Constant | MuJoCo | ForceBench | FR3 |
| --- | --- | --- | --- |
| Detector lag ℓ | 4mm | 5.3mm | 0.2mm |
| Contact σ | 3mm | 4mm | 0.3mm |
| Noise per tick σ<sub>noise</sub> | 0.5N | 0.5N | 0.0688N |
| Bias per episode σ<sub>b</sub> | 0.3N | 0.3N | 0.0747N |

18

<!-- page 19 of 31 -->

eraged over the last hold time or else the last half of the call, with $\sigma _ { m }$ of (B2). When $m   \geq   0 . 0 3$ kg and the load hangs, they also report the center-of-mass offset over the same window, printed as $| r |$ and the nearest base axis. A load of weight mg at horizontal lever r gives the torque $( - m g   r _ { y } , \: m g   r _ { x } , \: 0 )$ , and a sign constant $s _ { \tau }$ set on the simulator’s true lever maps the sensor’s raw torque rows to $r ,$

$$
r _ { x } = s _ { \tau } \frac { \bar { T } _ { y } } { m g } , \quad r _ { y } = - s _ { \tau } \frac { \bar { T } _ { x } } { m g } , \quad s _ { \tau } = - 1 ,\tag{B5}
$$

$$
\sigma _ { r } = \sqrt { ( \sigma _ { \tau } / m g ) ^ { 2 } + ( \left| r \right| \sigma _ { m } / m ) ^ { 2 } } ,\tag{B6}
$$

with $\sigma _ { \tau } = \sqrt { ( s _ { x } ^ { 2 } + s _ { y } ^ { 2 } ) / n   +   \tau _ { b } } ,$ , where $s _ { x }$ and $s _ { y }$ are the standard deviations of the two torque axes and $\tau _ { b } = 0 . 0 1   \mathrm { N }$ m is the torque bias. The lever is taken about the sensor’s vertical axis. Both quantities appear in MuJoCo and ForceBench. A reading is flagged not settled if the height varies over 3 mm and unsteady if $s _ { z } > 3 \sigma _ { \mathrm { n o i s e } } + 0 . 5   \mathrm { N }$ . The mass is flagged no load if $m < 2 \sigma _ { m }$ and pushing up not hanging if $\bar { F } _ { z } > 0$ without no load, and the offset is flagged within noise if $| r | < 2 \sigma _ { r }$ . The FR3 reports weight from the still reading of Sec. I instead of payload mass, which would use one zero point for every pose, and it omits the offset.

**Contact.** A call whose contact event passes the contact gate reports four quantities on all three backends. The gate admits an event outside grasp, release, and pull that lasts at least 0.15 s and peaks at or above $0 . 8 f _ { c }$ within 0.3 s, where $f _ { c }$ is the contact threshold. The contact height is $z _ { c } =$ $z ( t _ { c } ) - u _ { z } \ell \; \mathrm { i f } \; | u _ { z } | > 0 . 7$ , and otherwise the contact point is $s _ { c } = ( p ( t _ { c } ) - u \ell - p _ { 0 } ) \cdot u$ , both taken at the tick of $t _ { c }$ with $\sigma = \ell / 2 + v \Delta t ,$ , where ℓ is the detector lag of Table B1. The contact normal $\bar { F } / | \bar { F } |$ and its tilt arccos $( | \bar { F } _ { z } | / | \bar { F } | )$ use the 6 rows from $t _ { c } ,$ and the impact peak is max $| F |$ from $t _ { c }$ to $t _ { c } + 0 . 3   \mathrm { s } ,$ both without σ. The stiffness k is a least-squares fit $F _ { u } = k \delta + b$ with $\delta = ( p - p ( t _ { c } ) ) \cdot u$ and $F _ { u } = - \boldsymbol { F } \cdot \boldsymbol { u } ,$ on at least 4 of the 12 rows from $t _ { c }$ that have $\delta > 0 . 5$ mm, and it is reported if $k > 0 . 0 5   \mathrm { N / m }$ m with $\sigma _ { k }$ of (B4). The impact is labeled gentle below 5 N, moderate below 15 N, and hard above, and the stiffness soft below 1 N/mm, firm below 10 N/mm, and rigid above.

**Pressing and sliding.** push and slide report the normal force, the mean of $F _ { n } = - F \cdot u , \mathrm { o r } \; \mathrm { o f } \; { - F \cdot { \hat { n } } }$ on slide, over the rows of the last 80% of the call with $F _ { n } > 0 . 5 f _ { c } ,$ , with $\sigma$ of (B1). On the same rows slide adds the friction coefficient $\mu = \overline { { | F _ { t } | } } / \bar { F } _ { n }$ with $F _ { t } = F$ · u and $\sigma _ { \mu }$ of (B3), and the stickslip index std| $| F _ { t } | / \overline { { | F _ { t } | } }$ . All three backends report them. If fewer than 3 rows qualify, the result says lost contact and gives the mean of all rows with no σ and no friction line, and an index above 0.5 sets stick slip.

**Insertion.** search and insert report the advance $d \mathrm { ~ = ~ }$ $( p _ { \mathrm { e n d } }   -   p _ { 0 } )   \cdot   \hat { n }$ over the whole call and the lateral force $| \overline { { F - ( F \cdot \hat { n } ) \hat { n } } } |$ over its last 5 rows, without $\sigma ,$ on all three backends. search ends dropped in on a drop or a resistance drop, no hole found at full radius, and jammed if stalled. insert ends seated at depth, and after under 1 mm of advance in 1 s it ends jammed with $F _ { n }$ at least half of max force or stalled no progress with less, and otherwise not seated. Both give jammed hard at the force cap and lateral load if the lateral force at the end exceeds $f _ { c }$

Pulling. pull reports the pull force, the mean of $- F \cdot d _ { i }$ along the pull direction $d _ { i }$ of each tick, and the mean sideways force over the last 80%, with σ of (B2), or on the FR3 the mean of $- F ^ { \mathrm { r a w } } \cdot u .$ After 3 cm of travel it adds the path turn atan $\overline { { \vphantom { \overline { { \vphantom { \overline { { \alpha } } } } } } } } 2 ( q _ { 1 } \times q _ { 2 } , \: q _ { 1 } \cdot q _ { 2 } )$ between the first and last 1.5 cm of path $q _ { 1 }$ and $q _ { 2 } ,$ counter-clockwise positive, the radius and pivot of an algebraic circle fit if $\mathrm { { \bf { \cal { f } } } } \left| \mathrm { { t u r n } } \right| \ge 4 ^ { \circ }$ , and the free direction as the heading of $q _ { 2 }$ . They carry a σ of $2 ^ { \circ }$ , or the fit RMS plus 5 mm for the fit, in MuJoCo and ForceBench. The call ends mechanism stopped when it stalls, under 3 mm in 1 s with a mean pull force over 0.5 s of at least 95% of target, or on the FR3 under 0.2 mm in 1 s with a mean resistance of at least 95% of target, and a turn under $3 ^ { \circ }$ is flagged straight line.

**Turning on ForceBench.** The quantities of this paragraph and the next appear only under ForceHarness. On ForceBench, $\theta$ is the commanded turn about the axis a through the pivot $p _ { v }$ , and T is the torque row. press turn reports the resisting torque $\tau = \mathrm { s g n } ( \theta ) T _ { z }$ , positive against the turn, as its mean over the second half with $\sigma = s / { \sqrt { n } }$ and its peak over the turn. On Stove Press it reports instead the resistance per side, the median of $\tau$ on each side of the start angle over rows moving away from it, 8 to $2 5 ^ { \circ }$ out and since the last grasp, from 4 rows on, with $\sigma = 1 . 2 5 3 s / \sqrt { n }$ It also reports the press force, the mean upward reaction while turning, and the depth, the start minus the end tool height, both without σ. turn can also report the resisting moment $M = \mathrm { s g n } ( \theta ) \big [ \boldsymbol { a } \cdot \boldsymbol { T } - \boldsymbol { a } \cdot ( ( \boldsymbol { p } - \boldsymbol { p } _ { v } ) \times \boldsymbol { F } ) \big ]$ , smoothed over 4 rows, as its mean over the second half, its peak over the call, and its value at the last row, without $\sigma ,$ but this was off in all reported runs.

**On the FR3 only.** Every call reports the still reading of Sec. I and, while the hand holds a load, the weight $\hat { m } =$ $- \Delta F _ { z } / g$ above $2 0   \mathrm { g } ,$ , with $\Delta F$ the change since grasp ended. The reading takes 0.5 s after a 0.8 s relax, nominally, and carries a fixed ±400 g on grams. Every call also reports the net force peak max $| F |$ over the whole call. screw reports the advance $\Delta z = z _ { \mathrm { s t a r t } }   -   z _ { \mathrm { e n d } } ,$ , split into the sink while pressing and while turning, the advance per turn $\Delta z _ { \mathrm { t u r n } } / | \theta | \times 3 6 0 ^ { \circ }$ if $| \theta | \geq 1 0 ^ { \circ }$ , and as side load the size of the untared change

19

<!-- page 20 of 31 -->

in $\bar { F } _ { x y }$ between its first 3 and last 5 rows, without σ. It ends advancing, rising, or not advancing for a sink while turning above 0.2 mm, below −0.2 mm, or in between. The FR3 prints no quantity when the force calibration is not valid for the pose.

## B.3. Rendering levels

L1 prints each value with its unit and any fixed geometry, and L2, the default of ForceHarness, adds ± σ and a line with the window and flags, as Figure F2 shows.

## B.4. Accuracy

We check the estimators against the ground truth of the simulators. On HiddenPhys test episodes, the mass error against simulator truth has a mean of 0.004 kg and a standard deviation of 0.03 kg, which matches the printed σ, and the direction cosine of the center-of-mass offset is at least 0.97. On ForceBench, masses of 0.06 and 0.18 kg read 0.058 and 0.177 kg. The printed σ covers measurement noise only.

## C. The HiddenPhys benchmark

HiddenPhys is the benchmark behind Table 5 of the main paper, and Sec. D extends its results. Everything in this section and in Sec. G belongs to its MuJoCo backend.

## C.1. Simulation setup

Table C1 lists the constants of the simulation, and a few behaviors of the simulator decide what each row can sense. Tool results omit durations, simulated time, and the finger opening while the gripper is closed, because these reveal the load.

Table C1. Constants of HiddenPhys, shared by all tasks unless stated. Noise and bias are standard deviations of Gaussians.

| Item | Value |
| --- | --- |
| Simulator | robosuite 1.5.1 on MuJoCo 3.2.7 |
| Robot | Panda, BASIC composite controller, fixed impedance, damping ratio 1 |
| Action | operational-space pose delta and gripper at 20Hz, full scale 5cm and 0.5rad per tick, no gripper on Wipe |
| Stiffness kp | 150, 40 along a pressed axis, 10 on wrist yaw in pull, 1500 for the stiff arm of the Lock checks |
| Arrival tolerance | 3mm, 0.1mm on Lock |
| Force sensor | MuJoCo wrist sensor, rotated to the world frame |
| Force noise | 0.5N and 0.05N·m per axis and tick |
| Residual bias | 0.3N and 0.03N·m per axis and episode |
| Zero point | mean of 10 ticks at rest, at reset and after leveling the tool to within 0.5◦ |
| Scene noise | offset per object on the true pose, fixed within an episode, 4mm in x and y, 15mm on Peg, 2◦ in yaw |
| Contact, stop | contact at 3N, stop at 25N or 3N·m, force targets at most 20N |
| Episode budget | 40 turns, 80 tool calls, 300s simulated, 1200s wall clock |

The primitives level the tool from its initial pitch of about 9◦and re-zero, since the hand’s weight would otherwise leave about 0.1 N·m, or 1.3 cm of center-of-mass error.

**Stop and budget.** The stop of Table C1 binds the harness, the scripts, and the program rows. It acts when the unfiltered reading crosses the limit, and a no-force tool result says only that the motion was blocked. Scripts share the episode budget, except 500 turns and 1200 calls for the point-by-point probing of the position-only controller on Peg, and learned policies run for the horizon of Table G2.

## C.2. Task cards

Figure C1 gives the task cards, which fix what an episode hides, what the model is told, how the final state is scored, and what the expert force controller of Sec. C.3 does. Every render comes from a recorded episode of that task.

**Scoring on Lock.** Table 5 of the main paper uses the soft arm with kp 150, and the environment grasps the handle before every episode. Script rows also need a correct verdict, and harness rows do not but give 20/20 correct verdicts in both conditions. Learned policies give none, and doing nothing passes every locked door, so 10 of the 11 Diffusion Policy successes are locked doors.

## C.3. Expert and position-only controllers

The expert force controller runs one script per task, given on the cards of Figure C1, and shows how far hand-written control gets with the primitives, noise, and stop of the harness. Its descents stop at a literature default of 3 N. It was written knowing the hidden distributions, so it is a human expert’s upper bound rather than a task-free competitor. Most constants follow from the instruction. Tuning used seeds disjoint from the test seeds.

**Position-only and oracle scripts.** On the first four tasks the position-only controller runs the expert scripts on the position interface, without lift and the contact primitives. On Peg it probes along a spiral and reads the tool height for a drop, a position-only spiral heuristic. On Latch a stepped probe adds 2◦ of lever and tries a 2 mm pull until the door moves, and an alternating probe alternates a 0.5◦ hinge turn with 3◦ of lever. On Lock a stepped probe moves 1 to 5 mm along the tangent and calls the door stiff past 0.5◦, or 0.1◦. The oracle position script knows the unlock angle on Latch and the class on Lock.

## C.4. Evidence that the tasks depend on force

A task needs force only if kinematics leave the outcome ambiguous, as on Lock, where a still door may be locked or need more force. Figure C2 tests this with scripts on all 50

20

<!-- page 21 of 31 -->

Weight judge a held load

![Image block](doc:537a3fa/tier:basic/page:21/block:2)

**Hidden** cube mass, 0.15 or 0.60 kg.

**Told** both masses, place the cube in the matching 16 cm zone, give a verdict.

**Success** cube under 6 cm above the table and within 8 cm of the true zone center, fingers open over 3 cm, no gripper contact.

![Image block](doc:537a3fa/tier:basic/page:21/block:6)

![Image block](doc:537a3fa/tier:basic/page:21/block:7)

**Expert** grasp, lift 10 cm, hold 1 s, heavy above 0.375 kg, the class midpoint, from the instruction.

![Image block](doc:537a3fa/tier:basic/page:21/block:9)

**Wipe** find a surface

**Hidden** table height, the minimum of zero and a Gaussian with 3 cm deviation, lowest top 6.7 cm below the nominal 0.90 m.

**Told** table up to about 6 cm low, a spot comes off only at 5 N or more, more than 15 N fails.

**Success** all 12 spots wiped, each at a normal reading of 5 N or more, none at 15 N or more.

**Expert** fit a line through the remaining spots, slide along it at 9 N, reverse, at most 4 passes, from the instruction.

## Peg find a hole

**Hidden** hole position, reported with 15 mm Gaussian error per axis, square peg clearing the round wall by 4.5 mm.

**Told** clearance about 4 mm, position off by up to about 3 cm, feel for the hole, insert at least 2 cm.

**Success** peg center within 7.5 mm of the hole center, bottom over 2 cm below the reported hole height, release not checked.

**Expert** search a 5 cm spiral of 4 mm pitch at 4 N, insert 2 cm with a limit of 12 to 15 N, two retries at higher force, a prior.

**Lock** operate a mechanism

**Hidden** locked, or stiff and freed at a handle force uniform in 6 to 11 N, moving like a locked door until then.

**Told** handle already held, both classes, the 6 to 11 N, the 15 N line, open a stiff door at least 17◦ and answer stiff, leave a locked one and answer locked.

**Success** no damage, door above 17◦if stiff or below 2◦if locked.

**Expert** pull 15 cm along the tangent at 12.5 N, above 6 to 11 N and below 15 N, at most 5 times, stiff if the door moved over 1◦, from the instruction.

![Image block](doc:537a3fa/tier:basic/page:21/block:25)

![Image block](doc:537a3fa/tier:basic/page:21/block:26)

**CoM** judge a held load

![Image block](doc:537a3fa/tier:basic/page:21/block:28)

**Hidden** center of mass of a 12 cm, 0.8 kg bar, centered or 4 cm along the axis, in theory 0.31 N·m at the wrist.

**Told** the mass and both cases, grasp the bar at its geometric center, place it in the matching zone, give a verdict.

**Success** as Weight for the bar center, fingers open over 5 cm.

**Expert** as Weight with the tool along the bar, offset above 0.12 N·m, under half of 0.31 N·m, derived.

## Stack find a surface

**Hidden** green cube height, reported with an error uniform within ±1 cm.

**Told** top known to about ±1 cm, place the red cube gently.

**Success** robosuite check, red cube center over 4 cm above the table, not grasped, touching the green cube.

**Expert** grasp, lift, go 10.5 cm above the reported cube, descend until contact at 2.5 N, release, a prior.

**Latch** operate a mechanism

**Hidden** unlock angle uniform in 15 to 35◦, end stop 8◦beyond, detent torque rising to 0.45 N·m over the last 8◦, random hinge damping and friction.

**Told** the range, the end stop, the 15 N line, press the lever until it releases and keep it pressed while pulling, open at least 17◦.

**Success** hinge above 0.3 rad, no damage.

**Expert** turn the lever until resistance drops, pull 3 cm, turn it back, pull 12 cm at 8 N, at most 3 times, lever limit 45◦, from the instruction and derived.

## Shared by all tasks

Every check is applied in code to the final state. On Weight, CoM, and Lock the parity of the seed sets the class, and the verdict also counts as a decision. On Latch and Lock, damage is any wrist reading above 15 N while the gripper is commanded closed, counted on Lock from 1 s after closing. An expert constant comes from the instruction, is derived from it, or is a prior chosen knowing the task and its hidden distribution.

Figure C1. The seven tasks of HiddenPhys, each rendered from a recorded episode in MuJoCo. Hidden is what the seed draws and the model never sees, Told summarizes the instruction, Success is the check on the final state, and Expert is the script of the expert force controller with the constant that changes its outcome.

test seeds. On Weight and CoM the position-only controller gives one fixed answer, right on exactly half of the balanced seeds, while the expert answers all 50 on both and loses two CoM episodes on the final-state check. On Wipe the position-only controller never reaches the pressure band, and on Peg it runs out of time in 31 episodes. Stack depends on force least, as its check has no force term.

**Knowing the answer is not enough.** On Latch the oracle position script scores 40/50 against 47/50 for the expert, and 8 of its 9 damaged handles have unlock angles of 29.9◦ or more, where position targets overrun the 15 N line. On Lock with the stiff arm, the 0.5◦rule judges all 50 doors correctly

yet damages all 25 locked handles and one stiff one, since a 1 mm probe already presses about 25 N. Position targets put no bound on force.

**Choice of the position-only row.** For Latch and Lock, Table 5 of the main paper takes the variant that did best on check seeds disjoint from the test seeds.

**What the no-force control can still read.** We left kinematic cues in place. A heavy cube lowers the hand by 2 to 3 mm on Weight, and on Wipe scene lists only the remaining spots, a one-bit force reading. On Lock all 10 locked doors fail because its probe presses a locked handle to about 25 N

21

<!-- page 22 of 31 -->

![Chart block](doc:537a3fa/tier:basic/page:22/block:1)

Figure C2. Scripts without a model on the 50 test seeds 9100 to 9149. (a) Successes and (b) damaged handles out of 50. Daggers mark the rows of Table 5 of the main paper, lighter bars the variants, and hatched bars the oracle position script. On Lock a success also needs a correct decision.

unseen. On CoM we see no leak, as it answers offset in all 20 episodes.

## C.5. Seeds and protocol

Seeds pair the rows. Table 5 of the main paper uses the test seeds 9100 to 9119, the scripts and both Diffusion Policy rows also run 9100 to 9149, and Code-as-Policies and the other models of Sec. D run 9100 to 9104. Demonstrations use seeds 20000 to 20209, up to 20239 on Peg. Development used 9000 to 9019, and checks, tuning, and earlier model runs used blocks up to 7605, never a test seed.

A seed fixes the hidden variable and the layout, and the logged hidden variables match across rows.

A cell counts successes over 20 seeds by the checks of Figure C1, after done or the budget, or for learned policies after the full horizon with their last checkpoint. No test seed was dropped. Intervals are Wilson 95% intervals and tests are exact McNemar tests on paired seeds.

## D. Extended results

This section expands Tables 5 and 6 of the main paper on HiddenPhys, with test seeds 9100 to 9119 unless stated

otherwise. Costs are token counts times list prices without cache discounts.

## D.1. Per-task detail of the two harness rows

Figure D1 shows the effort behind the two GPT-6 Astra rows of Table 5 of the main paper. Force cuts both model calls and cost on every task except Weight and CoM. On Peg the search keeps running, so episodes take longer and three of them ran out of the 300 s budget. Without force Astra judges 15 of 20 Weight and 10 of 20 CoM episodes correctly against all 20 with force, and it judges all 20 Lock doors correctly in both rows, so its Lock failures without force are not wrong judgments.

## D.2. Peak force

Figure D2 shows how hard each row presses, beyond the damaged handles counted in Table 5 of the main paper. On the sorting, wiping, and stacking tasks every median peak lies between 2.2 and 19.1 N, so the figure gives only Peg, Latch, and Lock. ForceHarness has the fewest episodes above 25 N, 7 of 140, and its three on Lock, which the noforce row shares, come from the scripted pre-grasp, since every per-call peak there stays below 13 N.

## D.3. Learned baselines

Further runs test the learned rows of Table 5 of the main paper over more seeds and with fewer demonstrations. Over the 50 seeds 9100 to 9149, Diffusion Policy succeeds in 203/350 episodes without force and 204/350 with it, against 337/350 for the scripts that recorded its demonstrations. With force its median peak on Peg rises from 12.4 to 40.0 N, with 28 of 50 episodes past 25 N. Doing nothing succeeds on a locked door, so most of its Lock score is free, while stiff doors succeed in only 3 and 4 of 25 without and with force. Going from 200 to 50 demonstrations costs 21 of 140 episodes with and without force. Prompt Fusion makes π<sub>0</sub>.5 press more gently but leaves its total at 69 and 67.

## D.4. Other models

Figure D3 repeats the two harness rows on eight models with five seeds per task, and all eight improve with force. Sol nearly matches Astra at about a fifth of the cost, and GPT-5.4, GPT-5.2, and GPT-5.1 reach 19 to 21, close to Diffusion Policy at 20 and 18 without and with force on the same seeds. With force, damaged handles fall for every model, while the cost per episode falls for Astra and Sol and rises for the GPT-5.x models. Only Astra and Sol solve Latch, and Peg stays the hardest task for every model.

## D.5. Code-as-Policies per task

Table D1 splits the Code-as-Policies result of Figure 4c of the main paper by task. Without force, a program written in one call falls behind mostly on Stack and the two sorting

22

<!-- page 23 of 31 -->

![Chart block](doc:537a3fa/tier:basic/page:23/block:1)

Figure D1. Effort per episode of GPT-6 Astra on each task without force and with ForceHarness, the two harness rows of Table 5 of the main paper. Each bar is a mean over 20 episodes.

![Chart block](doc:537a3fa/tier:basic/page:23/block:3)

Figure D2. Peak wrist force on HiddenPhys. (a) Median peak per episode on the three tasks where the rows differ, with the 25 N stop of ForceHarness. (b) Episodes whose peak passes 25 N and (c) successes that stay below it, out of 140. The expert and position-only checks log handle force on Latch and Lock, so their hatched bars cover the other five tasks, 100 episodes.

![Chart block](doc:537a3fa/tier:basic/page:23/block:5)

Figure D3. Eight models in the two harness rows on seeds 9100 to 9104, five episodes per task. (a) Successes out of 35, with Diffusion Policy on the same seeds as the dashed line. (b) Cost per episode, where GPT-5.4-nano has no listed price. (c) Damaged handles out of the 10 Latch and Lock episodes.

tasks, where it cannot change course after a tool result. With force it reads our quantities, so its 27/35 measures such a program over our force interface, as Sec. G explains. None of the 70 program episodes had a program error or damaged

Table D1. Code-as-Policies per task, as successes out of 5 on seeds 9100 to 9104, against the harness rows and the expert.

<table><tr><td></td><td colspan="2">One program</td><td colspan="2">Step by step</td><td></td></tr><tr><td>Task</td><td>None</td><td>Force</td><td>None</td><td>Force</td><td>Expert</td></tr><tr><td>Weight</td><td>0</td><td>5</td><td>3</td><td>5</td><td>5</td></tr><tr><td>CoM</td><td>0</td><td>4</td><td>3</td><td>5</td><td>4</td></tr><tr><td>Wipe</td><td>3</td><td>5</td><td>4</td><td>5</td><td>5</td></tr><tr><td>Stack</td><td>0</td><td>3</td><td>5</td><td>5</td><td>5</td></tr><tr><td>Peg</td><td>0</td><td>0</td><td>0</td><td>2</td><td>4</td></tr><tr><td>Latch</td><td>2</td><td>5</td><td>3</td><td>5</td><td>4</td></tr><tr><td>Lock</td><td>3</td><td>5</td><td>2</td><td>5</td><td>5</td></tr><tr><td>Total</td><td>8</td><td>27</td><td>20</td><td>32</td><td>32</td></tr></table>

a handle.

## D.6. Form of force

Figure D4 and Table D2 expand Table 6 of the main paper per task and on paired episodes, with the same force primitives in every row except no force. Sol runs on seeds 9100 to 9109 and the GPT-5.x models of Figure D3 except GPT-5.4-nano on 9100 to 9104, reusing their no-force and L2 runs. For Sol,

23

<!-- page 24 of 31 -->

(a) GPT-6.1 Sol, successes of 10

|  | Weight | CoM | Wipe | Stack | Peg | Latch | Lock |
| --- | --- | --- | --- | --- | --- | --- | --- |
| No force | 7 | 5 | 10 | 10 | 0 | 7 | 5 |
| Event words | 6 | 5 | 10 | 10 | 3 | 10 | 10 |
| Force plot | 10 | 10 | 10 | 10 | 3 | 10 | 10 |
| Raw numbers | 10 | 9 | 10 | 10 | 2 | 10 | 10 |
| Derived signals | 10 | 9 | 10 | 10 | 2 | 10 | 10 |
| Quantities, L1 | 10 | 10 | 10 | 10 | 4 | 10 | 10 |
| + uncertainty, L2 | 10 | 9 | 10 | 10 | 4 | 10 | 10 |

(b) Smaller models, successes of 25

| Weight | CoM | Wipe | Stack | Lock |
| --- | --- | --- | --- | --- |
| 10 | 6 | 1 | 3 | 2 |
| 9 | 9 | 19 | 12 | 18 |
| 10 | 11 | 20 | 8 | 21 |
| 15 | 12 | 20 | 10 | 22 |
| 21 | 13 | 21 | 12 | 19 |
| 19 | 13 | 17 | 13 | 18 |
| 21 | 16 | 19 | 12 | 20 |

(c) Correct decisions of 25

| Weight | CoM | Lock |
| --- | --- | --- |
| 12 | 13 | 18 |
| 10 | 17 | 21 |
| 13 | 19 | 21 |
| 17 | 15 | 20 |
| 24 | 18 | 22 |
| 23 | 20 | 20 |
| 24 | 19 | 21 |

Figure D4. Form of force per task, the rows of Table 6 of the main paper. Each cell counts successes or correct decisions, and its shade gives the rate. Sol runs 10 episodes per task. The five smaller GPT-5.x models run five each, so 25 per task, and were not run on Peg and Latch.

![Chart block](doc:537a3fa/tier:basic/page:24/block:8)

Figure D5. Successes per task in distribution and under shift on seeds 9100 to 9119, out of 20. Their totals are the Shift column of Table 5 of the main paper.

Table D2. Paired comparisons with the same model, task, and seed. Each cell gives the episodes where only the first form succeeds, then those where only the second does, and the exact McNemar p.

| Comparison | Sol success | Smaller success | Smaller decisions |
| --- | --- | --- | --- |
| L2, raw | 4, 2, 0.69 | 23, 14, 0.19 | 18, 6, 0.023 |
| L2, events | 11, 2, 0.023 | 34, 13, 0.003 | 24, 8, 0.007 |
| L2, plot | 3,3,1 | 28, 10, 0.005 | 19, 8, 0.052 |
| L2, signals | 3, 1, 0.63 | 18, 16, 0.86 | 9,9,1 |
| L2, L1 | 1,2,1 | 20, 12, 0.22 | 8,7,1 |
| L2, no force | $2 0 , 1 , 2 \times 1 0 ^ { - 5 }$ | $7 1 , 5 , 5 \times 1 0 ^ { - 1 6 }$ | 29, 8, 0.0008 |
| Raw, no force | 18, 1, 8 × 10−5 | $6 3 , 6 , 4 \times 1 0 ^ { - 1 3 }$ | 23, 14, 0.19 |
| Signals, raw |  |  | 16, 4, 0.012 |
| L1, raw |  |  | 18, 7, 0.043 |

Peg is the only task that stays hard. For the smaller models, the gain of computed quantities in decisions is clearest on Weight, from 12 of 25 without force to 17 with raw numbers and 24 with L2. Since derived signals do as well as L1 and L2, we credit computing the quantity rather than its format. Raw numbers here are tared and show the last eight samples.

## D.7. Distribution shift per task

The shifted variants change only the hidden variables and the numbers in the instruction. Weight uses masses of 0.30 and 0.45 kg instead of 0.15 and 0.60 kg, CoM an offset of 2.5 cm instead of 4 cm, Wipe a table up to 3 cm lower than before, and Peg a hole noise of 25 mm instead of 15 mm, while object names, the scene format, and the success checks stay the same. No policy is retrained and the expert keeps its constants. Figure D5 expands the Shift column of Table 5 of the main paper per task.

Paired by seed, ForceHarness with GPT-6 Astra has 8 episodes that succeed only in distribution and 2 only under shift, and its 69 against the expert’s 68 is a tie. With Claude Fable 5.2 the counts are 5 and 1, while Fable without force loses on all four tasks. Over 50 seeds, the expert goes from 190 to 175 of 200 and the position-only script from 65 to 59, while Diffusion Policy goes from 128 to 107 without force and from 151 to 117 with it. On Wipe, $\pi _ { 0 . 5 }$ falls from 14 to 3 while force-reading rows hold.

Under shift ForceHarness misjudged two CoM episodes, reading offsets near 1.1 cm when the truth was 0 or 2.5 cm. On Peg, most of its failures begin with a first search that

24

<!-- page 25 of 31 -->

![Image block](doc:537a3fa/tier:basic/page:25/block:1)

![Image block](doc:537a3fa/tier:basic/page:25/block:2)

Figure E1. The web monitor while the robot runs wiggle during a USB-plug insertion. The strip at the top lists the calls so far, the left column follows the current turn, and the right column plots wrist force and torque over the last 60 s. The 8 N line guards free motion, so a force primitive may press above it, up to the 25 N cap. The page replays the recorded episode of Figure E2 in real time.

Figure E2. The page of the same episode after it ended, with the search turn selected. The operator interrupted the episode and labeled it a success by hand, so the page shows both a fault and a manual label, and the video panel is empty because the replay has no video.

25

<!-- page 26 of 31 -->

settles away from the true hole under the larger position noise. Only the frozen models read the new ranges from their instruction. Without force Astra loses most on Weight, while the image-only Diffusion Policy loses least because it never reads the noisier hole position and its wrist camera still sees the hole.

## E. Web monitor

The web monitor of Sec. 5.1 of the main paper lets a reader check any tool result against the force behind it. Figure E1 shows it while a primitive runs, and Figures E2 and E3 show the same episode after it ended.

The force panel plots the three components of F and |F|, live or over a whole episode, and marks each primitive with a band, each contact event with a triangle, and each guard stop of Sec. A.5 with a line. A second panel plots torque. The curves are raw and not tared, so a bias of about 1 to 2 N is normal. Each turn is a card from the model’s input and note to the returned tool result, and selecting it opens the action details, which is how we checked that each reported quantity matches the recorded force.

The monitor only reads run files and relays messages between the harness and the model, and all limits and checks stay in the harness.

## F. What the model sees

This section shows what a frozen model reads on Hidden-Phys under each form of force of Table 6 of the main paper and under the no-force control. Excerpts are copied from stored transcripts or source files, and three dots mark cuts.

## F.1. System prompt

All conditions share one system prompt, the template of Figure F1 with five placeholders in braces.

Here force note is the sentence of Table F1 for the form of force, interface is force or pose, and tool names lists the tools of the task and interface from Table F2. f cap is 25, and in every MuJoCo row of the main paper vision note reads “You receive NO camera images; rely on the scene description and the signals you are given.” The first user message adds the instruction, the scene as JSON, and the robot state, and a rejected call leaves the robot still and returns a tool error: line. The prompts of ForceHarness and the no-force control thus differ only in the interface, the tool list, and the force sentence, and both mention the wrist sensor and the same 25 N limit. The FR3 prompt adds that force is estimated from joint torques and that weights should be compared within an episode.

## F.2. Tool schemas

Table F2 lists the tools of HiddenPhys as the model receives them, and the interface changes only the closing sentence of

```txt
You control a Franka Panda arm in a tabletop
manipulation simulation through a small set of motion
primitives (tools).
You are not in the control loop: each primitive runs its
own closed-loop controller at 20 Hz and returns when
it finishes,
when it is blocked, or when a safety limit trips. You
decide what to do between primitives.
Action interface for this episode: {interface}.
Available tools: {tool_names}.
Sensing you receive: {force_note} {vision_note}
2. Prefer small, deliberate motions near objects. Any
motion whose wrist force exceeds {f_cap:g} N is
aborted by the safety layer.
3. When something is uncertain and cannot be seen, you
may probe it physically (touch, lift, push) if your
tools allow it.
4. Be honest in done: report success=true only if you
believe the task is actually complete; give a verdict
when the task asks for one.
```

Figure F1. The system prompt shared by all conditions, verbatim without its frames and units, its wrist-sensor line, and rules 1 and 5.

Table F1. The sentence that each form of force adds to the system prompt, verbatim. Every form after no force opens with “After each primitive you receive”, cut at the three dots, and L2 appends its sentence to that of L1.

| Form | Sentence |
| --- | --- |
| No force | You receive NO force or torque information; only motion results. |
| Event words | ... a list of detected contact EVENTS (contact start/end, safety cap, stall, slip, resistance drop) and whether the tool is in contact at the end. |
| Force plot | ... a PLOT (image) of the wrist force and torque versus time. |
| Raw numbers | ... the raw wrist force/torque numbers (current value and recent samples). |
| Derived signals | ... detected contact EVENTS plus derived signals (peak force, contact direction, payload estimate when lifting, mean normal force when pushing/sliding, torque peak when rotating). |
| Quantities, L1 | ... detected contact EVENTS plus physical MEASUREMENTS computed from the wrist force/torque (payload mass, centre-of-mass offset of the held load, contact height/point, ...), each with units. |
| + uncertainty, | Each measurement carries an uncertainty (±) and |
| L2 | quality flags. |

the motion tools and the rotations, marked G and H. Wipe has no gripper and drops grasp, release, and lift, while point tool and pull appear on Lock and Latch only and turn on Latch only. A third interface, the pose tools with a stop on contact, serves only the real-robot no-force control of Sec. I, and ForceBench adds the tools of Sec. H.

## F.3. Tool results under each form of force

Figure F2 shows one lift on CoM, where the bar’s true offset is 4 cm, under the seven forms of Table 6 of the main paper, with GPT-6.1 Sol at test seed 9100. All seven episodes

26

<!-- page 27 of 31 -->

Table F2. Tool schemas on HiddenPhys, as the model receives them. Descriptions are verbatim and cut at the three dots, a question mark marks an optional parameter, and G and H stand for the interface sentences in the last rows.

<table><tr><td>Tool</td><td>Parameters</td><td>Description</td></tr><tr><td colspan="3">Both interfaces, pose and force</td></tr><tr><td>scene</td><td>none</td><td>Get the current scene description ... and the robot state.</td></tr><tr><td>move_to</td><td>x, y, z, speed?</td><td>Move the tool centre point to an absolute world position ... G</td></tr><tr><td>move_by</td><td>dx, dy, dz, speed?</td><td>Move the tool centre point by a relative offset ... G</td></tr><tr><td>rotate Z</td><td>angle_deg, torque_cap?, stop_on_resistance_drop?</td><td>Rotate the tool about the vertical axis by angle_deg ... H</td></tr><tr><td>grasp</td><td>none</td><td>Close the gripper...</td></tr><tr><td>release</td><td>none</td><td>Open the gripper.</td></tr><tr><td>hold</td><td>seconds?</td><td>Stay still for seconds seconds (measures the wrist signal while holding).</td></tr><tr><td>point_tool</td><td>direction</td><td>Re-orient the tool so the fingers point along direction ... G</td></tr><tr><td>turn</td><td>axis, pivot, angle_deg, rate_deg-s?, torque_cap?, stop_on_resistance_drop?</td><td>Rotate the hand rigidly by angle_deg about the axis that passes through pivot .</td></tr><tr><td>done</td><td>success, verdict?, note?</td><td>Declare the episode finished. ...</td></tr><tr><td colspan="3">force interface only</td></tr><tr><td>move_until_contact</td><td>direction, max_dist, f-threshold?</td><td>Move slowly along a direction vector until the wrist senses contact ..</td></tr><tr><td>push</td><td>direction, target force, duration?</td><td>Press along a direction and regulate the contact force ..</td></tr><tr><td>slide</td><td>direction, distance, normal_force?, normal? normal?, radius_max?, pitch?, normal-force?, drop?,</td><td>Slide along a surface: keep normal_force newtons against the surface ...</td></tr><tr><td>search</td><td>mode?, hop?, speed?</td><td>Insertion search around the current position ... Stops with result &#x27;dropped&#x27; as soon as the tool advances more than drop metres along the normal ...</td></tr><tr><td>insert</td><td>depth?, normal?, max force?</td><td>Compliant insertion along normal (default downward): ...</td></tr><tr><td>lift</td><td>height?, hold_time?</td><td>Raise the tool vertically by height metres, then hold still ..</td></tr><tr><td>probe</td><td>direction?, amplitude?, f-threshold?</td><td>Tap: move along direction until contact ... and return to the start. ...</td></tr><tr><td>pull</td><td>direction, force?, distance?, speed?</td><td>Pull something you are holding that is attached to a mechanism ...</td></tr><tr><td colspan="3">Interface sentences</td></tr><tr><td>G, force</td><td colspan="2">The motion STOPS EARLY if the tool makes contact with anything (contact guard) and reports it.</td></tr><tr><td>G, pose</td><td colspan="2">The motion completes the full commanded displacement unless a safety force limit is hit.</td></tr><tr><td>H, force H, pose</td><td colspan="2">Optionally stop when torque exceeds torque_cap or when resistance peaks and then drops.</td></tr><tr><td></td><td colspan="2">Rotation completes fully unless a safety torque limit is hit.</td></tr></table>

answer OFFSET, which is correct, but without force the model answers OFFSET at every seed from 9100 to 9104, whatever the truth. Event words print |F| only at an event, so a still hold gets no reading, and on Weight GPT-6.1 Sol with event words then calls 4 of the 5 heavy cubes light. Derived signals give the torque in N m and leave the lever to the model, while L1 and L2 give the offset in centimeters.

**Proprioception.** Kinematics can leak force, so on HiddenPhys and ForceBench every form omits the primitive’s duration and, while carrying, the gripper opening, which change with the load, but keeps the end position and displacement. The hand still sags by 2 to 3 mm under a heavy cube and 1 mm under a light one, so the no-force control is not blind to force and may score above a truly force-blind model. We did not hide more, to avoid an artificially ignorant control. On the FR3, tool results keep the duration and the gripper opening in both conditions.

## F.4. One short episode as a trace

A short Lock episode at test seed 9100 shows both conditions with GPT-6.1 Sol and the same instruction and scene, and the model wrote no text besides its tool calls.

```jsonl
TASK: Your gripper is already holding the door handle.
This door is either LOCKED or just STIFF (if stiff, it
takes 6-11 N at the handle to get it moving). ... If
it is locked, do not force it: call done with verdict
’locked’. Any force above 15 N on the handle damages
it and fails the task. ...
== ForceHarness
call 1: pull {"direction": [-0.12, 0.993, 0], "force":
12, "distance": 0.115, "speed": 0.015}
result: stalled
measurements (pull):
mean pulling force 8.82 ± 0.84 N; sideways
force 1.75 N
[last 80%; flags: mechanism_stopped]
call 2: done {"success": true, "verdict": "locked",
"note": "The door did not rotate under a controlled 12
N pull, above the stated stiff-door breakaway range
and below the 15 N damage threshold."}
== No-force control: calls 1 to 4 completed
call 5: move_to {"x": -0.1402, "y": -0.247, "z": 1.078,
"speed": 0.001}
result: stopped early (motion blocked, safety stop)
call 6: done {"success": true, "verdict": "locked",
"note": "Handle remained immobile under small, slow
opening-direction probes; stopped probing at
motion-blocked indication."}
```

Under ForceHarness the hand does not move and the primitive stops itself at a mean of 8.82 N, and the record gives a 12.58 N peak handle force, no damage, and success. Under no force, tool results print no force, events, or contact

27

<!-- page 28 of 31 -->

```txt
== Event words, full result == Raw numbers
primitive=lift params={’height’: 0.12, ’hold_time’: 2} ...
result: completed current: Fx=-1.32 Fy=-0.33 Fz=-7.83 Tx=-0.09 Ty=-0.25
end position [-0.015, -0.004, +0.945] m; commanded Tz=-0.00
displacement [+0.000, +0.000, +0.127], actual [-0.003, last 8 samples (t_rel, Fx, Fy, Fz, Tx, Ty, Tz):
-0.000, +0.123] +8.10, -1.06, +1.01, -7.82, -0.11, -0.27, -0.00
tool yaw changed by -0.1 deg
note=a torque about a horizontal axis while holding == Derived signals
still means the load’s centre of mass is offset from ...
the grasp peak |F| during primitive: 9.0 N; peak |T|: 0.40 Nm
contact events: none (no contact detected during this derived signals: hold_force_z_N=-7.78,
primitive) hold_force_z_std_N=0.52, payload_estimate_kg=0.793,
contact state at end: IN CONTACT hold_torque_xy_Nm=[-0.054, -0.246],
robot state: tool at [-0.015, -0.004, +0.945] m, gripper hold_torque_xy_mag_Nm=0.252, ...
closed == Quantities, L1
== No force, a move_by since this interface has no lift ...
primitive=move_by params={’dx’: 0, ’dy’: 0, ’dz’: 0.12} measurements (lift):
result: completed payload mass 0.793 kg
end position [-0.015, -0.004, +0.938] m; commanded centre-of-mass offset 3.22 cm toward mostly +x
displacement [+0.000, +0.000, +0.120], actual [-0.003, (away from the base)
-0.000, +0.117]
== + uncertainty, L2, the default of ForceHarness
== Force plot ...
measurements (lift):
wrist force/torque during this primitive is shown in the payload mass 0.791 ± 0.032 kg
attached plot (world frame, N / Nm, bias removed; [last 40 samples, 2.0 s]
dotted vertical lines mark detected events). centre-of-mass offset 3.32 ± 0.29 cm toward
<image attached> mostly +x (away from the base)
[last 40 samples, 2.0 s]
```

Figure F2. The same lift on CoM at test seed 9100 under each form of force, verbatim, with one episode per form. Lines starting with == are our labels, three dots outside Event words stand for the shared kinematic, contact, and robot-state lines, and &lt;image attached&gt; stands for a PNG that plots force and torque against time.

state, and a stop on force loses its number, so the fifth says only that a safety stop ended the motion, which the prompt ties to 25 N. The model reports success with the correct verdict, but the record gives a 25.12 N peak, damage, and failure.

## G. Baselines

This section describes how the baseline rows of Table 5 of the main paper were built, with the training recipes in Table G1 and the demonstrations in Table G2. We train every learned baseline on HiddenPhys ourselves, each with one training seed.

## G.1. Demonstrations

All learned policies train on the successful episodes of the expert force controller, counted in Table G2, using all 185 on Peg. Lock uses the soft arm and records after the pre-grasp. The recorder stores the action that the environment received plus six stiffness values, because the primitives switch stiffness and a policy with fixed stiffness could not learn the expert’s contact behavior. Force is stored noise-free and receives the noise of Sec. C.1 at conversion.

## G.2. Training recipes

Each learned policy departs from its published recipe as described below, and one evaluation client applies the damage rules of Sec. C.2 to all of them.

**Diffusion Policy.** Our implementation of the CNN Diffusion Policy [6] departs from its robomimic image setup in using 16-step DDIM sampling, stiffness in the action, and images of 128 resized to 96 and cropped to 88 rather than 84 and 76. Force is concatenated with the low-dimensional state, matching the concatenation baseline in FoAR [15] and ForceVLA [39]. The images-only rows drop the scene vector but keep proprioception, as ForceBench does, and equal the full input on Wipe, whose scene vector is empty. The scene vector is computed once and holds no door or lever angle, while a frozen model can call scene at any time.

π<sub>0</sub>.5. We fine-tune π<sub>0</sub>.5 [24] with unmodified openpi from pi05 libero, changed so that the discretized state enters the prompt, the action includes stiffness, and one model at batch 32 serves all 7 tasks, each with one short fixed prompt without numbers. With force, the 6 force dims join the state and enter the prompt as tokens of 256 levels, as in the Prompt Fusion of ForceBench [14], which used it on other tasks and actions. About 1 to 2% of force frames fall outside the normalization range and take the limit tokens, as in openpi.

**ForceVLA.** The other learned rows concatenate force, so we added the force-aware ForceVLA [39] lest our conclusion on trained force reading rest on concatenation alone, keeping its model code but changing the tasks, the data, and the steps. ForceVLA also differs from π<sub>0</sub>.5 in backbone, state input, normalization, and batch size, so we do not compare these

28

<!-- page 29 of 31 -->

Table G1. Training recipes of the learned policies on HiddenPhys. All three see two camera views and predict a pose delta of 7 dims plus 6 stiffness values, and each is trained with one seed and evaluated at its last step, with no checkpoint selection.

|  | Diffusion Policy | π0.5 | ForceVLA |
| --- | --- | --- | --- |
| Model | ResNet18 per view from scratch, GroupNorm, spatial softmax, 32 keypoints, 1D U-Net 256, 512, 1024 with FiLM, 100 DDPM and 16 DDIM steps | pi05_base, all parameters trained | π0 from piO_base, force token, LIMoE of 4 experts, ForceBench code |
| State | proprioception 11, scene 5 per object | proprioception 11, scene 15, as prompt tokens after quantile normalization | continuous, with the noisy scene |
| Force | 6 dims in the world frame, zeroed at reset, noisy 6 pose dims on Wipe, 2 frames observed, 16 | 6 dims as prompt tokens 10 predicted, 5 executed, min-max for | force token |
| Action | predicted, 8 executed | stiffness, pose deltas widened to \|z\| ≤ 5 | 13 dims, not the 7-dim absolute pose of ForceBench |
| Data | one model per task, first 200 saved episodes by seed, 185 on Peg, 197 on Latch | one model for 7 tasks, all 1423 episodes, task selected by prompt | as π0.5 |
| Training | 40 000 steps, batch 64, AdamW at 1e-4, cosine, 500 warmup steps, EMA | 30 000 steps, batch 32, seed 42, AdamW, peak 5e-5, 10 000 warmup steps, EMA 0.999 | 30 000 steps, global batch 64, optimizer and seed of ForceBench |

Table G2. Expert episodes recorded per task and the successful ones saved for training, with lengths in ticks at 20 Hz. The rollout horizon of a learned policy is 1.5 times the longest demonstration of the task.

| Task | Attempts | Saved | Frames | Longest | Horizon |
| --- | --- | --- | --- | --- | --- |
| Weight | 210 | 210 | 144312 | 723 | 1084 |
| CoM | 210 | 208 | 140607 | 709 | 1063 |
| Wipe | 210 | 207 | 68960 | 453 | 679 |
| Stack | 210 | 206 | 121311 | 793 | 1189 |
| Peg | 240 | 185 | 122194 | 2310 | 3465 |
| Latch | 210 | 197 | 111329 | 673 | 1009 |
| Lock | 210 | 210 | 18154 | 254 | 381 |
| Total | 1500 | 1423 | 726867 |  |  |

two rows directly.

## G.3. Code-as-Policies

In the program rows of Sec. 5.4 of the main paper, GPT-6 Astra at low reasoning effort writes one Python program per episode in a single call without examples, and the program runs with no further model calls. Its API comes from the harness tool schemas with the same 25 N stop and returns kinematics, and our force API adds the force primitives, whose results carry contact events, peak and final forces, and the physics translation layer. The force row thus measures a single zero-shot program written over our force interface, closer in spirit to code generation over compliant primitives [4, 22] than to the original Code-as-Policies [20], which uses few-shot examples, hierarchical generation, and no force.

## H. ForceBench port

This section gives the details behind Sec. 5.2 and Table 4 of the main paper, all in simulation on the public ForceBench repository [14] at commit 31acf73 with Isaac Sim 6.0.1 and Isaac Lab 3.0.0b2.post1.

## H.1. Setup

The port keeps ForceBench’s evaluator and success criteria and adapts only the harness. ForceHarness serves as a websocket policy that returns a target for the tool center point and a gripper value per request, and an adapter gives the primitives the interface of the robosuite backend, so the primitives, the physics translation layer, and the agent loop are not edited. Each request also carries both camera calibrations, which only locate and locate object use. The arm controller is softer than in MuJoCo, so the lead cap while seeking contact is 15 mm instead of 6 mm, and the adapter compensates the sag of a held load. An episode ends after ForceBench’s 100 s of simulated time, 30 turns, or 60 tool calls.

Each turn the model sees the main and wrist RGB views at 224 by 224 pixels, with no depth and no object coordinates. Each instruction adds to ForceBench’s task sentence a hand-written paragraph of dimensions, key heights, and facts from the public task code, such as the two mass classes on Weight Sort, but never an object’s position or a hidden value. ForceHarness offers 16 to 19 tools per task and the noforce control 9 to 11, and on Stove Press and Nut Thread the control’s move turn takes the place of press turn, while Nut Thread offers move turn under ForceHarness too. Tools and perception thresholds were developed on seeds 300 to 307, and after first runs on seeds 300 to 309 Nut Thread gained locate object with a corrected instruction and press turn came to print the resisting torque, first as a mean and then per side.

## H.2. Tasks and results

Table H1 lists what each task hides and how the public evaluator decides success. On Faucet Turn, Stove Turn, and Drawer Pull it omits conditions that the ForceBench paper lists, shown in the last column. A scripted state machine per task on our tools succeeds in 50/64 episodes on seeds 300 to 307 and in 48/64 on seeds 308 to 315, which shows that the tasks are feasible from RGB.

29

<!-- page 30 of 31 -->

Table H1. The eight tasks. C is the sampled angle of the resistance peak, the detent, d the hidden sign of the easier direction, and $x _ { t }$ the opening at 2.35 N of designed resistance. The third column is the public evaluator at commit $3 l a c f 7 3 ,$ which produced our rows, and the last column what Appendix A of the ForceBench paper requires beyond it.

| Task | Hidden variable | Success in the evaluator we ran | Also required in the paper |
| --- | --- | --- | --- |
| Weight Sort | mass class | correct container, release, settling | - |
| CoM Sort | CoM centered or 2cm off | correct destination, release, settling | - |
| Faucet Turn | C from 20◦ to 30◦ | angle past C + 0.95◦, water visible | final angle in [C,C + 2.1◦], release, clearance, rest |
| Faucet Direction | C and d | final dθf ≥ C, water visible, release, clearance, rest | - |
| Stove Turn | C from 60◦ to 90◦ | crossing at C + 0.5◦, final angle within 3◦, flame visible | release, clearance |
| Stove Press | C and d | dθ ≥ C + 0.5◦ while pressed | - |
| Drawer Pull | resistance profile | opening within 2.5mm of x<sub>t</sub> for 15 slow frames, latch intact | release |
| Nut Thread | bolt position | center within 2.5mm, height -2 to +1mm, tilt, release, clearance | - |

Table H3. All ten trained methods of the ForceBench paper, success in percent as reported in its Table I and not reproduced by us, with 200 demonstrations and 200 evaluation episodes per task. Table 4 of the main paper shows six of these rows.

| Method | Weight | CoM | Faucet | Faucet dir. | Stove | Stove press | Drawer | Nut | Mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| π0.5, no force | 54.0 | 45.5 | 9.5 | 31.5 | 8.5 | 1.0 | 4.5 | 31.5 | 23.25 |
| π<sub>0.5</sub> + Token Fusion | 50.5 | 50.0 | 12.0 | 28.5 | 6.0 | 2.5 | 13.0 | 26.0 | 23.56 |
| π<sub>0.5</sub> + Prompt Fusion | 88.0 | 56.5 | 21.0 | 41.5 | 16.0 | 2.5 | 23.0 | 34.0 | 35.31 |
| π<sub>0.5</sub> + Cross-Attention | 54.5 | 51.0 | 6.5 | 33.0 | 5.5 | 2.5 | 6.5 | 17.5 | 22.13 |
| π<sub>0.5</sub> + Token Fusion + Pred | 78.0 | 48.5 | 13.5 | 32.5 | 6.0 | 1.0 | 4.0 | 26.5 | 26.25 |
| π<sub>0.5</sub> + Temporal Prompt | 90.0 | 69.5 | 17.5 | 31.0 | 7.0 | 21.5 | 2.5 | 22.0 | 32.63 |
| π<sub>0.5</sub> + Dual-Path Fusion | 99.5 | 72.0 | 22.0 | 30.0 | 17.5 | 19.0 | 4.5 | 16.5 | 35.13 |
| π<sub>0</sub> | 61.5 | 39.0 | 3.5 | 27.0 | 7.0 | 5.0 | 7.5 | 36.0 | 23.31 |
| ForceVLA | 35.5 | 45.5 | 12.5 | 14.0 | 10.0 | 12.0 | 20.0 | 26.0 | 21.94 |
| ForceFlow | 51.5 | 3.0 | 0.0 | 0.5 | 0.0 | 0.0 | 0.0 | 0.0 | 6.88 |

Table I1. The six real-robot tasks as in the final runs. Actions gives the largest number of tool calls in one episode.

| Task | Scene | What the hand must do | Actions |
| --- | --- | --- | --- |
| Wipe a whiteboard | marker writing, an eraser | find the board by touch and wipe until the images show no writing | 40 |
| Drive a screw | a screw and a gear with a threaded center hole, on a wooden board | find the hole by touch, press the tip in, and turn with the thread, grasping again at the wrist limit | 60 |
| Peel an eggplant | an eggplant on foam, a clamped peeler | keep the blade in the curved skin while moving along +x | 60 |
| Insert a USB plug | a USB-A plug on a cable, a power strip | reach a port, find it by a light sliding search, and seat the plug | 40 |
| Sort three bottles by | look-alike bottles of 294, 204, and $1 6 0 \mathrm { g } , \mathrm { a }$ | lift each by its cap, take the heaviest to the white plate and the | 80 |
| weight | white and a green plate | lightest to the green one |  |
| Tell soft from hard | two wrapped objects of one shape | say which is soft without grasping | 15 |

Table H2 gives the per-task counts behind Table 4 of the main paper, and Table H3 lists all ten trained methods of the ForceBench paper, which learn from 200 demonstrations per task, see only the task sentence, and are evaluated on seeds 300 to 499. The force gain is largest on Drawer Pull. In 19 of 20 episodes with force, the model calls pull once with a 2.4 N target, and the hand stops where the resistance balances it, 1.1 to 1.7 mm from the goal against a 2.5 mm window. Without force no fixed motion can work, since the target opening ranges from 63.5 to 90.7 mm, and all 20 episodes stop short.

## I. Real-robot system and tasks

This section summarizes how the harness runs on the real robot of Sec. 5.1 of the main paper, a Franka FR3 without a wrist force sensor, and every constant here belongs to that arm.

**System.** The stock joint-impedance controller of Polymetis runs unchanged at 1 kHz, with joint gains at four times the default and eight times on joint 7, and we move only its joint target. Beside it on the real-time computer, our guard runs at a nominal 50 Hz and enforces the hard limits of Sec. A.5. The harness on the workstation runs the primitives and estimators and sends the guard one step per tick, at a median rate of 14.9 Hz. The frozen model is called once per

30

<!-- page 31 of 31 -->

Table H2. Successes out of 10 per task on seeds 300 to 309, with ForceHarness as Force, the no-force control as None, and Pairs counting the seeds on which only ForceHarness succeeds, then only the control. On Stove Press, ForceHarness uses the mean report for Astra and the per-side report for Sol, and Nut Thread uses its revised version in all four conditions.

<table><tbody><tr><td rowspan="2">Task</td><td colspan="3">GPT-6 Astra</td><td colspan="3">GPT-6.1 Sol</td></tr><tr><td>Force</td><td>None</td><td>Pairs</td><td>Force</td><td>None</td><td>Pairs</td></tr><tr><td>Weight Sort</td><td>10</td><td>8</td><td>2 / 0</td><td>10</td><td>9</td><td>1 / 0</td></tr><tr><td>CoM Sort</td><td>7</td><td>5</td><td>3 / 1</td><td>6</td><td>5</td><td>4 / 3</td></tr><tr><td>Faucet Turn</td><td>10</td><td>10</td><td>0 / 0</td><td>10</td><td>9</td><td>1 / 0</td></tr><tr><td>Faucet Direction</td><td>5</td><td>5</td><td>4 / 4</td><td>4</td><td>3</td><td>4 / 3</td></tr><tr><td>Stove Turn</td><td>10</td><td>10</td><td>0 / 0</td><td>10</td><td>10</td><td>0 / 0</td></tr><tr><td>Stove Press</td><td>9</td><td>8</td><td>1 / 0</td><td>10</td><td>6</td><td>4 / 0</td></tr><tr><td>Drawer Pull</td><td>10</td><td>0</td><td>10 / 0</td><td>10</td><td>0</td><td>10 / 0</td></tr><tr><td>Nut Thread</td><td>9</td><td>10</td><td>0 / 1</td><td>10</td><td>10</td><td>0 / 0</td></tr><tr><td>All tasks</td><td>70/80</td><td>56/80</td><td>20 / 6</td><td>70/80</td><td>52/80</td><td>24 / 6</td></tr></tbody></table>

turn and answers with one tool call.

**Force estimate.** Polymetis forwards the filtered external joint torques that libfranka estimates but not a wrench, so the guard solves Eq. 1 of the main paper on every tick by least squares with the flange Jacobian of the controller’s own kinematic model. It also checks the joint torque that no flange wrench explains, and a residual above 3 N·m or a Jacobian near a singularity ends the episode with a fault. The estimate has little noise, 0.03 to 0.05 N at rest, but its zero shifts by a few newtons with the approach direction, joint friction in motion, and the wrist angle. Filtering cannot remove such an offset, so we calibrate only the offset and read force by the three rules of Sec. 3.2 of the main paper. The harness tares with the gripper open before each empty-hand primitive, and after every primitive it relaxes the target onto the measured pose for 0.8 s and then averages the wrench over 0.5 s. Relaxing lowers the median difference between consecutive still readings from 1.07 N to 0.08 N over 999 empty-hand windows, because a target that leads the joints pushes against joint friction, which the estimate reports as external force. Each still reading is reported relative to the same reading at the start of the episode, and with an object in the hand also as the change since the grasp. Lifted weights are therefore compared within an episode rather than read in grams.

**Safety.** Beneath the guard of Sec. A.5, the harness adds collision stops at 8 N, or 12 N when holding, in the guarded tools of Sec. A, and it ends an episode after the number of actions in Table I1. An operator stands at the emergency stop in every episode.

**Localization.** The exterior camera is uncalibrated, so every coordinate comes from fixed geometry on the wrist depth image, with no learned detector. A pixel maps to the base frame through the flange pose at capture and the hand–eye calibration of the wrist camera. Each turn, the harness groups points about 1 cm or more above the fitted table plane into objects and lists their center, top, and long axis with its size, and the model picks targets itself with locate. The hand–eye calibration has a median error of 4.1 mm and 0.28◦ on 6 held-out views. The model is told that coordinates are good to about 1 to 2 cm, so vision alone cannot reach a port, and its last millimeters are left to search.

**Tasks and conditions.** Table I1 describes each task as in the final runs, and Sec. 5.1 of the main paper gives the success criteria. ForceHarness offers all 22 tools of Sec. A except wiggle and press turn, USB insertion adds wiggle, and screw driving swaps lift for wiggle. The no-force control keeps the 11 tools that move by position only or not at all, and its tool results report only kinematics. In peeling it also gets move until contact, which stops at a fixed 5 N and reports no force.

## J. Limitations and reproducibility

## J.1. Limitations

Querying a remote frontier model adds seconds of latency between primitives, which suits quasi-static contact tasks but precludes fast reactive replanning during dynamic motion. Both the primitive menu and the physics estimators are specified by hand, and each task relies on a hand-written instruction, leaving automatic primitive synthesis and open-ended task specification to future work. Finally, our evaluation focuses on single-arm tabletop manipulation with a parallel gripper across models from two API providers, and our component ablations are conducted in simulation rather than repeated on real hardware.

## J.2. Reproducibility

HiddenPhys uses robosuite 1.5.1 and MuJoCo 3.2.7, ForceBench runs on Isaac Lab with the changes of Sec. H.1, and the real robot runs stock Polymetis with its hybrid joint impedance controller. Code and configurations will be released upon publication, together with logs and task assets.

Model settings are those of Sec. 5 of the main paper, and the GPT-5.x models use chat completions with reasoning off when tools are present.

31