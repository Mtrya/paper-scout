# Learning Beyond Gradients

Jiayi Weng  
URL: https://trinkle23897.github.io/learning-beyond-gradients/  
Artifact repository: https://github.com/Trinkle23897/learning-beyond-gradients

> [Jiayi Weng](https://trinkle23897.github.io/cv/)

Continual Learning has remained hard largely because of catastrophic forgetting in neural networks: learn something new, and old capabilities can get overwritten. But what if we do not put all of our attention on neural network weights? Is there another way to make progress?

As LLM agents get stronger, coding gets faster and better. But the phenomenon I find more interesting is different: a coding agent can keep reading failures, editing code, adding tests, and watching replays, and a program system can improve without training a new network or updating weights.

That made me rethink heuristics: hand-written rules and programmatic policies. Many heuristics were not useless; they were simply too expensive to maintain. Coding agents change that maintenance curve. Rules that used to be one-off patches may start to become code worth owning for the long term.

Anything that can be iterated on continuously starts to become more solvable. That is also what Continual Learning has always wanted. Could this become the next paradigm after pretraining, RLHF, and large-scale RL/RLVR?

## The Anomaly

While maintaining [EnvPool](https://github.com/sail-sg/envpool) in my spare time, I wanted a cheaper way to test whether game environments were behaving correctly. Running a neural network every time was too expensive for CI.

The initial question was:

```text
Can we write cheap, reproducible heuristics that are much stronger than a random policy,
and use them to drive environments into informative states?
```

I tried using Codex (`gpt-5.4`) to write a rule-based version, with no neural network at all. After a few rounds, the results were far more surprising than I expected:

- In Atari Breakout, a programmatic policy went from `387 -> 507 -> 839 -> 864`, eventually reaching the theoretical maximum score.
- In MuJoCo Ant, a pure Python policy first learned a rhythmic gait, then added short-horizon model planning, and finally reached `6000+`, in the range of common Deep RL results.
- In MuJoCo HalfCheetah, interpretable gait/posture rules plus online planning reached a five-episode evaluation mean of `11836.7`, also in the range of common Deep RL results.
- In VizDoom, a first-person visual-control task, cv2/NumPy screen CV without neural-network training reached `mean=557.0` and `min=440.0` over 10 seeds on D3 Battle.
- Across Atari57, I ran `57 games x 2 observation modes x 3 repeats = 342` coding-agent search trajectories. The results were uneven, but under a fixed environment-step budget, median HNS around `1M` environment steps was already far above PPO-style Deep RL baselines at the same step count.

The raw scores were surprising enough. But the more interesting part was: Codex was not training a neural network. It was maintaining a software system that could keep growing.

Breakout policy was far beyond "move left when the ball is on the left." The policy had grown action probes, state readers, ball and paddle detectors, landing prediction, stuck-loop detection, regression tests, video replays, and experiment logs. Ant policy was more than a gait formula too: it had rhythmic control, posture feedback, contact signals, and short-horizon model rollouts.

That is when I felt a new concept was needed. The thing being updated was no longer just a policy function. It was a software system with memory, feedback channels, and regression mechanisms.

## Heuristic Learning

After more iteration with Codex, I started calling this process **Heuristic Learning (HL)**:

- HL is built out of program code.
- Like Deep RL as commonly practiced today, it has a loop of state, action, feedback, and update; unlike that setup, the object being updated is software structure rather than neural-network parameters.
- Its feedback is consumed by a coding agent, and can come from environment reward, test cases, logs, videos, replays, or human feedback.
- Its updates do not use backpropagation. The coding agent directly edits policies, state detectors, tests, configuration, or memory.
- HL is the learning and update process. The object maintained by HL over time can be called a **Heuristic System (HS)**.
- An HS is more than an isolated `policy.py`. It contains at least a programmatic policy, state representation, feedback channels, experiment records, replays or tests, memory, and an update mechanism executed by a coding agent. A single rule is not enough. Rules, feedback, history, and the next update path all need to connect before it becomes an HS.

As a table:

| Axis | Deep RL | HL |
| --- | --- | --- |
| Policy | Neural network parameters | Code: rules, state machines, controllers, MPC, macro-actions |
| State | Usually explicit observations | Usually explicit variables, detectors, caches, and other readable representations |
| Action | Produced by a neural network forward pass | Produced by executing code logic |
| Feedback | Mainly fixed reward | Provided through coding-agent context: tests, environment feedback, logs, and replays all count |
| Update | Gradient-based updates to neural-network parameters in a Deep RL algorithm | Direct code edits by a coding agent |
| Memory | On-policy methods basically have none; off-policy methods have replay buffers | Can explicitly store trials, summaries, failure reasons, replays, and version diffs |

Heuristic Learning has several useful properties compared with Deep RL:

- Explainability: neural networks are hard to explain, while HL policies can often be translated into plain language.
- Sample Efficiency: one effective code update can jump directly to a new policy, rather than slowly climbing through learning-rate tuning.
- Regression-testability: old capabilities can become tests, replays, or golden cases.
- Overfitting can be constrained: code heuristics can still overfit to seeds, environment details, or test loopholes, but simplification, regression checks, and multi-seed evaluation provide an engineering form of regularization.
- It can avoid part of catastrophic forgetting: old capabilities do not have to live only inside model weights; they can be written into rule sets and tests.

The point is that a class of heuristics that used to be too expensive to maintain may now be worth owning.

## Why Heuristic Learning Did Not Take Off Earlier

If HL has ancestors, they are expert systems and rule systems. Before coding agents, their maintenance cost was brutal.

Human-maintained heuristics easily turn into this:

```text
Add one rule today to fix case A.
Tomorrow, case B breaks.
Add another if-statement the day after.
The day after that, nobody dares delete anything.
```

The problem was not that heuristics were useless. The problem was that humans could not afford to keep maintaining them. Maintaining expert systems by hand was a bit like spinning thread before the Industrial Revolution: one person can do it, but once the scale grows, stability and maintenance cost become crushing. Spinning machines changed the production curve; coding agents change the maintenance curve for heuristics. They act like a channel that can continuously deliver intelligence into a Heuristic System and let it keep evolving.

![coding agent connects feedback into software growth](ig_0c2dd0d2f07176560169fbc256930481969d3c6ba3316d5486.png)

The common agentic feedback loop today looks roughly like this:

```text
feature request -> agent writes code -> tests pass -> human gives some feedback -> next patch
```

As models improve, human intervention should shrink. In systems with clear boundaries, this feedback loop can start to close automatically, allowing HL to produce Heuristic Systems in bulk:

```text
environment feedback / test failure / log anomaly
-> coding agent reads context
-> edits policy / test / memory
-> reruns
-> writes results back into trials and summaries
-> continues to the next round
```

## How Heuristic Learning Does Continual Learning

In neural networks, catastrophic forgetting happens when new data pushes parameters toward a new task and old capabilities get overwritten. HL can forget too, just in a more engineering-shaped way:

- A new rule fixes one failure mode and breaks an old scenario.
- A new memory repeatedly steers the agent in the wrong direction.
- A test is too narrow, so the policy learns to exploit it.
- A patch changes a shared interface, and old callers quietly break.
- Rules keep piling up until even the agent can no longer maintain the system.

So HL does not automatically solve Continual Learning. It turns "avoiding forgetting" into a more engineering-oriented problem.

In HL, old capabilities can be fixed into:

- regression tests;
- fixed-seed replays;
- golden traces;
- failure videos;
- version diffs;
- explicitly written-down failed directions.

This is very different from compressing experience into neural network weights. HL history is explicit, readable, deletable, and refactorable. It is responsible for remembering, but also for compressing a pile of local patches into a simpler representation.

(An HS that only grows and never compresses will eventually become a big ball of mud. It may "remember" many things, but the form of memory is so poor that nobody dares touch it, and the system decays.)

A healthy HS therefore needs at least two operations:

1. Absorb feedback: write new failures, logs, and rewards back into the system.
2. Compress history: fold local patches back into simpler, more maintainable representations.

That turns Continual Learning from "how do we update parameters?" into "how do we maintain a software system that keeps absorbing feedback?"

## The Complexity of Heuristic Systems

Here I define **coupling complexity** as the level of strategy complexity a coding agent can maintain in order to support HL. More concretely, it is how many interdependent states, rules, tests, feedback signals, and historical constraints an update has to account for at the same time.

This cannot be measured by lines of code. A 500-line policy with clean module boundaries, good tests, reproducible state, and clear logs may be easy to maintain. An 80-line policy where every line affects every other line, with no logs or replays, may be a time bomb.

On the code side, coupling complexity is bounded by module boundaries, interface stability, test coverage, observability, rollback cost, and state reproducibility. Good modularity cuts global coupling into local coupling, reducing the complexity an agent must hold in its head. Good tests let the coding agent avoid simulating the whole system mentally on every change.

On the coding-agent side, coupling complexity depends on model capability, context length, memory quality, tool quality, and iteration speed. A stronger model can handle more interactions at once. Longer context means fewer lost threads. Memory preserves experience across rounds. Search, localization, execution, and replay tools move part of the cognitive load outside the model.

Putting the two sides together gives a few working hypotheses:

- Clearer feedback increases the coupling complexity that a fixed amount of agent intelligence can maintain.
- With the same tools and feedback, stronger models can handle higher coupling complexity.
- Modularity, tests, and replays move part of the coupling complexity into the environment.
- Memory and tools increase the agent's effective context.
- An HS that only grows and never compresses will keep increasing coupling complexity until it exceeds maintenance capacity.

Breakout policy reaching the full `864` score is partly because the rules are simple, but also because failures can be replayed on video, locally reproduced, and regression-tested. Ant is much more complex, but it decomposes into rhythm, posture, contact, and residual MPC modules.

Montezuma is a useful counterexample. In Atari57, one unattended run reached `400` points, but the route consisted of `86` macro-actions and was basically open-loop execution. That example shows that some environments need stronger program forms: composable macro-actions, recoverable search state, and long-term memory. Plain `if else` cannot solve everything.

## The Next Paradigm?

The current paradigm shift has gone from pretraining to RLHF, and then to large-scale RL / RLVR. Anything that can be verified starts to become solvable.

Online Learning and Continual Learning can be partially addressed by agents produced by RLVR, through Heuristic Learning. From that perspective, I would call it a candidate for the next paradigm: anything that can be continuously iterated on starts to become solvable.

Why only partially? Because Heuristic Learning cannot do everything neural networks can do. It is bounded by what code can express, especially in complex perception and long-horizon generalization. With what I know today, I cannot imagine an agent writing pure Python code, without a neural network, to solve ImageNet.

The real question is how to combine neural networks and HL to address Online Learning and Continual Learning together. The most promising direction seems to be: use HL to process online data quickly, turn online experience into trainable, regression-testable, filterable data, and then periodically update the neural network.

Take robotics as an example. If we borrow the System 1 / System 2 language, one possible division of labor is:

- Specialized shallow NNs: part of System 1, fast and cheap, responsible for perception, classification, and object-state estimation.
- HL: also part of System 1, responsible for fresh data handling, rules, tests, replays, memory, safety boundaries, and local recovery.
- LLM agent: System 2, responsible for giving feedback to HL, improving data, and periodically extracting HL-generated data to update itself.

This can be further decomposed into a hierarchy:

```text
joint-level HL -> limb-level HL -> whole-body balance HL -> task-level HL
```

Lower levels handle safety and low-latency control. Middle levels handle gait and contact. Higher levels handle tasks, recovery, and long-term memory. The coding agent does not necessarily "understand walking" directly. It is more like an update pipeline inserted into the system: continuously feeding failure videos, sensor streams, simulation results, and test results into the system, then rewriting the feedback into code, parameters, guard rules, and memory.

LLM agents can share what they learn, or they can learn inside robotics in isolated branches. The open problem is how the *specific data distribution* produced by HL can avoid destabilizing the LLM's periodic updates. This is a classic post-training problem, and there is already a lot of mature experience here. For reasons, I will not expand on it in this post.

Agentic coding changes the speed of writing code. It also changes which code is worth owning for the long term.

Many heuristics looked hopeless because the real issue was maintenance cost. They were not necessarily too weak. Coding agents change that maintenance curve. Rules, tests, logs, memory, and patches used to be scattered engineering materials. Now they can become a continuously updated Heuristic System, one that can genuinely address problems Online Learning and Continual Learning have struggled to solve.

Welcome to the next paradigm.

## Disclaimer

This article represents only my personal views. It does not represent any company position, and the discussion here is unrelated to any specific company project, product plan, or internal work.

## Acknowledgements

Thanks to [Costa Huang](https://costa.sh/), [Tairan He](https://tairanhe.com/), and [Hao Sheng](https://hsheng.org) for feedback.

## Citation

If you need to cite this article in LaTeX, you can use the BibTeX below.

```bibtex
@misc{weng2026learning_beyond_gradients,
  title = {Learning Beyond Gradients},
  author = {Weng, Jiayi},
  year = {2026},
  month = may,
  howpublished = {\url{https://trinkle23897.github.io/learning-beyond-gradients/}},
  note = {Blog post}
}
```

## Appendix: Experiment Notes and Reproduction

The full artifact repository is [https://github.com/Trinkle23897/learning-beyond-gradients](https://github.com/Trinkle23897/learning-beyond-gradients). The commands below assume you have cloned that repo and are running them from the repository root. The GitHub Pages site only serves the article and the static files it references; the full scripts, CSVs, videos, and experiment artifacts live in the repo.

The Codex model used in the experiments below was `gpt-5.4`; newer model versions have not been tested yet. The experiment reports below were written by Codex itself.

### A.1 Experiment Notes

At first I simply asked Codex: "Write a strategy that solves Breakout." The result was mediocre. A low score is not very informative: the action semantics could be wrong, the state detector could be wrong, the evaluation setup could be wrong, or the policy structure itself could be weak. Later I changed the task shape: do not just hand me a `policy.py`; maintain a complete loop.

The loop looked roughly like this:

```text
probe actions and observations
-> write state detectors
-> write policy
-> run full episodes
-> record trials.jsonl and summary.csv
-> generate videos or curves
-> inspect failure modes
-> edit policy
-> simplify code and run regressions
```

At that point, the task had changed shape. The output was no longer just a policy file. It was an experimental system that could keep being modified. It had probes, records, replays, failure modes, and clues for the next round.

#### Breakout

Related artifacts: [heuristic_breakout.py], [heuristic_breakout_trials.jsonl], [heuristic_breakout_trials_summary.csv].

Breakout looks like a geometry problem: where is the ball, where is the paddle, and where will the ball land after bouncing off the wall? The hard part comes later. The policy can keep returning the ball, but stop hitting new bricks, trapping the score in a stable loop.

In the first round, Codex confirmed the action space and observation shape, then found the paddle, ball, and brick colors from RGB frames, and used those image labels to scan 128 RAM bytes. Early experiment records looked like this:

```text
trial_name                 score   cumulative_env_steps   note
shape_action_probe          -      32                     inspect obs/info/action
ram_byte_corr_probe_v1      -      5,032                  correlate RAM bytes
ram_fit_action_probe_v2     -      9,532                  action 2=right, 3=left
baseline_v0                99      16,303                 initial RAM intercept
tunnel0_v1                387      43,303                 no tunnel offset
```

`387` is the kind of local high score that can fool you. The policy was already good at returning the ball, but it sent the ball into a periodic route: it would not die, but it would not clear more bricks either. A human writing by hand might keep tuning "return accuracy." Codex inspected the video and the last few dozen steps of trace, then localized the issue to a lack of perturbation in the ball trajectory.

<video controls src="atari/breakout/heuristic_breakout_score387_tunnel0_render210x160.mp4" width="360"></video>

Video artifact: [heuristic_breakout_score387_tunnel0_render210x160.mp4].

The first effective mechanism was loop breaking: if there had been no reward for a long time, periodically add an offset to the predicted landing point and knock the ball out of the local cycle. That moved the score from `387` to `507`.

```python
if steps_since_reward >= stuck_trigger_steps:
    phase = stuck_offset_index % 4
    if phase == 0:
        offset = +stuck_offset_px
    elif phase == 1:
        offset = -stuck_offset_px
    elif phase == 2:
        offset = +0.5 * stuck_offset_px
    else:
        offset = -0.5 * stuck_offset_px
else:
    offset = 0.0
```

<video controls src="atari/breakout/heuristic_breakout_score507_stuckbreaker_render210x160.mp4" width="360"></video>

Video artifact: [heuristic_breakout_score507_stuckbreaker_render210x160.mp4].

Then another failure mode appeared. For a fast low ball, chasing the ordinary intercept made the paddle over-lead and drift away. Codex added `fast_low_ball_lead_steps=3`, and the score jumped from `507` to `839`.

```python
if vy > 0.1 and ball_y <= paddle_y:
    steps_to_paddle = max((paddle_y - ball_y) / vy, 0.0)
    intercept_x = reflect_position(ball_x + vx * steps_to_paddle)
    target_x = intercept_x + stuck_offset
elif vy >= fast_ball_min_vy:
    target_x = ball_x + fast_low_ball_lead_steps * vx
else:
    target_x = ball_x + chase_lead_steps * vx
```

<video controls src="atari/breakout/heuristic_breakout_score839_fastlead_render210x160.mp4" width="360"></video>

Video artifact: [heuristic_breakout_score839_fastlead_render210x160.mp4].

The move from `839` to `864` was more like caring for a system that had already become complex. Codex tried deadbands, serve offsets, stuck offsets, brick-balance bias, and lookahead steps. Many directions did nothing. The final useful change was a late-game condition: after the first wall of bricks, the stuck offset only applies when the ball is still far from the paddle; when the ball is close, the offset is gradually released, otherwise the last few bricks pull the paddle away. It also added a tiny paddle-drift compensation for the one-step delay between action and paddle position.

```python
if score >= 432 and stuck_release_horizon_steps > 0:
    release_ratio = clip(steps_to_paddle / stuck_release_horizon_steps, 0.0, 1.0)
    offset *= release_ratio

if score >= 432 and ball_y >= 170 and last_action == RIGHT:
    control_paddle_x = paddle_x + 2.0
elif score >= 432 and ball_y >= 170 and last_action == LEFT:
    control_paddle_x = paddle_x - 2.0
```

<video controls src="atari/breakout/heuristic_breakout_ci3985ae2_score864_render210x160.mp4" width="360"></video>

Video artifact: [heuristic_breakout_ci3985ae2_score864_render210x160.mp4].

The final RAM default configuration verified at `864 / 864 / 864` across three episodes. Later, Codex migrated the same geometry controller back to pure image input: no RAM, only RGB segmentation for paddle, ball, and brick balance. The image-only version first scored `310`, then `428`, and finally reached `864` for the first time after seven local policy episodes, corresponding to `14,504` local policy environment steps, after lowering the threshold for the late-game "release stuck offset" behavior so it applied throughout.

[![Breakout sample efficiency](atari/breakout/heuristic_breakout_sample_efficiency.png)][heuristic_breakout_sample_efficiency.png]

This should not be described as "image-only from scratch to max score in 14.5K steps." The real process was that Codex first discovered the geometry controller, loop breaker, and late-game offset release in the RAM version. Once the structure was stable, it swapped the state-reading layer from RAM to RGB detectors. The `14.5K` number is the transfer budget for the image-only version.

#### Ant and HalfCheetah

Related artifacts: [heuristic_ant.py], [ant_envpool.xml], [heuristic_ant_trials.jsonl], [heuristic_ant_trials_summary.csv], [heuristic_halfcheetah_v5.py], [heuristic_halfcheetah_v5_log.md].

Ant gives a different signal from Breakout. Breakout has intuitive geometry. Ant is continuous control, with 8 joint actions, and its failures are body dynamics rather than "missed the ball."

I did not specify "use CPG" or "use MPC" at the beginning. The constraints were simply: do not train a neural network, make it locally reproducible, leave records for each round, and keep pushing the score up. Codex first read the EnvPool/Gymnasium Ant observations and rewards, confirmed action order, root velocity, torso orientation, joint positions, and joint velocities, then proposed the first rhythmic gait on its own.

The first version was a four-leg phase oscillator: left and right legs in opposite phase, hip and ankle joints tracking sinusoidal target angles, with actions produced by a PD controller. It was not elegant, but it was already much stronger than random: the mean score over five random seeds was `2291`.

```python
leg_phase = warp_phase(phase + LEG_PHASE, stance_duty(vx))
stance = leg_phase < pi

hip_wave = HIP_BIAS + stance_or_swing_scale * (
    HIP_AMP * sin(leg_phase)
    + HIP_H2_AMP * sin(2 * leg_phase + HIP_H2_PHASE)
    + HIP_H3_AMP * sin(3 * leg_phase + HIP_H3_PHASE)
)

action[0::2] = KP * (
    HIP_SIGN * hip_wave
    + HEADING_AXIS * (YAW_GAIN * yaw + YAW_RATE_GAIN * yaw_rate)
    - q[0::2]
)
action[1::2] = KP * (ANKLE_SIGN * (ankle_wave + balance) - q[1::2])
```

The early iterations looked like tuning a real controller: add yaw feedback to reach `2718`, retune phase speed, hip/ankle amplitude, and yaw angular-velocity gain to reach `3025`, then add second/third harmonics to reach `3162`. Codex also tried broad parameter search, but it did not reliably beat the current rhythmic policy, so it stopped expanding the search budget and moved to a different representation.

The jump came from residual MPC. Roughly speaking, MPC is "think a short distance into the future while walking." Keep the rhythmic gait as the base reflex; at each real environment step, sample dozens of small residual action sequences inside a local MuJoCo model, score them, execute only the first residual action, then reobserve and replan on the next step, using the unfinished previous plan as a warm start.

This way, the policy does not plan all 8 joints from scratch at every step. It starts with a stable gait, then uses short-horizon model planning to correct it.

```python
base = cpg_action(phase, q, dq, roll, pitch, yaw, rates, contacts, vx)

best_plan = previous_plan.copy()
best_obj = rollout_objective(obs, best_plan)
for _ in range(CANDIDATES - 1):
    residuals = clip(
        best_plan + rng.normal(0.0, MPC_SIGMA, size=(HORIZON, 8)),
        -MPC_CLIP,
        MPC_CLIP,
    )
    residuals[1:] = 0.6 * residuals[1:] + 0.4 * residuals[:-1]
    obj = rollout_objective(obs, residuals)
    if obj > best_obj:
        best_obj = obj
        best_plan = residuals

plan[:-1] = PLAN_DECAY * best_plan[1:]
return clip(base + best_plan[0], -1.0, 1.0)
```

```text
trial_name                               score_mean   cumulative_env_steps   note
ant_lr_cpgpd_v1                         2291.9       5,000                  left/right anti-phase CPG + PD
ant_yawaxis_grid_v2                     2857.9       20,000                 yaw feedback + retuned params
ant_h3_428_v1                           3162.0       50,000                 second/third harmonics
ant_mpc_residual_v1_ep1                 3635.5       62,000                 horizon=6, candidates=32
ant_mpc_residual_cfg4_eval5             3964.7       67,000                 horizon=8, candidates=48
ant_mpc_residual_cand07_eval5           4647.1       73,000                 local search around MPC config
ant_mpc_residual_narrow04_eval5         4871.3       79,000                 lower z target, larger kp/candidate count
ant_mpc_residual_warm02_eval5           5165.2       85,000                 warm-start residual plan
ant_mpc_fast065x060_sigma008_clip012    5759.4       95,000                 faster gait + larger residual
ant_mpc_term001_ep1                     6054.5       100,000                terminal velocity cost
ant_mpc_default_adaptive_ep1            6146.2       106,300                speed-adaptive phase + stance
```

By the end, the policy had oscillator phase, stance ratio, speed adaptation, roll/pitch/yaw feedback, foot contact, short-horizon model rollouts, residual smoothing, terminal velocity cost, and warm-start plan decay. A human can write one or two of these modules, of course. But handling experiment records, code, videos, and failure directions at the same time, under a short iteration budget, is a different level of difficulty.

[![Ant sample efficiency](mujoco/ant/heuristic_ant_sample_efficiency.png)][heuristic_ant_sample_efficiency.png]

<video controls src="mujoco/ant/heuristic_ant_mpc_default_6146_render480.mp4" width="480"></video>

Video artifact: [heuristic_ant_mpc_default_6146_render480.mp4].

HalfCheetah is another point of evidence in the same family. I reran a five-episode evaluation of `mpc-staged-tree-asym-pd-cpg` with seeds `100..104`; the mean was `11836.7`, with min `11735.0` and max `12041.2`. The policy uses interpretable gait/posture rules and online staged-tree MPC: first form a high-scoring gait through CPG/PD, then use short-horizon model scoring and a staged swing-amplitude schedule to adjust actions.

[![HalfCheetah sample efficiency](mujoco/halfcheetah/heuristic_halfcheetah_sample_efficiency.png)][heuristic_halfcheetah_sample_efficiency.png]

#### VizDoom

Related artifacts: [heuristic_vizdoom_d1_cv.py], [heuristic_vizdoom_d3_cv.py], [record_vizdoom_d3_cv.py], [d1_cv_10seed_render_35fps.mp4], [d3_cv_best_10seed_render_35fps.mp4].

D1 Basic is the small experiment that first checks whether waiting to collect the medikit at the right time can get close to the reward ceiling. I first used a stronger object/info policy to validate the behavioral ceiling, which reached about `1.01` over 10 seeds; then I removed the non-reproducible information and kept only pip EnvPool, screen pixels from `render()`, and the public `HEALTH` variable. The final pure-CV version finds the medikit with brightness thresholding, morphological close/dilate, and connected components, then uses the bounding-box center to decide turning or moving forward. If health is still high and the medikit is already close, it stages in place and waits until health has decayed before collecting it. Over 10 seeds it reaches `mean=0.9441`, `min=0.2900`.

<video controls src="vizdoom/d1_cv_10seed_render_35fps.mp4" width="480"></video>

Video artifact: [d1_cv_10seed_render_35fps.mp4].

D3 Battle is closer to an FPS smoke policy. It is not a memorized map route; it decomposes behavior into three loops: kill monsters when visible, seek supplies when health or ammo is low, and explore quickly when no target is visible. On the screen side it uses cv2/NumPy color thresholds and connected components to extract enemy candidates, ammo/health item candidates, and dark-region ratios for walls and doorways. From EnvPool info it only uses public game variables such as `HEALTH`, `AMMO2`, `HITCOUNT`, `DAMAGECOUNT`, and `KILLCOUNT`. The iteration loop was to run 10 seeds in parallel, record a 35fps 10-grid video directly through `render()`, and inspect the failure modes. Early versions tried map access, object info, and seed-specific branches; those were removed. The final version keeps screen CV plus public variables as the closed-loop controller. Over 10 seeds it reaches `mean=557.0`, `min=440.0`.

<video controls src="vizdoom/d3_cv_best_10seed_render_35fps.mp4" width="480"></video>

Video artifact: [d3_cv_best_10seed_render_35fps.mp4].

#### Atari57

Related artifacts: [atari57_prompt_template.txt], [atari57_aggregate_curve_steps.csv], [atari57_env_mode_summary.csv], [openrl_atari57_per_game_hns_comparison.csv], [atari57_hns_normalization_inferred.csv].

Breakout and Ant are single-point stories. Atari57 asks what remains when the workflow leaves a few pretty examples. The setup was direct: run the same Codex workflow over the full Atari57 suite, with both `ram` and `native_obs` observation modes for every environment, and three independent repeats for each mode:

```text
57 games x 2 observation modes x 3 repeats = 342 coding-agent search trajectories
```

No human gave step-by-step hints during this batch. Each agent received the same template and different `ENV_ID / OBS_MODE / REPEAT_INDEX`, then ran until it stopped. Each run had to write `policy.py`, `trials.jsonl`, `summary.csv`, `sample_efficiency.png`, and `README.md`.

The full prompt is in [atari57_prompt_template.txt]. The core summary was:

<details markdown="1">
<summary>Atari57 batch prompt summary</summary>

```text
Goal: for one EnvPool Atari environment and one OBS_MODE, design and iterate a handwritten heuristic policy.

Run style: do not ask for confirmation and do not report intermediate progress; continue until a stop rule is met, then summarize.

Hard constraints:
- Do not train a neural network.
- Do not read environment source, tests, ROM details, or hidden state.
- In native_obs mode, only use obs returned by reset/step.
- In ram mode, info["ram"] is allowed.
- Atari initialization parameters are fixed, including frame_skip=1, reward_clip=False, sticky action=0.
- All actual environment steps used for probe/debug/trial must count toward cumulative_env_steps.

Output files:
- policy.py: the current best heuristic, simplified where possible.
- trials.jsonl: score, environment steps, config, and notes for every trial.
- summary.csv: aggregate from trials.
- sample_efficiency.png: score curve by environment step and episode.
- README.md: best score, reproduction command, failed directions, and stop reason.

Stop rules:
- Atari frame budget = 20,000,000.
- Do not stop before budget just because the score has plateaued or exceeded a reference.
- After every new best score, simplify the code and verify that the score does not regress before keeping it.
```

</details>

First look at environment-step curves. HNS means human-normalized score: each game's score is normalized against the human baseline before comparison. In the fully unattended batch, `native_obs` Atari median HNS reached `0.32` around `1M` steps, and `ram` reached `0.26`, clearly above the early PPO2 / CleanRL EnvPool PPO curves in the figure. Around `9.7M` steps, `native_obs` reached `0.81` and `ram` reached `0.59`. In the same comparison, the OpenRL Benchmark PPO2 / CleanRL EnvPool PPO median HNS curves are roughly `0.88 / 0.92` at `10M` steps.

[![Atari57 sample efficiency vs OpenRL Benchmark](atari/atari57/atari57_openrl_sample_efficiency_context.png)][atari57_openrl_sample_efficiency_context.png]

This comparison is about environment interaction efficiency. It does not convert the cost of the coding agent reading logs, writing code, and watching videos into the total compute budget. The signal is still concrete: a very rough coding-agent batch workflow, with no human inspection of intermediate results, already pushes the Atari57 median into the neighborhood of these baselines.

If we aggregate by taking the best input mode for each game at the end, Codex median HNS is `0.83`, OpenAI Baselines PPO2 is `0.80`, and CleanRL EnvPool PPO is `0.98`. If we relax further to best single run, Codex median HNS is `1.18`. This does not replace a strict training-curve comparison, but it shows more directly what level the unattended search eventually covered.

Aggregate curves compress differences into one median, so I also looked at per-game HNS. On Breakout, Krull, DoubleDunk, Boxing, and DemonAttack, both heuristics and Deep RL baselines reach scores well above the human baseline. On Asterix, Jamesbond, Centipede, Bowling, Skiing, and Tennis, heuristics are relatively strong. On Atlantis, VideoPinball, UpNDown, Assault, RoadRunner, and StarGunner, PPO is clearly much stronger.

[![Atari57 per-game HNS comparison](atari/atari57/atari57_per_game_hns_comparison.png)][atari57_per_game_hns_comparison.png]

The most interesting part of Atari57 is that the source of sample efficiency changes. Traditional neural-network Atari learning has to relearn representation, credit assignment, and action meaning from high-dimensional inputs in every environment. Codex instead decomposes each environment into small maintainable program systems: aiming and dodging in shooters, bounce prediction in catching games, position rules in avoidance games, wrapper details, and each environment's own failed-experiment records.

#### Montezuma

Related artifacts: [heuristic_montezuma.py], [heuristic_montezuma_state_graph_search.py], [heuristic_montezuma_400_policy.py], [heuristic_montezuma_400_macros.json], [heuristic_montezuma_400_metadata.json].

Some environments are not a good fit for ordinary reactive heuristic policies. Montezuma's Revenge is the typical example.

In an earlier standalone Montezuma search, state-graph search moved the key distance from `72` to `28`, but reward stayed at `0`. Later, in the Atari57 native-image batch, one unattended Codex run reached `400.0` points: the repaired best replay was `repair_replay_r1_t19734`, seed `10001`, using `1769` environment steps. It is essentially an open-loop route made of `86` macro-actions.

<video controls src="atari/montezuma/montezuma_400_render_seed10001_h264.mp4" width="360"></video>

Video artifact: [montezuma_400_render_seed10001_h264.mp4].

Montezuma exposes an expressivity problem. A normal `policy.py` state machine has trouble representing this kind of route: actions must align with timing, failures need recovery, and intermediate states need ways to re-enter the plan. Some environments need composable macro-actions, recoverable search state, and perhaps a program structure better suited to long-horizon planning than ordinary `if else`.

This kind of failure is valuable for HL. It tells us where the boundary is, and hints at the next layer of abstraction. Some feedback needs a new representation and a new program form before it can enter the system. For Montezuma, the next interface probably needs macro-actions, recoverable state, search, and long-term memory.

### A.2 Reproduction Entrypoints

The commands below assume they are run from this article's directory, with dependencies already installed from `requirements.txt`. They check the representative results discussed above.

#### Pong 21

Reproduction entrypoint: [heuristic_pong.py].

```bash
python atari/pong/heuristic_pong.py \
  --policy ram \
  --episodes 1 \
  --seed 0
```

Expected output should include `episode=0 score=21.0` and `mean=21.000`.

#### Breakout Intermediate Nodes

These commands reproduce the `387 -> 507 -> 839` intermediate nodes discussed in the appendix. They are not the final policy; they pin down the mechanisms that made each jump happen.

<details markdown="1">
<summary>Reproduce 387: return the ball, but do not break the loop</summary>

```bash
rm -f /tmp/repro_breakout_387.jsonl /tmp/repro_breakout_387.csv
python atari/breakout/heuristic_breakout.py \
  --policy ram \
  --episodes 1 \
  --seed 0 \
  --max-steps 27000 \
  --deadband 3 \
  --chase-lead-steps 6 \
  --tunnel-offset 0 \
  --launch-offset 24 \
  --fast-ball-min-vy 1000000000 \
  --stuck-trigger-steps 1000000000 \
  --stuck-switch-steps 0 \
  --stuck-offset 0 \
  --stuck-release-horizon-steps 0 \
  --brick-balance-bias-min-score 1000000000 \
  --late-game-paddle-lag-px 0 \
  --trial-name repro_breakout_387 \
  --log-path /tmp/repro_breakout_387.jsonl \
  --summary-path /tmp/repro_breakout_387.csv
```

Expected output should include `score=387.0` and `mean=387.000`.

</details>

<details markdown="1">
<summary>Reproduce 507: add stuck-loop perturbation</summary>

```bash
rm -f /tmp/repro_breakout_507.jsonl /tmp/repro_breakout_507.csv
python atari/breakout/heuristic_breakout.py \
  --policy ram \
  --episodes 1 \
  --seed 0 \
  --max-steps 27000 \
  --deadband 3 \
  --chase-lead-steps 6 \
  --tunnel-offset 0 \
  --launch-offset 24 \
  --fast-ball-min-vy 1000000000 \
  --stuck-trigger-steps 1024 \
  --stuck-switch-steps 256 \
  --stuck-offset 12 \
  --stuck-release-horizon-steps 0 \
  --brick-balance-bias-min-score 1000000000 \
  --late-game-paddle-lag-px 0 \
  --trial-name repro_breakout_507 \
  --log-path /tmp/repro_breakout_507.jsonl \
  --summary-path /tmp/repro_breakout_507.csv
```

Expected output should include `score=507.0` and `mean=507.000`.

</details>

<details markdown="1">
<summary>Reproduce 839: handle fast low balls</summary>

```bash
rm -f /tmp/repro_breakout_839.jsonl /tmp/repro_breakout_839.csv
python atari/breakout/heuristic_breakout.py \
  --policy ram \
  --episodes 1 \
  --seed 0 \
  --max-steps 27000 \
  --deadband 3 \
  --chase-lead-steps 6 \
  --tunnel-offset 0 \
  --launch-offset 24 \
  --fast-ball-min-vy 3 \
  --fast-low-ball-lead-steps 3 \
  --stuck-trigger-steps 1024 \
  --stuck-switch-steps 256 \
  --stuck-offset 12 \
  --stuck-release-horizon-steps 0 \
  --brick-balance-bias-min-score 1000000000 \
  --late-game-paddle-lag-px 0 \
  --trial-name repro_breakout_839 \
  --log-path /tmp/repro_breakout_839.jsonl \
  --summary-path /tmp/repro_breakout_839.csv
```

Expected output should include `score=839.0` and `mean=839.000`.

</details>

#### Breakout 864

Reproduction entrypoint: [heuristic_breakout.py].

```bash
rm -f /tmp/repro_breakout_864.jsonl /tmp/repro_breakout_864.csv
python atari/breakout/heuristic_breakout.py \
  --policy ram \
  --episodes 1 \
  --seed 0 \
  --max-steps 108000 \
  --deadband 3 \
  --chase-lead-steps 6 \
  --tunnel-offset 0 \
  --launch-offset 24 \
  --fast-ball-min-vy 3 \
  --fast-low-ball-lead-steps 3 \
  --stuck-trigger-steps 1024 \
  --stuck-switch-steps 256 \
  --stuck-offset 12 \
  --stuck-release-horizon-steps 8 \
  --brick-balance-deadzone 0.01 \
  --brick-balance-bias-min-score 432 \
  --late-game-paddle-lag-px 2 \
  --late-game-lag-ball-y 170 \
  --trial-name repro_breakout_864 \
  --log-path /tmp/repro_breakout_864.jsonl \
  --summary-path /tmp/repro_breakout_864.csv
```

Expected output should include `score=864.0` and `mean=864.000`.

#### Ant Default MPC Policy

Reproduction entrypoints: [heuristic_ant.py], [ant_envpool.xml].

```bash
rm -f /tmp/repro_ant_6146_eval5.jsonl /tmp/repro_ant_6146_eval5.csv
python mujoco/ant/heuristic_ant.py \
  --policy mpc \
  --episodes 5 \
  --seed 0 \
  --max-steps 1000 \
  --mujoco-xml-path mujoco/ant/ant_envpool.xml \
  --trial-name repro_ant_6146_eval5 \
  --log-path /tmp/repro_ant_6146_eval5.jsonl \
  --summary-path /tmp/repro_ant_6146_eval5.csv
```

My local rerun produced `mean=6005.521`, `min=5776.805`, and `max=6146.208`.

#### HalfCheetah Staged-Tree MPC

Reproduction entrypoint: [heuristic_halfcheetah_v5.py].

```bash
python mujoco/halfcheetah/heuristic_halfcheetah_v5.py \
  --policy mpc-staged-tree-asym-pd-cpg \
  --eval-episodes 5 \
  --eval-seed 100
```

My local rerun produced a five-episode mean of `11836.693`.

#### VizDoom D1 Basic CV Policy

Reproduction entrypoint: [heuristic_vizdoom_d1_cv.py].

```bash
python vizdoom/heuristic_vizdoom_d1_cv.py --episodes 10 --seed 0
```

My local rerun produced `mean=0.9440999741666019` and `min=0.28999998047947884`. This version depends only on pip EnvPool; it does not need the `vizdoom` Python package or any locally compiled extension.

To regenerate the 35fps video:

```bash
python vizdoom/heuristic_vizdoom_d1_cv.py \
  --episodes 10 \
  --seed 0 \
  --record-mp4 vizdoom/d1_cv_10seed_render_35fps.mp4
```

#### VizDoom D3 Battle CV Policy

Reproduction entrypoints: [heuristic_vizdoom_d3_cv.py], [record_vizdoom_d3_cv.py].

```bash
python vizdoom/heuristic_vizdoom_d3_cv.py
```

My local rerun produced `mean=557.0` and `min=440.0`; the 10-seed rewards were `[545, 475, 480, 440, 690, 500, 600, 595, 530, 715]`.

To regenerate the 35fps video:

```bash
python vizdoom/record_vizdoom_d3_cv.py
```

#### Montezuma 400-Point Replay

Reproduction entrypoint: [heuristic_montezuma_400_policy.py].

```bash
python atari/montezuma/heuristic_montezuma_400_policy.py \
  --metadata-out /tmp/repro_montezuma_400.json
```

Expected output should include `"score": 400.0` and `"env_steps": 1769`. This is a boundary case; it should not be read as a general Montezuma policy.

## Comments

<div class="github-issue-comments" data-lang="en" data-repo="Trinkle23897/learning-beyond-gradients" data-issue-url="https://github.com/Trinkle23897/learning-beyond-gradients/issues/new?title=Comment%3A%20Learning%20Beyond%20Gradients"></div>

[heuristic_breakout.py]: https://github.com/Trinkle23897/learning-beyond-gradients/blob/main/atari/breakout/heuristic_breakout.py
[heuristic_breakout_trials.jsonl]: https://github.com/Trinkle23897/learning-beyond-gradients/blob/main/atari/breakout/heuristic_breakout_trials.jsonl
[heuristic_breakout_trials_summary.csv]: https://github.com/Trinkle23897/learning-beyond-gradients/blob/main/atari/breakout/heuristic_breakout_trials_summary.csv
[heuristic_breakout_score387_tunnel0_render210x160.mp4]: https://github.com/Trinkle23897/learning-beyond-gradients/blob/main/atari/breakout/heuristic_breakout_score387_tunnel0_render210x160.mp4
[heuristic_breakout_score507_stuckbreaker_render210x160.mp4]: https://github.com/Trinkle23897/learning-beyond-gradients/blob/main/atari/breakout/heuristic_breakout_score507_stuckbreaker_render210x160.mp4
[heuristic_breakout_score839_fastlead_render210x160.mp4]: https://github.com/Trinkle23897/learning-beyond-gradients/blob/main/atari/breakout/heuristic_breakout_score839_fastlead_render210x160.mp4
[heuristic_breakout_ci3985ae2_score864_render210x160.mp4]: https://github.com/Trinkle23897/learning-beyond-gradients/blob/main/atari/breakout/heuristic_breakout_ci3985ae2_score864_render210x160.mp4
[heuristic_breakout_sample_efficiency.png]: https://github.com/Trinkle23897/learning-beyond-gradients/blob/main/atari/breakout/heuristic_breakout_sample_efficiency.png
[heuristic_ant.py]: https://github.com/Trinkle23897/learning-beyond-gradients/blob/main/mujoco/ant/heuristic_ant.py
[ant_envpool.xml]: https://github.com/Trinkle23897/learning-beyond-gradients/blob/main/mujoco/ant/ant_envpool.xml
[heuristic_ant_trials.jsonl]: https://github.com/Trinkle23897/learning-beyond-gradients/blob/main/mujoco/ant/heuristic_ant_trials.jsonl
[heuristic_ant_trials_summary.csv]: https://github.com/Trinkle23897/learning-beyond-gradients/blob/main/mujoco/ant/heuristic_ant_trials_summary.csv
[heuristic_ant_sample_efficiency.png]: https://github.com/Trinkle23897/learning-beyond-gradients/blob/main/mujoco/ant/heuristic_ant_sample_efficiency.png
[heuristic_ant_mpc_default_6146_render480.mp4]: https://github.com/Trinkle23897/learning-beyond-gradients/blob/main/mujoco/ant/heuristic_ant_mpc_default_6146_render480.mp4
[heuristic_halfcheetah_v5.py]: https://github.com/Trinkle23897/learning-beyond-gradients/blob/main/mujoco/halfcheetah/heuristic_halfcheetah_v5.py
[heuristic_halfcheetah_v5_log.md]: https://github.com/Trinkle23897/learning-beyond-gradients/blob/main/mujoco/halfcheetah/heuristic_halfcheetah_v5_log.md
[heuristic_halfcheetah_sample_efficiency.png]: https://github.com/Trinkle23897/learning-beyond-gradients/blob/main/mujoco/halfcheetah/heuristic_halfcheetah_sample_efficiency.png
[heuristic_vizdoom_d3_cv.py]: https://github.com/Trinkle23897/learning-beyond-gradients/blob/main/vizdoom/heuristic_vizdoom_d3_cv.py
[record_vizdoom_d3_cv.py]: https://github.com/Trinkle23897/learning-beyond-gradients/blob/main/vizdoom/record_vizdoom_d3_cv.py
[heuristic_vizdoom_d1_cv.py]: https://github.com/Trinkle23897/learning-beyond-gradients/blob/main/vizdoom/heuristic_vizdoom_d1_cv.py
[d3_cv_best_10seed_render_35fps.mp4]: https://github.com/Trinkle23897/learning-beyond-gradients/blob/main/vizdoom/d3_cv_best_10seed_render_35fps.mp4
[d1_cv_10seed_render_35fps.mp4]: https://github.com/Trinkle23897/learning-beyond-gradients/blob/main/vizdoom/d1_cv_10seed_render_35fps.mp4
[atari57_prompt_template.txt]: https://github.com/Trinkle23897/learning-beyond-gradients/blob/main/atari/atari57/atari57_prompt_template.txt
[atari57_aggregate_curve_steps.csv]: https://github.com/Trinkle23897/learning-beyond-gradients/blob/main/atari/atari57/atari57_aggregate_curve_steps.csv
[atari57_env_mode_summary.csv]: https://github.com/Trinkle23897/learning-beyond-gradients/blob/main/atari/atari57/atari57_env_mode_summary.csv
[openrl_atari57_per_game_hns_comparison.csv]: https://github.com/Trinkle23897/learning-beyond-gradients/blob/main/atari/atari57/openrl_atari57_per_game_hns_comparison.csv
[atari57_hns_normalization_inferred.csv]: https://github.com/Trinkle23897/learning-beyond-gradients/blob/main/atari/atari57/atari57_hns_normalization_inferred.csv
[atari57_openrl_sample_efficiency_context.png]: https://github.com/Trinkle23897/learning-beyond-gradients/blob/main/atari/atari57/atari57_openrl_sample_efficiency_context.png
[atari57_per_game_hns_comparison.png]: https://github.com/Trinkle23897/learning-beyond-gradients/blob/main/atari/atari57/atari57_per_game_hns_comparison.png
[heuristic_montezuma.py]: https://github.com/Trinkle23897/learning-beyond-gradients/blob/main/atari/montezuma/heuristic_montezuma.py
[heuristic_montezuma_state_graph_search.py]: https://github.com/Trinkle23897/learning-beyond-gradients/blob/main/atari/montezuma/heuristic_montezuma_state_graph_search.py
[heuristic_montezuma_400_policy.py]: https://github.com/Trinkle23897/learning-beyond-gradients/blob/main/atari/montezuma/heuristic_montezuma_400_policy.py
[heuristic_montezuma_400_macros.json]: https://github.com/Trinkle23897/learning-beyond-gradients/blob/main/atari/montezuma/heuristic_montezuma_400_macros.json
[heuristic_montezuma_400_metadata.json]: https://github.com/Trinkle23897/learning-beyond-gradients/blob/main/atari/montezuma/heuristic_montezuma_400_metadata.json
[montezuma_400_render_seed10001_h264.mp4]: https://github.com/Trinkle23897/learning-beyond-gradients/blob/main/atari/montezuma/montezuma_400_render_seed10001_h264.mp4
[heuristic_pong.py]: https://github.com/Trinkle23897/learning-beyond-gradients/blob/main/atari/pong/heuristic_pong.py
