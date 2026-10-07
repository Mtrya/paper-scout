# Hybrid control grid (gpt-6-astra reviewed by a scripted cerebellum)

Cells finished: 36/36 (30/30 pre-registered + 6 exploratory). 34 ground-truth successes. Success is the sim evaluation, never the model's own `done`. Each cell: real codex agent, `--arm <arm>`, `--max-decisions 60`, 3 seeds per task.

## Success grid (successes / n)

| Task | direct | oracle | noisy | biased | biased-soft | degenerate |
|---|---|---|---|---|---|---|
| T1 gate-button | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| T2 align-insert | 2/3 | 3/3 | 3/3 | 3/3 | 3/3 | 2/3 |
| **total (6 cells)** | **5/6** | **6/6** | **6/6** | **6/6** | **6/6** | **5/6** |

## Channel metrics per arm (all cells)

| Arm | model calls (decision steps) | astra-authored actions (REPLACE) | of which diverge from the proposed waypoint | astra share | accepts | accept rate | longest accept run |
|---|---|---|---|---|---|---|---|
| direct | 152 | 0 | 0 | 0.0% | 0 | n/a | n/a |
| oracle | 85 | 65 | 61 | 76.5% | 20 | 23.5% | 7 |
| noisy | 79 | 75 | 74 | 94.9% | 4 | 5.1% | 1 |
| biased | 76 | 72 | 72 | 94.7% | 4 | 5.3% | 2 |
| biased-soft | 86 | 81 | 81 | 94.2% | 5 | 5.8% | 1 |
| degenerate | 129 | 129 | 129 | 100.0% | 0 | 0.0% | 0 |

## Per-cell detail

| Task | Arm | seed | success | decisions | accept/replace | accept rate | tokens (in/out/total) | term | category | progress events | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| T1 | direct | 0 | Y | 13 | 0/0 | n/a | 368956/1291/370247 | done | ok | button_pressed,gate_opened,grasp_attached | 97 |
| T1 | direct | 1 | Y | 11 | 0/0 | n/a | 309215/1281/310496 | done | ok | button_pressed,gate_opened,grasp_attached | 88 |
| T1 | direct | 2 | Y | 12 | 0/0 | n/a | 338633/1266/339899 | done | ok | button_pressed,gate_opened,grasp_attached | 116 |
| T1 | oracle | 0 | Y | 11 | 9/2 | 82% | 324132/678/324810 | done | ok | button_pressed,gate_opened,grasp_attached | 85 |
| T1 | oracle | 1 | Y | 8 | 3/5 | 38% | 237223/672/237895 | done | ok | button_pressed,gate_opened,grasp_attached | 68 |
| T1 | oracle | 2 | Y | 8 | 3/5 | 38% | 242779/697/243476 | done | ok | button_pressed,gate_opened,grasp_attached | 67 |
| T1 | noisy | 0 | Y | 8 | 0/8 | 0% | 237818/804/238622 | done | ok | button_pressed,gate_opened,grasp_attached | 64 |
| T1 | noisy | 1 | Y | 8 | 1/7 | 12% | 237594/756/238350 | done | ok | button_pressed,gate_opened,grasp_attached | 78 |
| T1 | noisy | 2 | Y | 8 | 0/8 | 0% | 240531/791/241322 | done | ok | button_pressed,gate_opened,grasp_attached | 64 |
| T1 | biased | 0 | Y | 8 | 0/8 | 0% | 237597/793/238390 | done | ok | button_pressed,gate_opened,grasp_attached | 66 |
| T1 | biased | 1 | Y | 8 | 0/8 | 0% | 237508/759/238267 | done | ok | button_pressed,gate_opened,grasp_attached | 82 |
| T1 | biased | 2 | Y | 8 | 0/8 | 0% | 240512/819/241331 | done | ok | button_pressed,gate_opened,grasp_attached | 69 |
| T1 | biased-soft | 0 | Y | 8 | 0/8 | 0% | 237565/771/238336 | done | ok | button_pressed,gate_opened,grasp_attached | 66 |
| T1 | biased-soft | 1 | Y | 9 | 0/9 | 0% | 266584/855/267439 | done | ok | button_pressed,gate_opened,grasp_attached | 81 |
| T1 | biased-soft | 2 | Y | 10 | 0/10 | 0% | 297460/832/298292 | done | ok | button_pressed,gate_opened,grasp_attached | 84 |
| T1 | degenerate | 0 | Y | 8 | 0/8 | 0% | 239533/821/240354 | done | ok | button_pressed,gate_opened,grasp_attached | 61 |
| T1 | degenerate | 1 | Y | 8 | 0/8 | 0% | 239268/773/240041 | done | ok | button_pressed,gate_opened,grasp_attached | 81 |
| T1 | degenerate | 2 | Y | 8 | 0/8 | 0% | 242195/826/243021 | done | ok | button_pressed,gate_opened,grasp_attached | 71 |
| T2 | direct | 0 | Y | 18 | 0/0 | n/a | 523423/1690/525113 | done | ok | grasp_attached,plug_pushed | 147 |
| T2 | direct | 1 | Y | 42 | 0/0 | n/a | 1477700/4592/1482292 | done | ok | grasp_attached,plug_pushed | 351 |
| T2 | direct | 2 | N | 56 | 0/0 | n/a | 2304525/7331/2311856 | give_up | terminated_give_up | - | 578 |
| T2 | oracle | 0 | Y | 22 | 2/20 | 9% | 716441/1797/718238 | done | ok | grasp_attached,plug_pushed | 172 |
| T2 | oracle | 1 | Y | 16 | 2/14 | 12% | 491811/1520/493331 | done | ok | grasp_attached,plug_pushed | 148 |
| T2 | oracle | 2 | Y | 20 | 1/19 | 5% | 656608/1892/658500 | done | ok | grasp_attached,plug_pushed | 145 |
| T2 | noisy | 0 | Y | 17 | 1/16 | 6% | 525533/1671/527204 | done | ok | grasp_attached,plug_pushed | 146 |
| T2 | noisy | 1 | Y | 17 | 1/16 | 6% | 522565/1485/524050 | done | ok | grasp_attached,plug_pushed | 131 |
| T2 | noisy | 2 | Y | 21 | 1/20 | 5% | 684303/2078/686381 | done | ok | grasp_attached,plug_pushed | 177 |
| T2 | biased | 0 | Y | 16 | 1/15 | 6% | 487729/1480/489209 | done | ok | grasp_attached,plug_pushed | 131 |
| T2 | biased | 1 | Y | 19 | 2/17 | 11% | 604521/1951/606472 | done | ok | grasp_attached,plug_pushed | 177 |
| T2 | biased | 2 | Y | 17 | 1/16 | 6% | 525817/1375/527192 | done | ok | grasp_attached,plug_pushed | 139 |
| T2 | biased-soft | 0 | Y | 17 | 1/16 | 6% | 525099/1644/526743 | done | ok | grasp_attached,plug_pushed | 139 |
| T2 | biased-soft | 1 | Y | 19 | 2/17 | 11% | 602208/1780/603988 | done | ok | grasp_attached,plug_pushed | 178 |
| T2 | biased-soft | 2 | Y | 23 | 2/21 | 9% | 761500/2048/763548 | done | ok | grasp_attached,plug_pushed | 206 |
| T2 | degenerate | 0 | Y | 28 | 0/28 | 0% | 976639/3166/979805 | done | ok | grasp_attached,plug_pushed | 272 |
| T2 | degenerate | 1 | Y | 17 | 0/17 | 0% | 527652/1537/529189 | done | ok | grasp_attached,plug_pushed | 136 |
| T2 | degenerate | 2 | N | 60 | 0/60 | 0% | 2778785/6401/2785186 | - | budget_exhausted | - | 577 |

## What gets accepted: consequence class x arm (accepted / proposals)

| Consequence class | oracle | noisy | biased | biased-soft | degenerate |
|---|---|---|---|---|---|
| free-space move | 12/69 (17%) | 2/67 (3%) | 1/65 (2%) | 3/74 (4%) | 0/0 |
| contact move | 4/4 (100%) | 0/2 (0%) | 0/2 (0%) | 0/3 (0%) | 0/129 (0%) |
| gripper | 4/11 (36%) | 2/10 (20%) | 3/9 (33%) | 2/9 (22%) | 0/0 |
| terminal | 0/1 (0%) | 0/0 | 0/0 | 0/0 | 0/0 |
| **all** | **20/85** (24%) | **4/79** (5%) | **4/76** (5%) | **5/86** (6%) | **0/129** (0%) |

## Reviewer behaviour detail

| Arm | cells | first decision = accept | mean decisions | mean tokens | tokens per decision | mean check_path per cell |
|---|---|---|---|---|---|---|
| direct | 6 | 0/6 | 25.3 | 889984 | 35131 | 2.3 |
| oracle | 6 | 3/6 | 14.2 | 446042 | 31485 | 0.8 |
| noisy | 6 | 1/6 | 13.2 | 409322 | 31088 | 0.8 |
| biased | 6 | 0/6 | 12.7 | 390144 | 30801 | 0.8 |
| biased-soft | 6 | 0/6 | 14.3 | 449724 | 31376 | 0.8 |
| degenerate | 6 | 0/6 | 21.5 | 836266 | 38896 | 0.7 |

## Acceptance split by task (pooled accepts / proposals)

| Arm | T1 gate-button | T2 align-insert |
|---|---|---|
| oracle | 15/27 (56%) | 5/58 (9%) |
| noisy | 1/24 (4%) | 3/55 (5%) |
| biased | 0/24 (0%) | 4/52 (8%) |
| biased-soft | 0/27 (0%) | 5/59 (8%) |
| degenerate | 0/24 (0%) | 0/105 (0%) |

## Failure signatures

| Category | direct | oracle | noisy | biased | biased-soft | degenerate | total |
|---|---|---|---|---|---|---|---|
| budget_exhausted | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| terminated_give_up | 1 | 0 | 0 | 0 | 0 | 0 | 1 |

**Failure attribution.** Both failures are T2 grasp-retry loops on the task's hidden precondition: the plug must be pushed within 1.5 cm of the alignment mark before a grasp succeeds, and the sim (deliberately, like real hardware) never leaks that reason - the model only sees 'fully closed, torque 0.25' and keeps changing height/posture/openness (35-39 set_gripper calls inside 56-60 decisions, the trailing world-event window all `grasp_failed`). Neither failure is attributable to the proposal channel: one is a direct cell with no channel at all, the other accepted 0 of 129 proposals. Note also that the same T2 + arm=none configuration scored 0/3 in the 2026-09-22 ablation (two first-decision hallucinated `done` terminals) and 2/3 today: gpt-6-astra is a live model, so cell-level numbers are not stable across weeks.


## Prediction verdicts (pre-registered rules)

Decision rules fixed before the grid ran: a success difference counts as support only when the gap is >= 3 of 6 cells and two-sided Fisher exact p < 0.10; a gap of 2 cells is 'inconclusive/weak'; <= 1 cell is a refutation. n = 6 per arm (2 tasks x 3 seeds), so all verdicts are necessarily coarse.

**P1** hybrid-oracle >= direct, and astra authors fewer actions. oracle 6/6 vs direct 5/6 (Fisher p=1.000) -> 支持 on success (both tasks at ceiling; the separating evidence is T2, where direct needed 18/42/56 decisions and failed once while oracle needed 22/16/20 and always succeeded); astra-authored actions oracle 65 (of which 61 diverge from the proposed waypoint) vs direct 152 - all of them; model calls oracle 85 vs direct 152 -> 支持 on the authorship clause.

**P2** hybrid-degenerate ~ direct. degenerate 5/6 vs direct 5/6 (delta +0 cells, Fisher p=1.000); degenerate accept rate 0% -> 支持.

**P3** hybrid-biased < direct. biased 6/6 vs direct 5/6 (delta +1 cells, Fisher p=1.000); biased vs degenerate 6 vs 5 -> 反驳.

   Mechanistically, the anchoring signature P3 predicts is absent: in every corrupted arm the longest run of consecutive accepts is <= 2 (oracle cells reach 7), the accepted proposals are free-space hovers and uncorrupted gripper commands, and each accepted wrong proposal is followed by a corrective REPLACE. The 5 cm and 2 cm biases were rejected 72/76 and 81/86 times.

**P4** acceptance falls with proposal quality, biased stays high. Pooled accept rate oracle 23.5% > noisy 5.1% > biased 5.3% > degenerate 0.0% (within the 2 pp tolerance the pre-registered ordering holds: True -> 支持); the anchoring clause needs biased to stay stubbornly higher than degenerate: gap +5.3% against a 15 pp bar -> 未达阈值(不支持).

**Reference point (paper)**: 14.4% of hybrid steps are Astra-generated or corrected on RoboDojo; our astra intervention share (REPLACE / model calls) is direct 100.0%, oracle 76.5%, noisy 94.9%, biased 94.7%, biased-soft 94.2%, degenerate 100.0%.

**Exploratory arm (not pre-registered)**: hybrid-biased-soft (pure +2 cm x offset) 6/6 vs direct 5/6, accept rate 6%, astra-authored 81/86 calls. Added because the smoke cell showed the 5 cm / 15 deg bias is rejected outright; it probes the same axis at an error size near the task tolerances.


## Token use (from usage.json, when the provider reported it)

| Arm | calls | input | cached input | output | total | total per cell |
|---|---|---|---|---|---|---|
| direct | 6 | 5322452 | 4416768 | 17451 | 5339903 | 889984 |
| oracle | 6 | 2668994 | 2158336 | 7256 | 2676250 | 446042 |
| noisy | 6 | 2448344 | 2065408 | 7585 | 2455929 | 409322 |
| biased | 6 | 2333684 | 1968256 | 7177 | 2340861 | 390144 |
| biased-soft | 6 | 2690416 | 2276224 | 7930 | 2698346 | 449724 |
| degenerate | 6 | 5004072 | 4208640 | 13524 | 5017596 | 836266 |
