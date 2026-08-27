# GigaBrain-0.7 (arXiv 2608.15875) 源表提取

从论文 markdown(papers/vla/gigabrain07-2608.15875.md)逐表提取,供跨巡航对照引用。
全部为论文自报数字,未独立核验(代码未放出)。

## Tab 4 — VLM 骨干对比(System 1,真机 SR%)

| Backbone | Size | Clean Desk | Fruit Picking | Shirt Folding |
|---|---|---|---|---|
| PaliGemma2 | 3.5B | 50 | 88 | 30 |
| Qwen3.5 | 5B | 60 | 12 | 0 |
| Gemma 4 | 8.5B | 100* | 92 | 0 |

*Gemma 4 Clean Desk 为限范围位置泛化设定。

## Tab 5 — VLM-动作专家耦合(PaliGemma2 骨干)

| Architecture | Clean Desk | Fruit | Shirt | Train s/step | Infer s |
|---|---|---|---|---|---|
| Dual Stream | 50 | 88 | 30 | 4.93 | 0.221 |
| Last-Layer CrossAttn | 20 | 40 | 0 | 4.65 | 0.073 |
| Multi-Layer CrossAttn | 20 | 36 | 0 | 6.59 | 0.108 |

## Tab 6 — 后训练语言跟随(SR%,六任务均值)

| Model | PiPER Avg | H01 Avg |
|---|---|---|
| π0.5 | 88.8 | 75.2 |
| GigaBrain-0.1 | 76.1 | 69.6 |
| G0.5 | 81.4 | 25.7 |
| Xiaomi-Robotics-1 | 72.3 | 58.7 |
| GigaBrain-0.7 | 91.5 | 84.2 |

## Tab 7 — 后训练复杂操作(SR%,均值)

| Model | PiPER Avg(5 任务) | H01 Avg(7 任务) |
|---|---|---|
| π0.5 | 76.6 | 45.2 |
| GigaBrain-0.1 | 64.8 | 43.2 |
| G0.5 | 37.8 | 8.3 |
| Xiaomi-Robotics-1 | 48.9 | 32.8 |
| GigaBrain-0.7 | 84.9 | 74.1 |

## Tab 8 — MiMo-Embodied( embodied VL,13 benchmark 均值)

GigaBrain-0.7: Spatial .5215 / Affordance .3669 / Overall .4621(对比 G0.5-base .3916)。
MiMo 数据并入预训练混合后 0.5704(论文自注:此后训练集与评测不再不相交,仅作诊断)。

## Tab 9 — RoboTwin 2.0(Co-Train,50 任务 × 50 演示)

| Model | Easy | Hard | Overall |
|---|---|---|---|
| π0 (Single) | 46.42 | 16.34 | 31.38 |
| Xiaomi-Robotics-0 | 62.9 | 18.2 | 40.55 |
| X-VLA | 68.0 | 20.9 | 44.45 |
| X-WAM | 70.0 | 25.8 | 47.90 |
| π0.5 | 70.7 | 46.0 | 58.35 |
| GigaBrain-0.7 | 66.8 | 67.9 | 67.35 |

## Tab 10 — EBench

| Model | SR% | Score |
|---|---|---|
| π0 | 23.59 | 37 |
| X-VLA | 23.72 | 35 |
| π0.5 | 28.08 | 42 |
| GigaBrain-0.7 | 33.30 | 46.1 |

附录榜单快照(2026-08-15):Qwen-RobotManip 45.58% / 60 居首,正文未对比。

## Tab 11 — RoboColiseum

| Model | Instr | Spatial | Robust | Manip |
|---|---|---|---|---|
| π0 | .3680 | .1300 | .3130 | .3470 |
| GR00T N1.7 | .6460 | .2490 | .5380 | .4380 |
| π0.5 | .7460 | .3560 | .6130 | .5820 |
| ACoT-VLA | .7570 | .3970 | .6220 | .4770 |
| GigaBrain-0.7 | .8166 | .4729 | .6800 | .6092 |

## Tab 12 — 经验 RL 三阶段(SR%)

| Task | SFT | Offline RL | Online RL |
|---|---|---|---|
| Link Installation (PiPER) | 20 | 40 | 100 |
| Gift Box Packing (PiPER-X) | 80 | 90 | 100 |
| Cable Tie (PiPER-X) | 0 | 40 | 100 |
| Bearing Installation (H01) | 20 | 60 | 100 |

## System 3 消融(6.5.2)

| 任务 | BASE | +SubImage | +Value | +两者 |
|---|---|---|---|---|
| 叠衣服(SR/分/秒) | 100 / 68.3 / 107 | 100 / 81.7 / 83 | 100 / 85.0 / 79 | 100 / 88.3 / 75 |
| 礼物包装(SR/分) | 0 / 61.1 | 20 / 71.4 | 60 / 93.3 | 80 / 96.7 |
| 方块分拣(SR/分) | 40 / 55.0 | 50 / — | 45 / — | 55 / 87.5 |
