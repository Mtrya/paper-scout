# Run Index

Compact coverage log and dedup source of truth, newest first. Before serious scouting, read this to avoid repeating papers or research threads already covered. After each delivered run, append one block. Run reports and preserved evidence live in `runs/<run-id>/`, not here.

Entry format:

```
## YYYY-MM-DD — <period covered>
- Doc: <url>
- Run: runs/<run-id>/
- Deep threads: <thread name> (<paper ids if any>), <thread name>
- Covered papers: <id>, <id>, ...
- Shortlisted papers: <id>, <id>, ...
```

## 2026-08-10 — 2026-08-08 to 2026-08-10
- Doc: https://fudan-nlp.feishu.cn/docx/ViE2dEgqmonnOqx3CnZcHAWOnDd
- Run: runs/2026-08-10-roundtrip-worldtrace-simwam/
- Deep threads: Round-Trip Consistency 精读 + 六机制 Lorenz/单摆对照探针("吸引子失明"猜想证伪为虚警,C_i 测的是逆向腿)(2608.00675), WorldTrace 精读 + Qwen3 真实权重 RoPE 相消三组测量(逐频率存活、logit 平凡性、softmax 读权重)(2608.07408), SimWAM 精读 + 推理路径审计(视频塔未退场)+ 双 checkpoint census/IL-RL delta + 4090 prefill 截断探针(表征第 ~15 层收敛、prefill 仅占延迟 4.5%)(2608.07468)
- Covered papers: 2608.00675, 2608.07408, 2608.07468
- Shortlisted papers: 2608.05424, 2608.06729, 2608.06375, 2608.06994, 2608.06799, 2608.01851, 2608.05219, 2608.05703, 2608.06756

## 2026-08-08 — 2026-08-06 to 2026-08-08
- Doc: https://fudan-nlp.feishu.cn/docx/TKxSdihK6oEffBxiCUhcnSlxnHh
- Run: runs/2026-08-08-mass-memorytrust/
- Deep threads: MASS 精读 + typed-carrier 核心消融独立复现（5.61M/2.85M 双模型）+ 三探针（合法平行世界线 70.6%、确定性流 tick 1 全分歧、世界死亡吸引子 24–50 唯一状态）(2608.06257), When Memory Lies 精读 + SpatialSTALE 测试床重建与校准 + 模态鸿沟开源复跑（Qwen3-VL-8B 文本 1.000 vs 视觉 0.158)+ 图像消融（"对图像敏感但读不懂"第三种失败模式）+ 四因素仲裁曲线（文本天花板 null result)(2608.04574)
- Covered papers: 2608.06257, 2608.04574
- Shortlisted papers: 2608.05369, 2608.01964, 2608.05013, 2607.23783, 2608.06197, 2608.05042, 2608.03392, 2608.06374

## 2026-08-07 — 2026-08-04 to 2026-08-06
- Doc: https://fudan-nlp.feishu.cn/docx/N50edTJGso5vkzxtW9FceyWJnzf
- Run: runs/2026-08-07-worldcycle-wamspace/
- Deep threads: WorldCycle 精读 + ABot-World-0 六协议可逆循环探针（sink 锚定骗过闭环指标；avatar 转身代逆平移；回归后功能衰减 3×）(2608.04964), WAM 未来监督空间之争：ST-WAM/SG-WAM 精读与代码审计 + 三元组诊断独立复现 99 组（核心主张成立、VAE 判别率数字翻转、DINO 不变性过剩新发现）(2607.28993, 2608.01397)
- Covered papers: 2608.04964, 2607.28993, 2608.01397
- Shortlisted papers: 2608.03207, 2608.00486, 2608.02603, 2608.02580, 2608.02713, 2608.01127, 2607.29613, 2608.03994

## 2026-08-05 — 2026-07-20 to 2026-08-05
- Doc: https://fudan-nlp.feishu.cn/docx/Mc34d9LkeounK4xcyrLcpqIDnmf
- Run: runs/2026-08-05-visual-memory-posthoc-faithfulness/
- Deep threads: VLA 视觉记忆：NativeMEM/SOMA 精读 + MemoryVLA 固定槽 consolidation 代码审计与完整 checkpoint 结构核验 + History-Swap benchmark 设计（2607.06678, 2605.22283）, hindsight 合理化：Post-Hoc Reasoning 分阶段 Qwen3-VL-8B activation steering + Faithful Self-Evolvers 扰动代码审计与 uptake×robustness 二维重定义（2603.01437, 2601.22436）
- Covered papers: 2607.06678, 2605.22283, 2603.01437, 2601.22436
- Shortlisted papers: —（四篇均进入两条深挖线；本条为同日初稿经用户反馈后的唯一 canonical 大修版）

## 2026-08-03 — 2026-08-02 to 2026-08-03
- Doc: https://fudan-nlp.feishu.cn/docx/IOIwdEw2hoSguvxr1qNcs3YunWg
- Run: runs/2026-08-03-vtla-vipe-spatialcli/
- Deep threads: N₀-VTLA 精读 + 代码解剖(放出权重≠论文架构)+ 真实权重合成批次 z 敏感性探针 (2607.23782), VIPE 精读 + Wan2.2-5B 条件帧编辑反向检验(8 场景 × 3 编辑真实生成) (2607.25537), SpatialCLI 精读 + 放出范围核验(Internalize 数据未放出) (2607.27703)
- Covered papers: 2607.23782, 2607.25537, 2607.27703
- Shortlisted papers: 2607.28415, 2607.22561, 2607.21848, 2607.22393, 2607.23806, 2607.25308

## 2026-08-02 — 2026-07-31 to 2026-08-02
- Doc: https://fudan-nlp.feishu.cn/docx/Pfa4dqGlVoC9q4xmUd6cudzDnzh
- Run: runs/2026-08-02-ace-phizero-shadowdancer/
- Deep threads: ACE-Data-0 测量引擎规格核验 + 发布状态三角验证 (2607.28625), PhiZero 精读 + Wan2.2-5B 8 场景物理失败模式解剖实验 (2607.28624), ShadowDancer 精读 + sprites cross-shadow 正则/配对机制探针 (2607.28362)
- Covered papers: 2607.28625, 2607.28624, 2607.28362
- Shortlisted papers: 2607.27180, 2607.26056, 2607.26037, 2607.26754, 2607.27380, 2607.26760, 2607.28227, 2607.28568, 2607.22798, 2607.23402

## 2026-07-31 — 2026-07-18 to 2026-07-30
- Doc: https://qcn0umnxrmj2.feishu.cn/docx/B10IdsJp0okcqXxwei2cPQlGnkg (本地副本: runs/2026-07-31-abotworld-umi-turbovla/report.docxxml)
- Run: runs/2026-07-31-abotworld-umi-turbovla/
- Deep threads: ABot-World-0 4090 实跑 + 640-block 可控性阵发崩溃实验 (2607.19191), UMI 数据 fidelity×scale + HiFi-UMI-2K recovery 含量测量 (2607.15330, 2607.25895), 实时策略口径核验 + πR² staircase 玩具复现 (2607.27205, 2607.26055)
- Covered papers: 2607.19191, 2607.15330, 2607.25895, 2607.27205, 2607.26055
- Shortlisted papers: 2607.26754, 2607.26037, 2607.18367, 2607.23909, 2607.19343, 2607.18703, 2607.17977, 2607.27180, 2607.13429, 2607.24744, 2607.11498, 2607.25337, 2607.16401, 2607.14183, 2607.21655, 2607.24653

## 2026-07-18 — 2026-07-17
- Doc: https://fudan-nlp.feishu.cn/docx/P3S7dtwhyomKNZxULvocMplknX6
- Run: runs/2026-07-18-badwam-robotttt-gamestate/
- Deep threads: BadWAM world-action drift toy-WAM reconstruction + attack probe (2607.15207), RoboTTT fast-weight context probe (2607.15275), Pixels-to-States game-engine taxonomy triangulation (2607.14076)
- Covered papers: 2607.15207, 2607.15275, 2607.14076
- Shortlisted papers: 2607.13399, 2607.14777, 2607.15038, 2607.14187, 2607.14935, 2607.14952

## 2026-07-17 — 2026-07-16
- Doc: https://fudan-nlp.feishu.cn/docx/DVafdVjjvowpWPxo7x8cIJzmnQe
- Run: runs/2026-07-17-gigaworld05-harness-spear/
- Deep threads: GigaWorld-Policy-0.5 action-centered WAM code+weights verification (2607.13960), Harness Handbook behavior localization + mini-handbook probe on real Terminus-2 (2607.13285), SPEAR UE reflection-driven simulator code trace (2607.06701)
- Covered papers: 2607.13960, 2607.13285, 2607.06701
- Shortlisted papers: 2607.13104, 2607.12747, 2607.07702, 2607.12625, 2607.13921, 2607.12477, 2607.12395, 2607.13125, 2607.13639, 2607.09786

## 2026-07-16 — 2026-07-14 to 2026-07-15
- Doc: https://fudan-nlp.feishu.cn/docx/UIt3d2fE0oc8F7xZ0aAcPaFMnyf
- Run: runs/2026-07-16-densereward-terrazero-flowwam/
- Deep threads: DenseReward failure synthesis for dense rewards (2607.13033), TerraZero procedural driving sim and zero-demo self-play (2607.13028), FlowWAM optical flow as unified action representation for WAMs (2607.13017)
- Covered papers: 2607.13033, 2607.13028, 2607.13017
- Shortlisted papers: 2607.12992, 2607.12931, 2607.12892, 2607.12659, 2607.12571, 2607.12356

## 2026-07-14 — 2026-07-08 to 2026-07-14
- Doc: https://fudan-nlp.feishu.cn/docx/CQBSdrQELoOXcExGR83cZzasnvc
- Run: runs/2026-07-14-genception-lhtb-robodojo/
- Deep threads: GenCeption generative video as universal vision prior (2607.09024), Long-Horizon-Terminal-Bench dense grading for long-horizon agents (2607.08964), RoboDojo sim-and-real manipulation diagnosis (2607.04434)
- Covered papers: 2607.09024, 2607.08964, 2607.04434
- Shortlisted papers: 2607.09657, 2607.09661, 2607.06403, 2607.06291, 2607.05373, 2607.02403, 2607.04988, 2607.04425, 2607.06838, 2607.08716

## 2026-07-13 — 2026-07-08 to 2026-07-13
- Doc: https://fudan-nlp.feishu.cn/docx/UC4pdUL3bo8OS1xXokIch228nKb
- Run: runs/2026-07-13-lingbot-kinematic-memory/
- Deep threads: LingBot world models (2607.07675, 2607.07534), Imagined Rollouts kinematic diagnosis (2607.05966), MIRA multiplayer world models (2607.05352)
- Covered papers: 2607.07675, 2607.07534, 2607.05966, 2607.05352
- Shortlisted papers: 2607.07608, 2607.06442, 2607.06018, 2607.04434, 2607.03723, 2607.05765, 2607.08716, 2607.08768, 2606.30111, 2607.02501, 2607.07508, 2607.06987

## 2026-07-12 — 2026-07-08 to 2026-07-11
- Doc: https://fudan-nlp.feishu.cn/docx/XjeTdcYjroyDJUxikhPcEAE3nld
- Run: runs/2026-07-12-lamem-rynnworld-sieve/
- Deep threads: LaMem-VLA latent-memory probe (2607.07608), RynnWorld-4D code trace and 4D-policy analysis (2607.06559), SIEVE structure-aware selection probe (2607.06442)
- Covered papers: 2607.07608, 2607.06559, 2607.06442
- Shortlisted papers: 2607.08716, 2607.04988, 2607.03751, 2607.06558, 2607.06291, 2607.05352, 2607.06018, 2607.05765, 2607.05390, 2607.07534, 2607.07675, 2607.02466, 2607.04434, 2607.06403, 2607.02646, 2607.07508, 2607.08763, 2607.03748, 2607.03723, 2607.08768, 2607.06838

## 2026-07-11 — 2026-06-28 to 2026-07-11
- Doc: https://fudan-nlp.feishu.cn/docx/JaRGdeEawoAvAKxPQCgc3ZHFnUg
- Run: runs/2026-07-11-gigaworld-vla-corrector-physis/
- Deep threads: GigaWorld-1 world-model policy evaluation (2607.02642), PhysisForcing physics-reinforced world simulator (2606.28128), VLA-Corrector detect-and-correct inference (2607.01804)
- Covered papers: 2607.02642, 2606.28128, 2607.01804
- Shortlisted papers: 2607.06559, 2607.05390, 2607.02403, 2607.02517, 2607.05966, 2607.04434, 2607.07608, 2607.03751, 2607.00678, 2607.00272, 2607.02501

## 2026-06-27 — 2026-06-26 to 2026-06-27
- Doc: https://fudan-nlp.feishu.cn/docx/RXoKdXYYtowcXjxSELKckfghnRb
- Run: runs/2026-06-27-icwm-fastlew-hallucination/
- Deep threads: In-Context World Modeling for Robotic Control (2606.26025), Fast LeWorldModel (2606.26217), Hallucination in World Models is Predictable and Preventable (2606.27326)
- Covered papers: 2606.26025, 2606.26217, 2606.27326
- Shortlisted papers: 2606.27364, 2606.26790, 2606.26907

## 2026-06-25 — 2026-06-23 to 2026-06-25
- Doc: https://fudan-nlp.feishu.cn/docx/NXqedB7DAo2mx3xLcyhct759ndd
- Run: runs/2026-06-25-robotwin-foresight-beyond-gradients/
- Deep threads: RoboTwin 2.0 synthetic bimanual data engine (2506.18088), Foresight failure detection with action-conditioned world-model latents (2606.23085), Learning Beyond Gradients heuristic-learning paradigm (blog)
- Covered papers: 2506.18088, 2606.23085
- Shortlisted papers: 2606.20092, 2606.24742, 2606.22540

## 2026-06-22 — 2026-06-15 to 2026-06-22
- Doc: https://fudan-nlp.feishu.cn/docx/WA6hdqAhWox75Zx0DkRcAxULnTG
- Run: runs/2026-06-22-humanscale-playful-imagewam-wrbench/
- Deep threads: HumanScale egocentric vs. real-robot pretraining (2606.20521), ImageWAM image-editing world-action model (2606.19531), WRBench persistent-state diagnosis (2606.20545), Playful RATS code-as-policy play learning (2606.19419)
- Covered papers: 2606.20521, 2606.19531, 2606.20545, 2606.19419
- Shortlisted papers: 2606.15133, 2606.20515, 2606.19980, 2606.16122, 2606.20083, 2606.19495, 2606.18847, 2606.17480, 2606.18558, 2606.17030, 2606.14667, 2606.19338, 2606.00793, 2606.18180

## 2026-06-18 — 2026-06-16 to 2026-06-18
- Doc: https://fudan-nlp.feishu.cn/docx/Yc1HdNNXEoTGk2xuG3rcVtpEnSb
- Run: runs/2026-06-18-guava-kairos-omniagent/
- Deep threads: Guava harness for embodied manipulation (2606.18363), Kairos native world model stack (2606.16533), OmniAgent active perception (2606.19341)
- Covered papers: 2606.18363, 2606.16533, 2606.19341
- Shortlisted papers: 2606.18375, 2606.18208, 2606.18180, 2606.18216, 2606.17628, 2606.17861

## 2026-06-17 — 2026-06-16 to 2026-06-17
- Doc: https://fudan-nlp.feishu.cn/docx/B5IxdNyLAoqHcNxyoZHc7CGenMJ
- Run: runs/2026-06-17-aceego-actworld-motionvla/
- Deep threads: ACE-Ego-0 egocentric-robot VLA pretraining (2606.17200), ActWorld action-aware interactive world model (2606.17730), MotionVLA dual-stream humanoid motion tokenizer (2606.15142)
- Covered papers: 2606.17200, 2606.17730, 2606.15142
- Shortlisted papers: 2606.17054, 2606.17030, 2606.15768, 2606.17043, 2606.16519

## 2026-06-16 — 2026-06-15 to 2026-06-16
- Doc: https://fudan-nlp.feishu.cn/docx/NtxjdBv3AoNSQPxWD4XcqFpqnce
- Run: runs/2026-06-16-gam-apt-dreamx/
- Deep threads: Geometric Action Model (2606.17046), APT action-expert pretraining (2606.12366), DreamX-World 1.0 (2606.16993)
- Covered papers: 2606.17046, 2606.12366, 2606.16993
- Shortlisted papers: 2606.15631, 2606.09813, 2606.06194, 2606.14777, 2606.16295, 2606.17030, 2606.16519

## 2026-06-15 — 2026-06-15
- Doc: https://fudan-nlp.feishu.cn/docx/SVnEd0IsyobD8fx71yPcHPCynsd
- Run: runs/2026-06-15-mu0-hyvla/
- Deep threads: μ₀ 3D interaction-trace world model (2606.13769), Hy-Embodied-0.5-VLA full robot learning stack (2606.14409)
- Covered papers: 2606.13769, 2606.14409
- Shortlisted papers: 2606.13679, 2606.14249, 2606.14579, 2606.12384

## 2026-06-07 — 2026-05-28 to 2026-06-07
- Doc: https://fudan-nlp.feishu.cn/docx/G8dFd8ry2oC3yZxdIb3cGvwonnc
- Run: runs/2026-06-07-cosmos3-grail-qwenvla/
- Deep threads: Cosmos 3 omnimodal world models (2606.02800), Qwen-VLA text-to-action pretraining (2605.30280), GRAIL synthetic humanoid loco-manipulation (2606.05160)
- Covered papers: 2606.02800, 2605.30280, 2606.05160
- Shortlisted papers: 2606.03603, 2606.01247, 2606.03985

## 2026-06-10 — 2026-06-08 to 2026-06-10
- Doc: https://fudan-nlp.feishu.cn/docx/I2QadXXR6o0i9Kx8U5UcKbb5npf
- Run: runs/2026-06-10-oasis-ahawam-tbdvla/
- Deep threads: OASIS sim-to-real humanoid loco-manipulation (2606.08548), AHA-WAM async world-action modeling (2606.09811), TBD-VLA temporal block diffusion VLA (2606.07895), QGF test-time gradient guidance for flow policies (2606.11087)
- Covered papers: 2606.08548, 2606.09811, 2606.07895, 2606.11087
- Shortlisted papers: 2605.25077, 2606.06556, 2606.09669, 2606.11129, 2606.09828, 2606.07723, 2606.06476

## 2026-06-14 — 2026-06-08 to 2026-06-14
- Doc: https://fudan-nlp.feishu.cn/docx/VpjwdvelZo2tW6xUjP3ccv8lnp5
- Run: runs/2026-06-14-weaver-eurekagent-repwam/
- Deep threads: WEAVER latent world model for manipulation (2606.13672), RepWAM representation visual-action tokenizer (2606.13674), EurekAgent environment engineering for autonomous research (2606.13662)
- Covered papers: 2606.13672, 2606.13674, 2606.13662
- Shortlisted papers: 2606.12072, 2606.01027, 2606.11482, 2606.09426, 2606.08039, 2606.13681, 2606.11926, 2606.11119

## 2026-06-12 — 2026-06-11 to 2026-06-12
- Doc: https://fudan-nlp.feishu.cn/docx/YT9Rd6YeLoHZxQx1TczcBWpCn1e
- Run: runs/2026-06-12-spatialclaw-moverse-labvla/
- Deep threads: LabVLA scientific-lab VLA (2606.13578), MoVerse panoramic Gaussian world model (2606.13376), SpatialClaw code-as-action spatial reasoning (2606.13673)
- Covered papers: 2606.13578, 2606.13376, 2606.13673
- Shortlisted papers: 2606.12195, 2606.13681, 2606.13662, 2606.11926, 2606.12373

## 2026-06-11 — 2026-06-08 to 2026-06-11
- Doc: https://fudan-nlp.feishu.cn/docx/Li5YdEumooOkIrxTQgfcBivRnKe
- Run: runs/2026-06-11-alebench-worldpilot-embodiedr1-nextforcing/
- Deep threads: ALE-Bench long-horizon algorithm engineering (2506.09050), World Pilot steering VLAs with WAM priors (2606.12403), Next Forcing multi-chunk prediction for causal world models (2606.11187), Embodied-R1.5 unified embodied foundation model (2606.11324)
- Covered papers: 2506.09050, 2606.12403, 2606.11187, 2606.11324
- Shortlisted papers: 2606.09828, 2606.12072, 2606.07100

---
