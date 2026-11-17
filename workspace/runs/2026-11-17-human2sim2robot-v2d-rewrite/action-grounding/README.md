# 动作接地：物体轨迹与接触力矩

本目录继承 `runs/2026-11-15-human2sim2robot-v2d/` 的文献调查与探针，供 2026-11-17 报告重写核查。此次重写未新增实验，旧运行包与旧报告保持不变。

## 调查对象

- [Crossing the Human-Robot Embodiment Gap with Sim-to-Real RL using One Human Demonstration](https://arxiv.org/abs/2504.12609)，本地全文 `papers/robotics/human2sim2robot-2504.12609.md`，[官方代码](https://github.com/tylerlum/human2sim2robot)。
- [X-SIM: Cross-Embodiment Learning via Real-to-Sim-to-Real](https://arxiv.org/abs/2505.07096)，本地全文 `papers/robotics/x-sim-2505.07096.md`，[官方代码](https://github.com/portal-cornell/X-Sim)。
- [Learning Dexterous Manipulation Using Contact Wrench Guidance From Human Demonstration](https://arxiv.org/abs/2607.00033)，本地全文 `papers/robotics/chord-2607.00033.md`，实现位于 V2D 的 `robotic_grounding/flash_chord/`。

## 机制结论

Human2Sim2Robot 将人类演示分成两种作用不同的信号。物体逐帧 SE(3) 轨迹定义任务，转换成物体锚点的指数跟踪奖励；完整人手轨迹并不作为监督，只有物体开始运动前的一帧手姿经 IK 变成探索初始化。公开训练代码虽保存完整重定向轨迹，实际只读取第一帧预操作姿态；物体轨迹则逐帧读取。RL 因而可以找到与人不同的抓法，而非把不可达的人手动作硬映射到机器人。论文训练为单任务 PPO/LSTM，4096 个并行环境、单 A100 约 5–24 小时；真机策略仍输入在线物体 6D 位姿，而非 RGB。

X-Sim 保留物体中心奖励，补上“状态专家 → 成功仿真图像动作轨迹 → RGB Diffusion Policy”的蒸馏。每任务采集 500 条成功视觉轨迹。校准版再用 10 条由初始策略自行执行的真实 rollout（失败也可）与仿真重放组成逐帧图像对，以 InfoNCE 对齐表征；不需要遥操作，但真实采集、仿真配对和再训练仍是串行阶段。

CHORD 将中间变量推进到接触的作用效果。每个接触点与法向产生摩擦锥，锥边力 f 与接触位置 p 形成 wrench `[f, p×f]`。人手的接触集合与机器人的接触集合可以具有不同点数与位置，因此不直接比较 wrench 矩阵的列，而在 512 个预采样六维方向上分别取 support function，再比较两组 support 是否落在容差带内。实现还按物体包围半径归一化力矩维度，并另罚漏接触与非预期接触，见 `objectives/wrench.py` 与 `objectives/contact.py`。它不估计视频中不可见的绝对接触力，只比较单位力预算下可产生的力—力矩方向。

CHORD 的物体跟踪、手部模仿、接触引导仍与 virtual object control、参考帧重置、残差动作共同工作。论文报告 4739 个任务库、顺序训练其中 1831 个、同超参平均 82.12%；单任务 4096 环境、L40S 约 2 小时。接触噪声很大时改用 force-closure 覆盖奖励，稳定性提高但牺牲灵巧动作。真机闭环使用外部运动捕捉状态。

[DexMachina](https://arxiv.org/abs/2505.24853) 的虚拟物体控制以弹簧阻尼驱动物体，再随课程撤去；[SPIDER](https://arxiv.org/abs/2511.09484) 直接采样优化控制序列，可选虚拟接触引导维持预期手物接触点的相对位置，两者不可混写为同一种辅助机制。SPIDER 的规模化实验以 3,827 段已有手物三维演示为输入，跨四种手计划 15,308 个任务，15,302 条任务记录中 15,041 次完成展开。按参考最终抬升 ≥15cm、仿真最终抬升 ≥10cm、最终位置误差 ≤10cm 保留 6,520 条，占计划任务 42.59%，保存 1,291,575 个 100Hz 帧，共 3.5877 小时。论文报告单任务中位墙钟约 220 秒（含预处理）；粗乘任务记录得约 935.1 L40S GPU 小时，或每保留轨迹小时约 260.6 GPU 小时。完整逐任务耗时与并发数未公开，因此这只是量级估算；输入也不是原始 RGB，保留率不可外推为完整视频管线通过率。

## 对规模化的约束

- 这条路线恢复的是任务参考，不是真实人体动作标签；物理优化负责产生机器人动作。
- 物体轨迹比完整人手轨迹更易获得，但对倾倒、旋拧、滑动物体等任务，接触方式本身是不可省略的信息。
- 并行仿真加速单任务动作搜索，并不自动消除逐视频资产重建、逐任务训练、视觉蒸馏和质量验收。
- 从状态专家到视觉策略是独立阶段；把“仿真轨迹可执行”直接等同于“视觉模型已学会”会漏掉最关键的数据生成与泛化问题。
