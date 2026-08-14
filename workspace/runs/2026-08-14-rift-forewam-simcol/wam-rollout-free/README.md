# wam-rollout-free — RIFT/ForeWAM 免 rollout 未来读:玩具尺度机制重建

## 线程问题

RIFT(2608.11521)与 ForeWAM(2608.11605)同日提出同一主张:WAM 的动作专家在推理时需要的不是迭代 rollout 的轨迹,而是一个**一次性生产、全程复用**的未来 K/V 缓存。RIFT 用四模型 × 40 任务 × 2000 配对试验的干预电池(掩码/乱序/时间交换/终态缓存重放)证明"消费与生产分离";ForeWAM 用噪声未来槽 + σ=1 单趟 prefill 实现生产侧。两个方案都没放代码。

本线程的目标:在受控合成环境里独立重建 RIFT 的干预协议,比较五种"未来读"形态(joint rollout / currentonly / RIFT-L2 / RIFT-FM / ForeWAM 噪声槽),并回答一个两篇论文都没问的问题:**当指令(观测侧条件)与注入的未来缓存(计划侧)冲突时,动作跟谁走?**(计划注入探针,把"动作读未来"从相关推到因果。)

## 设计

- 数据(`gen_joint_data.py`):64×64 合成双连杆臂搬运,16 帧视频 + 32 步动作块的完整回合,两种策略模式(直搬 mode 0 / 高弧绕行 mode 1)让给定观测的未来**双模**——这是 FM-vs-L2 监督形态差异能显现的前提;planprobe 分割保存共享首帧与目标的模式对。训练时做**滑窗采样**(`JointDataset`):观测帧 f∈0..7,未来 8 帧,16 步动作块——闭环执行第 2 块起喂的是中途状态,窗口采样是必需的前置(第一版只用完整轨迹的静止首帧,闭环成功率恒为 0,教训见 README 底部)。
- 模型(`joint_wam_probe.py`):联合视频+动作流匹配 transformer。token 布局 obs(16 patch + 1 模式 token)/ fut(128 = 8 帧 × 16 patch)/ act(16),10 层共享块;注意力掩码与 RIFT 一致(obs→obs;fut→obs+fut;act→全部;视频侧永远看不到动作 → 未来缓存动作无关,可记录-编辑-重放)。流匹配沿用 RIFT 约定 x_σ=(1−σ)Y+σε,目标 ε−Y。
- 五变体(同骨干、同数据、同 12000 步预算):
  - `joint`:rollout 基线,视频+动作共去噪,动作读演化缓存
  - `currentonly`:纯当前观测(Fast-WAM current-only 类比)
  - `rift-l2` / `rift-fm`:学习型 anticipation token(240 个,对齐未来位置),单趟 prefill;未来监督分别为 L2 回归与条件 FM(+stopped-grad L2 探针)
  - `noiseslots`:ForeWAM 式,未来槽填纯噪声、σ=1 单趟 prefill,动作损失反传穿过缓存
- 干预电池(`replay_action`/`edit_cache`):配对种子、同 ε_a0、同 σ 调度;掩码(未来源移出注意力)/ 噪声值 / 冻结当前值 / 空间乱序(帧内 patch 换位)/ 时间交换(帧间换位)/ 终态缓存重放;EE-ADE = 干预轨迹与原轨迹末端执行器平均漂移(像素),SR = 任务完成率。
- 计划注入(`run_plan_probe`):mode-A 回合第 0 块注入 mode-B 的**未来部分**缓存(保留 A 的观测/指令侧),测第 4 步 EE 到专家 A/B 的距离、整回合成功率、以及"跟随注入计划"的比例;control 重放自缓存验证协议保真(应为 0 漂移)。
- 延迟:每块 wall-clock(record rollout vs prefill+action)。

## 执行环境

- 启智 notebook `wam-rift-probe`(4090-2 组,1×4090,镜像 ngc-pytorch:25.02,cuda12.8 驱动)。
- venv `.venv2`(python3 -m venv --system-site-packages;pip 装 numpy==1.26.4 pillow;继承系统 torch 2.7+cu128)。
- **坑**:geniworld-probe 的旧 venv 里 torch 2.13 为 cuda13.2 节点编译,在 4090-2(cuda12.8 驱动)上 CUDA 不可用——新建 venv 继承系统 torch 解决。
- 重跑:`bash run_wam.sh`(数据生成 + 训练 + 评估);评估可独立重跑 `python joint_wam_probe.py --data joint_data --eval-only --tag v2`(读 results/*.pt)。

## 结果

见 `results/eval_v1.json` 与 `assets/` 下绘图;总结见报告"RIFT/我们的玩具重建"节。

## 局限

- 玩具尺度(无 VAE 潜变量、像素空间流匹配、12M 参数级)、脚本化合成数据、从头训练——机制级结论,不能外推到数值级。
- 干预的"冻结当前/噪声"是 out-of-distribution 编辑,只作对照读。
- RIFT 的 key/value 分离消融与首帧扰动课程未复刻;ForeWAM 的动力学寄存器未复刻(noiseslots 只含 Future-KV 半边)。
- 论文缓存:`papers/world-models/rift-2608.11521.md` 与 `forewam-2608.11605.md` 已存;simulator-collapse 的 markdown 缓存因本机内存 OOM 未生成(全文经 HTML 版读取)。
