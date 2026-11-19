# ForceHarness：快闭环交回慢模型的“物理量 ±σ”在真实控制轨迹上是否可信

本目录是 2026-11-19 巡航中 ForceHarness 外部实验线程的证据包。对象是 arXiv 2611.21099《ForceHarness: Turning Force into Action and Physics for Frozen Frontier Models》（Show Lab, NUS）的官方实现，commit `733407c1ea90d8b474dff7809d9349d0f5f0b1d5`。全部实验在官方 HiddenPhys（MuJoCo/robosuite）里跑官方 scripted expert，不调用任何付费模型，不接触真机，不修改模型面对的文本、原语与安全限制。

## 一句话结论

ForceHarness 的第三层（physics translation）把力轨迹变成带 `±σ` 的物理量交给慢模型，但在同一份工具结果里，质量与质心这两个量的可信度并不相同：质量通道在论文噪声下诚实（20/20 判决正确，1σ 覆盖 80%），质心通道的 σ 只把扭矩残差按 0.01 N·m 计入，而仿真器每 episode 实际抽取 `0.1 × force_bias_std`（论文设置 0.03 N·m、1 N 通道 0.10 N·m，即这个常数小 3 倍与 10 倍）；实测整体 COM 误差 RMS 与报告 σ 之比因此是 1.22（论文通道）与 3.17（1 N 通道），平衡杆读数 0.47 cm（真实 0.19 cm）且 40% 没有任何标记；一个**未声明**的 0.5 N 残余偏置就能把质量的 1σ 覆盖率打到 0%。规模瓶颈不是每拍噪声，而是“要分辨的物理差落到 1–3σ”这一点：官方 OOD 质量档（0.30/0.45 kg）在 1 N 力通道下错误率 20%，与 σ 预测的 22% 一致；而同一估计器在三种控制层（完整力控 / 仅接触保护 / 无力控原语）下的辨识误差没有差别，差别在抓取瞬态的峰值接触力（13.6 / 1.9 / 1.9 N，质量任务）。

## 1. 问题，以及它在 DASH 对照中的位置

DASH 的主线是快慢系统的紧耦合：慢的规划/决策与快的控制回路互相约束。ForceHarness 是另一种形态——慢模型（每步数秒）只选择原语，快侧在控制器频率（约 20 Hz）闭环，然后把**物理量与不确定度**送回慢侧。这条回传通道是“紧耦合”真正发生的地方：慢模型此后的每一个动作都建立在“0.6 kg ± 32 g”“质心偏 3.0 ± 0.3 cm”这样的句子之上。

因此本线程要回答的不是“任务成功率高不高”，而是三个更窄的问题：

1. 这些 `物理量 ±σ` 在**真实控制轨迹**（专家抓取—抬升—保持的完整动作）上是否可信？即：报告的 σ 是否描述真实误差，区间是否覆盖，符号/方向是否正确。
2. 它们**何时成为规模化的瓶颈**？即：当要分辨的物理差缩小（OOD 质量档、2.5 cm 的质心档）、或力通道变差（1 N 偏置）、或标定在负载后失配（未声明残差）时，判决错误率如何变化，能否由 σ 预测。
3. 快侧的控制层本身对“未知物理量的辨识”有没有贡献？同一估计器、同一决策规则下，完整力控 / 仅接触保护 / 无力控原语三者的辨识误差差多少。

边界说明：没有付费模型，所以“慢模型”的角色由官方 scripted expert 的固定阈值扮演；本线程测量的是**工具结果层**（layer 3）的性质，不是模型如何使用它。这一点在第 6 节“局限”里再次说明。

## 2. 环境、种子与复现

- 官方仓：`code/ForceHarness`，commit `733407c1`，Python 3.11 venv，`pip install -e ".[dev,sim]"`，mujoco 3.2.7 / robosuite 1.5.1，CPU，`MUJOCO_GL` 不参与（全程 `render=False`，HiddenPhys 论文协议本身不用图像）。
- 门禁：Gate 0（505 tests）与 A0 已通过；A1/A2 见 `code/ForceHarness/runs/setup_report.md`（A1 的 3/5 `force_cap` 是 check_sim 自身盲降 20 cm、开口夹爪的单拍冲击，已在同一文件里用实验定位，未改动任何代码去绕过）。
- 种子：全部用开发种子 **9000–9019**（20 个/臂/任务）。测试种子 9100–9119 未参与任何调参；9100 只在 setup 报告里复现了一次 A1 的失败模式，不涉及调参。
- 样本量：实验一 4 任务 × 3 臂 × 20 种子 = **240 episodes**；实验二 2 任务 × 3 臂 × 20 种子 = **120 episodes**。总计 360 episodes，约 20 分钟 CPU，零 API 成本。
- 同一 seed 下物体摆放由 `np.random.seed(seed)` 决定，各臂一致；力噪声共享同一 `default_rng`，轨迹分岔后不再逐拍对应，因此臂间是“同场景、不同力通道”的准配对比较。
- 本次运行的原始日志（`e1_full.jsonl` 3.6 MB、`e2_full2.jsonl`、`figs/`）留在 `code/ForceHarness/runs/dev/`，那是会被清理的临时目录，不属于交付物；本包只带汇总、逐 episode CSV，以及能重建全部产物的脚本（第 8 节）。

## 3. 实验一：`±σ` 在真实控制轨迹上

### 3.1 设置

官方 `ScriptedAgent`（force 接口）跑完整 episode：`scene → rotate_z 对准 → move_to 上方 → move_until_contact 下探 → move_by 1.8 cm → grasp → lift(height=0.10, hold_time=1.0)`；估计值取自 `lift` 原语结束时的 `payload_mass` / `com_offset`（估计窗口为最后 20 拍、1.0 s 的静止段，`n=20`）。真值取自同一时刻：质量是 `task_state["mass"]`（reset 时由 seed 奇偶决定），质心是 `truth_fn` 在同一拍算出的 `com_lever_xy_mm`（世界系下质心相对腕部力传感器的水平杠杆）。

三个力通道臂，均为官方参数管道（`EpisodeConfig.force_noise_std / force_bias_std` 一路传到 `EstimatorContext`）：

| 臂 | 每拍噪声 (N) | 每 episode 偏置 σ (N) | 额外扰动 | 含义 |
|---|---|---|---|---|
| `paper` | 0.5 | 0.3 | 无 | 论文 HiddenPhys 设置（check_sim 与专家脚本的 EP 常量） |
| `bias1n` | 0.5 | 1.0 | 无 | 更差但**如实声明**的力通道：估计器的 σ 会随 `force_bias_std` 一起变大 |
| `undeclared` | 0.5 | 0.3 | +0.5 N 沿载荷轴、+0.02 N·m 横向扭矩 | 静止标定正确、负载/移动后出现残差，而估计器不知道；探针以 `DriftedSim` 子类在 `wrench()` 返回后叠加（不改仓库代码） |

四个任务：`weight_sort`（0.15/0.60 kg，判决界 0.375 kg）、`weight_sort_ood`（官方 OOD 质量档 0.30/0.45 kg）、`com_sort`（质心偏 4 cm 或居中）、`com_sort_ood`（2.5 cm 或居中）。

判决规则固定为估计器自己会给的规则：质量以 0.375 kg 为界；质心以估计器自己的 `balanced_floor = 2σ` 为界（`r > 2σ` 判 offset）。另记录官方专家使用的 0.12 N·m 扭矩阈值作为对照。

### 3.2 质量：论文噪声下诚实，但诚实依赖于噪声模型被如实声明

| 任务 | 臂 | 有效抓取 | 误差 均值 ± SD | 报告 σ | RMS/σ | 1σ / 2σ 覆盖 | 判决正确 |
|---|---|---|---|---|---|---|---|
| weight_sort | paper | 20/20 | +9.5 ± 21.3 g | 32.3 g | 0.71 | 80% / 100% | 20/20 |
| weight_sort | bias1n | 18/20 | +28.7 ± 61.4 g | 102.2 g | 0.65 | 89% / 100% | 18/20 |
| weight_sort | undeclared | 20/20 | **+60.7 ± 21.5 g** | 32.3 g | **1.99** | **0% / 60%** | 20/20 |
| weight_sort_ood | paper | 20/20 | +8.6 ± 20.7 g | 32.3 g | 0.68 | 80% / 100% | 20/20 |
| weight_sort_ood | bias1n | 18/20 | +27.2 ± 60.3 g | 102.3 g | 0.63 | 89% / 100% | 16/20 |
| weight_sort_ood | undeclared | 20/20 | +59.1 ± 20.8 g | 32.3 g | 1.93 | 5% / 70% | 18/20 |

线性拟合（对两类均值回归 `est = a + b·m`，剔除抓空 episode）：`paper` 为 `a = +6.9 g, b = 1.007`；`bias1n` 为 `a = +18.9 g, b = 1.028`；`undeclared` 为 `a = +58.2 g, b = 1.006`。即：估计器本身几乎是线性、无标度误差的；出问题的只有**偏置**项。

三件事值得写进报告：

1. **σ 的算术是对的。** 质量 σ = `sqrt((std/√n)² + bias_std²)/g`，在 0.3 N 时约为 32 g，在 1.0 N 时约为 102 g；实测误差的 RMS 分别是 σ 的 0.71 与 0.65 倍，1σ 覆盖 80–89%（名义 68%），也就是略保守而不过度。论文噪声下的 20/20 判决正确不是运气：半档距 0.225 kg 相当于 7σ，6.9 g 的系统偏移只占它的 26%。
2. **一个未声明的 0.5 N 残差就足以让区间失效。** 同一个估计器、同一个 σ 公式，只因为工作位姿里多了一个 0.5 N 的常量残差，估计整体偏 +61 g（+1.9σ），1σ 覆盖 0%，2σ 覆盖 60%——区间仍然“看起来”很窄（32 g），但它已经不在误差所在的位置。名义档判决仍是 20/20（偏置只吃掉 0.059/0.225 = 26% 的半档距），这正是危险之处：宽档位上偏置无症状。
3. **失分有两个来源，性质完全不同。** 名义档 `bias1n` 的 2 个错误全部是“抓空”：1 N 偏置让 `move_until_contact` 的 3 N 接触阈值误判，专家连续重试 3 次后夹爪闭合在空气上（`closed_on_something = False`，`opening = 2.1 mm`），`lift` 读到空手（-1.46 N / +0.80 N），两次都被 `no_load` 标记；**控制失效，但信号诚实**。OOD 档的 4 个错误里另有 2 个是真正的窄档位误判：0.30 kg 的方块被读成 0.381 / 0.437 kg（偏置 +28 g 加噪声），越过 0.375 kg 边界，且**没有任何 flag**——同一偏置在 0.225 kg 半档距下无症状，在 0.075 kg 半档距下就把判决推过边界。这正是“规模”一词的精确含义。

### 3.3 质心：σ 的扭矩残差项被低估 3–10 倍，整体误差为 σ 的 1.2–3.2 倍

| 任务 | 臂 | 误差 均值 ± SD | 报告 σ | RMS/σ | 1σ / 2σ 覆盖 | 平衡→offset 误报 | offset→平衡 漏报 | 专家 verdict |
|---|---|---|---|---|---|---|---|---|
| com_sort | paper | +0.09 ± 0.40 cm | 0.33 cm | 1.22 | 60% / 85% | **40%** | 0% | 20/20 |
| com_sort | bias1n | +0.55 ± 1.37 cm | 0.45 cm | **3.17** | 28% / 50% | **75%** | 0% | 14/20 |
| com_sort | undeclared | +0.00 ± 0.52 cm | 0.31 cm | 1.64 | 35% / 65% | 40% | 0% | 20/20 |
| com_sort_ood | paper | +0.09 ± 0.38 cm | 0.32 cm | 1.20 | 55% / 85% | 40% | 0% | 18/20 |
| com_sort_ood | bias1n | +0.66 ± 1.27 cm | 0.41 cm | 3.40 | 44% / 56% | 75% | 10% | 14/20 |
| com_sort_ood | undeclared | +0.04 ± 0.47 cm | 0.30 cm | 1.53 | 40% / 70% | 40% | 0% | 19/20 |

机制可以直接从代码里读出来，并被数据证实：

- 质量的 σ 使用 `ctx.bias_std`，也就是**被声明的** `force_bias_std`；质心的 σ 不用它，只加一个写死的 `0.01 N·m`（`translation/estimators.py:123`：`s_tau = hypot(stx, sty) + 0.01`）。而仿真器每 episode 的扭矩残差是 `0.1 × force_bias_std`（`backends/mujoco/env.py:321`）：论文设置下 0.03 N·m（3× 于公式），1 N 臂下 0.1 N·m（10× 于公式）。
- 一个 0.03 N·m 的残余扭矩，在 0.8 kg 的杆上就是 `0.03/(0.8·9.81) = 0.38 cm` 的假杠杆——与报告的 σ（0.33 cm）同量级。实测平衡杆的平均读数是 **0.47 cm**（其真实杠杆平均只有 0.19 cm，来自抓取点偏离几何中心），并且 **40% 的平衡杆没有 `within_noise` 标记**，也就是慢模型看到的是“质心偏 0.5–0.9 ± 0.3 cm”这样一条自信的、错误的句子。
- 在 1 N 通道上，残余扭矩 0.1 N·m → 1.27 cm 假杠杆，平衡杆平均读成 **1.66 cm**，75% 未加标记；专家 verdict 掉到 14/20，任务成功率 10/20（杆在指间滑落/放不进区域）。
- 方向本身是可信的：对 offset 杆，报告方向与真实杠杆方向的夹角中位数 5.3°（paper），100% 在 30° 以内；“哪一端重”的符号判断在所有臂上都是 10/10 正确（夹角均 < 90°）；paper 臂的夹角最大 12°（OOD 21°），1 N 臂中位数 14–28°、最大 78°，已接近把这一端判错。`(x_cm, y_cm)` 与真值都按世界系比较（这一点由数据确认：仿真扭矩读数与 `com - site` 的世界系水平分量一致）。
- 但“方向可信”不等于“量值可用”：杆在指间会倾斜（平均 10.4°，最大 45°），真实杠杆因此从名义 4.0 cm 缩到平均 3.38 cm（30% 的 offset episode 低于名义值的 75%）。这是物理衰减，不是估计误差——估计器准确跟踪的是“当前这个杠杆”，而任务问的是“这根杆的质心偏多少”。σ 对这两者的差别一无所知。

### 3.4 什么时候成为瓶颈：把“物理差 / σ”画出来

按“判决错误率 vs 判决余量（以 σ 计）”整理，规模瓶颈有一条很干净的判据：

| 设定 | 要分辨的半档距 | 报告 σ | 半档距/σ | σ 预测错误率 | 实测错误率 |
|---|---|---|---|---|---|
| weight_sort, paper | 225 g | 32 g | 7.0 | ~0% | 0% |
| weight_sort, undeclared | 225 g | 32 g（但偏 +59 g） | 7.0 → 有效 5.1 | 0%（区间已失效） | 0% |
| weight_sort_ood, paper | 75 g | 32 g | 2.3 | 1.2% | 0% |
| weight_sort_ood, bias1n | 75 g | 102 g | 0.74 | 22% | 20% |
| weight_sort_ood, undeclared | 75 g | 32 g（偏 +59 g） | 2.3 → 有效 0.5 | 15% | 10% |

也就是说：**当要分辨的物理差落到 1–3σ 时，`±σ` 从“背景信息”变成“瓶颈”**；在 7σ 时它可以完全错（1σ 覆盖 0%）而任务仍然全对——只测成功率会完全看不见这个问题。质心更早进入瓶颈，因为它的真实误差已经是 σ 的 1.2–3.2 倍（通道越差倍数越大），而且物理杠杆还会随倾斜缩水：4 cm 档在 `bias1n` 下，真实杠杆（3.38 cm）与真实误差（1.37 cm）之比只有 2.5，实测 14/20。

另外两个由 σ 派生的规则也会在规模上失效，值得单列：

- **`no_load` 标记（`m < 2σ`）会与真实载荷碰撞。** 1 N 通道下 σ = 102 g，2σ = 204 g > 真实轻方块 150 g，于是 **45% 的 episode 被标 `no_load`，其中大多数是正确测到的 150 g 方块**。标记本来用于“没抓到东西”，在大 σ 下变成了“轻的东西”。
- **`within_noise`（`r < 2σ`）同样如此**：平衡杆残余 0.47 cm 已经接近 2σ = 0.66 cm，于是 40% 的平衡杆没有任何标记；而在 1 N 通道下残余 1.66 cm 远大于 2σ，反而 75% 都被标成“有偏置”。

## 4. 实验二：控制层对未知物理量辨识的影响

三个控制臂，**同一估计器、同一判决规则、同一场景**，只在接触与抬升的处理上不同：

| 臂 | 下探 | 抓取后抬升 |
|---|---|---|
| `force` | `move_until_contact`（接触 3 N 停）+ `move_by` 1.8 cm | `lift(0.10, hold=1.0)`（力调制的柔顺上升 + 静止保持） |
| `guard` | `move_to` 名义高度（接触保护：碰到即停） | `move_by` 10 cm（2 cm/s）+ `hold(1.0)` |
| `pose` | 同上，但**无接触保护**（论文的 `none|pose`） | 同上 |

结果（20 seeds/格）：

| 任务 | 臂 | 辨识误差 RMS | 判决正确 | 抓取成功 | 接近段峰值力 | 物体抬升 |
|---|---|---|---|---|---|---|
| weight_sort | force | 20.7 g（0.64σ） | 20/20 | 20/20 | **13.6 N** | 10.2 cm |
| weight_sort | guard | 25.6 g（0.75σ） | 20/20 | 20/20 | 1.9 N | 9.6 cm |
| weight_sort | pose | 20.7 g（0.60σ） | 20/20 | 20/20 | 1.9 N | 9.6 cm |
| com_sort | force | 0.41 cm（1.25σ） | 16/20 | 20/20 | **20.4 N** | 9.7 cm |
| com_sort | guard | 0.51 cm（1.24σ） | 18/20 | 20/20 | 1.9 N | 8.7 cm |
| com_sort | pose | 0.43 cm（1.00σ） | 19/20 | 20/20 | 1.9 N | 8.7 cm |

结论是一个明确的否定结果，并且它比“力控更好”更有信息量：

- **辨识质量与是否使用力控原语无关。** RMS 误差的抽样标准误约为其值的 15%（n=20），20.7/25.6/20.7 g 与 0.41/0.51/0.43 cm 的差别都在一个标准误附近；判决比例的 1 个标准误约 8–11%，80/90/95% 同样不可分辨。原因很直接：辨识只需要“抓稳之后静止保持 1 秒”，而这一点在三条臂里都成立。
- **快环买到的是瞬态，而它的瞬态并不温和。** 峰值出现在 `move_until_contact` 这一步（不是随后的 `move_by`）：均值 13.6 N（质量）/ 20.4 N（质心），而 `guard/pose` 只有 1.9 N；最差 episode 达到 **37.2 N / 33.9 N**，并且 **2/20（质量）与 5/20（质心）的下探以 `force_cap` 终止**——也就是说，原语自己的 25 N 上限是被撞击之后才生效的。机制是检测器的 4 拍滑动平均 + 2 拍确认对单拍冲击有洞：目标以 6 mm 的 lead 压在刚性边缘上，一拍就能到 20–35 N。官方 expert 在实验一里同样如此：20 个 episode 的 40 次 `move_until_contact` 下探中，质量任务有 3 次、质心任务有 4 次以 `force_cap` 结束。
- 这条观测和 A1 的 `check_sim` 失败是同一件事：开口夹爪盲降到桌面时，单拍冲击到 27–30 N，`force_cap` 先于 `contact` 触发。所以 A1 的 3/5 “FAIL”不是检查脚本的孤例，而是“接触停止有一个拍宽的洞”的直接后果；在本实验的抓取下探里，这个洞的代价是 20–37 N。
- **`guard` 臂在这一场景下等于 `pose`**：名义抓取高度（桌面上方 2 cm）配合 7.7 cm 开口的平行夹爪，让手指正好从物体两侧掠过（`contact_fraction = 0`），接触保护从未被触发。所以本实验**没有**测到“接触保护”的价值；要测它需要场景高度误差（`wipe_hard` 式的桌面高度随机化），这超出本线程范围，如实记为未完成项。

这条结果给 DASH 对照一个具体说法：快环的贡献可以分解为（i）接触瞬态/安全、（ii）物理量测量、（iii）测量可信度声明。本实验里（ii）与（iii）由“静止保持”和“估计器”决定，与（i）解耦；（i）只在力控原语真正压入接触时才变差。

## 5. 慢模型实际读到什么

`phys2` 渲染的 `lift` 工具结果（`com_sort` seed 9007，paper 臂，**平衡**杆，真实杠杆 0.9 mm）：

```
primitive=lift params={'height': 0.1, 'hold_time': 1.0}
result: completed
...
contact events: none (no contact detected during this primitive)
contact state at end: IN CONTACT, |F|=7.7 N, mainly along -z
measurements (lift):
  payload mass               0.783 ± 0.032 kg
      [last 20 samples, 1.0 s]
  centre-of-mass offset      0.85 ± 0.33 cm  toward mostly +y (robot's left)
      [last 20 samples, 1.0 s]
```

这条结果把三件事同时摆在了慢模型面前：

1. **一条自信的、错误的质心读数**：0.85 ± 0.33 cm（距零 2.6σ），没有 `within_noise` 标记。真实杠杆 0.9 mm；多出来的约 0.4 cm 来自未声明的扭矩残差，见 3.3 节。40% 的平衡杆都是这种呈现方式。
2. **同一段文字里的接触矛盾**：`contact events: none` 与 `contact state at end: IN CONTACT, |F|=7.7 N` 并存。前半句来自力事件检测器（抬升期间 `suppress_contact` 抑制了“载荷≈接触”的误报），后半句来自 `gripper_contact_state`（手指确实夹着物体）。240 个 `lift` 结果里有 **229 个**同时出现这两句。对一个只有文本的慢模型，这是“没有接触”和“正在接触”的同一句话。
3. **σ 是唯一的误差信号**：240 条质量估计里有 109 条 |z| > 1，其中 **101 条没有任何 flag**。flags（`no_load`、`within_noise`、`not_settled`、`unsteady`）并不承担“系统性误差”的检测。

另外，`weight_sort` 里报告的 `centre-of-mass offset`（例：150 g 方块读到 3.31 ± 1.27 cm）对一个小方块并不是“物体质心偏了多少”，而是“抓取点相对腕部偏了多少”：4 cm 的方块允许抓偏 ±2 cm，而 σ 只描述传感器噪声。这是 `com_offset` 在非 com_sort 任务上的定义问题，报告若引用该字段应说明。

## 6. 局限

- 20 seeds/格，判决比例的 1 个标准误约 8–11%；`guard` 与 `pose` 之间、`com_sort` 的 80/90/95% 差异均在噪声内，不能当作排序证据。
- “慢模型”由官方 expert 的固定阈值代替；真实前沿模型如何权衡 `±σ`、`no_load` 与“IN CONTACT”矛盾文本，本实验没有测，也不应外推。
- `undeclared` 臂是对 `wrench()` 的探针式扰动（+0.5 N、+0.02 N·m 常量），模拟“静止标定正确、负载后出现残差”，不是官方配置；数值大小是我选的，不应读成任何具体硬件的规格。
- 没有真机、没有热漂移、没有跨 episode 的时变漂移；仿真的偏置每 episode 恒定，因此“未声明残差”的下界被低估（真机的残差会随时间与位姿变化）。
- 质心真值取“估计时刻质心相对力传感器的水平杠杆”，它包含抓取点偏离几何中心的那部分（平衡杆平均 0.19 cm）。这正是任务变量与可测量之间的定义差，我在正文里把它当作物理混淆而不是估计误差处理。
- 实验二只覆盖“场景位姿准确”的情形；`pose` 臂免费获得了正确的高度，因此“无力控原语”的代价被系统性低估。接触保护未被触发（见 4 节），这是未完成项而非阴性结果。
- 未跑训练型策略、未做多噪声重复（每个 seed × 臂只有一条噪声实现），因此区间统计量本身也有抽样误差（20 个样本的覆盖率标准误约 10%）。
- A1 的 `check_sim` 3/5 `force_cap` 来自开口夹爪盲降时的单拍冲击（与实验二里下探到棱边的机制相同），不是力符号错误；三条对照与保留的诊断脚本见 `code/ForceHarness/runs/setup_report.md`，未为此改动任何仓库代码。

## 7. 对报告的启示

1. **`±σ` 是模型，不是测量。** 它的可信度等于它所依据的噪声模型的可信度。ForceHarness 自己的两个量恰好分居两侧：质量的 σ 使用被声明的 `force_bias_std`，因此诚实（甚至略保守）；质心的 σ 只计 0.01 N·m，而仿真器按 `0.1 × force_bias_std` 抽取扭矩残差：这个常数小 3 倍（论文设置）到 10 倍（1 N 通道），净效果是整体误差为 σ 的 1.22 倍（论文噪声）与 3.17 倍（1 N 通道）。同一份工具结果里的两个数字，可信度可以差一个数量级，而慢模型从文本上分辨不出来。
2. **快→慢交接的主要失效模式是系统性偏移，不是方差大。** 0.5 N 的未声明残差让 1σ 覆盖变成 0%，但在 0.45 kg 档上判决仍然全对。只报告成功率的评测会把这种失效完全隐藏；它在窄档位（OOD 0.15 kg）才变成 20% 的错误率。这与 DASH 的“紧耦合”直接相关：慢侧的每一个决策都建立在快侧自报的不确定度上，而自报的不确定度只在“标定假设成立”的范围内有效。
3. **规模瓶颈有一条可写的判据：半档距 / σ。** 7σ → 实测 0%；2.3σ → 0–10%；0.74σ → 20%（σ 预测 22%）。质心的真实误差是 σ 的 1.2–3.2 倍，按报告 σ 估出来的余量会同样倍数地偏乐观；再加上杆倾斜把物理杠杆从 4.0 缩到 3.4 cm，2.5 cm 的 OOD 档已经贴着边界。
4. **σ 派生的规则会继承并放大 σ 的错误。** `no_load`（2σ）、`within_noise`（2σ）、以及任何“r > 2σ 判 offset”的规则，都假设 σ 是无偏的；在 1 N 通道下 `no_load` 对真实轻方块误报 45%，`within_noise` 对平衡杆漏报 75%。对慢模型来说，应该把“偏置上界”与“随机噪声”分开报告（例如 `±σ_random` 与 `bias_bound`），而不是合成一个 1σ。
5. **快环的贡献要分开计。** 同一估计器下，完整力控、仅接触保护、无力控原语三者的辨识误差无差别；差别在接近段的峰值接触力（13.6/20.4 N vs 1.9 N）。所以“把接触交给快环”买到的是瞬态安全与（在噪声下）重试机会，而不是更好的物理量。这与 DASH 的对照一致：快环的价值可以被隔离到“接触瞬态”这一项上，而不必假设它改善了慢侧的物理理解。

## 8. 重跑

脚本的规范位置是本包的 `code/`；官方克隆里的 `runs/dev/` 只是本次运行的暂存拷贝，会随 `code/` 一起清理。全部命令约 20 分钟 CPU、零 API。以下两种方式跑的是同一批脚本（包内 `code/` 与克隆内 `runs/dev/` 内容一致）。

**A. 从证据包运行（推荐；不必把脚本拷进克隆）**

```bash
FR=/path/to/ForceHarness    # 官方仓 commit 733407c1，.venv 已装 dev+sim
PKG=/path/to/paper-scout/workspace/runs/2026-11-19-dash-fast-slow/forceharness-2611.21099
mkdir -p /tmp/fh && cd "$PKG"

# 实验一：3 力通道 × 4 任务 × 20 开发种子（240 episodes，约 13 min）
"$FR/.venv/bin/python" code/estimation_probe.py --repo "$FR" --seeds 9000-9019 --out /tmp/fh/e1.jsonl --out-dir /tmp/fh/runs
# 实验二：3 控制层 × 2 任务 × 20 种子（120 episodes，约 4 min）
"$FR/.venv/bin/python" code/control_ablation.py --repo "$FR" --seeds 9000-9019 --out /tmp/fh/e2.jsonl --out-dir /tmp/fh/runs
# 汇总（JSON + 逐 episode CSV）
"$FR/.venv/bin/python" code/analyze.py --e1 /tmp/fh/e1.jsonl --e2 /tmp/fh/e2.jsonl --out-dir /tmp/fh/analysis
# 四张图（含报告主图 force-sigma-reliability.png）
"$FR/.venv/bin/python" code/figures.py --e1 /tmp/fh/e1.jsonl --e2 /tmp/fh/e2.jsonl --out /tmp/fh/figs
```

只重画报告主图，不需要克隆、不需要 JSONL——只读本包内的逐 episode 表：

```bash
cd "$PKG"
python3 code/figures.py --episodes code/results/episodes_e1.csv --out /tmp/fh/figs
```

该命令逐字节复现包内 `assets/force-sigma-reliability.png`（本次已验证）。图脚本自带三项版面自检并在不通过时以非零码退出：文本两两不重叠、没有数据点被文字压住、导出 PNG 四边留白 ≥ 2 px。

**B. 在官方克隆内运行（若脚本已拷到 `ForceHarness/runs/dev/`）**

```bash
cd /path/to/ForceHarness
.venv/bin/python runs/dev/estimation_probe.py --seeds 9000-9019 --out runs/dev/e1_full.jsonl
.venv/bin/python runs/dev/control_ablation.py --seeds 9000-9019 --out runs/dev/e2_full.jsonl
.venv/bin/python runs/dev/analyze.py --e1 runs/dev/e1_full.jsonl --e2 runs/dev/e2_full.jsonl --out-dir runs/dev/analysis
.venv/bin/python runs/dev/figures.py --e1 runs/dev/e1_full.jsonl --e2 runs/dev/e2_full.jsonl --out runs/dev/figs
```

与 A 相同，只是省掉 `--repo`（脚本向上找到 `pyproject.toml`）。这条路径依赖克隆内的临时副本，`code/` 清理后即失效；包内 `code/` 是规范副本。

`estimation_probe.py` 的 `--arms` 只接受 `paper bias1n undeclared`；`control_ablation.py` 只接受 `force guard pose`。两个探针都会把每个 episode 的 transcript 写到 `--out-dir/--run-name`（可删）。

## 9. 文件

```
forceharness-2611.21099/
├── README.md                       # 本文件
└── code/
    ├── estimation_probe.py         # 实验一：官方 expert × 3 力通道 × 4 任务 × 20 seeds
    ├── control_ablation.py         # 实验二：force / guard / pose 三臂
    ├── analyze.py                  # 汇总：误差、σ、覆盖率、判决、机制分解 → JSON/CSV
    ├── figures.py                  # 四张图；--episodes 只用包内 CSV 画报告主图
    └── results/
        ├── summary_e1.json         # 12 个 (任务, 臂) 单元的完整统计
        ├── summary_e2.json
        ├── episodes_e1.csv         # 240 行逐 episode（真值、σ、flag、抓取结果、方向误差、倾斜）
        └── episodes_e2.csv         # 120 行
```

图（在 `../../assets/`，均 < 200 KB）：

- `force-sigma-reliability.png`（报告主图）：左质量、右质心；横轴报告 σ、纵轴真实误差 RMS，虚线 y = x；实心点 = 名义档、空心环 = 官方 OOD 档；单位分别为 g 与 cm。生成时自带版面自检（文本不重叠、无点被文字压住、导出四边留白 ≥ 2 px），可用 `figures.py --episodes code/results/episodes_e1.csv` 单独复现。
- `force-estimation-calibration.png`（保留的细节图）：a/b 报告 σ 与真实 RMS 误差；c 1σ/2σ 覆盖 vs 名义 68/95%；d 判决正确率 vs σ 计的判决余量。
- `force-control-ablation.png`：a 三控制层的辨识误差（以 σ 计）；b 接近段峰值接触力（柱 = 各 episode 峰值的均值，横线 = 最差 episode，虚线 = 原语自身的 25 N 上限）。
- `force-com-tilt.png`：`com_sort` 报告杠杆 vs 真实杠杆，颜色为杆倾斜角；对角线之下即 σ 未建模的物理衰减。

数据规模：`summary_e1.json` 16.4 KB，`episodes_e1.csv` 52.0 KB，`episodes_e2.csv` 21.5 KB，`force-sigma-reliability.png` 117 KB；原始 JSONL（3.6 MB）与逐 episode 渲染文本留在本次运行的 `code/ForceHarness/runs/dev/`（临时目录，会被清理），可由第 8 节命令重建。
