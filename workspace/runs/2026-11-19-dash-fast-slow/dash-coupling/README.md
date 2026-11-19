# DASH 时间接口核验：Eq.(11)–(12) 的语义、过移与锚点

DASH 深度线的独立核验。问题：DASH 的 Eq.(12) 用累计量 `n_k = ⌊Σ_{j≤k}ρ_j⌋ − ⌊Σ_{j<k}ρ_j⌋`、`f_k = {Σ_{j≤k}ρ_j}` 对齐重叠动作块；论文自己的例子是 `ρ = 0.6` 两块，第一块 `n=0, f=0.6`，第二块 `n=1, f=0.2`。若每块都在自己的推理开始/到达时刻锚定，第二块到达时的年龄是 0.6 而非 1.2。这里检验：1.2 到底是什么量，哪种 buffer 语义下 Eq.(12) 精确，哪种会过移，以及它与 DuoCore-FS、VLASH、RTC 各自解决的那一笔 staleness 是什么关系。

结论先行：

1. **Eq.(12) 没有算错，但它算的是「写位置」，不是「位移」。** `S_k = Σ_{j≤k} ρ_j` 等于 `(a_k − a_0)/T_exec`，即第 k 块到达时刻在输出时间轴上的绝对坐标。读者自己的位置是 `P(t) = (t − a_0)/T_exec`，读取下标是 `P − S_k = (t − a_k)/T_exec`。论文的第二块到达时 `S = 1.2`、读者在 `0.6`、`1.2 − 0.6 = 0.6`，恰好是它的年龄。**1.2 与 0.6 不是同一类量，二者并不矛盾。**
2. **只有累计相位这一种写法会过移**：把 `n_k + f_k` 当作「在第 k 块到达时对已有 buffer 的额外位移」，叠加在本就每 tick 前进一格的读指针上。此时第 2 块被多推 `S_2 = 1.2` tick（60 ms），第 k 块被多推 `S_k`，只有第 0 块仍按真实年龄老化，整条流在 **0.8 s** 后饥饿（H=16, 20 Hz）。这不是 Eq.(12) 的说法，但正文「a chunk received ρ ticks ago must be shifted by ρ」的年龄措辞为它留了门。
3. **精确性只覆盖「到达节拍」，不覆盖「观测锚点」。** ρ 由到达间隔测得，因此 Eq.(12) 对齐的是到达时钟。块内容是观测时刻 `s_k` 的函数，二者相差动作头自身延迟 `L_k`：常数 `L` 变成整 tick 滞后，抖动 `L` 变成时间弯折。按 DASH 自己的 Table II，动作头 35 ms + 预处理 15 ms = 50 ms，恰为 20 Hz 的一个 tick，**按到达锚定时残差正好是 1.00 tick（50 ms）**；整数前缀丢弃（ceil）把它压到 0.3–0.5 tick，压不到 0。这一笔是 δ_exec，不是 Eq.(12) 解决的。
4. **与直接前作的分账**：DuoCore-FS 处理慢语义 staleness（bridge buffer + 跨时标共训），VLASH 与 RTC 处理动作/状态在整数控制步上的错位（未来状态、冻结前缀 + inpaint），据论文文本，三者都按整数节拍对齐，未描述小数、逐次到达的 ensemble 相位。Eq.(12) 显式处理「到达间隔为非整数控制步」时的重叠块相位，这在 DASH 里来自 III-B 的 un-gated action server。第三笔是记账，不是学习。

## 1. 材料与口径

- DASH：`papers/vla/dash-2611.11854.md`，重点 §III-B（un-gated action server、前缀丢弃）、§III-E（Eq. 11–12、Fig. 5）、Table I–III。
- DuoCore-FS：`papers/vla/duocore-fs-2512.20188.md`，重点 §3.1（bridge buffer、快慢频率）、§3.3（跨时标采样 `Δ ~ U[0, Δ_max]`）、§3.4（异步推理：latest-value）。
- VLASH：`papers/vla/vlash-2512.01031.md` §IV-A/B；官方代码 `code/vlash`，commit `22cbabfee0f57874987c75a35a7dac129e695db0`，部署路径 `vlash/run.py` 的 `VLASHAsyncManager`，训练偏移 `vlash/datasets/vlash_dataset.py`。
- RTC：`papers/vla/rtc-2506.07339.md` §2–3（`d = ⌊δ/Δt⌋`，冻结前 d 个动作 + soft mask inpaint）。
- 模拟口径：`T_exec = 50 ms`（20 Hz，DASH Table I 的控制率）、`H = 16`（论文未给 horizon；8/16/32 做过敏感性）、指数 ensemble 权重 `exp(−0.1·age)`（对齐 ACT [14] 的默认形式），并以均匀权重复核。

真实动作序列：LeRobot `lerobot/aloha_sim_transfer_cube_human` episode 0，50 Hz、400 帧、14 维关节位置指令，两次抓放（左夹爪在 4.66 s 合拢、5.94 s 张开）。这是与 DASH 模拟平台相同的 ALOHA 构型、真实遥操作示范，只取了 229 KB 的 episode 0 行数据（`code/data/aloha_transfer_rows.json`），未下载完整数据集。解析轨迹为分段最小 jerk 的二维 pick-and-place（含停顿、慢放、快速转移），用于把机制与数据噪声分开。

## 2. 时间模型

连续时间 `t`；执行 tick `τ_i = i·T_exec`。

- 第 k 块：观测/推理开始时刻 `s_k`（内容锚点），到达时刻 `a_k = s_k + L_k`（可被消费的时刻），内容 `C_k[j] = g(s_k + j·T_exec)`，`j = 0..H−1`。
- `g` 是参考轨迹；`C_k` 是它的精确拷贝。**预测误差被人为置零**，所以执行误差全部来自时间记账。
- 一个对齐方案归结为一件事：在时刻 t，读第 k 块的第几个下标 `A_k(t)`。执行指令 `u(t) = Σ_k w_k·interp(C_k, A_k(t)) / Σ w_k`，取 `0 ≤ A_k ≤ H−1` 的块。
- ensemble 的有效时间与超前量：`t_eff(t) = Σ w_k (a_k + A_k(t)·T_exec) / Σ w_k`，`lead = (t_eff − t)/T_exec`。对齐正确时所有块给出一致的绝对时间，`lead ≡ 0`。

因为 `C_k` 精确，任何 `lead ≠ 0` 都直接是时间误差；`time_err_ticks = ‖u − g‖ / ‖Δg‖`（向量范数之比），单位是 tick，1 tick = 50 ms。

## 3. Eq.(12) 的三种语义

记 `S_k = Σ_{j≤k} ρ_j`，`ρ_j = (a_j − a_{j−1})/T_exec`，`S_0 = 0`。以下例子按论文的叫法从第 1 块计；代码里的 `k` 从 0 计，`S_0 = 0`。

**R1 绝对放置（精确）**：buffer 按输出时间轴索引，第 k 块写在 `S_k`，读者在 `P(t) = (t − a_0)/T_exec`，`A_k = P − S_k`。因为 `S_k = (a_k − a_0)/T_exec`，所以 `A_k ≡ (t − a_k)/T_exec`。`n_k` 是「写指针这次跨过了几个整槽」，`f_k` 是「写指针在槽内的相位」。这就是 ACT `all_time_actions[t]` 的小数推广：ACT 把块写在整数槽 `t`，DASH 把它写在实数槽 `S_k`。

**R2 相对 buffer + 相位（精确）**：buffer 的 slot 0 是下一 tick，读指针每 tick 自动前进一格。`n_k` 只是记录「时钟自己走过了几个整格」，`f_k` 是写入相位；公式是**一致性关系**，实现只用 `f_k` 写块、不用 `n_k` 再推一次。与 R1 同构，数值相同。

**R3 把 `n_k + f_k` 当额外位移（过移）**：读指针已每 tick 前进一格，再在每次到达时把整条 buffer 推开 `n_k + f_k`。第 k 块被多推 `S_k`，论文的第二块多推 1.2 tick = 60 ms，且随 k 线性增长。

**对偶解释**：把输出时间轴的原点放在第一块到达处，`S_k` 就是第 k 块写位置在该轴上的坐标；`n_k` 是这次跨过的整槽数，`f_k` 是槽内相位。读指针独立按 tick 推进，因此不需要把累计量交给任何一块。等价地，只要每块到达时记下 `a_k` 就能直接算年龄；累计只是「拿不到绝对时间戳、只有到达间隔」时的替代品。R1 与 R2 都成立，区别只在写指针的整数推进由谁完成（buffer 显式移，还是时钟隐式走）。

### 裁决

- `Σ(n_k + f_k) = S_k` 恒等（数值验证：`max|Σ(n+f) − S| = 0`；`max|A_placement − A_age| = 2.8×10⁻¹⁴` tick）。
- 论文例子里的 0.6、1.2 只能读作**位置**：按论文叫法，第一块写在 0.6，第二块写在 1.2；两块到达时自己的年龄都是 0，第二块到达时第一块的年龄是 0.6。「先不移动、插值 0.6；再移动一格、插值 0.2」描述的是写指针的状态轨迹，不是两次位移。
- 正文「a chunk received ρ ticks ago must be shifted by ρ」是**年龄**语言，与 Eq.(12) 的**位置**语言在第 0 块之外不重合。两者各自都能实现精确对齐（R1/R2），但把两句缝在一起就得到 R3。
- **锚点的另一种取法（一行改动）**：若 `T_k` 取 `s_k − s_{k−1}`（推理开始间隔）而非 `a_k − a_{k−1}`（到达间隔），`S_k` 就直接是观测锚点：数值上该恒等式成立到 `2.1×10⁻¹⁴` tick（`identity.observation_clock`），锚点实验的误差从 1.00 tick 降到 0.01–0.03 tick。论文写的是 arrival interval，所以默认读法对齐到达；这两种取法只差一个时间戳，却决定 50 ms 残差是否存在。这是本核验里最可操作的一条。
- R3 不是可以长期运行的系统：第 k 块被多推 `S_k`，只有第 0 块（`S_0 = 0`）仍按真实年龄老化，它在 `t − a_0 > (H−1)T` 时越界，其余块更早。模拟中它在 **0.8 s** 后进入 88–90% 的饥饿（`first_starve_s = 0.8`，与 ρ 无关：0.8 s = 16 tick = H）。DASH 在真机上以 23/24 的成功率运行，若实现真是 R3，0.8 s 后就会整段饥饿，与此不符；R1 或 R2 是自洽的解释。

Fig. 1（`assets/dash-eq12-bookkeeping.png`）左图为 `S_k` 与「第 k 块自己的年龄」，右图为若把 `S_k` 当位移的过移量（0/30/60/…/270 ms）。

## 4. 各臂定义

`A_k(t)` 即执行时刻 t 对第 k 块的读取下标；`age_k(t) = (t − a_k)/T_exec`。

| 臂 | 定义 | 语义 |
|---|---|---|
| `exact_anchor` | `(t − s_k)/T` | 观测锚定的真值 |
| `exact_arrival` | `(t − a_k)/T` | Eq.(12) 的 R1/R2 等价形式（精确） |
| `shiftpos` | `age_k + S_k` | R3：把累计位置当位移 |
| `ceil_clock` | `age_k + Σ_{m>k}(⌈ρ_m⌉ − ρ_m)` | 逐块整数 ceil 时钟（论文点名的 rounding 失败类） |
| `floor_clock` | 同上，`⌊·⌋` | 整数 floor 时钟 |
| `fixed_clock` | `age_k + Σ_{m>k}(m̄ − ρ_m)`，`m̄ = round(mean ρ)` | 固定整数偏移（ACT 式 fixed offset） |
| `prefix_ceil` | `age_k + ⌈L_k/T⌉` | 到达锚定 + 整数前缀丢弃（DASH §III-B 的措辞） |

Eq.(12) 本身不单列一臂：用 `n_k + f_k` 逐次累加出的写位置表与 `exact_arrival` 的读下标逐 tick 相同，恒等式在 `identity` 中断言（最大差 `2.8×10⁻¹⁴` tick）；`exact_arrival` 就是它的等价实现。

`ceil/floor/fixed` 用「逐块」模型：每次到达对整个 buffer 施加同一整数位移，因此第 k 块累计的是它到达之后的各次位移，误差随块年龄增长，但被 horizon 截断。若改用单一全局整数时钟（不重锚），误差无界：`ρ=0.6` 时 ceil 的漂移是每块 +0.4 tick，即每 tick 快 0.67 tick，约 1.2 s 后同样饥饿。逐块模型更接近「每到达一次就 shift 一次」的实现，本文报它。

## 5. 关键数值

解析轨迹（2 维），指数权重，H=16，误差单位 tick；括号内为 ALOHA 14 维仿真遥操作序列。

| 臂 | ρ=0.6 | 抖动 ρ∈[0.5,1.6] | 跨整数 0.9/1.1 + 漂移 |
|---|---|---|---|
| `exact_arrival` | 0.02 (0.04) | 0.03 (0.05) | 0.03 (0.05) |
| `ceil_clock` | 1.96 (1.89) | 1.63 (1.54) | 1.48 (1.46) |
| `floor_clock` | 11.63 (9.64)，速度 0.02–0.03（冻结） | 4.03 (3.44) | 3.96 (3.45) |
| `fixed_clock` | 1.96 (1.89)，与 ceil 同值 | 0.37 (0.28) | 0.24 (0.36) |
| `shiftpos` | 11.58 (9.66)，0.8 s 后饥饿 | 12.53 (10.16) | 12.63 (10.19) |

Fig. 2（`assets/dash-trajectory-arms.png`）给出恒定与跨整数两种节拍下的执行轨迹：`exact_arrival` 与参考重合，`shiftpos` 在 0.8 s 处截断（之后是饥饿保持），`ceil_clock` 表现为带超前的平滑。Fig. 3（`assets/dash-arm-error.png`）是上表的对数坐标版本，1 tick 线标出 50 ms。

其他量：

- `ceil_clock` 的稳态超前 `lead`：约 +2.1（ρ=0.6）、+1.6（抖动）、+1.5（跨整数）tick；jerk 比在解析轨迹上为 1.3× / 3.3× / 8.9×（ρ=0.6 / 抖动 / 跨整数），ALOHA 上为 0.6× / 1.3× / 2.4×。误差随 horizon 增长：H=8/16/32 时为 0.77/1.48/2.12 tick。
- `fixed_clock` 在均值 ρ 为整数时近乎精确（0.24–0.37 tick），在 ρ=0.6 时与 ceil 同值。**「固定偏移」本身不是病；病在假定的偏移与实际节拍不符。** DASH 的 sync 基线在仿真里的块率是 4.1–9.7 Hz（随 backbone 变化），与 20 Hz 的 tick 不是一回事，因此 Table III 的 16/24 vs 23/24 同时改变了控制率、运行时和 aggregator，无法单独归因于 Eq.(12)。
- ALOHA 抓取时刻：`ceil_clock` 在 ρ=0.6 时提前 100 ms，`floor_clock` 在 ρ≈1 时落后 150–200 ms；`exact_arrival` 为 0 ms（节拍精确，锚点问题另计）。

### 锚点实验（决定性的那一笔）

固定观测节拍 20 Hz（`s_k = k·T`），动作头延迟 `L_k = 1 tick + 抖动`，`a_k = s_k + L_k`。三臂都对齐到达，只有锚点不同：

| 臂 | 高斯抖动 σ_L=0.25 tick | 每 25 块一次 +0.9 tick 尖峰 | ALOHA 抓取偏移 |
|---|---|---|---|
| `exact_anchor` | 0.01–0.03 | 0.01–0.03 | 0 ms |
| `exact_arrival` | **1.00** | **1.02** | **+50 ms** |
| `prefix_ceil` | 0.50–0.52 | 0.32–0.41 | 0 ms（被 tick 量化掩盖） |

`exact_arrival` 的 1.00 tick 不是数值巧合：它就是把块内容整体滞后 `L = 1 tick` 的解析结果。也就是说，**Eq.(12) 把到达节拍对得完全精确，却把整条时间轴锚在比观测晚一个动作头延迟的位置**。DASH 的 50 ms 动作路径在 20 Hz 下正好是一个 tick；若不丢前缀，残差 50 ms；用 ceil 丢前缀，残差 0.3–0.5 tick（取决于 `L/T` 落在整数哪一侧）。这与论文「removes most of the inference-to-execution delay」并不矛盾——δ_exec 是被 50 ms 的快路径压小的，不是被 Eq.(12) 消掉的——但 Eq.(12) 的「exact」只在到达时钟内成立。

Fig. 4（`assets/dash-anchor-latency.png`）左图为 ALOHA 左夹爪在 4.3–5.2 s 的三条曲线，到达锚定整体右移一格；右图为三臂误差。

## 6. 与 DuoCore-FS / VLASH / RTC / ACT 的准确关系

三笔 staleness 必须分开记：

| 机制 | 解决哪一笔 | 时间基 | 小数/整数 | 有无 ensemble |
|---|---|---|---|---|
| DASH §III-C 新鲜 ResNet + §III-D 去同步训练 | 慢语义 `δ_ctx` | 图像帧，`Δ ~ U{0..Δ_max}` | 整数帧 | — |
| DASH §III-B 前缀丢弃 + 50 ms 快路径 | 动作头执行 `δ_exec` | 到达时刻 | 整数 tick（ceil） | — |
| DASH §III-E Eq.(12) | 重叠块 ensemble 对齐 | **到达间隔** | **小数 tick** | 有 |
| DuoCore-FS §3.1/3.3/3.4 | 慢语义 `δ_ctx` | 帧，`Δ ~ U[0, Δ_max]` | 整数帧 | 无（论文未描述） |
| VLASH §IV-A/B + `run.py` | 动作/状态 `δ_exec` | 控制步，`δ ∈ {0..Δ_max}` | 整数步 | 无（硬切换） |
| RTC §3 | 动作头延迟，块边界 | 控制步，`d = ⌊δ/Δt⌋` | 整数步 | 无（inpaint 代替） |
| ACT [14] | ensemble 平滑 | 绝对 tick | 固定整数偏移 | 有 |

因此：

- **DuoCore-FS 与 DASH 在第一条腿上同构**：都是 latest-value 缓存 + 在训练里注入跨时标偏移。差别在 DASH 冻结 VLM、外挂新鲜 ResNet，DuoCore 端到端共训并保留 action tokenizer；DuoCore 用 `Δ_max ≈ T_slow + ⌊T_slow/T_fast⌋·T_fast`（约两个慢周期）从两个频率推出上界，DASH 用部署实测的最大观测差除以训练数据周期；两者都是帧级均匀采样，粒度不同、来源不同。二者都没有小数 aggregator：DuoCore 的快环按 25–30 Hz 运行，论文未讨论其推理时长浮动。
- **VLASH 与 DASH 在第二条腿上选了相反的通道**。VLASH 把状态滚到执行起点、保留触发时刻的图像（`run.py` 中 `observation["observation.state"] = current_chunk[n_action_steps-1]`，图像仍是触发帧），即接受视觉 staleness、消掉状态/动作时间差；DASH 反过来，用 ResNet 消掉视觉 staleness，却把块的时间锚点留在到达（这是就 Eq.12 以到达间隔计 ρ 而言；§III-B 的整数前缀丢弃会补偿其中整数 tick 部分，实际残差取决于取整方向与延迟分布，见 §5 锚点实验）。VLASH 的偏移全是整数控制步，训练时 `random.randint(0, max_delay_steps)` 抽整数；`run.py` 是固定 `n_action_steps` 的预取 + 硬切换，没有 ensemble；推理超过 overlap 窗口时不会崩，但切换点的 `.cpu()` 会阻塞控制环（退化成停顿），`overlap_steps=0` 时根本不预取、每个块结束同步推理一次。因此 VLASH 根本没有提出 Eq.(12) 要解的问题。
- **RTC 用 inpainting 替代 ensemble**，且明确声明不考虑 sub-timestep（footnote 1），`d` 是整数。它把「新块从到达 tick 起可用」这件事作为构造前提，而不是事后对齐。
- **ACT 是 Eq.(12) 的整数前身**：块写在绝对整数槽 `t`，读 `all_time_actions[:, t]`，指数加权。它没有「逐次到达、小数相位」的概念。

一句话：**Eq.(12) 的贡献是「把 ACT 的绝对整数槽推广到实数槽」这一行记账；它成立的前提是动作服务器 un-gated，这是 DASH III-B 的运行时选择；DuoCore/VLASH/RTC 的部署循环按定频或整数控制步设计，分数相位在它们那里不出现。**

## 7. 结论的边界

- 本核验只检验**公式语义**，不是真机失败报告。模拟里预测误差为零、控制是开环指令比对，没有接触动力学、没有闭环反馈放大。`ceil_clock` 的 1.5–2 tick 误差与 `floor_clock` 的冻结是记账错误的上界展示，不等于 DASH 真机会出现同等幅度的失败。
- `H = 16` 是假设；论文未给 horizon。误差随 H 增长（0.77→2.12 tick，H=8→32），趋势比绝对值可靠。
- 没有 DASH 官方代码。R1/R2 是从公式与例子反推的、自洽的读法，R3 会被 0.8 s 饥饿排除；论文未说明 buffer 是绝对索引还是相对索引，也未说明前缀丢弃是 `floor` 还是 `ceil`。这两个未定点不影响「1.2 是位置」的结论，但影响残差是 0、0.3 还是 1.0 tick。
- 动作序列来自 ALOHA 仿真遥操作，不是 DASH 的人形真机；它检验的是「抓取时刻随锚点平移」这一条，不是 DASH 的硬件时延分布。

## 8. 复现

```bash
cd runs/2026-11-19-dash-fast-slow/dash-coupling/code
# 事件模拟 -> results/results.json（含 identity 断言、bookkeeping、closed loop、anchor、H 敏感性）
python sim_timing.py --out results
# 图 -> ../../assets/dash-*.png
python make_figures.py
```

只依赖 numpy + matplotlib，可使用已安装这两个依赖的 Python 环境。数据 `data/aloha_transfer_rows.json` 由 HF datasets-server 的 rows 接口取 `lerobot/aloha_sim_transfer_cube_human` episode 0 的 400 行（`https://datasets-server.huggingface.co/rows?dataset=lerobot%2Faloha_sim_transfer_cube_human&config=default&split=train&offset=0&length=400`），字段为 `episode_index/frame_index/timestamp/action`（14 维），保留在包内以便离线重跑。

## 9. 对报告的启示

1. **不要把 Eq.(12) 写成「DASH 修正了异步推理的时间错位」**。它修正的是**块到达节拍**的对齐，且只在到达时钟内精确；动作头自身的 50 ms 延迟仍在，实测残差 0.3–1.0 tick。慢语义 staleness 由 §III-C/D 负责，与 Eq.(12) 无关。
2. **1.2 vs 0.6 是一个好的「概念区分」素材**：位置与年龄、写指针与读指针、增量与水平量。报告里可以用一张小图（Fig. 1）把这三个区分讲清楚，比复述公式有效。
3. **DASH 在这里的贡献是运行时选择与相位记账，不是学习成分**：un-gated action server 才产生小数相位；一旦定频，DuoCore 的 latest-value 和 VLASH 的整数偏移都与各自的整数时间基匹配。这条线索可以把三篇论文放在同一张时间基表格里，而不是并列成「都是快慢系统」。
4. **Table III 不能作为 Eq.(12) 的证据**：sync 基线同时改变了控制率（仿真同规模下 9.7 Hz vs 20 Hz）、运行时和 aggregator 假定；模拟显示 fixed offset 在偏移等于平均节拍时近乎精确，所以「fixed vs adaptive」不是自明的差异。报告应把这一点写成证据强度问题，而不是指控。

## 10. 未解决

- DASH 的 buffer 实现（绝对索引 / 相对索引、前缀丢弃的取整方向、`S_0` 的定义）无法从论文确定；三者组合会产生 0 / 0.3 / 1.0 tick 的不同残差。若能拿到官方代码，只需检查 `write_pos` 的更新式与读指针是否独立推进即可裁决。
- DuoCore-FS 的代码未公开（论文称向商业用户提供），其快系统是否真的定频、是否有隐式 ensemble 无法核实；本核验只依据论文文本。
- horizon 与真实动作头延迟分布（35 ms 是否稳定、15 ms 预处理是否与推理串行）只能从 Table II 的单点值推断，无法做分布层面的结论。
