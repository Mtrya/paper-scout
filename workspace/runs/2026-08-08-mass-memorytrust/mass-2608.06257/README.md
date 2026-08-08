# 线程:MASS(2608.06257)typed-carrier 复现与权威状态失真探针

论文:MASS: Multiplayer World Models with Authoritative Shared State。精读文本:`papers/world-models/mass-2608.06257.md`。

## 做了什么

1. **核心消融独立复现**:48×48 双人 Snake,按论文 B.2 的 canonical typed codec(447 token 前缀、schema 生成 decoder mask、转移规则不进 mask)训练 5.61M decoder-only Transformer(20k 步,batch 8,AdamW 2e-4/wd 1e-4,bf16);对照 2.85M independent-head CNN 稠密基线,同数据同预算。指标:semantic/active/position/count/full_exact/contradiction × H∈{1,8,16,32,64,128}。
2. **512-episode 诊断**:数据量 ×4,区分"数据不足"与"结构性难点"。
3. **单步误差解剖**:512 次从真值状态的一步预测,逐字段定位残差错误。
4. **探针 1 漂移定位**:自回归 rollout 逐 tick 对比真值,首次分歧分布 + 事件/平滑 tick 条件错误率 + 分歧后世界合法性检验。
5. **探针 2 确定性隔离**:ε=0 策略 + hash-of-state 定点 respawn,动力学完全确定时是否仍分歧。
6. **探针 3 吸引子分析**:no-op + 空 spawn 流 H=1024/4096,唯一状态数与循环周期。

## 核心数字

- typed 结构性质完整复现:全 horizon contradiction = 0;count 除 H=128(81.2)外全 100。
- typed 语义保持弱于论文:position H=1 87.5(论文 99.1)、H=32 6.2(90.2)、H=128 3.1(72.3);full_exact 恒 0(论文 H=1 41.7)。
- dense 是更强基线:H=1 position 100(论文 2.7)、contradiction 0(论文 100);递归下延迟崩溃,H=16 position 6.2、H=32 归零、contradiction 100——论文"稠密载体递归下保不住实体身份"的机制主张以更强形式成立。
- 512ep 诊断:full_exact H=1 0→50,food 错误 98.0→26.8;但 head/body 错误不随数据量改善(head 14.3→13.2,body 25.5→26.4)——瓶颈特异性集中在蛇移动/生长的学习。
- 探针 1:food 分歧几乎全在 tick 1,头部首次分歧中位 tick 2;on-track 条件错误率事件 tick 50%(n=4)vs 平滑 tick 17%(n=12);分歧后 70.6% 蛇步进与"从模型自己的状态经真实引擎推一步"一致——分歧后是引擎合法的平行世界线,不是垃圾。
- 探针 2:确定性流下 16/16 episode 仍 tick 1 分歧——残差是动力学学习误差,不是不可约随机性。
- 探针 3:no-op 流 24–48 tick 进入吸引子,1024 tick 仅 24–50 个唯一状态(论文 20k ckpt 在 H=4096 是 277–306),alive=2、invalid=0——"世界死亡"复现且更早更重。H=4096 确认性复跑(`probe3_attractor_4096.json`):唯一状态冻结在 [38,24,50,49],与 1024 完全重合,吸引子不随 horizon 增长解套。

## 与论文的已知差异

数据来自自写启发式策略(论文语料未公开);只复现 N=2(论文混合 N∈{2,4,8});引擎边界规则自定义;未实现论文的 coordinate embeddings;dense 输入含动作 one-hot 广播(H=1 完美与此有关)。详见 `code/README.md`。

## 执行环境

typed 20k 在本机 CPU(启智 Sii-Proxy 全天故障);其余在 andromeda(4060 Ti)用既有 torch 环境。保留 checkpoint:`ckpt_typed.pt`(128ep)、andromeda 侧另有 `ckpt_typed512.pt`、`ckpt_dense.pt`。

## 目录

- `code/`:复现与探针全部脚本(engine、codec、双模型、训练、评测、三探针、单步解剖),README 含复跑说明。
- `code/results/`:全部指标与探针 JSON。
- 报告图在 `../assets/mass-*.png`。
