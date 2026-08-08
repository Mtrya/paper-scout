# mass-repro — MASS (2608.06257) 核心消融的独立复现 + 扩展探针

复现论文核心主张:类型化 token 状态作为递归载体的世界模型,在自回归 rollout 下
保持实体身份与位置(论文 Table B.2/B.4:H=1 head-position 99.1%,H=128 72.3%,
contradiction 0%),而稠密网格载体第一步就丢失位置(≤2.7%)且每步结构矛盾(100%)。

## 文件

| 文件 | 作用 |
| --- | --- |
| `snake_engine.py` | 多人 Snake 引擎:48×48、N=2、64 food 槽;墙/身体碰撞致死、吃食生长(上限 40)、死亡锚点;启发式最近食策略(ε=0.1);确定性变体(hash-of-state 定点 respawn)供探针 2 |
| `codec.py` | 类型化 token codec(对齐论文 B.2/A.2):447 token 前缀 = BOS+STATE+状态段(421=4 tick 半字节+64 排序 food CELL+player_count+8×44 玩家槽)+ACTION+8 动作+EXOG+4 半字节+spawn_count+8 spawn CELL+OUTPUT;schema mask + 动态选择器(food 严格递增唯一、身体坐标唯一、规范 padding、死亡/存活一致性);转移规则不进 mask |
| `gen_data.py` | 数据生成:train 128 eps×192 转移 / val 16×160 / test 32×160 / det(确定性)16×160;按 episode 划分;存 `data/*.npz`(int16 压缩) |
| `model_typed.py` | 模型 A:decoder-only Transformer,宽 256/6 层/8 头/MLP×4/tied embedding/学习位置,**5.61M** 参数(论文 5.66M);含 KV-cache 增量解码 |
| `model_dense.py` | 模型 B:independent-head CNN(27 通道稠密网格输入,逐格 6 类 softmax + 朝向/存活全局头),**2.85M** 参数(论文 3.26M) |
| `train_typed.py` / `train_dense.py` | 训练:20k 步、batch 8、AdamW lr 2e-4/wd 1e-4、clip 1.0、单转移上下文 teacher forcing;bf16 仅 CUDA;显式 seed |
| `rollout.py` | 贪心自回归 rollout(带 mask 约束解码)+ 六项直接状态指标(semantic/active/position/count/full_exact/contradiction) |
| `eval_rollout.py` | B.4 式评测:val 16 eps,H∈{1,8,16,32,64,128},输出 `results/eval*.json` + per-tick 曲线 `.npy` |
| `probe1_drift.py` | 探针 1 漂移定位:首次分歧 tick 分布、事件/平滑 tick 错误率、分歧后"引擎合法性"(模型自己的状态经真实引擎推一步是否与模型预测一致 = 合理平行世界) |
| `probe2_determinism.py` | 探针 2 确定性隔离:det.npz(策略 ε=0 + 确定性 respawn,动力学无随机性)上重跑,分歧即动力学学习误差 |
| `probe3_attractor.py` | 探针 3 吸引子:no-op 动作流 + 空 spawn,H∈{1024,4096},唯一状态数(剔除 tick)/循环进入点/周期/存活/结构有效性(论文 C.3:20k N=2 H=4096 仅 ~277 唯一状态) |
| `run_cpu_pipeline.sh` | 全量 CPU 管线(幂等,可重入):typed 训练→typed 评测→探针 1/2→dense 训练∥探针 3(H=1024)→dense 评测→探针 3(H=4096) |
| `inspire_watch.sh` | 每 10 min 探一次启智连通,恢复时 touch `results/INSPIRE_UP` |

## 复跑

```bash
cd code/mass-repro
uv venv .venv && uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python 'numpy<2' matplotlib
.venv/bin/python gen_data.py            # ~2 min,生成 data/*.npz
./run_cpu_pipeline.sh                   # CPU 全程 ~20h(12 线程);各阶段幂等可断点续跑
```

远程 GPU(启智)复跑:`scp` 本目录到 workroot,venv 用 `--system-site-packages`
继承 NGC torch(注意 `numpy==1.26.4 + scipy==1.14.1`),同跑 `run_cpu_pipeline.sh`
(torch 自动用 CUDA,~40 min 训完)。

## 结果位置

- `results/eval.json` / `eval_dense.json` — 核心复现表(六指标 × 六 horizon)
- `results/typed_curve.npy` / `dense_curve.npy` — per-tick 指标曲线
- `results/probe1_drift.json` + `probe1_curve.png` — 漂移定位
- `results/probe2_determinism.json` — 确定性隔离
- `results/probe3_attractor.json` / `probe3_attractor_4096.json` — 吸引子
- `results/*.log` — 各阶段日志;`ckpt_typed.pt` / `ckpt_dense.pt` — 权重(每 5k 步覆盖保存)

## 与论文的已知差异(如实记录)

1. **数据分布不同**:论文语料未公开,我们的轨迹来自自写的最近食启发式策略
   (ε=0.1 随机),事件率(eat ~0.27/tick,death ~0.4%/tick)与论文语料未必一致。
2. **引擎规则细节自定义**:同时头碰头双死、身体上限 40(超过不再生长但仍吃食)、
   撞墙死亡锚点记为原头位置;论文未公开这些细节。
3. **参数量略小**:5.61M(vs 5.66M)、2.85M(vs 3.26M),词表/宽度取整差异。
4. **只用 N=2**:论文逻辑评测混合 N∈{2,4,8},我们只复现 N=2(任务指定)。
5. **dense 指标定义自洽但与论文未必逐项等价**(论文的 count/contradiction 精确
   定义未公开);typed 侧指标定义与 B.3 文字描述对齐。
6. **exogenous spawn 是"真实净增 food"**:与论文 B.2 兼容通道语义一致(记录是否
   发生 spawn 及其坐标)。

## 状态(2026-08-08 12:30 +08)

启智 Sii-Proxy 持续故障(~8.5h),远程改走 SSH 直连的 andromeda(RTX 4060 Ti
16GB,无公网,用既有 `~/Projects/chess-transformer/.venv`,torch 2.10+cu128)。
typed 20k 在本机 CPU 完成(8.5h),其余全部阶段在 andromeda GPU 完成。
注意:andromeda 上 numpy 的 `ma` 子包在 c2p venv 被裁剪,`np.unique` 不可用,
codec 的 contradiction 检查用 Python set 实现;c2p 目录后被并行任务移走,
 runner 已切到 chess-transformer venv。

## 结果一览(详见 results/*.json)

- typed(128 ep 训练,论文预算):H=1 position 87.5% / H=128 3.1%,contradiction 全 0
- dense(同预算):H=1 position 100%(我们的 dense 输入含动作广播,一步很容易),
  H=16 6.2%,H=32+ 0%,contradiction 到 H=32 达 100%——递归下崩溃,与论文机制一致
- typed 512 ep 诊断:full_exact H=1 0→50%,semantic/active 显著提升,
  head-position 未提升——位置误差不完全是数据量问题(见 onestep_anatomy)
- 探针 1:首次头部分歧中位 tick 2;on-track 单步条件错误率事件 tick 50%(2/4)
  vs 平滑 tick 17%(2/12);分歧后 70.6% 的蛇步进与引擎合法一致("合理平行世界")
- 探针 2:确定性动力学下仍于 tick 1 全态分歧——残差是动力学学习误差,非随机性
- 探针 3:no-op H=1024 仅访问 24-50 个唯一状态,~50 tick 内进入循环吸引子,
  alive=2,invalid=0——论文 C.3 的"世界死亡"现象复现(我们崩塌得更早)
