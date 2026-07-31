# Thread: 实时操作策略 — TurboVLA (2607.27205) vs πR² (2607.26055)

## 研究问题

大 backbone 的 action-chunking flow policy 又慢又开环。两篇同期论文给出互补的解法:TurboVLA 走"换掉 LLM 中心架构"的路线(V+L→A 直接映射),πR² 走"保留大 backbone、改造推理时序"的路线(proprioception-reactive diffusion forcing + latency-adaptive schedule)。代码核验:它们的数字到底声称了什么?

## 尝试了什么

1. 端到端追踪 TurboVLA 官方代码(github.com/H-EmbodVis/TurboVLA,浅克隆于 `code/turbovla/`):架构、推理图、训练 recipe、评测 harness,并对照论文全文。
2. 从 HF checkpoint 文件字节数独立验证部署参数量 —— `code/verify_turbovla_size.py`(可重跑,只查 API 元数据)。
3. 精读 πR² 全文(arXiv HTML v1 含附录),提取 staircase schedule 形式化、实验表与消融。
4. **真实验(猜想检验,负结果)**:`code/pir2_modality_toy.py` —— πR² 的 fast channel 只有 proprio,猜想其 staircase 调度优势是模态特异的:对 proprio 可见扰动(手臂冲量)显著,对纯视觉扰动(目标跳变)消失。1D 点质量玩具环境 + diffusion-forcing flow policy(per-slot τ、delay embedding),sync vs staircase × 延迟 {1,3,5} × 扰动模态,8 seeds。

## 保留的证据

- `code/verify_turbovla_size.py` — 2026-07-31 重跑结果:4 个 LIBERO fp32 checkpoint 各 ~426.5 MB(≈106.6M 参数,与代码计算的 DINOv3-B 86.6M + 21M heads 精确吻合);RoboTwin bf16 safetensors 868.2 MB(≈434M = 21M heads + DINOv3-L 304M + BERT-base 110M)。
- `code/pir2_modality_toy.py` — 2026-07-31 扰动模态实验,最终版(last-minute finalization staircase,8 seeds)。

## 实验:πR² staircase 的扰动模态盲区(猜想被证伪)

**猜想**:πR² 的 fast channel 只有 proprio,其 staircase 调度优势应当模态特异——proprio 可见扰动(手臂冲量)上显著,纯视觉扰动(目标跳变)上消失。

**结果(8 seeds,mean±std)**:

| 延迟 | sync arm | staircase arm | sync target | staircase target |
|---|---|---|---|---|
| 1 tick | 0.551±.200 | 0.635±.193 | 0.641±.263 | 0.700±.244 |
| 3 tick | 0.630±.190 | 0.672±.183 | 0.713±.215 | 0.754±.216 |
| 5 tick | 0.722±.198 | 0.711±.192 | 0.784±.255 | 0.808±.173 |

- **猜想证伪**:两种模式下 err@arm ≈ err@target(所有延迟档),没有任何模态不对称。玩具误差由观测-规划回路的固有延迟主导,与扰动来自哪个通道无关。
- **但有一个与论文叙事一致的方向性结果**:延迟低时(d=1)staircase 付质量税(err_all 0.645 vs sync 0.544,+19%)——sync 此时每 tick 重规划,staircase 的 1-NFE 调度没有反应优势可挖;延迟高时(d=5)staircase 反超(err_all 0.687 vs 0.734)——last-minute finalization 在观测变陈旧时开始兑现价值。这正是 πR² 论文"随延迟预算增长 πR² 持平、RTC 崩塌"的玩具级回响,但幅度温和得多。
- **实现脆弱性(意外的主要发现)**:在得到上表之前,连续 3 种看似自然的 staircase 实现全部灾难性失败(err 为 sync 的 2–6 倍):
  1. *τ=1 OOD*:rolling buffer 前槽钉死在 τ=1,而训练 τ~U[0,1) 从不含 1,τ embedding 外推出 ±4 的野动作;
  2. *去噪网格错配*:把每 tick 步长设为 1/(H-1),高 τ 区流场增益 1/(1-τ) 达 15,欧拉迭代进入临界不稳定,τ≈0.9 槽位内容腐化为 ±3 的垃圾再被发射;
  3. *干净槽冻结*:全 buffer 每 tick 步进使动作在发射前 12 tick 就定型,等效 12-tick 开环执行,误差发散。
  能工作的设计必须同时满足:与 sync 相同的 K 层去噪网格、只有前 K 槽每 tick 步进(动作的 K 步去噪恰好落在发射前最后 K tick)、前槽在当 tick 观测下完成最后一步后立即发射(last-minute finalization)。πR² 论文对这些实现选择零消融——"1 个 Euler 子步"的优雅叙事背后是一片容易踩空的实现空间,这对复现者是真实风险,也可能解释为什么 naive-async 基线在论文真机表里表现不稳定。
- **玩具局限**:1D、单一 MLP、专家是无记忆 PD——隔离的是调度问题而非感知问题;真机视觉场景的结论可能不同。


## 关键发现(报告用)

### TurboVLA

- **"0.2B 参数"的口径**:LIBERO 部署图只有 ~107M;BERT 文本编码器被离线缓存(scripts/libero/build_text_cache.py),推理时一次 index_select 查表,从不执行。0.2B 只在把不运行的 BERT 算进去时成立。
- **"32 Hz"的口径**:这是 chunk 预测率(1/31.2ms);每个 12 步 chunk 开环执行,真实 replan 率 ~2.7 次/秒。仓库里没有任何延迟/显存 benchmark 脚本,31.2ms/0.9GB 无法从发布代码复现;评测是朴素 bf16 eager。
- **架构真相**:核心"双向视觉-语言交互"是 Grounding DINO 的 feature enhancer 整体移植(同一个 BiAttentionBlock,单个 QKᵀ 双向 softmax),且默认必须从 GroundingDINO 预训练权重初始化;动作解码是 tanh 压缩的 L1 回归(没有 CVAE 的 ACT),全程无 flow/diffusion。97.7% LIBERO 均值里有多少来自 grounding 预训练初始化,消融没有分离。
- **两个延迟数字口径不对称**:LIBERO 31.2ms 不含 BERT(缓存),RoboTwin 43.4ms 含 live frozen BERT;RoboTwin 评测还叠加了 ≤50 个重叠 chunk 的 adaptive temporal ensembling,60.2% 不是单 pass 策略的原始分数。论文称混合套件联合训练,HF 却发布 4 个 per-suite checkpoint。
- **值得肯定**:把"感知-语言融合"从 LLM 权重里剥离、用 grounding 预训练替代,这个方向本身干净且被 97.7% 的 LIBERO 均值支持;<1GB VRAM 量级从文件大小可独立验证为真(bf16 权重仅 215MB)。

### πR²(论文精读,代码未发布)

- **机制**:per-position staircase 噪声级(前 d 个 in-flight 动作 clamp clean 作 inpaint 条件,ramp 内部,d 个纯噪声尾槽),每次调用只做 1 个 Euler 子步就发射 d 个 clean action;fast channel(proprio,每 tick 新鲜)与 slow channel(V+L,异步 + 6 项零初始化 delay embedding lookup)。架构改动只有一行:AdaLN 从全局共享改 per-position。是 Train-Time RTC 的 diffusion-forcing 推广。
- **结果**:xArm6+XHand 真机 4 任务均值 61% vs RTC 40% / naive-async 35% / sync 29%;仿真中随延迟预算增长,πR² w/ async 基本持平(0.43→0.45)而 RTC 崩塌(0.36→0.19)。
- **核验发现的保留项**:(a) "+30%" 是单任务最大值,均值 +21pp,Don't Spill +5pp 在 N=20 下不显著;(b) 25Hz 部署用了第二张 GPU(VLM 与 DiT 分卡),附录承认;(c) 真机表没有 w/o-async 行,两个贡献在真机上无法分离;(d) delay embedding 从未被单独消融(仿真不用它,真机没对照);(e) Insert Box 上 naive-async 反而赢 RTC——作者归因于 RTC 的 inpaint 前缀锚定阻碍 recovery,这是对 RTC 核心机制的有趣反证;(f) fast channel 只有 proprio,纯视觉扰动是盲区。

### 合在一起的图景

两条路线互补且都不完美:TurboVLA 用小模型换绝对速度,但牺牲了语言理解的在线性(文本缓存假设指令集封闭)且数字口径需打折;πR² 保留大模型语义、用调度数学换反应性,但依赖 proprio 传感与第二张卡。共同的诚实结论:实时性的瓶颈不在 FLOPS 而在"观测陈旧度管理"。

## 如何重跑

```bash
python3 code/verify_turbovla_size.py   # HF 元数据核验,CPU 即可
python3 code/pir2_modality_toy.py      # 扰动模态玩具实验,CPU ~5 分钟
```

代码追踪基于 `code/turbovla/`(已清理的浅克隆);关键引用见线程笔记(本 README 已浓缩)。

## 对报告意味着什么

TurboVLA 段应写"方向有价值、数字要重述"——明确区分 chunk 率与反应率、含 BERT 与不含 BERT 的口径;πR² 段应写"调度层面的优雅推广 + 真机证据强但消融有洞"。两者对比本身就是报告论点:实时 VLA 的两条互补路线。
