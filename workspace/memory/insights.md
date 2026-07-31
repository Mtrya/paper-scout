# 研究洞察记忆(进 git)

按巡航累积的、可跨巡航复用的洞察。每条注明来源(巡航日期 + 论文/实验)。不同于 `runs/INDEX.md`(覆盖日志,回答"读过什么"),这里回答"学到了什么"。

## 2026-07-31 巡航(ABot-World-0 / UMI 数据扩缩 / 实时策略)

### UMI 数据

- **UMI 采集范式有结构性的 recovery 盲区**:对 HiFi-UMI-2K 全量 1,125 episodes 的运动学测量显示 90.2% 的 episode 双手零方向反转(38 个任务中位数全部为 0)。演示者追求一次流畅完成,犯错-纠偏片段根本不会进入数据——fidelity 再高也记录不到不存在的行为。含义:pose 精度(~3mm)之后,数据侧的下一个 frontier 是 recovery/接触扰动覆盖,它需要刻意设计的犯错协议或仿真扰动增广,不会在自然采集中免费出现。(HiFi-UMI 2607.25895,实验:umi-data-scaling/code/)
- **fidelity × scale 坐标系**:HiFi-UMI(保真)与 Xiaomi-Robotics-1(规模)互补——fidelity 决定 UMI-only 能否"收官"(去掉真机 anchor),scale 决定 out-of-box 泛化能走多远。parity 可以用 ~10× 数据买到;成本从不消失只转移(小米:标注→prompt 分布桥接;HiFi-UMI:遥操作→硬件工程+逐条仿真验证)。(2607.15330 / 2607.25895)
- **action↔state 语义核验很便宜**:HiFi-UMI-2K 的 action[t] 与 state[t+1] 逐位精确相等。做运动学分析前先验证这一假设,否则指标全错。

### 实时策略

- **diffusion-forcing staircase 调度的实现空间是雷区**,三种看似自然的实现全部灾难性失败(误差 2–6×):(i) rolling buffer 前槽钉在 τ=1,而训练 τ~U[0,1) 不含 1,τ embedding 外推出野动作;(ii) 每 tick 步长 1/(H−1) 时高 τ 区流场增益 1/(1−τ)≈15,欧拉迭代临界不稳定,τ≈0.9 槽位腐化后被发射;(iii) 全 buffer 每 tick 步进导致动作提前 12 tick 定型冻结 = 长开环执行。能工作的设计:与 sync 相同的 K 层去噪网格 + 只步进前 K 槽(动作的 K 步去噪落在发射前最后 K tick)+ 前槽用当 tick 观测完成最后一步后立即发射。(πR² 2607.26055,实验:realtime-policies/code/pir2_modality_toy.py)
- **staircase 的优势不是模态特异的**(我们的猜想被玩具实验证伪):proprio 可见扰动与纯视觉扰动的误差几乎重合;真正的自变量是观测延迟档——d 小时 staircase 付质量税,d 大时 last-minute finalization 反超。玩具局限:1D、无记忆专家、不计算力价格。
- **实时 VLA 数字的口径三问**:chunk 率还是反应率(TurboVLA "32Hz" 实为 chunk 率,replan ~2.7Hz)?离线缓存的模块算不算参数量("0.2B" 含不运行的 BERT,部署图仅 ~107M)?不同 benchmark 的延迟是否同口径(含/不含文本编码器、是否叠加 temporal ensembling)?(2607.27205)

### 世界模型

- **ABot-World-0 的"无限"= 21 latent 帧滚动窗(~7s)+ 5 张预生成参考图负 RoPE sink**;超过 7 秒的世界一致性只活在 5 张图里。动作注入是 270M 参数 conv adapter(5.1%),控制带宽 ~1.3Hz(每 block 一次采样)。demo 实际 12FPS vs 论文 16FPS(后者是 5090 DiT 生成速率)。(2607.19191)
- **论文只测了 60s 视觉伪影,从未测长程 action-following**——我们 4090 实跑的 640-block(10.7min)实验发现:可控性**不是长期衰减而是阵发性崩溃**——yaw 跟随率首尾同为 ~0.95,但 block 130-180 塌到 0.25、460-500 塌到 0.75,之后自行恢复;崩溃窗同时伴随画面涂抹,两者同步恶化同步恢复。滚动窗 + 参考图 sink 让 rollout 统计意义上无限续命,但模型会反复进入退化态再被 sink 拉回。(2607.19191,实验:abotworld0-2607.19191/code/)

### 方法论

- **"长/大/快"声明应配衰减曲线**:可控性-vs-时长、成功率-vs-数据量、质量-vs-延迟——单点数字不足以评估这个时代的系统声明。
- 猜想驱动实验与 claim 核验各占一半时,巡航产出最好:本轮三个实验一个证实(recovery 盲区)、一个证伪(模态特异)、一个测了论文没测的维度(可控性衰减),每个都产生了论文文本里没有的知识。
