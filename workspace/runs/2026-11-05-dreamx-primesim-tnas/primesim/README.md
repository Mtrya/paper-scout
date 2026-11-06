# PrimeSim (arXiv:2611.04870) 线程包

## 问题

PrimeSim 声称:30,110 小时跨本体语料后训练的 Wan2.2-TI2V-5B 底座通用模拟器,WorldSimProbe 66.6 五 suite 赢四、EWMBench 6.09、策略排序 ρ=1.0、GRPO 模型内训练 61.7→73.2,并在摘要中以现在时声称 "We release the model weights, inference code, and training code"。

本线程的问题:这些数字的可信度与口径是什么?开源声明是否属实?

## 调查动作

1. 全文精读(本地缓存 `papers/world-models/primesim-2611.04870.md`)。
2. 外部核查(子代理,报告 `code/primesim-audit/audit.md`,含逐项来源 URL):
   - 语料表 Table 12 逐数据集对照官方口径,重算隐含 fps;
   - WorldSimProbe Table 1 基线数字与原文 (arXiv:2608.09298) Table 2 逐位对照;
   - EWMBench Table 2 基线溯源(Genie Envisioner 2508.05635、Thinker-WM 2610.18447、Qwen-RobotWorld 2606.17030、EWMBench 2505.09694);
   - GitHub 仓库与项目主页的开源状态快照(2026-11-05)。

## 发现(全部经外部核验)

1. **语料表严谨**:30,110h = 帧数/fps 严格换算。此前可疑的 RT-1(350h 对 4M 帧,隐含 3.17Hz)与 Bridge V2(110h 对 2M 帧,5.05Hz)精确等于两者的原生控制频率(3Hz/5Hz),反而证明其元数据管道忠实。DROID 表内 510h vs 官方常说 350h:官方 350h 只计 76k 条成功轨迹,PrimeSim 计入了发布中保留的 16k 条未标注成功轨迹(92k 全量 × 15Hz ≈ 510h)。唯一真正的口径问题:InternData-A1 原作者定性为全仿真合成,PrimeSim 把其中 4,000h 重划为 Real。In-House 9,540h(32%)私有不可审计。
2. **基线数字非自测**:WorldSimProbe 五个基线的全部 T1–T5 分数 100% 照录自 WorldSimProbe 原论文(逐位一致),PrimeSim 只有自己的一行是自测。赢四输一(T1 输 LingBot-VA 70.2 vs 77.0)的框架下,基线行是文献值而非同炉复跑。
3. **EWMBench 口径继承**:GE-Sim 两行(5.82/5.79)是 Genie Envisioner 原论文 8 项指标归一化求和的精确还原(求和复核通过);其余基线继承 21-episode 社区划分(Thinker-WM/Qwen-RobotWorld 协议)。EWMBench 训练分布来自 AgiBot——PrimeSim 的训练语料含 AgiBot World,作者自己承认 in-domain。
4. **假开源实锤**(截至 2026-11-05):摘要与引言均为现在时 "We release...",但 GitHub 仓库(PrimebotTech/PrimeSim,创建于 2026-11-04)只有 3 个占位文件(LICENSE/README/"Code coming soon"),0 releases,0 tags;项目主页无任何权重链接。精确障碍:想复现/探针均不可能,只能审计文本。
5. **方法论真实贡献(文本层面)**:四阶段后训练(chunk-causal 化 → Causal CD 蒸馏 50→4 步 → clip 自 rollout 非对称 DMD 纠 compounding → episode 级 + 分层 KV 记忆 3+9+9);记忆训练用模型自生成历史——部署时信息集匹配,与我们此前 CMD/EVOKE/DreamX-Phi 的记忆规则同向。值得注意的张力:amortized supervision 里教师看干净窗,学生滚自己的历史——信息集不完全对等。

## 判读

数据侧的严谨性(帧率级吻合)与基准侧的"照录+继承"形成对照:论文的工程可信度高于其基准对比的证据强度。GRPO 61.7→73.2 与 ρ=1.0 是全文唯一完全自产自销的硬数字,但既无权重也无代码,无法第三方核验。

## 证据位置

- 审计全文:`code/primesim-audit/audit.md`(本包归档副本见 `code/audit.md`)
- 论文缓存:`papers/world-models/primesim-2611.04870.md`
