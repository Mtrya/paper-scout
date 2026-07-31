# Thread: UMI 数据扩缩 — Xiaomi-Robotics-1 (2607.15330) vs HiFi-UMI (2607.25895)

## 研究问题

高保真 robot-free(UMI 式)数据能否取代真机遥操作,成为可部署操作策略的训练基底?两篇同期论文给出了看似不同实则互补的答案。

## 尝试了什么

1. 精读两篇全文(arXiv HTML:Xiaomi v2 含附录,HiFi-UMI v1 含 Limitations),提取 scaling 证据、数据管线、真机结果与失败模式。
2. 核验 HiFi-UMI 公开数据集 simple-world-lab/HiFi-UMI-2K 的实际记录信号 —— `code/probe_hifi_umi_2k_meta.sh`(可重跑),元数据原件存于 `code/hifi-umi-2k-meta/`。
3. 对照原始 UMI(Chi et al. 2024)的已知弱点做三角验证。
4. **真实验(猜想验证)**:HiFi-UMI 评测中观察到的"恢复弱"失败模式,其根源是策略问题还是数据问题?猜想:UMI 演示数据本身几乎不含 recovery 分布——演示者追求流畅完成,犯错-纠正的片段被天然筛掉。下载 HiFi-UMI-2K chunk-0000(478,810 帧 / 1,125 episodes 全量)到本地,用 `code/hifi_umi_recovery_probe.py` 对双手 6D 位姿做运动学分析(方向反转检测:120°/0.2s 平滑速度;action↔state 语义核验)。

## 保留的证据

- `code/probe_hifi_umi_2k_meta.sh` + `code/hifi-umi-2k-meta/{info.json,modality.json}` — 2026-07-31 核验:LeRobot v3.0 格式,25 fps,6 路 512×640 相机,20 维双臂 EEF state/action(每手 [xyz, rot6d, gripper_rad]),action 为 absolute next-state target,逐帧 validity mask(valid.frame / state_valid / action_valid)与论文"validity masks"声明一致;单 chunk 1125 episodes、38 tasks。
- `code/hifi_umi_recovery_probe.py` + `code/episode_kinematics.csv` — 2026-07-31 全量运动学分析结果(每 episode 一行:双手反转率/抖动/时长/任务标签)。

## 实验:UMI 数据里的 recovery 含量(猜想证实)

**猜想**:HiFi-UMI 的 WAM Remote Insertion "恢复弱"失败,根源在数据分布——UMI 演示天然排斥 recovery 片段,策略学不到犯错后的纠偏。

**方法**:对 1,125 条 episode 的双手 EEF 轨迹做方向反转统计(平滑速度 120° 阈值,归一化为 reversals/10s)。先核验数据语义:`action[t] == state[t+1]` 逐位精确成立(max diff = 0.0),确认 action 即 next-state。

**结果**(全量 1,125 episodes,详见 `code/episode_kinematics.csv`):

- **90.2% 的 episode 双手零方向反转**;reversals/10s 中位数 = 0;仅 2.0% 的 episode 超过 1 次反转/10s。
- 按任务分解,所有 38 个任务的中位数都是 0;最"富恢复"的 Clear the tablecloth 最大值 2.2/10s 但中位数仍为 0——即没有任务系统性地包含纠偏行为。
- 附带测量:关节抖动中位 7.22mm(注意:该指标混淆真实运动加速度与传感器噪声,不能用来反驳论文的 3mm SLAM 精度声明,报告中不作此用途)。

**结论**:猜想证实。UMI 数据确实几乎不含 recovery 分布——"恢复弱"不是 HiFi-UMI 的管线缺陷,而是 UMI 采集范式的结构性盲区:演示者追求一次流畅完成,fidelity 再高也记录不到不存在的纠偏行为。这给线程主结论补上了数据侧证据:pose 精度(~3mm)之后,下一个 fidelity frontier 是 **recovery/接触扰动的覆盖**,而它在"演示者自然操作"的采集范式下无法免费获得——需要刻意设计"故意犯错"的采集协议或仿真扰动增广。

## 关键发现(报告用)

### Xiaomi-Robotics-1

- UMI pretraining 的迁移证据是全文最有分量的图:同一 post-train recipe 下,out-of-box(unseen 环境+物体)成功率随 UMI 数据量 26%(无 pretrain)→53%(12.5%)→75%(100%),"无饱和迹象";contact-rich 任务(shoe tidying)从 0%→75%。
- 但 scaling 曲线本身不干净:12.5%/25% 档 validation MSE 先降后升(过拟合早停),只有 4 个数据点;scaling 研究只用 20k/100k 小时。模型档位(2B/5B/10B)间差距小于数据档差距 —— "数据量是主要瓶颈"。
- Auto-labeling:等长切片(非语义切分)+ Qwen3.5-27B 标注 state-transition 描述,2 周标完 100k+ 小时;**全文无任何标注质量控制**。代价被转嫁:post-training 必须额外桥接 "state-transition 描述 → imperative instruction" 的 prompt 分布鸿沟(用 1k+ 小时人工标注 UMI + 人工切分的真机标注)。
- **论文从未做"去掉真机数据"的 ablation** —— 7,200 小时真机数据(post-train 采样占 85%)是否必要,他们根本没问。这正是 HiFi-UMI 切入的缝隙。
- 有意思的 negative result:DiT attend 到 VLM 的 action tokens 会掉点(shortcut 学习),因此把 action tokens 从 DiT attention 排除。
- 内部数字不一致:RoboCasa365 Abstract 57.4% vs Introduction 57.6%;VLABench Instruction track 输给 ERVLA(55.8 vs 58.0)但正文不提;真机 downstream 每任务仅 10 trials。

### HiFi-UMI

- 保真升级(对照 UMI 2024):头戴离线 stereo-inertial SLAM(刻意放弃全局 loop closure,改 dynamic sliding window)→ workspace-local ~3mm(UMI ~6mm);µs 级 GPIO 硬件同步(<40µs);每手 2 个非平行 fisheye ~200°;双手相对位姿同帧原生测量(非事后重建);full-palm glove。
- WBC replay validation:每条轨迹在仿真中对目标 embodiment 回放,丢弃运动学/动力学不可行者;重建 98% × replay 98% ≈ 96% 累计可用率。小米 pipeline 无任何 replay 验证环节。
- 核心结果:三 backbone(StarVLA-QwenPI、OpenPI-π0.5、LingBot-VA)× 4 个 tabletop 双臂任务,UMI-only vs in-domain teleop 各 40 rollouts —— aggregate parity(−2.5pp / +3.1pp / −0.6pp),Remote Insertion 85% 是全场最高且来自 UMI-only。**但 parity 用 ~10x 数据买**(3200 vs 300 条/任务,作者诚实声明)。
- 失败结构差异:UMI 策略 nominal 执行更流畅,但 contact 不完美时恢复弱(WAM Remote Insertion −7.5pp);WAM oracle 诊断**排除了 gripper 通道**(teleop 默认全张 vs UMI 保留手势角,语义未校准)——20 维动作空间里 2 维的跨接口对齐靠评测回避而非解决。
- Scaling 证据更干净:log-log power-law α=0.268/R²=0.993(in-distribution),OOD α=0.095,按 interaction-dynamics 覆盖率解释(textile <1% frames → garment folding 最差);~3200 条/任务饱和。
- 未做 fidelity 因子分解(作者承认):"high fidelity suffices, but not how much of each property is required"。

### 合在一起的图景

不是正面冲突:Xiaomi 做 mobile manipulation breadth(瓶颈是 scale/覆盖率),HiFi-UMI 做 stationary bimanual precision(瓶颈是轨迹/接触精度)。证据链互补:**保真度(~3mm/µs-sync/宽视场 + replay 验证)决定 UMI 数据能否"收官"(去掉真机 anchor),scale 决定它能走多远(广度与 out-of-box 泛化)**。pose 精度到 ~3mm 后边际价值让位给 contact/recovery 覆盖与 force/tactile 信号 —— 后者两篇都没有,是下一个 fidelity frontier。

## 如何重跑

```bash
bash code/probe_hifi_umi_2k_meta.sh /tmp/hifi-verify   # 数据集元数据核验
python3 code/hifi_umi_recovery_probe.py                 # recovery 含量分析(需先下载 chunk-0000,见脚本头注释)
```

## 对报告意味着什么

这条线程给报告一个真正的研究结论:把两篇放在同一坐标系(fidelity × scale)里,"zero-robot"是有条件的 yes——只能换掉 post-training 的最后一公里,且成本没有消失而是转移(小米:标注→对齐;HiFi-UMI:遥操作→硬件工程+逐条仿真验证+人工核验)。
