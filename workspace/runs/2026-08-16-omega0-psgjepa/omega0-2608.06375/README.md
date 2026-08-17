# ω-0 / ω-HOME 数据集探针(2608.06375)

## 尝试了什么

论文宣称 ω-HOME 的核心价值是"并发 loco-manipulation"(concurrent loco-manipulation)数据,
但全文没有任何量化。代码未放出,数据集(HF `keycharon/omega-HOME`,39GB,4827 集)是唯一可验证的产物。
我们下载了 5 个任务的全部 state_action.hdf5(1000 集:mop_floor 200、pick_and_place_apple 207、
push_chair 205、retrieve_from_the_upper_fridge 201、wipe_table 187),直接测量:

1. **并发度**:腿部 12 关节与臂部 14 关节的关节速度(30Hz),外加基座 yaw 角速度与 VR 手部点速度。
   活动阈值 0.15(rad/s 或 m/s),报告 P(下肢动 ∧ 上肢动 | 有任何运动)。
2. **遥操作保真度**:`body_q_target` vs `body_q_measured` 逐关节互相关时延 + RMSE。
3. **模态与格式核验**:fps、视频帧数与状态序列逐帧对齐、base_trans 全局位移是否存在。

## 保留了什么

- `code/omega_probe.py`:逐集统计脚本(并发度/时延/RMSE/抓取周期)。
- `code/omega_probe.csv`:1000 集逐集指标。
- `code/omega_plots.py`:报告图生成。
- `../assets/timeseries_concurrency.png`:mop_floor vs pick_and_place_apple 的活动时间序列对照。
- `../assets/task_summary.png`:五任务并发度条形 + 时延/跟踪误差箱线。
- `../assets/mop_exo_frame.png`:第三人称视角实拍帧(G1 持拖把)。

## 如何重跑

```bash
# 数据(HF,公开):
hf download keycharon/omega-HOME --repo-type dataset <task>/<ep>/state_action.hdf5 --local-dir code/omega-home
code/scout-exp/bin/python code/omega_probe.py out.csv
code/scout-exp/bin/python code/omega_plots.py
```

## 结果对报告意味着什么

- 并发度谱系: mop_floor 0.866 > wipe_table 0.499 > push_chair 0.376 > retrieve_fridge 0.138 > apple 0.048。
  "并发 loco-manipulation"在拖地/擦桌类任务上确实成立(运动帧的近九成上下身同动),
  而拿取类任务退化为串行——数据集的差异化价值集中在接触型长工具任务。
- **发现 1(论文未提)**: 遥操作命令→实体的时延 70–230ms,且与任务机械负载同序
  (拖地 197ms、冰箱取物 228ms vs 拿苹果 81ms);跟踪 RMSE 0.107–0.206 rad。
  长工具/高负载任务的"意图-动作"错配最大,模仿学习继承的是被低层延迟滤波后的演示。
- **发现 2(数据边界)**: `body_q_target` 只在 6 个腕部关节上变化——腿与腰没有命令流,
  由 SONIC 内部平衡/步态自主生成;`base_trans_measured` 全数据集恒定(未记录全局里程计),
  `delta_heading` 恒零(死通道)。用这份数据学全局定位/世界坐标轨迹是不可行的。
- 30Hz 与论文一致(时间戳量化到 0.02s 网格,中位 dt 会误读为 25Hz,须用时距/样本数算);
  视频帧数与状态序列逐帧对齐(248=248)。
- `token_state`(64 维)94% 可由当前关节状态线性读出——它是 SONIC 状态 token,不是动作潜变量。
