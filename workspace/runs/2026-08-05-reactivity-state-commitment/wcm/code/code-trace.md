# WCM 代码路径核验

- 核验日期：2026-08-05
- 官方仓库：https://github.com/sylvestf/WCM
- commit：`c126057d6d00ceff48f885f96bbf5a3dda4d2bc8`

关键路径：

- `world_critic/model.py:285-294`：history trunk 是标准
  `TransformerEncoderLayer` / `TransformerEncoder`，对历史长度 K 的 attention
  复杂度为 O(K²)，不是论文摘要所说的 exponential complexity。
- `world_critic/model.py:411-512`：instruction-conditioned value 分支与
  action-conditioned latent dynamics 分支共用历史 context。
- `world_critic/data.py:262-265`：数据窗为 `history_size + prediction_horizon`，当前
  实现显式要求 prediction horizon 为 1。
- `world_critic/training.py:146-211`：value、prediction 与 SIGReg loss 联合；DDP
  下对 global valid-token count 和 SIGReg 做专门归一化。

发布边界：核心 critic 训练/评测实现、小型真机资产与部分 checkpoint 可见；论文
149-task 的全部 RL 集成、全量 checkpoint 与真机复现脚本没有完整打包。
