# LAWA 线程:潜动作作为 WAM 意图接口 + 玩具锚点实验(实验 C)

论文:LAWA (arXiv 2608.24882,2026-08,无代码)。问题:世界动作模型(WAM)的
动作生成到底被什么引导——LAWA 让冻结的潜动作分词器产出"意图 token"作为
动作生成的条件。

## 论文要点(全文通读)

- 分词器:ViPRA 式——DINOv2 帧差特征 → VQ 码本;SAM2 掩码辅助损失;
  在 action-free 机器人数据 + 第一人称视频上预训练(机器人样本加权 ~20%)。
  分词器在下游训练中**冻结**。
- 三专家联合去噪:video / latent / action 三流,结构化注意力掩码
  (latent 只看 obs+latent;action 看 obs+latent+action);推理时丢视频分支。
- 结果:RoboCasa few-shot 65.6% / 全量 80.8%;Fast-WAM 对照低 9.6 / 4.5 点。
- **承重墙消融(Table 4)**:无第一人称预训练时,LAWA 反而比 Joint-WAM 低
  3.4 / 2.0 点——潜动作接口的收益几乎完全来自分词器预训练,不是结构本身。
- 功能使用证据:推理时扰动 latent,高斯噪声 80.8→52.2,乱序 56.4——
  动作生成确实在读 latent,不是绕过它。

## 实验 C 设计(玩具 WAM,启智 hydra-probe 4090)

动机:08-14 玩具实验里"学得槽 0% 打不过噪声槽 7.5%"——当时解释为
"未来槽不该学"。LAWA 给了另一个假说:**失败是锚点缺失**——学得槽没有
内容约束,自由漂流;LAWA 的码本量化恰好提供了锚点。

- 脚本:`drafts/wam_if_toy.py`(远端 wam-if-toy/,数据符号链接自
  wam-rift-probe 存档,joint 数据 2000 集)。
- 变体 lawatok:rift-fm 目标改为 k-means 码本(K=256,614400 维特征拟合)
  量化后的离散 token——给学得槽钉上码本锚点。
- 对照基线(已有):joint 26.3% / currentonly 3.7% / rift-fm 0% /
  noiseslots 7.5%。
- 两组规模:cfull(全量 2000 集,12000 步)+ cfs(10% 数据 200 集,
  8000 步,四变体)——假说预测锚点收益在 few-shot 下放大。
- 判定:lawatok SR 显著超过 rift-fm(0%)与 noiseslots(7.5%)则锚点
  假说成立;若仍接近 0,则"学得未来槽"问题比锚点更深。

## 结果

- 全量(2000 集,12000 步):lawatok SR 3.0%(n=100)。对照(08-14 存档,n=80):
  joint 26.25% / currentonly 3.75% / rift-fm 0% / noiseslots 7.5%。锚点把学得槽
  从 0 抬到 3,追平"无未来"但不如纯噪声槽——"失败全因锚点缺失"在全量下证伪。
- 少样本(200 集,8000 步,n=100):currentonly 3% / rift-fm 7% / noiseslots 7% /
  **lawatok 16%**——唯一显著跑赢噪声槽的学得未来变体,收益恰好在数据稀缺处显现,
  与 LAWA 论文 few-shot 放大模式同向(+9.6 few-shot vs +4.5 全量)。
- 未解释倒挂:lawatok 少样本 16% > 全量 3%(单种子,n=100,差约 3σ)。
  猜测:数据充足时观测直解任务、码本目标变干扰。留作悬案。

## 置信度边界

玩具 WAM(合成数据、小模型)结论只可迁移定性方向;码本由 k-means 充当,
与 DINOv2+SAM2 的语义分词器有本质差距——若假说成立,只证明"锚点有用",
不证明"LAWA 的特定分词器必要"。

## 文件

- `code/wam_if_toy.py`:实验脚本(远端运行,训练+评估一体)。
- `code/eval_cfull.json` / `code/eval_cfs.json`:结果(待回传)。
