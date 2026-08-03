# SpatialCLI (arXiv:2607.27703) — 线程记录

## 这条线做了什么

放出范围核验(纯代码/数据审查,无 GPU 实验)。框架三阶段 Call→Learn→
Internalize,最新颖的是 Internalize(轨迹言语化 + 双视图训练)。

## 核验结果

- repo IANNXANG/SpatialCLI(13 stars):agent 框架、全部评测代码、RL 训练脚本
  (verl)、6350 条 RL 数据(下载实测:多选题格式,规则奖励押选项字母,能力
  标注如 'DP'=深度)。clone 在 `code/` 实验台,RL parquet 抽样读过。
- 模型权重 ZYT-MFM/SpatialCLI-8B 在 HF 可用。
- **Internalize 阶段(言语化 + 双视图)的数据与代码未放出**,Table 3 消融
  (Final Answer Only 52.7 / CoT+Answer 45.0 / 单视图 71.1|62.6 / 双视图
  72.7|91.3)无法复现,只能引用。

## 线程产物

- 报告图:`../../assets/spatialcli-fig1.jpg`(论文 Figure 1)。
- 本线程无自研代码与补丁;障碍已精确陈述(Internalize 数据/代码未放出)。
