# PhiZero (arXiv:2607.28624) — 线程记录

## 这条线做了什么

PhiZero 报告 Physics-IQ Verified IQ-Score 41.2,基座 Wan2.2-5B 仅 21.2,但聚合
分数不回答"基座到底错在哪:不会动,还是动错后果"。本线程把基座模型拉到
Physics-IQ 真实场景上做失败模式解剖。

## 实验:Wan2.2-TI2V-5B 物理探针(8 场景)

- 数据:Physics-IQ benchmark,8 场景各对应一种物理失败模式(碰撞后果/两体
  碰撞/动量守恒/流体体积/流体边界/热力学/物体恒存/静力学),perspective-center
  take-1,24FPS。
- 生成:I2V(条件视频末帧 + 原版 prompt),121 帧(5s),1280×704,seed 42,
  官方 Wan2.2 代码;启智 1×4090 48GB(NGC PyTorch 25.02,venv 继承系统 torch)。
- 评估:复刻官方协议顺序——运动掩码在原生分辨率生成(背景差分阈值 10,
  见 physiq/binary_mask_generator.py),再缩放到官方目标尺寸 960×540(4K//4),
  算 ST-IoU / 加权 S-IoU / MSE;运动能量曲线(帧差均值);6 帧帧条。
- 协议保真度:自实现掩码算法在真实视频上重跑 vs 官方掩码视频,IoU=1.000
  (见 `code/sanity_mask_check.py`)。

## code/

- `gen_probe.py`:8 场景批量生成(管道只加载一次;含场景 prompt 表)。
- `eval_probe.py`:指标 + 帧条 + 能量曲线。
- `sanity_mask_check.py`:协议自洽性检验。
- `metrics.json`:8 场景全部指标(2026-08-02 运行)。

## eval/

8 场景帧条(`*_strip.png`,上真实下生成)与能量曲线(`*_energy.png`)。
报告精选在 `../../assets/`。

## 结果要点

- 平均 ST-IoU 0.016 / S-IoU 0.077(original 协议估计 IQ≈4.6;场景刻意选难,
  有选择偏差;与 Verified 分数不可比)。
- 失败模式三分类:后果缺失(0032 不弹开、0053 不摆、0089 不溢、0182 不倒)、
  欠动画(0065 液面不涨)、现实破裂(0140 纸消失+工具手闯入、0146 球瞬移)。
- 核心结论:基座"太会动但动得没有后果"(7/8 场景生成能量高于真实)——
  reason-then-render 要补的是"果"不是"动"。

## 远端环境(复现)

启智 Notebook(本线为 `wan22-physprobe`,用完即弃,重建按 workspace/INSPIRE.md):
workroot `<workroot>/embodied-research/wan22-probe/`,内含 models/(权重 25GB,
ModelScope 镜像 curl 直链下载,fp32 磁盘格式)、data/(Physics-IQ 8 场景)、
code/(本目录脚本的远端副本)、outputs/(gen/ 8 条生成视频,eval/)。
注意:NGC 镜像 cv2 缺 codec,视频读取用 decord;48GB 跑 720p 需
`offload_model=True` + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。
