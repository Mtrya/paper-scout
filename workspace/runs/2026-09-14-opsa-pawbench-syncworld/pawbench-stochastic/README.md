# WAM 概率对齐线程(2608.27345 PAWBench + OpenWAM 邻域)

PAWBench 把"分布形状轴"(本实例 08-14 的监督信号第二轴)从训练侧推广到评估侧:世界模型应当重现同一 (观测, 动作) 下的**结局分布**,而不只是单条可信轨迹。

## 论文核心主张(PAWBench,上海 AI Lab)

- 形式化:概率对齐 = support 对齐(有效结局都能出现)+ probability-mass 对齐(出现比例正确);TVD 测后者,coverage 测前者,SPR(场景通过率)单独报。
- 50 场景 × 8 机制族,25 Calibration(有解析参考分布)/ 25 Coverage(只知支持集);K=50 重复 rollout,Gemini 3.5 Flash rubric 判读终局(人机一致 81.3%)。
- 11 个视频生成器无一全过;最优 Calibration TVD 20.5(Cosmos3),最优 Coverage ~90%(LTX-2.3);有限样本零假设 TVD 均值 8.33 vs 实测 31.2(99% 分位 9.22)。
- 因果/非因果配对控制:物理因果干预下分布变化不足,非因果提示(文字)却显著搬动概率质量——模型不追踪"什么改变了物理转移"。
- 三接口干预:语言(PE:VLM 预测的分布本身就错;Oracle PE 大幅改善但生成器只实现 37.6-58.1% 的指定结局)、耦合噪声 C2C(扩覆盖、校准混杂)、LoRA 数据配比(单调但粗糙——训练配比把两个场景朝同方向推,无法同时满足 upright 50/50 与 leaning 100/0;全局频率控制 ≠ 场景条件分布学习)。
- 局限自述:终局标签丢轨迹动力学;K=50 有限;不含具身交互。

## OpenWAM(邻域,设计空间研究)

- Infra 把 WAM 设计空间分解:视觉编码器(冻结 VAE/表示+S-VAE)、流骨干(5 视频 + VLM + ActionDiT)、可见性掩码(4 模式)、架构 6 变体。
- 三问结论:Q1 继承什么——更强生成骨干单调涨(但 14B vs 5B 只差 1.4pp);latent 紧凑且富含世界信息比"重建 vs 表示"之辩重要(S-VAE 压缩后 DINOv3 90.18 追平 Wan2.2-VAE 90.30)。Q2 交互——独立 ActionDiT+联合自注意 92.36 ≫ 动作混入共享序列 85.50;世界→动作通路必需(+5pp)、反向从零训练几乎无关;同步联合去噪 > 异步(93.0 vs ≤92.3);detach 梯度 91.85 已近端到端。Q3 跨域——预训练主要买 OOD(+12.12pp)、ID 小;预训练后 mutual 信息流翻正(最终采用)。
- 发布:46 个 HF 模型 = 14 个评测 checkpoint + 全部架构/信息流/骨干/编码器消融 + pretrain/sft 变体 + 基础模型。预训练 128×H200 7 天,数据 1.33B 帧→518.5M 帧(5 源 21 embodiment)。
- **与 PAWBench 的互补**:OpenWAM 全文没有"重复 rollout 恢复结局分布"类评估——设计空间研究完全没碰分布形状,而 PAWBench 恰好只测这个。两者拼起来是完整的"接口谱系 × 分布形状"坐标纸。

## 我们的研究动作

- **B 玩具随机结局世界(远程 4090,已完成)**:pachinko 式下坠(T=8 行,p=0.4,结局=Bin(8,0.4) 平移,动作=初始列偏移 a∈{−2..2}),16 帧 32×32,参考分布解析可得。同一解码器家族三种训练目标:det(逐帧二值交叉熵)/ VAE-Gaussian(ELBO + free bits 0.2 nats/dim)/ z-diffusion(在同一 VAE 隐空间上建模 p(z | 首帧, 动作));oracle = 模拟器采样。每个动作 500 次 rollout,末帧落点读列。结果(五动作平均):TVD det 0.721 / VAE 0.208 / z-diffusion 0.102 / oracle 0.039;有效支持覆盖率 det 1/9、两条隐变量臂 6/9、oracle 8/9。det 的 TVD 精确等于 1 − 众数概率(1 − 0.2787),即点目标的解析上限;两条隐变量臂缺失两端尾部(以动作 0 为例缺 7/21/23 列)且质量整体右移(13 列欠采、15 列过采)。
- **两个过程发现**:① 后验坍缩——未加约束的 VAE KL 归零、解码器完全无视 z,先验采样 256 次全部落在同一列、后验均值解码同样(探针证据 `code/vae_collapse_probe.md`);线性探针仍能从坍缩的 z 读出 R²=0.897 的结果信息,但方差仅 1e-5、KL 与解码器都不用它——"信息在 z 里" ≠ "模型能采样出分布"。② 评测 readout 若在 logits 空间做列 argmax,未训练边框列的 logits≈0 会打败真实落点列(背景行强负),三臂全判边缘列、TVD=1.0 的假失败;改概率空间 + 3 像素匹配平滑后恢复理论预期,对真实视频 64/64 精确。
- **PAWBench 代码审计**:50 outcome + 50 trustworthiness YAML rubric,判读协议(readout policy/priority rules/unreadable vs out-of-schema)设计严密;evaluate.py 与生成器解耦;Gemini 判官 + 人机一致 81.3% 是已知噪声底。

## 关键文件

- 本地:code/pawbench(clone)、papers/world-models/pawbench-2608.27345.md、openwam-2609.07398.md(pdftotext)。
- 远端:workroot `opd-pawb/` 下 b_toy_wam.py / vae_probe.py / b_render.py / b_hist.py;产物 results/toywam/（det.pt、vae.pt、zdiff.pt、eval_report.json、agg_summary.json)与 results/plots/plot_b_tvd.png(已回收到 `assets/`);本地同源脚本在 `code/opd-exp/`。
