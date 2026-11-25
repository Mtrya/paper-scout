# AnyS1 真实权重与数据集独立核验

## 研究问题

AnyS1（arXiv 2611.23751）训练单一 4B System-1 指针模型接管智能体工作流的可枚举决策点，运行时用校准置信度门控决定何时升级到 System-2；cold 模式把单一全局阈值 τ_cold=0.918 从 153 个 held-out 开发样本零样本迁移到三个未见基准。本核验回答两个问题：(1) 官方发布的 Any-DP 数据集统计是否与论文 Table 1 一致；(2) 官方发布的真实权重在 held-out test 抽样上，τ_cold=0.918 是否真能达到论文定阈设计所要求的 ≥95% 一致率。

## 结论

1. **数据集审计吻合**：train 25,986 条样本、1,072 技能，与论文 Table 1 完全一致；test 2,537 条、84 技能；none-of-above 标签占 6.92%（论文报 6.7%）；字段结构与论文决策点模板 (q, O, d) 一致。
2. **τ_cold=0.918 复核通过**：held-out test 随机抽 600 条（seed 7 洗牌，0 条因 label 不在 options 被跳过），τ=0.918 时 System-1 接管 24.8% 的决策（149/600），与金标准一致率 **95.3%**——正好落在论文"阈值取到 ≥95% 一致"的设计目标上；覆盖率偏低与论文 cold 模式"保守委派"的描述一致（SOPBench cold 40% 委派 @ 100% 一致）。
3. **裸准确率与难度口径**：不加门控的 top-1 准确率 78.3%；held-out test 技能难于论文开发集（论文 Dev 83.8%）。分类型：literal 83.0%（n=141）、inferential 76.9%（n=420）、planning 76.9%（n=39）。
4. **操作曲线形状支持标签平滑设计**：τ 从 0.85 升到 0.918，覆盖率 77.7%→24.8% 而一致率 87.8%→95.3%，高置信带陡而薄；τ≥0.97 后无样本。这与论文"标签平滑防止置信度饱和、保留可迁移置信度余量"的机制解释一致（硬标签 CE 的 τ 会虚高至 0.998）。
5. **权重元数据与论文附录一致**：head.pt meta 记录 base=Qwen/Qwen3.5-4B-Base（revision 1001bb4d）、LoRA-16、head_dim=256、head_type=oc、label_smoothing=0.1、lr 1e-4、1 epoch、bf16。

## 材料与口径

- 论文精读缓存：`papers/llm-agents/anys1-2611.23751.md`；官方代码：`code/AnyS1/`（s1.py、runtime.py、serve_s1.py 已通读，TAU_COLD=0.918 与 CalibratedGate 在源码核实）。
- 数据：官方 Hugging Face `anys1/Any-DP` train/test jsonl 全量下载审计。
- 权重复核在启智平台 L40S Notebook（ps-anys1-verify-1120）执行：官方 AnyS1-4B 适配器 + modelscope 下载的 Qwen3.5-4B-Base，`code/verify_tau.py` 复刻官方 serve_s1.py 的加载与打分路径（fim/box 特殊 token、指针头前向、softmax），单次前向输出选项分布。
- 结果：`code/tau_result.json`（含 τ 扫描曲线与逐条记录）。
