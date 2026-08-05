# DEFT-RLVR（arXiv:2608.01755）— 初稿阶段的 HLD 因果探针

本目录保留同一巡航初稿阶段已经完成、且仍有跨问题价值的负结果。最终报告不再把问题限定为自动驾驶；这组证据作为“自由文本事后计划未必因果约束未来行为”的一个具体先例并入 hindsight 线程，而不是单独构成报告。

## 这条线做了什么

全文审读 trajectory anchoring、AD-MCQ、两轮 DEFT agent loop 与 RLVR reward；
clone 官方 repo 后追踪 `blind_mcq_agent_loop.py`、reward parser 和内部完整评测说明；
再用真实 Qwen3-VL-8B 权重、公开 AD-MCQ 视频设计 HIGH_LEVEL_DECISION 因果干预。

## 代码解剖发现

- Turn 2 候选作为 user turn 注入；两轮 assistant token 的 response_mask 都为 1，
  共享同一个 GRPO 标量优势。
- Turn 2 因 server 无状态重新传入三路视频；“不要修改决定”仅是 prompt 文本，没有
  parser / constrained decoder / candidate filter 程序性绑定 HLD 与 FINAL_CHOICE。
- 论文 HLD 指标只比较 Turn-1 文本与 GT，没有直接替换 HLD 后观察选择是否随动。

## 实验

- 脚本：`code/run_commitment_probe.py` 与 `code/run_remote.sh`。
- 数据：公开 AD-MCQ 的 6 个真实 Waymo 场景，oracle 覆盖 A–F；真实三路视频。
- 模型：Qwen3-VL-8B-Instruct，bf16，启智 1×RTX 4090 48GB，greedy decoding。
- 条件：同一个 Turn 1 分别以 original / withheld / counterfactual HLD 进入 Turn 2；
  counterfactual HLD 由与 oracle ADE 最远的候选生成。
- 结果：移除 HLD 后 3/6 选择改变；反向覆写后仅 1/6 改变，且没有改到注入目标；
  反向条件有 2/6 选择反向目标，但两例 original 本来就如此，因干预新跟随目标为
  0/6。original / withheld / counterfactual 的 oracle 命中 1/6 / 2/6 / 1/6，
  小样本不作准确率估计。结论：HLD 的存在影响收敛路径，但自由文本内容未形成
  对 Turn-2 的语义绑定。
- 原始输出：`commitment_probe_final.jsonl`；汇总：`commitment_probe_summary.json`。

## 精确障碍与边界

作者发布的 DEFT-RLVR checkpoint 本次经 HF 主站/镜像无法完整下载：镜像的大分片
重定向到当前网络不可达 endpoint，第一分片只取到 97MB。实验因此针对论文明确报告
的 training-free DEFT base（同款 Qwen3-VL-8B + 两轮协议），不能外推到 RLVR 后
77.9% checkpoint；一旦权重可达，首要复现是原样比较 RL 前后的 commitment effect。
