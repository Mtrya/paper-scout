# Democratizing Robot-Use Agents:同一个模型,不同 harness

论文:Yuan 等,2026,arXiv [2611.01560](https://arxiv.org/abs/2611.01560)。代码:[Open-RUA/uni-robot-agent](https://github.com/Open-RUA/uni-robot-agent)。

## 我们测了什么

论文的主实验是 RoboDojo 42 个任务与 19 个真机任务,需要模拟器、机器人、模型 API 与任务预算。我们没有复现这些;我们做的是把 harness 的**调度器**单独拿出来,在无模型、无机器人的条件下量它的并发边界。

`code/sched_probe.py` 用公开包里的 `Toolkit`、`SerialToolExecutor`、`ParallelToolExecutor` 与 `ToolExecutionPolicy` 构造两个合成工具:

- `slow_exclusive`:睡眠 1 秒,资源 `arm`,exclusive;
- `slow_parallel`:睡眠 1 秒,资源 `vision`,parallel。

每个条件跑三次,墙钟时间逐次记录在 `code/out/run1.json`、`run2.json`、`run3.json`。三次之间相差不超过 2 毫秒,下表取第一次。

| 条件 | 串行(s) | 流水线(s) | 隐藏的推理(s) |
| --- | ---: | ---: | ---: |
| 4×1 秒动作,每次 0.25 秒推理 | 5.00 | 4.00 | 1.00 |
| 4×1 秒动作,每次 1.5 秒推理 | 10.00 | 6.00 | 4.00,另 2.00 暴露 |
| 4 个并行视觉调用 | — | 1.00 | — |
| 4 个 exclusive + 4 个 parallel | — | 4.00 | 并行工作全隐藏 |

## 读法

流水线能隐藏的推理时间上限就是动作时间。推理比动作短时,串行相加变成取最大值,收益等于推理时间;推理比动作长时,只能隐藏动作时间,剩下的模型延迟仍然暴露。论文报告的 84.5–93.8% 动作时间与模型请求重叠,是"模型调用没有成为瓶颈"时的数字,不是一般加速保证。

论文的归因也有同样的边界。RoboDojo 上,同一个 Qwen3.8-Flash-Next 换 harness,Score 从 1.90(RoboProbe)到 31.39(uni-robot-agent),跨度 29.49;同一个 uni-robot-agent 内,GPT-6 Astra 51.90 对 Qwen 31.39,模型跨度 20.51;摘要里"开源超过闭源"的 31.39 对 28.97 只有 2.42 分,是跨 harness 比较。真机 19 任务上,Qwen 49.50 对微调 π0.5 47.75,但论文自己的设置里 agent 额外用了 D405 深度,π0.5 与 Dream-1 只吃 RGB;模拟器的深度消融给 Qwen 带来 4.50 分,大于真机的 1.75 分优势。这个"追平"没有被识别。

## 复现

```bash
uv venv code/rua1103/.venv --system-site-packages
uv pip install --python code/rua1103/.venv/bin/python --index-url https://mirrors.aliyun.com/pypi/simple/ jsonschema matplotlib
git clone --depth 1 https://github.com/Open-RUA/uni-robot-agent.git code/rua1103/uni-robot-agent
code/rua1103/.venv/bin/python code/rua1103/sched_probe.py
code/rua1103/.venv/bin/python code/rua1103/plot_rua.py
```

## 边界

- 这是调度器的合成测试,不是 RoboDojo 或真机 rollout;没有模型 API、机器人或任务成功率的测量。
- 合成工具是固定 1 秒睡眠,真实工具的运动时间随任务变化;结论是调度语义的边界,不是论文数字的复现。
- 论文的 84.5–93.8% overlap、四能力失败率、真机分数都按论文口径转述,未独立复核。
