# gptpolicy-2609.19138 线程:GPT-Policy 精读 + 仿真上下文消融

锚点论文 GPT-Policy(2609.19138,冻结 GPT-6 Astra + 演示进上下文)与本巡航实验一的证据包。

- 论文缓存:`papers/agents/gpt-policy-2609.19138.md`
- 官方仓通读记录:`code/gpt-policy/`(工作区,零改动;真机回路完整,README 承诺的 RoboDojo 仿真评测未发布)
- 仿真实现:`code/gpt-policy-sim/`(sim_world/sim_render/tasks/demo_record/run_episode/ablate/summarize;`.venv` 未入库,重建:`uv venv && uv pip install matplotlib numpy pillow`,决策后端用本机 codex CLI 的 gpt-6-astra)
- 结果:`code/gpt-policy-sim/results/`(cells.jsonl 36 格权威数据 + summary.md + failure_modes.md 逐格归因 + ablation_grid.png + logs/ 36 份 stdout)
- 演示包:不入库(体积策略);用 `demo_record.py --task <T1|T2|T3> --seed 1042 --mode video|video+action` 重建
- 抽样 traces:不入库(体积策略);逐格归因与权威数据在 `results/`(failure_modes.md、cells.jsonl、logs/)

核心结果(3 任务 × 4 档上下文 × 3 种子,1800s/60 决策上限,编排 124.9 min):

- 成功率:T1 none 3/3、target 3/3、demo-video 0/3、demo-action 0/3;T2 仅 demo-action 1/3;T3 仅 demo-action 1/3——demo 害简单任务、帮难任务
- 失败归因(28 格逐格人工核对):首步幻觉 done 9 格(原始输出核对,非 harness 伪影;demo 臂 6/9);上下文/prompt 伪影 0 格;真失败 17 格(6 格假阳性 done 收尾,释放偏差 2.5–10.4cm 阈值 2cm);超时 2 格
- 机制三则:坐标轴交换自陷(仅 demo 臂,模型自述"坐标轴误换");精度窗口打不中(度量读数误差 2–5cm vs 窗口 1.5–2.5cm);策略缺口(T3 none/target 零次想到钩子,demo-video 32/19 次提到)
- demo-action 的度量锚点:T3 s1 闭合位置与钩子真值差 1.4mm(demo-video 差 5cm)——论文"视频+动作 > 纯视频"的仿真机制级复现
- 边界:单模型 + 2D 渲染 + n=3/格,方向性结论有分量,具体成功率不可外推真机
