# RynnValue (2608.09853) — 价值基础模型的捷径压力测试

论文:阿里 DAMO。价值基础模型,监督目标为 cost-to-go 秒数(非归一化进度),7000h/3M clips 训练,时序打乱 + value-isolation attention 两件防捷径设计。真开源(代码 + 4B/8B 权重)。

## 我们的研究动作

官方 RynnValue-4B 权重在启智 4090 实例真实加载(bf16,官方 prefix 协议,前缀重采样 16 帧,读最后价值槽),对同一段 28.5s 示范视频(854 帧)构造八种受控扰动,逐前缀扫 150 点:

- **forward / reversed / truncate90% / shuffle**:倒放 v 从 0.40 升至 10.44(ρ=0.69);截断终点 v=2.37 vs 正向 0.38;打乱后 ρ(v,末帧真实剩余)=0.76、ρ(v,位置)=-0.03 → 内容接地,数帧捷径被打掉。
- **frozen40 / frozen80**:画面冻结后 v 仍沿时序惯性下滑(残留时序先验泄漏,部分捷径签名),但终点读数(1.28/2.00)仍显著高于正向(0.38)。
- **rewind(60%→30%→end)**:回退腿 v 上跳 5.3s,方向正确、幅度被压缩(校准天花板 ~10s vs 真实 28.5s)。
- **loop(中段 10% 循环 3 遍)+ loopdense(循环区加密采样)**:每次重播 v 精准上跳 +2.3s/+3.9s,循环结束曲线回落,终点 0.44 ≈ 正向 0.38。多尺度回退检测,对 RL credit 与重试片段标注直接可用。
- **语言分支**:Match 对/错指令判定正确;Success 在正确指令且任务确实完成时判 No(256 token 复跑排除截断),自生成描述系统性漏掉收尾动作(关抽屉)→ 语言头当 verifier 的假阴性模式。

## 核心发现

- 论文的防捷径设计声明被独立证实(shuffle 条件下位置信息被物理摧毁,ρ=0.76 只能来自内容)。
- 新发现:价值函数在多时间尺度上识别动作回退,小尺度校准良好。
- 两个使用警告:frozen 时序泄漏(防捷径不完全)、Success 判定假阴性(描述漏收尾)。

## 内容

- `code/probe_rynnvalue.py`:八条件压力测试脚本(含 `--conditions` 与 `--skip_analysis`)
- `code/plot_report.py`:报告图绘制
- `code/results/`:probe_results.json(六条件)、probe_results_extra.json(frozen80/loop + 256 token 语言分支)、probe_results_dense.json(loopdense)
- 运行环境:启智 notebook `rynnvalue-probe`,workroot `/inspire/qb-ilm/project/cq-scientific-cooperation-zone/ky26021/embodied-research/rynnvalue-probe/`
