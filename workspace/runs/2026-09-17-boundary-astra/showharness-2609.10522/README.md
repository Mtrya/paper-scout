# showharness-2609.10522 线程:Show-Harness D 条件复测(实验 B)

锚点论文 Show-Harness(2609.10522)与实验 B(动作语义命名×约定五条件复测)的证据包。

- 论文缓存:`papers/agents/showharness-2609.10522.md`
- 探针代码:`code/`(robot_showharness.py 单文件 CLI server、run_trial.py codex 驱动)
- 证据:五条件(A 语义+约定/B 仅语义/C 任意+约定/D 任意/E 任意+仅图像)各 5 回合的 result.json 与关键 trace(D/t0 的 6 调用探测序列、E 条件的 observe-probe-observe 节奏)

核心结果:5 条件 × 5 回合 25/25;论文 D 条件 1/20、自写映射 23.3% 正确的负结果是被试(Gemini-3.1 Pro)的,不是范式的。
