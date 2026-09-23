# parts-2609.21788 线程:PARTS 精读 + 玩具复现与 verifier 噪声拷问

锚点论文 PARTS(2609.21788,瓶颈子任务残差 RL)与本巡航实验二的证据包。

- 论文缓存:`papers/robotics/parts-2609.21788.md`
- 玩具实现:`code/toy-parts/`(env/base/rl/run_all/plot/summarize,纯 CPU,`.venv/bin/python run_all.py --out results` 复现;README 含算法偏差清单:REINFORCE-with-baseline 替代 TD3 critic、正优势截断、30 步 fresh-actor 重训)
- 结果:`code/toy-parts/results/`(24 run JSON + summary.md + fig_training.png + fig_noise.png)

核心结果(60k 环境步,3 种子,评估 100 eps):

- ε=0:base 27% → PARTS 97±1%(无重训 86±7%),全任务稀疏 RL 79±23% 且中途退化(0.82→0.46)——方向与论文一致
- verifier 翻转噪声:ε≤0.3 平台(0.92–0.95),0.3→0.5 陡崩(0.37±0.41)——阈值型容错;代价先在样本效率(3.8k→11.0k 步)与方差;误标成功→更多重试(346→373 次尝试),部分补偿污染
- 计划外发现:正优势截断是必需品(对称优势曾把 93% 策略打到 0%);fresh-actor 重训 ≥150 步过拟合摧毁策略,30 步保留 97%(论文中此职能由 operator 选 checkpoint 隐性承担)
