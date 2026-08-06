# 线程 A:WorldCycle 闭环奖励盲区——ABot-World-0 可逆循环探针

论文:WorldCycle(2608.04964),自验证 RL:可逆动作循环 → 空间闭环奖励 + 时间一致性奖励,DiffusionNFT post-train WorldPlay 8B;CycleBench(T1–T4,RoMa 评测,ESC/RPS/RCS)。

问题:闭环指标(全局图像匹配)测量的状态变量,是否就是动作作用的变量?在 sink 锚定 + 第三人称 avatar 的模型(ABot-World-0,不同谱系)上,闭环会不会"因错的原因"变好?

## 实验

环境:启智 abot-cycle(4090 48GB,ngc-pytorch:25.02),ABot-World-0-5B-LF bf16 全量,streaming 推理(12 帧/块 @12fps,704×1280,5 张参考图 sink)。脚本 `code/cycle_probe.py`(探针)、`code/analyze_cycle.py`(SIFT 归一化位移 → ESC/RPS/RCS + 偏航光流率)、`code/run_cycle_probe.sh`(远端运行器)。

协议(块=12 帧):p0 J×4(偏航基线);p1 W×4→S×4;p2 J×4→L×4;p3 (W×2→S×2)×4;p4 W×8→S×8;p5 W×4→S×4→J×4(回归后功能测试)。p0/p5 用两组独立噪声复跑(seed1 在 code/cycle_metrics_seed1.json,seed2 在 code/cycle_metrics_seed2.json)。

## 关键数字

| 协议 | ESC | RPS | RCS | 备注 |
|---|---|---|---|---|
| p1 平移循环 | 0.0047 | 0.0048 | — | 漂移剖线全程 ~0.005:相机没动 |
| p2 偏航循环 | 0.0166 | 0.0397 | — | 漂移弧 0.078→0.226→0.017:真转出去真回来 |
| p3 重复循环 | 0.0032 | — | 0.0046 | 同上,相机不动 |
| p4 长程循环 | 0.0198 | 0.0091 | — | W→S 切换处尖峰 0.165(avatar 瞬间转身) |
| p5 vs p0 | — | — | — | 回归后偏航响应 1.03/1.27 vs 新鲜 3.30 px/帧,衰减 ~3× |

## 结论

1. 闭环指标在 ABot-World-0 上"近乎完美",但 W/S 移动的是 avatar、相机/背景被 sink 钉死——指标的状态变量不响应动作。
2. 语义可逆性失败:p1 S×4 末 avatar 转身 180° 面对相机(code/keyframes/p1_trans_cycle/block_007.jpg),W⁴∘S⁴≠I。
3. 功能不等价:循环后同动作响应衰减 3×。
4. 动作切换边界是阵发失控的结构性触发点(接 07-31 的 640 块阵发崩溃发现)。
5. 对 WorldCycle:R_act(动作跟随分)不可省;CycleBench 式评测应加状态变量审计;verifier 的盲区就是 reward hack 的方向。
