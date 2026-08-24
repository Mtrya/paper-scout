# 帧表示对照实验 B2a/B2b:位姿条件化 + 跨本体

承接实验 B(exp-action-repr):Base 头(基座系动作)在视角偏移下脆断(单视角训练 19 点中 8 点成功率 0,平均 0.48),Cam 头(相机系动作 + 执行时显式旋转合成)全 1.0。本实验问:**这个差别是"表示几何"的差别,还是"信息可得性"的差别**——把相机位姿作为输入给 Base 头,能否追平 Cam 头?

## 环境

```
cd code/camcalib-exp          # 仓库内已有 venv:code/camcalib-exp/.venv(torch CPU + numpy + matplotlib)
```

## 跑法

```
cd runs/2026-08-21-camvla-calibration/exp-frame-rep/code
../../../../../code/camcalib-exp/.venv/bin/python run_b2.py            # 串行全流程(含 DENSE;SCAN 1k/4k/16k)
../../../../../code/camcalib-exp/.venv/bin/python par_run.py           # 并行编排:顺序训练 + 并行评测(实测更快)
../../../../../code/camcalib-exp/.venv/bin/python eval_group.py g1     # 单组评测(独立进程,输出 checkpoints/eval_cache/)
../../../../../code/camcalib-exp/.venv/bin/python merge_results.py     # 合并 eval_cache → results.json + 出图
```

注意:本机 CPU 上小张量 torch 多线程会 barrier thrash(>4 线程慢 ~20 倍),评测池若用 `multiprocessing` fork 会因 torch 线程池锁死锁——评测务必用独立进程(`eval_group.py`),每进程 `torch.set_num_threads(1)`。本实验最终用 `par_run.py`(训练顺序 4 线程)+ 7 个 `eval_group.py` 进程并行评测完成。

产出:
- `figures/fig_b2a1_yaw_generalization.png` — B2a:19-yaw 泛化曲线(T1/T2/DENSE 三面板 × Base/Base+位姿/Cam)
- `figures/fig_b2a2_sample_complexity.png` — B2a:样本复杂度(T2 协议,1k/4k/4.5k/16k 演示,19-yaw 平均成功率)
- `figures/fig_b2b_cross_embodiment.png` — B2b:2×3 柱状(成功率/终距 × E1/E2/E3)
- `figures/fig_b2b_cam_flatness.png` — B2b 补充:Cam 头跨本体 19-yaw 曲线
- `results.json` — 全部数值(每 yaw 成功率/终距 ± 标准误,n=50/点)
- `run_log.txt` — 运行日志

## 文件

| 文件 | 职责 |
|---|---|
| `blob_world.py` `models.py` `train.py` `evaluate.py` | 实验 B 原样拷贝的世界/模型/训练/评测(协议基准,未改动) |
| `models_pose.py` | `PolicyX`:同架构 CNN,可拼接额外条件输入(位姿/本体标签);extra=0 时与实验 B 的 Policy 逐位同权 |
| `b2.py` | 本体(mount)参数化外参/渲染/位姿向量、数据集、训练(4 头)、闭环评测 |
| `run_b2.py` | 编排 B2a/B2b,保存 results.json 与图 |

## 协议

全部超参与实验 B 一致(15 epoch, lr 1e-3, batch 256, Adam wd 1e-5, 训练 seed=1, 评测 n=50/点,闭环逐帧重规划 K=1),只改输入/输出条件:

**B2a(单本体 E1 = 原始相机,位姿输入 = R_bc(9) + t_b/2.5(3))**:
- T1:yaw=0 训练,5000 eps(实验 B T1 协议)。**注意:位姿输入在训练中恒定,不携带任何学习信号**——这是退化的对照,如实报告;
- T2:yaw∈{-30,0,30} 训练,4500 eps(实验 B T2 协议),位姿输入有 3 个值;
- DENSE:yaw~U(-45,45) 连续训练,4500 eps,位姿输入稠密;
- SCAN:T2 协议下 {1k,4k,16k} eps(1k/4k 是 4.5k 流的嵌套前缀),看样本复杂度。

**B2b(跨本体)**:本体 = 手眼安装几何(mount)。E1(h=0.6,d=2.5,即实验 B 相机)、E2(h=1.0,d=2.2)、E3(h=0.8,d=2.35,未见)。yaw=0 单视角,5000 eps 两本体 50/50 交替混合训练,四条件:
- `cam`:相机系头,无本体标签(执行时用各本体自己的 GT 外参合成);
- `base_pose`:基座系头 + 位姿输入(位姿编码了本体);
- `base_pose_label`:基座系头 + 位姿 + 本体 one-hot 标签;
- `base_ref`:基座系头,无位姿无标签(参考地板)。

评测:19-yaw 网格 × E1/E2/E3,执行用 GT 外参;E3 的标签输入为 OOD(零向量)。

## 设计要点

- **相机系表示跨本体"免费"的机制**:相机系 delta 目标是 R_cb(视角,本体) @ Δp_b,即"物理动作在该相机的轴系下的表达"——图像与该目标都在同一相机的轴系里,映射 `图像→相机系delta` 与本体无关;执行时旋转合成用各本体自己的外参。因此 Cam 头混合训练不需要本体标签,且对未见本体(E3)也应泛化。
- **基座系表示需要知道本体**:同样的图像,不同 mount 对应不同的基座系 delta;位姿输入给了全部信息(12 维,含 mount),标签在数学上被位姿蕴含——B2b(iii) 检验标签是否在实践中补充位姿。
- 单视角训练下位姿输入恒定(B2a-T1),网络无法学习位姿依赖——该条件检验"位姿输入存在但不携带信息"时是否仍能救 Base,预期 ≈ Base 基线,不作为"表示几何差异"的证据。
