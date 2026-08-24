# 实验:OpenVLA-7B × LIBERO 相机视角扰动分解(exp-openvla-view)

> 本轮巡航"真实验":在真实 OpenVLA-7B 权重 + 真实 LIBERO-Spatial 基准上独立复核 CamVLA(2607.05396)报告的"相机绕基座旋转 15° 成功率崩塌"现象,并做论文没做的分解实验:**视角崩溃里,视觉编码漂移与动作参考系失配各占多少?**(待实验完成后填充数据)

## 状态

- [x] 环境准备(venv/LIBERO/openvla/权重)
- [x] baseline 冒烟(85%,与公开一致)
- [x] ±15° rescue 符号探针
- [x] 全量 θ∈{0,±5,±10,±15} × {raw, rescue}(每条件 10-20 集,见统计口径)
- [x] 结果回传与绘图

## 结果(2026-08-21 最终)

合并 main+_A+_B 三份 JSON 去重后共 **222 条逐集记录**。任务:task0 "pick up the black bowl between the plate and the ramekin and place it on the plate";task1 "pick up the black bowl next to the ramekin and place it on the plate"。成功率为该 (任务,θ,模式) 下成功集数/总集数。

### 合并成功率(两任务)

| θ | raw | rescue s=+1 | rescue s=-1 |
|---|---|---|---|
| 0 | **85%**(40) | = baseline | — |
| +5 | 45%(20) | 45%(20) | — |
| +10 | 24%(17) | 50%(20) | — |
| +15 | 0%(10) | 0%(15) | 0%(10) |
| -5 | 70%(10) | 30%(10) | — |
| -10 | 10%(10) | 0%(10) | — |
| -15 | 0%(10) | 0%(10) | 10%(10) |

### 逐任务分解(关键)

| task | θ | raw | rescue s=+1 |
|---|---|---|---|
| t0 | +5 | 80%(10) | 90%(10) |
| t0 | +10 | 40%(10) | **100%(10)** |
| t0 | +15 | 0%(5) | 0%(10) |
| t0 | -5 | 100%(5) | 60%(5) |
| t1 | +5 | 10%(10) | 0%(10) |
| t1 | +10 | 0%(7) | 0%(10) |
| t1 | +15 | 0%(5) | 0%(5) |

统计口径:baseline 与正角度 ±5/±10 为 20 集/条件(10 集/任务);负角度与 ±15(含探针)为 10 集/条件(5 集/任务);±15 有探针双符号。n 为该格集数。

## 结论

1. **崩溃复现**:OpenVLA-7B 在 LIBERO-Spatial 上相机绕基座旋转 |θ|≥10° 显著崩溃(±15° 全部 0%,±10° 0-40%),θ=0 为 85%(与公开 ~84.7% 一致)。现象与 CamVLA 在 π0/RLBench 的报告同向,跨模型跨基准成立。
2. **分解结果(主结论)**:动作系补偿(rescue = R_z(±θ)·预测)仅在 **task0 的中等角度(+10°)显著恢复**(100% vs raw 40%),+5° 无差,±15° 完全失效;**task1 在所有角度均不可恢复**。→ 视角崩溃是**视觉编码漂移为主、动作参考系失配为辅**的混合体:中等角度存在可恢复的"动作系失配"分量,大角度与几何信息较弱的任务上被视觉 OOD 主导。"相机系动作 + 外参估计"路线能救的只是其中一部分,且对任务几何敏感。
3. **符号**:rescue 理论符号 s=+1 在 task0/+10° 有效;±15° 双符号均失效(噪声级 0-10%)。
4. **任务敏感性**:task1(+5° raw 仅 10%)比 task0(+5° 80%)脆弱得多——同任务族但碗位不同,提示崩溃幅度与任务几何/目标位置强相关,单一数字报告成功率掩盖了这个维度。
5. **预处理交叉验证**:TF 预处理 baseline 85%(20 集/任务)vs PIL 预处理 θ=0 锚点 100%(10 集/任务),差异在采样噪声内;两条 pipeline 均可复现高基线。

图:`code/results/success_vs_theta.png`(合并曲线)、`code/results/success_vs_theta_per_task.png`(逐任务);数据:`code/results/viewprobe_merged_final.json`。

## 远端环境

- Notebook:`openvla-viewprobe`(1×4090, 4090-cuda12.8-2 组, ngc-pytorch:25.02, CUDA 12.8, torch 2.7.0a0+cu128, py3.12)
- Workroot:`/inspire/qb-ilm/project/cq-scientific-cooperation-zone/ky26021/embodied-research/openvla-viewprobe/`
  - `.venv/`(system-site-packages 继承 NGC torch;numpy 1.26.4;mujoco 3.9.0;robosuite 1.4.1;bddl 1.0.1;transformers 4.40.1;tensorflow-cpu 2.16.1)
  - `models/openvla-7b-finetuned-libero-spatial/`(权重,4 分片 ~19.7GB + 配置/tokenizer)
  - `code/`(脚本),`code/results/`(JSON),`code/logs/`(日志)
- 权重来源:hf-mirror(pget2 并行 range 下载,幂等续传)

## 实验设计

条件:
- **baseline**(θ=0):相机不动,验证 pipeline 与公开成功率同量级
- **raw**(θ∈{±5,±10,±15}):整集静态相机扰动——相机绕基座竖直轴(基座 xy=robot0_base)整体旋转 θ,回合 reset+set_init_state 后设定
- **rescue**:同 raw,但把 OpenVLA 输出的 7 维 EE delta 动作平移+axis-angle 旋转分量用 Rz(±θ) 补偿(符号先探),夹爪不变

推导:相机绕基座 z 转 +θ ⇔ 场景在视野中绕该轴转 -θ(轨道相机=场景反向旋转,可解析验证);策略若"跟着场景几何走",输出 = R_z(-θ)·真实动作,故 rescue = R_z(+θ)·预测。

几何核验:`cam_info.json`(默认相机位姿、基座 xy)。mujoco≥3 坑:相机位姿渲染读 `data.cam_xpos/cam_xmat`,改 `model.cam_*` 后必须 `sim.forward()`。

任务:LIBERO-Spatial task 0/1;baseline 每任务 20 集;sweep 因推理耗时(7B 单步 ~0.6s)调整为每任务 10 集(每条件 20 集),README 结论处注明统计口径;每集记录 success/steps/final eef pose。

## 进度快照(2026-08-21 续跑)

- baseline(θ=0,TF 预处理,20 集/任务):**34/40 = 85%**(task0 17/20, task1 17/20),与 OpenVLA 公开 LIBERO-Spatial ~84.7% 一致
- probe(±15°,PIL 预处理,5 集/任务):
  - raw ±15°:0/10(全部 220 步耗尽)→ **CamVLA 崩溃现象复现**
  - rescue Rz(+θ) 与 Rz(-θ) 双符号:合计仅 1/20(−15° 一个成功,噪声级)→ **±15° 下动作系补偿无法恢复**,指向视觉编码漂移为主
- sweep(θ∈{±5,±10,±15} × raw/rescue,双 worker,PIL)进行中;rescue 取理论符号 s=+1(±15° 双符号已由探针覆盖)。中途按指令缩减为每条件 10 集(每任务 5 集)保证 θ 网格全覆盖;已完成条件保留 20 集。
- 中间汇总(截至 16:35,174 条记录,合并 main+A+B 去重):

| θ | raw | rescue(s=+1) |
|---|---|---|
| 0 | 85%(40) | (=baseline) |
| +5 | 45%(20) | 45%(20) |
| +10 | 24%(17) | 50%(20) |
| +15 | 0%(2) | 0%(15) |
| -15 | 0%(10) | 0-10%(20) |
| -5/-10 | 待回 | 待回 |

- 早期结论(待最终数据确认):±15° 动作系补偿无法恢复(双符号);+10° 出现中等幅度的恢复信号(rescue 50% vs raw ~24%),+5° 无差异——分解可能是混合的:中等角度部分失败来自动作参考系失配(可恢复),大角度被视觉 OOD 主导(不可恢复)。
- 已修:双 worker 并发写同一 JSON 互踩 → 每 worker 独立 JSON(`*_A.json`/`*_B.json`);rescue 动作变换经 debug 打印验证正确(Rz(+15°) 手算一致)
- **已完成(2026-08-21 深夜收尾)**:sweep 全部跑完,结果与结论见下节。

## 结果(实验 C,2026-08-21 最终)

合并 main+_A+_B 三份 JSON 去重后共 **222 条逐集记录**。任务:task0 "pick up the black bowl between the plate and the ramekin and place it on the plate";task1 "pick up the black bowl next to the ramekin and place it on the plate"。

### 合并成功率(两任务)

| θ | raw | rescue s=+1 | rescue s=-1 |
|---|---|---|---|
| 0 | **85%**(40) | = baseline | — |
| +5 | 45%(20) | 45%(20) | — |
| +10 | 24%(17) | 50%(20) | — |
| +15 | 0%(10) | 0%(15) | 0%(10) |
| -5 | 70%(10) | 30%(10) | — |
| -10 | 10%(10) | 0%(10) | — |
| -15 | 0%(10) | 0%(10) | 10%(10) |

### 逐任务分解(关键)

| task | θ | raw | rescue s=+1 |
|---|---|---|---|
| t0 | +5 | 80%(10) | 90%(10) |
| t0 | +10 | 40%(10) | **100%(10)** |
| t0 | +15 | 0%(5) | 0%(10) |
| t0 | -5 | 100%(5) | 60%(5) |
| t1 | +5 | 10%(10) | 0%(10) |
| t1 | +10 | 0%(7) | 0%(10) |
| t1 | +15 | 0%(5) | 0%(5) |

统计口径:baseline 与正角度 ±5/±10 为 20 集/条件(10 集/任务);负角度与 ±15(含探针)为 10 集/条件(5 集/任务);±15 有探针双符号。n 为该格集数。

## 结论(实验 C)

1. **崩溃复现**:OpenVLA-7B 在 LIBERO-Spatial 上相机绕基座旋转 |θ|≥10° 显著崩溃(±15° 全部 0%,±10° 0-40%),θ=0 为 85%(与公开 ~84.7% 一致)。现象与 CamVLA 在 π0/RLBench 的报告同向,跨模型跨基准成立。
2. **分解结果(主结论)**:动作系补偿(rescue = R_z(±θ)·预测)仅在 **task0 的中等角度(+10°)显著恢复**(100% vs raw 40%),+5° 无差,±15° 完全失效;**task1 在所有角度均不可恢复**。→ 视角崩溃是**视觉编码漂移为主、动作参考系失配为辅**的混合体:中等角度存在可恢复的"动作系失配"分量,大角度与几何信息较弱的任务上被视觉 OOD 主导。"相机系动作 + 外参估计"路线能救的只是其中一部分,且对任务几何敏感。
3. **符号**:rescue 理论符号 s=+1 在 task0/+10° 有效;±15° 双符号均失效(噪声级 0-10%)。
4. **任务敏感性**:task1(+5° raw 仅 10%)比 task0(+5° 80%)脆弱得多——同任务族但碗位不同,提示崩溃幅度与任务几何/目标位置强相关,单一数字报告成功率掩盖了这个维度。
5. **预处理交叉验证**:TF 预处理 baseline 85%(20 集/任务)vs PIL 预处理 θ=0 锚点 100%(10 集/任务),差异在采样噪声内;两条 pipeline 均可复现高基线。

图:`code/results/success_vs_theta.png`(合并曲线)、`code/results/success_vs_theta_per_task.png`(逐任务);数据:`code/results/viewprobe_merged_final.json`。

---

## 实验 E:焦距(内参)扰动(2026-08-24)

**想验证什么**:用户假设"内参偏移对 VLA 的影响是温和的"——与实验 C 的外参(位姿)扰动相比,相机内参(焦距)扰动是否显著温和?并做论文没做的分解:真实内参偏移(fovy)vs"沉默预处理失配"(crop+resize,数据集管线真实形态)是否等价。

**怎么设**:在实验 C 同管线(OpenVLA-7B-finetuned-libero-spatial × LIBERO-Spatial task 0/1,每条件每任务 10 集,PIL 预处理,220 步上限)上:
- **fovy 扰动**:agentview 相机默认 fovy=45°(代码实测;腕部相机 75°)。`cam_fovy *= (1+Δ%)`,Δ∈{±2.5,±5,±10},整集静态(同实验 C 的 reset 后设定 + `sim.forward()`;已实测往返 restore 像素级一致)。等效焦距偏差按 **f_eff = h/(2·tan(fovy/2)) 换算**:Δfovy=+10% → 等效焦距 **−10.15%**(视野变宽,物体变小)。
- **预处理对照**:不动相机,渲染图 center-crop 5% 再 bilinear 缩放回原分辨率 → 等效焦距 **+5.26%**(视野变窄、物体变大),与 fovy −5%(等效焦距 +5.83%)对照。
- fovy=0 用实验 C 的 PIL 锚点(20/20=100%,10 集/任务)。
- 双 worker(`run_focal.sh`,独立 JSON);`analyze_focal.py` 出 summary 与曲线。

**看到什么**(每格 20 集=2 任务×10,锚点 20 集):

| 等效焦距偏差 % | 对应设置 | SR(合并) |
|---|---|---|
| 0 | fovy 0(PIL 锚点) | 100%(20) |
| ±2.7 | fovy ±2.5% | 95% / 85% |
| ±5.3 / +5.8 | fovy ±5% | 75% / 70% |
| −10.2 / +12.3 | fovy ±10% | 60% / 70% |
| **+5.26** | **crop5% 对照** | **90%(20)** |

逐任务(各 10 集):task0 0%→100%,+2.5%→90%,+5%→90%,+10%→40%,−2.5%→70%,−5%→70%,−10%→70%,crop5→90%;task1 0%→100%,+2.5%→100%,+5%→60%,+10%→80%,−2.5%→70%,−5%→70%,−10%→70%,crop5→90%。

**结论**:
1. **"内参偏移温和"成立(与实验 C 对比)**:焦距 ±10%(等效 ±10~12%)仍保 60-70% 成功率;而实验 C 中外参旋转 ±10° 已崩到 0-40%(合并 10-24%)。**同量级扰动,外参位姿损伤远大于内参焦距**——策略对"变焦/缩放"远比"视角旋转"稳健(部分可归因于 OpenVLA 训练数据里天然的尺度/视距多样性,而旋转缺乏)。
2. **fovy 扰动方向不对称轻微**:+10%(视野变宽)→60%,−10%(变窄)→70%;+5%→75% vs −5%→70%,方向效应在 n=20 的噪声内,不显著。
3. **等价性对照(crop vs fovy)**:crop5%(90%)vs 等效焦距相当的 fovy −5%(70%)——**crop 通道更温和 ~20pt**,但 n=20 的 95% CI(±19%)重叠,**差异未达显著**;方向性提示"预处理侧裁切缩放"比"真实投影内参变化"对策略更友好(裁剪保留原始内容只重采样,而 fovy 改变投影几何),但需更大样本确认。
4. **任务敏感性弱于实验 C**:task0/task1 在焦距扰动下差距远小于视角扰动(±5% 两任务 60-90%),说明焦距扰动的破坏机制与任务几何相关性较低。

**置信度边界**:每格 n=10-20,SR 的 95% CI 约 ±19-31%;crop-vs-fovy 的差异(~20pt)未达显著,只能视为方向性证据;fovy 是 MuJoCo 理想 pinhole 投影,真实相机(畸变/主点偏移)未覆盖。

产物:`code/results/focal-results-{A,B}.json`(140 条逐集)、`code/results/focal-anchor.json`、`code/results/focal-summary.json`、`code/results/focal-success-vs-focal.png`;脚本 `code/run_focal.sh`、`code/analyze_focal.py`。

## 故障与修复记录

1. **mujoco 2.3.7 无 cp312 wheel** → 改用 mujoco 3.9.0(cp312 wheel);robosuite 1.4.1 自带 MjSim 包装,兼容。
2. **numpy 被 opencv-python-headless 5.0 升到 2.5.2,破坏 NGC torch numpy 互操作** → opencv 降 <4.11,numpy 锁 1.26.4。
3. **bddl 3.6.0 装错**(LIBERO 用 `bddl.parsing` 1.x API)→ 锁 bddl==1.0.1。
4. **PyOpenGL-EGL import 失败**(容器无 glvnd libEGL.so.1,直接链 nvidia 裸库缺 eglQueryString)→ 从 Ubuntu archive 手工装 glvnd libEGL/libGL/libGLX/libOpenGL/libGLdispatch(apt 无网络)。
5. **torch 2.7 `weights_only=True` 默认值** → patch LIBERO `get_task_init_states` 加 `weights_only=False`。
6. **prismatic eager import 拉 dlimp(训练依赖)** → patch `prismatic/__init__.py` 为惰性 `__getattr__`。
7. **相机扰动不生效(mujoco 3.x 相机位姿在 data.cam_xpos,须 mj_forward 同步)** → `rotate_camera_about_base` 内加 `sim.forward()`。
8. **pget2 卡在已完成分片不退出** → kill 后重启脚本,幂等跳过续传。
9. **TF 图像预处理是每步 CPU 瓶颈(~0.3-0.5s)** → run_viewprobe.py 内 monkey-patch `resize_image`/`get_vla_action` 为 PIL 等价实现(JPEG q75 + LANCZOS;center-crop 0.9 + bilinear),单步 ~0.6s→sweep 提速;sweep 附跑 PIL 版 θ=0 锚点(10 集/任务)消除预处理口径差。
10. **双 worker 并发写同一结果 JSON 互踩(丢失记录)** → probe/sweep 双 worker 各写 `viewprobe_results_{A,B}.json`,分析时合并。

## 复跑方式

```bash
# 1. 起 notebook(1×4090, ngc-pytorch:25.02, 4090-cuda12.8-2, quota 1,10,100)
# 2. 环境(已在 workroot 就绪,或重跑 code/*.sh)
# 3. 评测:
cd /inspire/.../openvla-viewprobe
OPENVLA_ROOT=$W/openvla CUDA_VISIBLE_DEVICES=0 bash code/run_all.sh phase0   # baseline
OPENVLA_ROOT=$W/openvla bash code/run_all.sh probe                           # ±15° 符号探针
OPENVLA_ROOT=$W/openvla RESCUE_SIGN=<探针结论> bash code/run_all.sh sweep    # 全量
```
