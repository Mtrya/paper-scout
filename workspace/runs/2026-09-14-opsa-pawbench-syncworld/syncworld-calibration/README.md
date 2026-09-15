# SyncWorld 标定×世界模型线程(2609.09155,UMass)

SyncWorld 把"相机外参×动作表示"战线(本实例 08-21/08-24)推进到世界模型侧:不估外参、不学回归头,而是把一段**视觉标定交互**作为上下文喂给世界模型,in-context 解析"动作→像素运动"的 setup 专有映射。这是外参来源谱系(AX=XB → 探索运动 → 滤波+可观性管理 → 学习式回归头)的第五种形态:**标定不进参数,进上下文**。

## 论文核心主张

- 问题:数值动作不是像素空间通用语言——相机、基座、本体都改变同一动作的视觉表现;混训=冲突监督,部署=映射漂移。
- 方法:标定 episode(±6 DoF 各一方向,论文 12 段)→ 顺序前缀;C_s 显式给出映射;训练期坐标增强(符号翻转/轴置换/平移缩放,det S=+1)防止背死约定;标定蒸馏(teacher=带标定,student=null 标定)使部署无标定时用交互历史近似映射。
- 结果:LIBERO/ManiSkill/真机(xArm 未见过本体)PSNR/SSIM/LPIPS/FID 大幅超过 IRASim/WorldGym/Ctrl-World(w/ Calib PSNR 28.3/27.0/29.2 vs 基线 21.8–25.2);Met3r 跨视角一致性逼近 oracle;GPC-Rank 测试时策略改进(oracle headroom 三个任务上 0.52/0.56/0.48 → 0.58/0.72/0.60,即 +0.06~+0.16,逼近真实模拟器 oracle 的 0.60/0.80/0.66)。w/o Calib 与 w/ Calib 差距小——历史即隐式标定。

## 代码审计发现(发布 vs 论文)

**github.com/UMass-Embodied-AGI/SyncWorld(19★)+ yyuncong/SyncWorld 权重 + yyuncong/SyncWorld-Evaluation。仓库 docs/paper_differences.md 坦承四处分歧:**

1. **主干换了**:论文 Wan2.2-TI2V-5B → 发布 Cosmos3-Nano(许可统一)。因此论文主表数字无法用发布权重复现——发布权重是"同方法、不同主干"的另一个模型。
2. **动作单位**:论文米+度 → 发布厘米+度(平移量级差 100×,喂错单位即失效)。
3. **条件方式**:论文跨注意(视频 attend 动作)→ 发布 MoT 联合自注意(动作成一等 token 流,双向可见,同栈可做逆动力学);动作头是唯一新初始化参数。
4. **标定段数**:论文 12 段(±6DoF)→ 发布 6 段(每 DoF 一槽,正方向,按 body-frame 打分选取——坐标增强会置换轴,必须全局 12 候选打分)。

- 标定构造细节:夹爪开合在**未录制**warmup 完成(避免伪线索);扫动符号/幅度随机化+move_range.pkl 记录实现方向(反记忆化);每轴 out-停顿-back-停顿 以留干净边界。
- 评估脚本(eval_gripperhead_fdm_rollout.py):全 episode 自回归 rollout(16 帧/chunk),round 0 teacher-forcing,后续轮用自生帧闭环;N=2 默认暴露一步误差累积;快速采样预设(-0.19dB PSNR 换 3× 速度)与论文口径参数(--num-steps 35 --action-cfg-scale 5.0)分开注明。
- 环境门槛:CUDA≥12.8、~60GB 环境(uv)、Wan2.2 VAE + Cosmos3-Nano + Qwen3VL assets;评测集按 episode 存 calibration/expert mp4。

## 与既有主线的接口

- 与 08-21 判据互补:相机系动作的价值=构造性解耦(策略侧);SyncWorld=世界模型侧的同构解耦。两者都绕开显式外参。
- 空白点:标定是一次性 episode,无"热维护"——相机中途漂移只能靠历史隐式吸收,没有显式滤波/重标定通道(与 FastCal 的滤波热维护形成对照)。
- 与本轮 PAWBench 的对照:SyncWorld 用 PSNR/SSIM/LPIPS 单条轨迹指标评估——按 PAWBench 口径,它同样没测结局分布。

## 研究动作与结论口径

- 纯代码审计线程(权重可下载但全量 cosmos_framework 环境 ~60GB + CUDA12.8 门槛,巡航预算内不上模型实测;如实注明)。
- 审计已产出可入报告的硬结论:发布权重 ≠ 论文模型(主干替换),引用数字必须分清口径。
