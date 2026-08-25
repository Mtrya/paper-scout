# LD4WAM 线程:冻结 LDM 的 OOD 运动探针(实验 B)

论文:LD4WAM (arXiv 2608.22403;官方只放出 LDM 一半:github.com/stubborn111/LD4WAM,
LDM.pt 4.54GB)。问题:论文声称 latent dynamics 是跨本体的运动桥,但全部运动证据是
域内的(RoboTwin 回归、语料内数据集检索)。桥在分布外还剩多少可解码运动?

## 实验设计

- 域:DROID 21 集(ep000-ep020,外视角 320×180@15fps)。双重 OOD:不在 5086h 语料内,
  且语料 76.4% 第一人称人类视频,DROID 为第三人称固定机位。
- 协议:严格复刻论文附录 E——8 帧 clip 产 7 迁移,目标按 0.01m/0.08rad 缩放,
  MLP 探针,train/val 划分;12 维双臂目标换成 DROID 6 维单臂 ΔEE(基座系)。
- 三重对照:
  1. DINOv3 特征差分作过拟合对照(同数据量同协议;train≈1.0 而 LDM train=0.13
     → LDM 低 val 是信息上限而非探针容量)。
  2. 逐维分解定位可解维度。
  3. 幅度对照(v4):||Δxyz||/||Δrot|| 对任意固定逐集外参不变——排除"逐集外参错位"
     这一替代解释。
- 中途诊断:DROID cartesian_position 第 4 维(roll)在 ±π 环绕,np.diff 产生 ±2π
  毛刺(stride1 1.36% / s2 2.71% / s4 9.29% 的迁移),目标方差爆炸(dim3 scaled std
  9.13 vs 其他维 0.15-0.65);剔除 |scaled|>8 迁移后为干净结果。预处理已核验与官方
  逐字一致(模型内部做 ImageNet 归一化,dinov3.py if_normalize_img=True)。

## 结果(probe_results.json)

- 主结果:LDM ΔEE 回归 val R² 全地板(s1 all 0.048 / s1 top50 0.020 / s2 all 0.072 /
  s2 top50 0.068);DINOv3 train≈0.99-1.00、val 全负。
- 逐维:只有 z(垂直)可解,top50 上 0.20(s1)→0.39(s2),随时距上升;其余五维 ≈0。
- 检索(motion bank):same_ep 0.376 vs cross_ep 0.162(随机基准 ≈0.048)→ 外观泄漏。
- 幅度对照判决:all 子集 ||Δxyz|| R²=0.36(靠静止/运动一比特),运动帧内部
  (mov05/top50)||Δxyz|| 与 ||Δrot|| 双双 -0.09~-0.13 → 排除外参错位解释。
- 范数伪影:corr(||z_latent||, |ΔEE|)=-0.61(SoftVQ 对输入与码本均 L2 归一化,
  范数不含幅度信息);corr(||dv3diff||)=+0.51。

## 置信度边界

样本 4.4k-5.4k 迁移 vs 论文域内 676k@50Hz(少两个数量级);但 DINOv3 对照在同样本量
过拟合,信息上限论证成立。措辞应为"OOD 域+该样本量下 latent 运动内容远低于域内",
而非"LDM 无用"。

## 文件

- `code/probe_ldm_droid.py`:远端探针 v4(含幅度对照;cache_s*.npz 在远端,
  本包只收脚本与最终 JSON)。
- `code/probe_results.json`:全部指标(逐维 R²、幅度对照、检索、诊断统计)。
- `code/fig_expb_ldm.py`:绘图(产出 assets/expb_probe.png)。
