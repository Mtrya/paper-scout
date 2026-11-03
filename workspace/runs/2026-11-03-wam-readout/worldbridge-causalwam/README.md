# WorldBridge × CausalWAM:动作头读什么

合成世界实验:一个共享视觉编码器 + 视频预测解码器 + 稠密或 top-k 稀疏动作头。场景里有一个带白色标记的目标和一个同色无标记的干扰物;训练时干扰物紧贴目标,测试时两者拉开。动作是四步走向目标,与干扰物无关。

## 条件

- `base`:仅分布内视频与动作。
- `vid`:加 48 条分布外视频/步,无分布外动作。
- `vid1` / `vid5`:在 `vid` 上再加每任务 1 / 5 条分布外动作演示,动作头微调 80 步(带分布内 replay)。
- 动作头:`dense`(36 token)与 `k=1/2/4/8` 的硬 top-k 路由。

## 结果

- 稠密头:分布外 2.1%,与零动作/随机动作的 1.19%/1.24% 同量级。
- k=1:五个种子里三个分布外 42–49%,两个 4%;好种子选中 token 分散在边缘,坏种子挤在中心。
- 加分布外视频把好种子抬到 47–66%,坏种子仍 4.2–4.7%;模式比例不变。
- 一条/五条演示让 k=1 的分布内掉到 40–60%/32.3%,分布外 15.2%/14.4%。
- 留出样本的 cue probe 47.1%(机会 50%);选中 token 的目标位置读出 R² 分布内 0.96–0.99,分布外 −0.09/0.22。

## 复现

```bash
python synth_wam.py --seed 0 --topk 1 --train_steps 1500 --out out/grid/base_k1_s0.json
python run_grid.py main      # 28 runs: base/vid/vid1 各 8,vid5 4
python run_grid.py k1seeds   # k=1 的 seeds 2-4
python plot_grid.py
python cue_probe.py --ckpt <model.pt> --out out/cue_probe.json
```

模型 checkpoint 不保留(可再生,且体积不符合运行包预算)。`out/grid_raw.csv` 是逐 run 指标,`out/grid_summary.csv` 是均值与种子范围。
