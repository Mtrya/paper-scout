# ShadowDancer (arXiv:2607.28362) — 线程记录

## 这条线做了什么

检验论文核心论断 "regularization does not substitute for pairing"(Table 5 的
机制探针)在最小可控玩具尺度是否成立,并追问配对的收益到底是信息上的还是
优化上的。

## code/

- `sprites_probe.py`:双球 sprites 实验。48×48 两球(一静一匀速),影子对共享
  位姿/速度、重采样颜色/背景/半径;三变体(自重建 β=0.01 / 自重建 β=1.0 /
  cross-shadow β=0.01)同架构(z=32);冻结 z 上做岭回归探针(动球色/静球色/
  方向)+ cross/self MSE 比。本机 CPU venv(`code/scout-exp`),约 35 分钟跑完。
- `shadow_sprites_results.json`:三变体全部探针数字(2026-08-02 运行)。
- `make_figs.py`:读 JSON 画报告图(资产在 `../../assets/`)。

## 结果要点

- B(配对):方向 R²=0.87、泄漏最低(0.07/0.01)——最干净的潜在空间。
- A(自重建 β=0.01):方向 0.84、泄漏小(0.12/0.05);玩具外观太简单,
  论文的大幅外观泄漏差距未复现。
- Ahb(β=1.0):解码器层面坍缩(MSE=模糊复制基线 0.0105),但 z 仍可被外部
  线性读出(方向 0.61、颜色 0.25)——正则断开通道而非纯化信息。
- 坍缩经济学:z 存活需重建收益 > β×KL;前四次尝试因模糊复制太便宜
  (吸收态)全部坍缩,最终有效的组合是方向预热(前 600 步,特权信息,
  三变体共享)+ 提高运动幅度(8-14 px/帧)使收益反超。

## 复现

```bash
cd <workspace>/code/scout-exp  # 或任何带 torch/numpy/matplotlib 的 py3
python sprites_probe.py   # 生成数据+训练+探针,写 shadow_sprites_results.json
python make_figs.py       # 生成 sprites_probe_bars.png / sprites_pairs_example.png
```

局限:玩具尺度、单 seed、预热使用玩具特权信息(对三变体相同,不构成偏置)。
下一步:给 sprites 加多种纹理外观,使"以貌索引动作"有利可图,检验 A 泄漏
是否放大而 B 保持干净。
