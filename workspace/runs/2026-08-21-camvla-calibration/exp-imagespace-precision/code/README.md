# 实验代码副本

`precision.py` — 全部闭式公式(横向/立体深度/平面/旋转/标定传播)+ Monte Carlo 验证器 + 数字表
输出(`numbers.json` 为数字留痕,其中 `mc` 字段为闭式 vs MC 偏差,全部 <5%)。

`figs.py` — 五张图件,输出到 `figures/` 并 PIL 按非白像素 bbox 裁白边(dpi=180):

| 图件 | 内容 |
|---|---|
| `fig1.png` | 横向精度 vs 深度:1 源像素 / 1 输入像素(224²) |
| `fig2.png` | 立体深度精度 vs 深度 × 视差误差三档(对数轴) |
| `fig3.png` | **核心**:3D 定位误差 (X,Z) 切面地图,<2mm/1cm 边界(三台立体 + 单目+已知平面) |
| `fig4.png` | 旋转可观性:两点平面内误差 vs 间距;出平面倾斜误差 vs 深度 |
| `fig5.png` | 标定传播:(a) 基线/焦距 bias (b) 光轴旋转 (c) 绝对定位 vs 相机系 delta 对照 |

运行(在 `code/camcalib-exp/imagespace-precision/`,venv 为 `../.venv`):

```bash
../.venv/bin/python precision.py
../.venv/bin/python figs.py
```

无外部依赖(numpy / matplotlib / pillow 已在 venv)。所有数字纯解析 + MC,无真实数据。
