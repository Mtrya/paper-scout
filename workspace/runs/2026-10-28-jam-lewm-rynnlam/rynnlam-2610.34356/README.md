# RynnLAM 真实权重解耦探针(2610.34356 / RynnVLA-Latent)

## 问题

RynnLAM 把帧对编码成 608 维描述符:z^m(8×64 运动 token)、z^a(64 全局动作潜变量)、
z^c(32 相机潜变量)。论文用 GRL(梯度反转)把相机自运动从 z^a 中驱逐出去,
但明确声明只有 z^a 受解耦约束,z^m 与整个描述符"不假定相机不变"。
本探针用发布的 RynnLAM 权重直接测量:各类相机运动下,三个分量各自动多少?

## 设置

- 权重:HF `Alibaba-DAMO-Academy/RynnLAM`(RynnLAM.pt,1.46GB,sha 未钉;fp32 CPU 推理)。
- 数据:HF `physical-intelligence/libero` chunk-000 前 6 集,每集 3 个帧对(t, t+4),
  共 18 对;图像缩放到 224×224,与推理 API 约定一致。
- 条件:identity(不变)、content_later(同集再晚 4 帧的真实运动)、
  content_other(同集远处的另一帧)、合成相机运动(平移 8–24px、旋转 ±5–10°、缩放 1.1–1.25×)。
- 指标:Δ = ||z_cond − z_base|| / ||z_base||,按 z^m / z^a / z^c 三个分量分别报告。

## 结果(18 对均值)

| 条件 | Δz^m | Δz^a | Δz^c |
|---|---|---|---|
| identity | 0 | 0 | 0 |
| content_later(+4 帧真实运动) | 0.087 | 0.020 | 0.038 |
| content_other(另一帧) | 0.388 | 0.085 | 0.184 |
| 平移 8–24px | 0.341 | 0.166 | 0.249 |
| 旋转 ±5–10° | 0.404 | 0.092 | 0.224 |
| 缩放 1.1–1.25× | 0.469 | 0.045 | 0.114 |

(逐条件数值见 `code/decoupling_results_summary.json`;图为 `code/decoupling_fig.png`。)

## 读法

- **运动 token z^m 被相机运动强烈污染**:平移/旋转/缩放引起的 Δz^m(0.34–0.47)
  不低于、甚至高于换一帧真实内容的 Δz^m(0.388)。论文自己的 LARY 消融也显示 z^m
  单独做控制回归最好(0.54),z^a 单独最差(0.93)——两个事实放在一起说明:
  控制预测性最强的分量恰好是相机最敏感的分量。
- **z^a 的解耦是不均匀的**:缩放时最稳(0.045,远低于 content_other 0.085),
  但平移 24px 时 Δz^a 达 0.257,是 content_other 的 3 倍。GRL 并没有给 z^a
  换来普遍的视点鲁棒性,这与论文 LIBERO-Plus 相机扰动档弱于 π0.5 一致。
- **z^c 不是纯相机通道**:content_other 也使它移动 0.184,内容信息同样泄漏进去。
- 合成相机运动是屏幕空间变换,不含视差/遮挡变化,真机连续 ego-motion 的污染
  只会更复杂;因此这些数字是解耦强度的**下界证据**(合成都这么敏感),
  不是上界。

## 复跑

```bash
# 依赖:torch(CPU 即可)、pandas、pyarrow、Pillow、opencv-python-headless、safetensors、einops、timm
# 数据:code/rynnlam-probe/data/episode_00000{0..5}.parquet(HF physical-intelligence/libero chunk-000)
# 权重:code/models/RynnLAM.pt(HF Alibaba-DAMO-Academy/RynnLAM)
cd code/rynnlam-probe && python probe_decoupling.py --device cpu --episodes 6 --pairs-per-ep 3
python plot_decoupling.py
```

仓库:`github.com/alibaba-damo-academy/RynnVLA-Latent`(rynnlam.inference.RynnLAMEncoder)。
原始数据与权重不进运行包(体积);上面两条下载路径即可重建。
