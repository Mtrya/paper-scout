# Invisible Shortcuts (2608.05424) — 三真实编码器的元数据痕迹实测

论文:CTU Prague + 大阪大学。视觉编码器从预训练 metadata–语义相关性学到像素级不可见捷径(JPEG/EXIF/曝光);诊断指标 MP(痕迹可解码性)与 SPD(语义预测翻转);缓解可泛化到未见属性;敏感性双刃剑(生成图检测)。

## 我们的研究动作

论文评测矩阵未覆盖 SigLIP(当下 VLA 视觉塔热门选项),我们按其诊断框架实测三个真实编码器(SigLIP-SO400M / CLIP-B/16 / DINOv2-B 对照):

- 数据:imagenette 验证集 500 图(10 类),每图**随机指派**一档 JPEG 质量(30/50/70/85/95)重压缩——随机指派物理摧毁质量–语义相关性,MP 因此量到纯痕迹可解码性。
- MP(70/30 逻辑回归):SigLIP 56.7%、CLIP 52.7%、DINOv2 18.0%(随机基线 20%)。
- 位移:同图 q95↔q30 特征距离 / 平均图间距离:SigLIP 10.2%、CLIP 11.2%、DINOv2 3.3%。
- SPD(kNN 语义预测翻转):三编码器均 ≈0(0.2–2.0%),随机指派下的机制性预期。

## 核心发现

- 论文"语义监督诱导元数据敏感"的机制声明在论文未测的 SigLIP 上成立;语言监督 vs 自监督的梯度(57/53/18)干净利落。
- 敏感性与伤害解耦:痕迹在(MP 高)但无相关性时不翻语义(SPD≈0)——论文核心论点的最小实证。
- 实用:SigLIP/CLIP 系视觉塔的特征里躺着采集管线指纹,换相机/压缩即是一笔隐藏分布偏移债;DINOv2 在此轴干净。

## 内容

- `code/shortcut_probe.py`:MP/SPD/位移探针(支持 `ENCODER_ROOT` 本地权重)
- `code/plot_shortcut.py`:报告图绘制
- `code/results/`:shortcut_siglip.json、shortcut_clipdinov2.json
- 运行环境:启智 notebook `rynnvalue-probe`(同 RynnValue 线程),编码器权重经 ModelScope/hf-mirror 下载
