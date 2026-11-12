# SLA：空间监督怎样进入动作策略

## 研究问题

《SLA: Scaling Spatial-Language-Action Model via Continual Pre-training》将 Spatial Forcing 的三维表征对齐从任务后训练扩展到持续预训练，并在两个阶段保留空间监督。本线程追问：这种训练期几何教师改变了策略的哪一部分；公开实现怎样处理离散状态；实际策略对视觉和本体状态的扰动作何响应。

论文缓存：`papers/vla/sla-2611.08373.md`。官方代码快照为 OpenHelix-Team/Spatial-Forcing 的 `405ca88`，实现位于 `SLA/`。基础权重为 `haofuly/SLA_base`，LIBERO 后训练权重为 `haofuly/SLA_benchmark_ckpts` 中的 `SLA_libero_plus_ckpt`。

## 代码路径得到的机制判断

- π0.5 将归一化机器人状态离散为 256 个 bin，写入 `Task: ..., State: ...; Action:` 文本提示。`StateDropout` 设置状态掩码，`TokenizePrompt` 据此移除整段状态文本；数值置零是不同的反事实。
- SLA 将多层视觉词元经共享 LayerNorm 与 MLP 投影，使用逐词元、逐通道门控融合，再与同视角、同图像位置的 OmniVGGT 特征做 cosine 对齐。总目标是动作损失加 α 倍空间对齐损失。教师、投影与门控仅在训练期出现。
- 当前公开 `pretrain.py` 与下游配置均设置 `vla_adapt_layers_align=False`、`vla_layers_align=12`。论文报告了自适应层融合与固定层的比较，公开默认配置则固定第 12 层；两者不能混称。
- 动作输出经过 `AbsoluteActions` 时会使用当前状态重建绝对动作。探针将 `_output_transform` 设为恒等，在反归一化和绝对动作重建之前比较模型动作，避免把输出变换中的状态通路误算成网络敏感性。

## 真实权重反事实协议

探针 `code/sla_counterfactual.py` 对同一输入分别执行原样、状态置零、状态跨样本置换、状态词元删除、图像置黑与图像跨样本置换。每个条件的流匹配噪声完全相同，比较归一化模型动作前 active action dimensions 的 RMSE、相对 RMSE 与余弦相似度。相对 RMSE 的分母是原始动作 RMS。

正式实验包括：

1. `SLA_base` 在官方真实机器人 `handover_vege1` 样本上取 64 帧，使用 `SLA_base` 的归一化统计量，比较前 14 个动作通道。
2. `SLA_base` 在 LIBERO 单集取同样的 64 帧，使用 LIBERO 后训练权重的归一化统计量，比较前 7 个动作通道。
3. `SLA_libero_plus_ckpt` 对 LIBERO 同帧、同噪声重复上述测量，再用 `code/compare_counterfactuals.py` 计算后训练减基础权重的配对差值。

LIBERO 对比描述从经空间对齐的 SLA 基础权重到任务后训练权重的输入敏感性变化；两个权重之间包含完整后训练过程，差值不归因于空间对齐或状态 dropout 单项。所有结果描述输入敏感性与动作通路，不估计任务成功率。

实验使用固定种子 20261112。探针首先断言状态词元删除后的提示词元数严格小于原始提示，并将实际计数写入 `tokenization_check.json`。逐模型 `actions.npz` 是原始输出，只留在远程临时空间，不进入证据包。

## 实际结果

三组实验均完成 64 帧、五种扰动与原始输入的配对推理。下表为相对 RMSE 的样本均值；分母是每帧原始动作的 RMS，因此数值可以大于 1。所有比较都在反归一化与绝对动作重建之前进行。

| 数据与权重 | 状态置零 | 状态置换 | 状态词元删除 | 图像置黑 | 图像置换 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 真实机器人，SLA_base | 0.7031 | 0.5657 | 0.7010 | 0.6263 | 0.8849 |
| LIBERO，SLA_base | 0.3628 | 0.2266 | 0.1339 | 0.4201 | 1.2202 |
| LIBERO，后训练权重 | 0.0693 | 0.0797 | 0.0294 | 0.7495 | 1.4863 |

LIBERO 后训练减基础权重的配对差值依次为 -0.2935、-0.1470、-0.1046、+0.3294、+0.2661，五项配对 bootstrap 95% 区间均不跨零。这描述完整任务后训练配方的输入敏感性变化：在该单集样本中，状态通路的响应减弱，视觉通路的响应增强。公开 `pretrain.py` 与 LIBERO 后训练配置的状态 dropout 都是 0.25，因此这组差值不单独估计 dropout 或空间对齐的效应。

`code/analyze_sla_response.py` 进一步确认，三组实验的 64/64 状态置换都改变了实际提示词元，64/64 图像置换都改变了图像。真实样本的置换对最少改变 3 个、中位改变 34.5 个提示词元，LIBERO 的对应值为 10 与 25。状态词元删除也经过独立计数断言，不依赖对代码路径的猜测。

真实样本的 50 步动作块揭示了不同的时间结构。状态置换在前 10 步与末 10 步的 RMSE 分别为 0.2056 与 0.1550；图像置换则从 0.1125 增至 0.3878。图像置黑同样从前 10 步的 0.0769 增至末 10 步的 0.2888。第 6、13 通道为两臂夹爪，图像置换在这两个通道上的 RMSE 为 0.5330，其余通道为 0.1906。近端动作保留了对本体状态的直接依赖，后续轨迹更多随视觉输入改变。

LIBERO 的 10 步动作块中，图像置换在夹爪第 6 通道的 RMSE 由基础权重的 1.1351 增至后训练的 1.4662，其余六通道由 0.0755 增至 0.4147。后训练的动作 RMS 均值为 0.4850，基础权重为 0.3780，因此夹爪的相对响应增长并非单纯由原动作接近零造成。真实样本的原始动作 RMS 均值为 0.3459。精确首尾定义、逐条件数值与输入检查见各组 `response_analysis.json`。

这些测量限定于所取单集与归一化模型动作，支持输入通路与动作时程的解释，不估计闭环任务成功率，也不把输入敏感性等同于三维表征质量。LIBERO 基础与后训练均使用 10 步动作配置；真实样本使用 50 步，跨数据域只比较结构，不直接比较数值大小。

证据位于 `code/results/`：`summary.json` 保存帧号、权重、样本数、动作时程与汇总，`paired_metrics.csv` 保存 320 行逐帧配对指标，`comparison.json` 保存同帧差值及区间，`response_analysis.json` 保存输入置换检查与动作结构。面向报告的两张图在 `assets/sla-counterfactual-real.png` 与 `assets/sla-counterfactual-libero.png`。原始 `actions.npz` 只用于远程分析，验证后随临时实验空间删除。

## 重跑

实际远程环境为启智 Workspace `可上网GPU资源`、Project `前沿课题探索`，单张 L40S，镜像 `ngc-pytorch:25.02-cuda12.8.0-py3`。准备官方代码、上述权重和对应样本，并保留相应数据集的 `meta/tasks.jsonl`；LIBERO 使用 `physical-intelligence/libero` 的任务表。实际验证使用 Python 3.12、JAX 0.5.3、Flax 0.10.2、orbax-checkpoint 0.11.13、NumPy 1.26.4、SciPy 1.14.1、Transformers 4.53.2、pin 2.7.0 与 augmax 0.4.1；基础镜像提供 PyTorch/CUDA，openpi 与 openpi-client 以 editable、无依赖模式安装。`patches/inference-no-lerobot.patch` 删除 checkpoint 恢复路径对完整训练数据加载器的导入，推理环境因而无需安装 LeRobot/RLDS。LIBERO 探针将数据集的 `image`、`wrist_image` 与 `state` 字段适配到 `LiberoPlusInputs` 所需的 `observation/image`、`observation/wrist_image` 与 `observation/state`。随后在以下环境变量下，从本线程目录运行：

```bash
export OPENPI_DATA_HOME=/path/to/openpi-cache
export FFMPEG_BINARY=/path/to/ffmpeg
export PYTHONPATH=/path/to/Spatial-Forcing/SLA/src:/path/to/Spatial-Forcing/SLA/packages/openpi-client/src

python code/sla_counterfactual.py \
  --mode real --checkpoint /path/to/SLA_base \
  --norm-assets /path/to/SLA_base/assets --norm-id SLA_base \
  --data /path/to/sla_demo --episode 1 --num-samples 64 \
  --output code/results/sla-real-base

python code/sla_counterfactual.py \
  --mode libero --checkpoint /path/to/SLA_base \
  --norm-assets /path/to/SLA_libero_plus_ckpt/assets --norm-id omnisf_11.1_jax \
  --data /path/to/libero --episode 0 --num-samples 64 \
  --output code/results/sla-libero-base

python code/sla_counterfactual.py \
  --mode libero --checkpoint /path/to/SLA_libero_plus_ckpt \
  --norm-assets /path/to/SLA_libero_plus_ckpt/assets --norm-id omnisf_11.1_jax \
  --data /path/to/libero --episode 0 --num-samples 64 \
  --output code/results/sla-libero-post

python code/compare_counterfactuals.py \
  --base code/results/sla-libero-base/paired_metrics.csv \
  --post code/results/sla-libero-post/paired_metrics.csv \
  --output code/results/sla-libero-comparison

python code/analyze_sla_response.py \
  --mode real --checkpoint /path/to/SLA_base \
  --norm-assets /path/to/SLA_base/assets --norm-id SLA_base \
  --data /path/to/sla_demo --run-dir code/results/sla-real-base

python code/analyze_sla_response.py \
  --mode libero --checkpoint /path/to/SLA_base \
  --norm-assets /path/to/SLA_libero_plus_ckpt/assets --norm-id omnisf_11.1_jax \
  --data /path/to/libero --run-dir code/results/sla-libero-base

python code/analyze_sla_response.py \
  --mode libero --checkpoint /path/to/SLA_libero_plus_ckpt \
  --norm-assets /path/to/SLA_libero_plus_ckpt/assets --norm-id omnisf_11.1_jax \
  --data /path/to/libero --run-dir code/results/sla-libero-post
```
