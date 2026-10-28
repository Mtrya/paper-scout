# Ropedia W0：图像对齐的动作条件是否控制生成

论文：*Ropedia W0: Scaling Loco-Manipulation World Models across Embodiments with Human Experience*，arXiv 2610.26855。代码与权重：https://github.com/Ropedia/ropedia-w0 、https://huggingface.co/ropedia-ai/Ropedia-W0 。本线程使用的上游 commit 为 759b192f。实际 midtrain 权重与本地手部检测模型的 SHA-256 记录在 `code/results/model_manifest.json`。

## 问题

模型接收一张第一视角初始图、文字和逐帧运动条件，预测未来视频。运动条件不是关节控制量，而是相机位姿转成的 Plücker 射线图，以及投影到图像的手、臂或夹爪骨架。问题不是它能否生成好看的视频，而是这两个条件能否改变生成的运动，以及文字、首帧和视频底座能补偿多少。

## 论文与代码核查

- 相机射线图每像素六个通道：射线方向 d 与相机中心、方向的外积 o×d。位姿相对于第一帧。骨架由各本体正向运动学投影，再经人工修正。跨本体统一发生在图像表示层，各本体几何没有消失。
- 视频、相机和骨架经过同一个视频 VAE，再分别投影并相加。发布权重的两个条件投影已经训练，不能把训练前的零初始化当作发布模型没有条件作用。
- 训练经历约一万小时人类视频的预训练，再用约三千小时机器人数据做 1:1 人机混合训练。我们实验使用发布的 5B midtrain 教师模型，不把 5B pretrain 或 A14B 的评测数字安到它头上。
- 论文有去相机、去骨架的重训变体，400 窗口、8 本体，主要报视觉保真。它没有报告这些变体的相机或手部运动跟随指标。测试时关闭已训练模型的条件与重训一个没有该条件的模型，是两个不同问题。
- 发布的 `--disable_camera` 和 `--disable_skeleton` 分别跳过整个加性条件分支，不是输入黑帧。采样的正负文本分支共用相机与骨架条件，所以文本无分类器引导不是动作条件的增益旋钮。
- 论文还描述腕部位姿与宽视场分支；5B midtrain 权重的 827 个张量里没有这两组参数，实际只有相机和骨架输入分支。不能声称这些额外分支的效果可由发布的该权重复现。
- 论文的跨机器人结果并非每一维都领先。人类视频的零样本结果中，5B pretrain 的回估相机轨迹误差为 1.793cm，DreamDojo 14B 为 2.652cm；手部位置误差分别为 32.7px 和 65.9px。它们来自 200 个未见人类视频，前者不是物理控制误差。
- 论文的运动条件按各方法的原生接口提供：Ropedia 用显式相机位姿与投影骨架，DreamDojo 用学习潜动作，其他基线用相机与手部联合条件。首帧、文本与标注来源相同，但接口的信息形式和归纳偏置不同，因此对比同时测接口与模型，不能把差值全归因于底座能力。

## 本机离线证据

所有本轮采用的结果来自本机 CPU 和 RTX 3060 12GB。其他机器上产生的生成数字、联系表与结论已废弃，不属于本线程证据。

### 公开示例的条件剂量

`code/results/offline.json` 记录 18 个官方公开示例。相机平移路径的中位数为 0.091m；5/18 的平移路径小于 2cm，6/18 的最大旋转小于 1°。骨架墨迹占画面的中位数为 0.0097。18 个文本都含手、臂或操作相关词，但其中一些描述静止握持，不能都当作大运动。

这些示例与论文的 400 窗口、8 本体评测集不是同一总体。其分布只告诉我们公开示例给了多少条件变化，不能直接解释论文消融的数字。

### 真实 VAE 与条件投影

`code/results/vae_rice.log` 记录 rice cooker 示例的真实 VAE 往返：骨架墨迹的交并比为 0.951。将骨架平移 2px 和 32px，潜变量的变化量分别是“有骨架与空白骨架之差”的 0.052 和 0.216。这个静态例子的相机外积项精确为零，射线方向项的能量 RMS 为 1.1795；不能外推为所有 AgiBot 示例都精确为零。

`code/results/branch_scale.log` 使用真实 midtrain 条件投影，输入标准正态视频潜变量。相机与骨架分支的输出 RMS 分别是视频 token 投影 RMS 的 0.7374 和 0.6371。探针只取条件的前 21 帧；合成相机在这一个窗口内累计 3.75° 偏航、0.125m 横移和 0.025m 前移，完整 81 帧末端才是 15°、0.5m 与 0.1m。该窗口的投影变化量是相机分支 RMS 的 0.1299，是视频 token RMS 的 0.0958；骨架右移 40px 对应 0.1913 和 0.1218。VAE 先输出 6×30×40 的潜网格，再由 2×2 空间合并得到 15×20 的 token 网格。

这些值表明条件投影有非零输出，但总幅值包含各去噪步之间不变的分量，不能当成被利用的运动信息；干预差分也不能直接推断最终视频跟随。`code/results/probe_summary.json` 与 `assets/ropedia_condition_summary.png` 汇总上述测量。

## 本机生成实验

资格运行成功加载全部 827 个张量，没有缺失或意外键；9 帧、320×256、8 步的单次生成耗时 1095.6s。画面有畸变，只证明推理链路可运行，不作效果证据。逐层卸载文本编码器使 DiT 能整模留在显卡。正式八例均已按完整尺寸完成，未用资格运行的缩小配置代替。

正式对照为八个有明确判别目的的病例，配置在 `code/cases_local.json`：原条件、去骨架、双条件关闭、合成相机、骨架右移、骨架时序重排的 rice cooker 六组，以及原条件、去相机的 cut cardboard 两组。全部 81 帧、640×480、50 步、seed 0；机器人文本引导系数为 3、输出 30 fps，人类为 5、输出 16 fps，与官方示例配置一致。81 帧含被固定的首帧，后续 80 帧是预测。原始视频是原条件轨迹的记录，不是合成相机或移位骨架的真实未来；与这些变体比较时，不能把画面差异直接解释为正确的运动跟随误差。

替换相机或骨架而固定另一路，是刻意解耦的反事实，不保证物理一致。切纸板关闭相机后，骨架投影仍可能携带视点变化，只测相机射线分支的增量作用。固定机位的电饭煲按压中，去骨架与双关更适合检查文本和首帧的补偿。骨架时序重排保持第 0 帧，用随机种子 0 重排其余 80 帧，实际排列保存在 `code/results/skeleton_order.json`；它检验时序敏感性，但不连续输入不是运动理解得分，也没有对应的真实未来。

### 已完成的前两例：原条件与合成相机

`rice_full` 用时 494.7s、峰值显存 11.675GB，`rice_synthcam` 用时 484.6s、峰值显存 11.673GB，均为单张 RTX 3060 上的完整 81 帧、640×480、50 步生成。第一例的机器人、电饭煲和背景在中段仍保持可辨认的形状，没有资格运行中的明显崩坏；但这只说明画面可用，不能证明条件跟随。

`code/results/rice_camera_metrics.json` 记录特征点匹配与鲁棒单应拟合得到的图像四角平均位移。第 60 帧时，原条件的水平位移为 0.06px，合成相机为 +9.80px；第 80 帧时分别为 0.00px 与 +8.37px。合成相机使用 −15° 偏航，背景向右移动与几何预期一致。第 60、80 帧的匹配内点比例为 85.5% 与 77.5%，位移是图像空间代理，不是相机的 ATE，也不能直接换算成角度跟随误差。合成条件的原视频不是它的真实未来，不能把与原视频的 PSNR 差异当作控制误差；当前原条件为 23.22dB、合成相机为 17.90dB，只描述各自与原记录的像素相似度。

### 骨架位移与时序重排

`rice_skelshift` 用时 497.5s，`rice_skelshuffle` 用时 503.1s，峰值显存均为 11.673GB。两例都保持可辨认的场景和机器人，没有明显画面崩坏。与原条件生成视频逐帧比较（排除固定的首帧），骨架右移 40px 的 PSNR 为 24.366dB、平均 RGB 绝对差为 0.01828，3.51% 的像素平均通道差超过 0.1；时序重排分别为 21.499dB、0.02613 和 5.75%，合成相机分别为 18.376dB、0.06629 和 19.77%。这些是条件变化造成的画面差异，不是动作跟随正确率。

骨架右移主要改变手臂局部形状与位置，没有看到清晰的整只手右移 40px 的证据。时序重排改变了一部分输出，却没有提供精确遵循骨架顺序的证据，也没有明显动作崩坏。它们提示文本与首帧承担了很大的任务先验，不能把 VAE 与投影层的非零响应直接当成生成控制。`code/results/condition_pairs.json` 保存逐对、逐帧差异，且明确区分生成视频之间的比较与相对真实未来的误差。

### 本地手部检测器的校准

公开仓库没有论文手部回估指标的实现。本机另用 MediaPipe 0.10.32 的 HandLandmarker 视频模式和官方 [hand_landmarker.task](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task) 做补充测量，三个置信度阈值均为 0.3。每只检测手与标注手按 21 个关键点的平均距离做最优匹配，再报告匹配点的欧氏距离；最近匹配本身不等于正确检测，误配会表现为较大的位置误差。它不是论文 HandL2 的复刻，不能与论文表格数值混算。

`code/results/hand_source.json` 是切纸板原视频校准：81 帧共有 162 个标注手，检测到 137 个且都能匹配，数量比例 84.6%；匹配关键点的平均误差为 22.201px，中位数 21.886px，第 90 百分位为 37.546px。检测覆盖与标注偏移共同限制了可分辨的最小效应，几像素的差别不作解释。`code/audit_hand_landmarker.py` 是实际运行的探针，独立虚拟环境只需 MediaPipe、ImageIO 与其依赖，不改模型推理环境。生成后的 `cut_full` 和 `cut_nocam` 也会用同一检测器；只有原条件轨迹可以把原标注当未来参考，去相机结果与原标注的距离只作描述，不是跟踪误差。

### 切纸板：去相机改变视点并加大手部偏差

`cut_full` 用时 502.6s、峰值显存 11.673GB，`cut_nocam` 用时 487.2s、峰值显存 11.662GB，均按人类示例的 16fps 写出。去相机后，纸板、桌面和双手的视点都发生明显变化，不再只是局部纹理不同；这是相机分支确实影响生成的直接证据。两例与原条件生成视频的 PSNR 为 15.118dB，平均 RGB 绝对差为 0.1053，32.23% 的像素平均通道差超过 0.1。这里的比较仍只描述条件造成的变化，不是跟随误差。

同一本地手部检测器下，全条件检测到 139/162 个标注手，匹配比例 85.8%，平均关键点误差 19.843px；去相机检测到 140/162，匹配比例 86.4%，平均误差 24.569px。全条件与原视频校准的 137/162、22.201px 处于相近的测量水平；去相机误差更大，但检出率没有塌陷。这说明相机分支更像是在改善视点和几何对齐，而不是决定画面中是否出现手。去相机仍保留投影骨架和详细任务文本，因此不能把结果解释为“没有视点信息”，只能说明射线分支的增量作用。

`code/results/cut_full_hand.json`、`cut_nocam_hand.json`、`cut_camera_metrics.json` 保存测量。动态切纸板场景中，单应拟合的内点比例在后段可低于 50%，不把这类图像代理当成相机轨迹真值；主要判读依据是实际画面、生成视频之间的差异与经原视频校准的手部测量。`assets/ropedia_cut_camera_comparison.png` 是全条件与去相机的逐帧对照。

### 去骨架：任务语义仍在，几何轨迹改变

`rice_noskel` 用时 496.0s，峰值显存 11.673GB。去掉骨架后，模型仍能生成电饭煲和手臂，按压任务没有消失；文本与首帧确实提供了任务语义。但与原条件相比，手臂轨迹和动作节奏明显不同：两例的 PSNR 为 19.371dB，平均 RGB 绝对差为 0.04260，9.92% 的像素平均通道差超过 0.1。逐帧像素变化量从原条件的 0.00326 增至 0.02129，约为 6.5 倍；这只表示画面变化更活跃，不代表动作更快或跟随更准确。

因此，前面的弱位移响应不能被解释成“骨架无关紧要”。它说明几何条件参与动作生成，但文本和首帧可以补回任务的大致语义，精确的几何执行需要另外测量。`assets/ropedia_noskel_comparison.png` 保存全条件与去骨架的逐帧对照；`code/results/condition_pairs.json` 保存同口径数值。

### 双关：本体外观与几何执行一起失去约束

`rice_none` 用时 481.2s，峰值显存 11.648GB。同时关闭相机与骨架后，模型仍能生成电饭煲和手臂，但末端从外观到轨迹都发生显著变化：黑色夹爪在部分帧中变为白色杆状末端与袖口、手套状结构，遮挡关系也明显改变。与原条件生成相比，PSNR 为 13.554dB，平均 RGB 绝对差为 0.13878，42.28% 的像素平均通道差超过 0.1，是八例中最大的一组变化。

这说明文本与首帧能保留任务语义，却不能独立维持机器人本体的外观、几何与动作执行。去骨架仍保留静态相机射线，因此不能把双关的差异全部归因于缺少骨架；更合适的是把相机与骨架看成共同约束生成的两路条件，分别检查替换、关闭和时序破坏，而不是要求每一路在所有案例中都产生同等效果。`assets/ropedia_condition_ablations.png` 给出完整条件、去骨架和双关的对照，`assets/ropedia_condition_effects.png` 给出逐帧差异曲线。

`code/results/generation_log.jsonl` 记录八例的真实尺寸、步数、帧率、种子、耗时与峰值显存，`code/results/metrics.json` 记录全量汇总，`code/results/condition_pairs.json` 记录全部六组生成视频之间的比较。原始视频保留在 ignored 工作区供最终看图，不进入运行包。至此八例全部完成，最终报告只使用这些本机输出，不使用其他机器的结果。

## 承重前作与相邻方法

本线程不是只读 Ropedia 自己的相关工作段落，另核查了三篇原文。

- [Precise Action-to-Video Generation Through Visual Action Prompts](https://arxiv.org/html/2508.13104) 已把人手与机器人夹爪画成图像骨架，用视觉条件驱动视频。它的骨架经三维卷积编码，再通过 ControlNet 式分支与视频 token 通路注入。其 EgoVid 训练子集为 200k 段；RT-1 上专用骨架模型的时空 IoU 为 0.604，重实现的原始状态条件为 0.507，联合模型为 0.576。这里的 IoU 比较运动区域与真实视频的时空重合，不是关节跟踪误差。它过滤显著视角变化，不能把移动相机留给生成器自行猜测的问题与 Ropedia 的显式相机控制混为一谈。
- [OSCAR](https://arxiv.org/html/2606.04463) 也将正向运动学投影的二维骨架经 VAE 和独立投影加到视频 token，统一机器人手臂与人手，公开原始机器人数据约 10.9k 小时，人类数据约 929 小时。它明确过滤运动相机，并把相机运动留作未来工作。因此 Ropedia 对 OSCAR 的 fixed-base 概括不能覆盖其所有数据，但 OSCAR 的模型条件确实不负责相机运动控制。其 200 个机器人片段的自动指标主要是视觉保真、时序一致性与分布距离，另在 65 个会话、7 个策略上比较虚拟和真实评测；不能把这些说成逐帧手部轨迹跟随精度。
- [DreamDojo](https://arxiv.org/html/2602.06949) 用 44,711 小时人类视频预训练，以相邻帧学习出的连续潜动作作为统一代理条件；到目标机器人时替换动作输入层并后训练。它提供了人类经验迁移的另一条路线，但不是一个权重直接接受各本体相同的显式几何轨迹。其自动消融仍以视觉保真为主，反事实控制主要依赖 12 名评测者的偏好；这些证据不直接回答物理坐标中的轨迹跟随。

由此，Ropedia 的新意不应写成“第一次用骨架统一本体”或笼統的“统一第一”。更准确的定位是：把已有的图像骨架接口与显式相机射线条件合并，覆盖移动与操作，同时以人类预训练和人机混合训练共享同一几何接口。真正值得审计的是这两个条件在发布权重中能否改变结果。

## 继续好奇的问题

- 关闭条件后，文字和首帧能恢复多少动作语义，几何精度损失在哪里？
- 合成相机或骨架位置改变时，生成是否按方向响应，还是仅发生画面扰动？
- 条件经过视频 VAE 后，位置变化的保留是否与最终跟随误差相关？
- 骨架帧序破坏后，生成是否改变，区分时序信息与姿态、位置分布的作用？
- 论文未显式建模接触力与物体动力学，哪些任务仍能安全使用它做策略评测或规划？

前四个问题已由上述离线探针与正式生成设计承接；最后一个仅作文献证据边界，不在本机八例上夸大外推。

## 重跑

在仓库的 ignored `code/` 中准备 Ropedia W0 代码、Wan2.2-TI2V-5B 底座与 W0 midtrain 权重。所有权重、原始视频与缓存仅用于本轮实验，运行包不保存。线程包中的 `audit_*.py`、`make_*.py`、`probe_*.py`、`compare_conditions.py`、`plot_condition_effects.py` 与 `cases_*.json` 复制到克隆仓库根目录，`patches/local-offload.patch` 应用于上游提交；`finish_local_experiments.sh` 和 `summarize_ropedia_probes.py` 放在工作区的 ignored `code/` 下。下列离线命令在克隆仓库根目录执行。

```bash
# 离线剂量、VAE 与条件投影
mkdir -p results
python audit_offline.py --examples examples --out results/offline.json
python probe_vae_condition.py --clip examples/agibot_rice_cooker --out /tmp/ropedia-vae --frames 9 \
  > results/vae_rice.log
python probe_branch_scale.py --ckpt weights/w0-5b-midtrain.safetensors \
  --clip examples/agibot_rice_cooker \
  --alt_camera variants/synthcam_RICE/data.npz \
  --alt_skeleton variants/skelshift_RICE/skeleton.mp4 \
  > results/branch_scale.log
python ../summarize_ropedia_probes.py --results results --out results/probe_summary.json --figure condition_summary.png

# 真实生成；从当前 workspace 执行
bash code/finish_local_experiments.sh

# 生成视频之间的对照，不把反事实当真实未来
code/ropedia-w0/.venv-torch/bin/python code/ropedia-w0/compare_conditions.py \
  --dir code/ropedia-local/audit --out code/ropedia-local/condition_pairs.json
code/ropedia-w0/.venv-torch/bin/python code/ropedia-w0/plot_condition_effects.py \
  --data code/ropedia-local/condition_pairs.json --out code/ropedia-local/condition_effects.png
```

`code/finish_local_experiments.sh` 先等待本机 SuperMimic 的最终配对世界模型探针，再调用 `audit_run.py --text_layer_offload` 和 `audit_metrics.py`。本机适配见 `patches/local-offload.patch`。`audit_run.py` 支持仅检查输入的 `--dry_run`，每个病例记录尺寸、步数、种子、输出帧率、耗时与峰值显存。上游 commit 为 `759b192fcb0aed48695ace42a4eb2fd7e48c12b9`；环境、底座下载与示例许可见上游 README。

手部校准是独立补充工具，在 workspace 下安装与执行，不进入 Ropedia 推理环境：

```bash
uv venv code/ropedia-hand-venv --python 3.12
uv pip install --python code/ropedia-hand-venv/bin/python \
  'mediapipe==0.10.32' 'imageio==2.37.3' 'imageio-ffmpeg==0.6.0'
curl -fL -o code/ropedia-w0/models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
code/ropedia-hand-venv/bin/python code/ropedia-w0/audit_hand_landmarker.py \
  --video code/ropedia-w0/examples/xp10m_cut_cardboard/video.mp4 \
  --labels code/ropedia-w0/examples/xp10m_cut_cardboard/data.npz \
  --model code/ropedia-w0/models/hand_landmarker.task \
  --out code/ropedia-local/hand_source.json \
  --overlay code/ropedia-local/hand_source.png
```
