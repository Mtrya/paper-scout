# 有效相机内参 K' 离散度审计:主流机器人数据集与 VLA 管线预处理链

> 实验 F · 2026-08-24 · 目标:回答"内参从不进入控制循环"这一共识建立于其上的前提——混合数据集训练时,各来源经预处理后的有效内参 K' 差异有多大。
> 方法:论文 + GitHub 代码核验(resize/crop 逻辑逐行确认);焦距按相机标称 FOV 用针孔模型估算(fx = W/2·tan⁻¹(hFOV/2)),均为估算值,非实测。

## 1. 核心结论(先读)

1. **预处理就是内参变换,但没有任何主流管线记录 K'**。DROID 是唯一逐轨迹发布内外参标定的数据集;Bridge/Fractal 不记录;所有 VLA 管线(resize/crop/pad)都确定性改变焦距与主点,却从不把变换后的 K' 写进数据或喂给模型。
2. **预处理分两族,两族都制造失配**:
   - **naive 拉伸族**(OXE 数据层 `rlds_dataset_mod` 256×256、OpenVLA RLDS 224×224、RT-1-X 300×300):非等比,把 16:9/4:3 源图直接压成方框,图像被**纵向拉长 33%–78%**(各向异性 sy/sx = 1.33–1.78),fx′ 与 fy′ 被不同因子缩放;
   - **等比+pad 族**(openpi 的 `resize_with_pad` 224×224):保持 fx/fy 比例,但 pad 平移主点并改变内容占比(16:9 源内容只占高度 56%,大量黑边)。
   - 同一份数据(如 DROID)进两家管线,有效 K′ 完全不同;族内跨源数据集的缩放因子跨度达 10×(0.2–2.0)。
3. **有效焦距 fx′ 离散度达 2–3 倍量级**。以 OpenVLA 管线计:DROID 外相机 fx′≈78 px vs Bridge(D435)fx′≈162 px;以 π0 混训(openpi 管线)计:DROID 78.4 vs LIBERO(128 源放大 1.75×)270,跨度 3.4×。**几十个百分点是保守说法,同管线内跨源差异 ≥100%。**
4. **同数据集内跨机型也有结构性差异**:DROID 的 ZED 2(外)与 ZED Mini(腕)标称 hFOV 110° vs 102°,原生 fx 448 vs 518,差 +15.7%——即便不 resize,同一数据集内部有效 K′ 也不是一个值。
5. **模拟与真实混训差距大**:robosuite 系 fovy=45°,84×84 渲染 f′=101,128×128 渲染 f′=155;π0 混合训练(9.1% 开放数据 + 大规模 sim)中模拟数据与 DROID→224 的 fx′=78 差 30%–100%。

**量级判断**:混合训练中有效 K′ 的离散度是 **2 倍量级(约 2–3.4× 跨度)**,各向异性(图像拉长)33%–78% 是常态。这不是"内参标定误差百分之几"的细粒度问题,而是预处理链系统性、沉默地重写了每个来源的相机几何。

## 2. 逐对象审计

### 2.1 DROID(2403.12945)

| 项 | 内容 | 来源 |
|---|---|---|
| 相机 | Franka Panda + **2× ZED 2(外,可调三脚架)+ 1× ZED Mini(腕)**,均为立体相机 | [论文 §3.2](https://arxiv.org/abs/2403.12945) |
| 原始分辨率 | 3 路立体 RGB @ **1280×720**,15Hz | [论文 Appendix B](https://arxiv.org/abs/2403.12945) |
| 标定 | **逐轨迹记录内参与外参**(1417 个视角;附录 G 发布 36k 场景的相机-to-base 与相机-to-camera 标定矩阵集),场景布置时用棋盘格+OpenCV | [论文 Appendix G](https://arxiv.org/abs/2403.12945) |
| 原生焦距(估算) | ZED 2(hFOV 110°×vFOV 70°):fx≈**448**、fy≈514;ZED Mini(102°×57°):fx≈**518**、fy≈663 → 外/腕 fx 差 +15.7% | [ZED 2 规格](https://www.stereolabs.com/products/zed-2)、[ZED Mini 规格](https://www.stereolabs.com/store/products/zed-mini) |
| 论文基线 DP 预处理 | 1280×720 → **128×128 拉伸** + 训练期 random crop **116×116**(主点随机平移) | [论文 Appendix F](https://arxiv.org/abs/2403.12945) |
| openpi(π0.5)预处理 | `resize_with_pad` → 等比 s=1/5.714≈**0.175**,resize 到 224×126,上下 pad 49 → fx′=**78.4**(外)/**90.7**(腕),主点回到 (112,112) | [openpi config.py](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/config.py)、[image_tools.py](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/shared/image_tools.py) |

### 2.2 Open X-Embodiment / RT-X / Octo(2310.08864、2405.12213)

| 项 | 内容 | 来源 |
|---|---|---|
| 官方表述 | "select one canonical camera view from each dataset … **resize it to a common resolution**"(无具体数值) | [OXE 论文](https://arxiv.org/abs/2310.08864) |
| Octo 生态数据层 | `rlds_dataset_mod` 的 `ResizeAndJpegEncode`:MAX_RES=**256**,dlimp `resize_image` 用 Lanczos3 **直接拉伸**(不保持纵横比、不 pad)到 256×256 | [mod_functions.py](https://github.com/kpertsch/rlds_dataset_mod/blob/main/rlds_dataset_mod/mod_functions.py)、[dlimp utils.py](https://github.com/kvablack/dlimp/blob/main/dlimp/utils.py) |
| Octo 训练增强 | 256 图上 random_resized_crop(scale 0.8–1.0,**ratio 0.9–1.1**)再拉回 256 → 额外 ±10% 各向异性 + 主点抖动;腕部 128×128 不裁剪 | [Octo 论文](https://arxiv.org/abs/2405.12213)、[octo_pretrain_config.py](https://github.com/octo-models/octo/blob/main/scripts/configs/octo_pretrain_config.py)、[dlimp augmentations.py](https://github.com/kvablack/dlimp/blob/main/dlimp/augmentations.py) |
| RT-1-X 推理 | `tf.image.resize(image, (300, 300))` **直接拉伸**(注释明言数据管线同样用 tf.image.resize) | [rt1_inference_example.py](https://github.com/google-deepmind/open_x_embodiment/blob/main/models/rt1_inference_example.py) |
| 源分辨率离散 | Fractal(RT-1)320×180;DROID 1280×720;Bridge 640×480;Berkeley autolab 等 128×128 → 纵横比 1:1 到 16:9 并存 | [DROID 论文](https://arxiv.org/abs/2403.12945)、[BridgeData V2](https://arxiv.org/abs/2308.12952)、[RT-1 论文](https://arxiv.org/abs/2212.06817) |

### 2.3 BridgeData V2(2308.12952)

| 项 | 内容 | 来源 |
|---|---|---|
| 相机 | WidowX 250 + **Intel RealSense D435**(固定 over-the-shoulder RGBD)+ **2× Logitech C920**(位姿随机)+ **Raspberry Pi 相机模块**(腕部,型号未写明) | [论文 Appendix C](https://arxiv.org/abs/2308.12952) |
| 原始分辨率 | **640×480** @ 5Hz | [论文](https://arxiv.org/abs/2308.12952) |
| 标定 | 论文未提及内参记录 | [论文](https://arxiv.org/abs/2308.12952) |
| 原生焦距(估算) | D435(hFOV 69.4°):fx≈**462**;C920(对角 78°):fx≈**494**;RPi(按 V2,62.2°H):fx≈**530** → 数据集内相机间差约 15% | [Intel D435](https://www.intelrealsense.com/depth-camera-d435/)、Logitech C920 标称 |
| 论文实验预处理 | 640×480 → **128×128**(naive,sx=0.2、sy=0.267),只用 over-the-shoulder 视角 | [论文](https://arxiv.org/abs/2308.12952) |

### 2.4 OpenVLA / openpi(推理端与训练端)

| 项 | OpenVLA(2406.09246) | openpi / π0·π0.5(2410.24164) |
|---|---|---|
| 目标尺寸 | 224×224(论文明示) | 224×224(代码 `ResizeImages(224,224)`) |
| resize 方式 | RLDS 阶段 dlimp `resize_image` **直接拉伸**(各向异性);名义上默认 letterbox 策略,但输入已是正方形,实际 no-op | `resize_with_pad` **等比缩放 + 黑边 pad**(各向同性) |
| 训练/部署一致 | 一致(同 transform);image_aug 默认关;部署可选 center_crop 0.9 | 一致(训练/推理共用同一 transforms Group) |
| 来源 | [论文](https://arxiv.org/abs/2406.09246)、[obs_transforms.py](https://github.com/openvla/openvla/blob/main/prismatic/vla/datasets/rlds/obs_transforms.py)、[datasets.py](https://github.com/openvla/openvla/blob/main/prismatic/vla/datasets/rlds/dataset.py)、[dinoclip_vit.py](https://github.com/openvla/openvla/blob/main/prismatic/models/backbones/vision/dinoclip_vit.py)、[deploy/openvla_utils.py](https://github.com/openvla/openvla/blob/main/experiments/robot/openvla_utils.py) | [config.py](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/config.py)、[image_tools.py](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/shared/image_tools.py)、[π0 论文](https://arxiv.org/abs/2410.24164)(9.1% 混合 = OXE+Bridge+DROID+sim) |

### 2.5 RoboMimic / robosuite / LIBERO / MimicGen

| 项 | 内容 | 来源 |
|---|---|---|
| robosuite fovy | 相机 fovy 取自 MuJoCo `cam_fovy`,**默认 45°**;f = 0.5·H/tan(fovy·π/360) ≈ **1.207·H** | [camera_utils.py](https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/utils/camera_utils.py)、[table_arena.xml](https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/models/assets/arenas/table_arena.xml)(无显式 fovy→MuJoCo 默认 45°) |
| robosuite 分辨率 | 默认 84×84(旧版文档)/ **256×256**(当前 master `robot_env.py`)→ f′=101 / 309 | [文档](https://robosuite.ai/docs/modules/environments.html)、[robot_env.py](https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/environments/robot_env.py) |
| robomimic | 数据 84×84(或 128×128)→ f′=101 / 155;K 精确已知(理想针孔,无畸变) | [robomimic 文档](https://robomimic.github.io/docs/datasets/robosuite.html) |
| LIBERO | **128×128** → f′=155 | [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) |
| MimicGen | **84×84** → f′=101 | [MimicGen 论文](https://arxiv.org/abs/2310.17596) |
| 与真实混训差距 | π0 混合中:模拟 101–155 vs DROID→224 的 78.4 → 差 29%–97% | [π0 论文](https://arxiv.org/abs/2410.24164) |

## 3. 汇总表:预处理 → 有效焦距变换

相机原生焦距为标称 FOV 估算;fx′/fy′ = 原生焦距 × 缩放因子(naive 拉伸时 x/y 各乘各的)。

| 数据集 | 原生分辨率(纵横比) | 原生 fx(fy) | 管线 | 目标 | sx | sy | fx′ | 各向异性 sy/sx | 效果 |
|---|---|---|---|---|---|---|---|---|---|
| DROID 外(ZED 2) | 1280×720(16:9) | 448(514) | openpi | 224 letterbox | 0.175 | 0.175 | **78.4** | 1.00 | 信箱,内容占高 56% |
| DROID 腕(ZED Mini) | 1280×720(16:9) | 518(663) | openpi | 224 letterbox | 0.175 | 0.175 | **90.7** | 1.00 | 同上 |
| DROID 外 | 1280×720 | 448 | OpenVLA | 224 naive | 0.175 | 0.311 | **78.4**(fy′=160) | **1.78** | 图像拉长 78% |
| DROID 外 | 1280×720 | 448 | Octo 数据层 | 256 naive | 0.200 | 0.356 | 89.6(fy′=183) | 1.78 | 拉长 78% |
| Bridge(D435) | 640×480(4:3) | 462 | openpi | 224 letterbox | 0.350 | 0.350 | **161.7** | 1.00 | 信箱,内容占高 75% |
| Bridge(D435) | 640×480 | 462 | OpenVLA | 224 naive | 0.350 | 0.467 | **161.7**(fy′=216) | **1.33** | 拉长 33% |
| Bridge(C920) | 640×480 | 494 | OpenVLA | 224 naive | 0.350 | 0.467 | 172.9 | 1.33 | 拉长 33% |
| Fractal(RT-1) | 320×180(16:9) | 未知 | RT-1-X | 300 naive | 0.938 | 1.667 | —(相机内参未公开) | **1.78** | 拉长 78% |
| Fractal | 320×180 | 未知 | Octo 数据层 | 256 naive | 0.800 | 1.422 | — | 1.78 | 拉长 78% |
| Berkeley autolab 等 | 128×128(1:1) | 未知 | Octo 数据层 | 256 naive | 2.000 | 2.000 | — | 1.00 | 放大 2× |
| LIBERO(模拟) | 128×128 | f′=155(渲染) | openpi | 224 letterbox | 1.750 | 1.750 | **270.4** | 1.00 | 放大 1.75× |
| MimicGen(模拟) | 84×84 | f′=101(渲染) | openpi | 224 letterbox | 1.750 | 1.750 | **177.5** | 1.00 | 放大 1.75× |

**离散度区间(有效 fx′)**:
- OpenVLA 管线跨源:DROID **78.4** → Bridge **161.7**(2.1×)
- openpi 管线跨源:DROID 78.4 → LIBERO 270.4(3.4×)
- 同源跨管线:DROID fx′ 均为 78.4,但 fy′ 在 openpi 下 78.4(等比)vs OpenVLA 下 160(拉伸 1.78×)

## 4. 判断

1. **量级 = 2–3 倍,不是百分之几**。预处理链把有效焦距跨度推到 2.1×–3.4×,各向异性(图像被拉长)33%–78%。若"混合训练 = 隐式容忍 K′ 离散",其容忍幅度已达"不同相机"级别,远超标定噪声级别。
2. **各向异性拉长是真实存在的、且方向一致的沉默偏差**:16:9 源(占主流)全部被纵向拉长 78% 压成方框(naive 族),或被打回信箱(等比族)。它对依赖精确像素↔3D 映射的任务(相机系动作、深度、抓取点)是系统性几何偏差;对仅需语义的任务可能被网络统计吸收。
3. **可弥补性成立**:DROID 证明大规模数据逐轨迹带内参是可行的;openpi 的等比+pad 至少在族内保持 fx/fy 比例(仍改变尺度与主点)。缺失的只是"把 K′(或变换参数)与观测一同记录/注入"这一环——正是 CamVLA 家族"内参从不进入控制循环"论断里那个未被验证的前提。

## 5. 查不到的空缺清单(诚实标注)

1. **DROID 发布数据的逐机器人内参分布未公开**:论文给了 36k 场景标定矩阵集与质量指标(重投影误差、MAD 离群),但没有 fx/fy 的分布直方图;ZED 出厂 per-device 标定公差官方未公布(估算 <1%,未验证)。
2. **Fractal/RT-1 相机型号与内参未公开**(Google 内部机器人;RT-1 论文仅给 300×300 输入与 EfficientNet-B3)。
3. **Bridge V2 腕部 RPi 相机模块型号/版本未写明**,其 fx 按 RPi Cam V2 假设,不确定。
4. **OXE 全量 60+ 数据集的完整分辨率清单未逐一核实**(只核实了代表:128² ~ 1280×720,1:1 ~ 16:9;官方 dataset spreadsheet 为 Google Sheets,未抓取)。
5. **RT-X 官方数据管线的目标分辨率存在不一致**:OXE 论文只说 "common resolution" 不给数值;RT-1-X 推理代码 300×300 拉伸;Bridge 论文称 RT-1 用 320×256。未决。
6. **openpi 中 DROID 的实际内参未从数据读取验证**(本文用标称 FOV 计算;ZED SDK 出厂内参可能偏离标称)。
7. 各相机标称 FOV 与具体分辨率模式(如 ZED 720p 是否 binning 保持 FOV)未逐一用 SDK 文档核实——ZED 2 标称 110°×70° 为 Max FOV,720p 模式按全幅读出处理。

## 6. 主要来源

- DROID:[arXiv:2403.12945](https://arxiv.org/abs/2403.12945)([HTML v2](https://arxiv.org/html/2403.12945v2))、[github.com/droid-dataset/droid](https://github.com/droid-dataset/droid)
- rlds_dataset_mod:[mod_functions.py](https://github.com/kpertsch/rlds_dataset_mod/blob/main/rlds_dataset_mod/mod_functions.py)(`ResizeAndJpegEncode.MAX_RES = 256`)
- dlimp:[utils.py](https://github.com/kvablack/dlimp/blob/main/dlimp/utils.py)(`resize_image` = `tf.image.resize` Lanczos3 拉伸)、[augmentations.py](https://github.com/kvablack/dlimp/blob/main/dlimp/augmentations.py)(random_resized_crop 拉回原尺寸)
- Octo:[arXiv:2405.12213](https://arxiv.org/abs/2405.12213)、[octo_pretrain_config.py](https://github.com/octo-models/octo/blob/main/scripts/configs/octo_pretrain_config.py)
- Open X-Embodiment:[arXiv:2310.08864](https://arxiv.org/abs/2310.08864)、[rt1_inference_example.py](https://github.com/google-deepmind/open_x_embodiment/blob/main/models/rt1_inference_example.py)
- BridgeData V2:[arXiv:2308.12952](https://arxiv.org/abs/2308.12952)、[rail-berkeley/bridge_data_v2](https://github.com/rail-berkeley/bridge_data_v2)
- OpenVLA:[arXiv:2406.09246](https://arxiv.org/abs/2406.09246)、[github.com/openvla/openvla](https://github.com/openvla/openvla)
- openpi:[github.com/Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)(config.py / image_tools.py);π0:[arXiv:2410.24164](https://arxiv.org/abs/2410.24164)
- ZED 2:[stereolabs.com/products/zed-2](https://www.stereolabs.com/products/zed-2);ZED Mini:[stereolabs.com/store/products/zed-mini](https://www.stereolabs.com/store/products/zed-mini)
- Intel RealSense D435:[intelrealsense.com](https://www.intelrealsense.com/depth-camera-d435/)
- robosuite:[github.com/ARISE-Initiative/robosuite](https://github.com/ARISE-Initiative/robosuite)(robot_env.py、camera_utils.py)
- LIBERO:[github.com/Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO);MimicGen:[arXiv:2310.17596](https://arxiv.org/abs/2310.17596);robomimic:[docs](https://robomimic.github.io/docs/datasets/robosuite.html)
- RT-1:[arXiv:2212.06817](https://arxiv.org/abs/2212.06817)
