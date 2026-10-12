# DreamTrue(arXiv 2610.12468)标定发布物:独立复算与实现走查

**一句话结论**:DreamTrue 随论文放出的标定 mini 包(condition-mini.tar.gz,268MB)在几何上**完全自洽且经得起独立复算**——我们把 DROID 数据集自带标定(公开 GCS 原始 metadata + ZED 工厂内参)与论文精标定逐相机对齐,外参差仅 0.5–6.2cm / 1.2–6.6°,外参相机 fx 差 5/6 在 ±1.3% 内(一例 +4.2%),wrist 相机 fx 系统性 +3.2~+6.9%;从 ModelScope 48GB 参数 tar 里用 HTTP range 猎出 3 集 AgiBot 原始标定,解开其未文档化的坐标系约定后,原始与精标定在每集 ~0.6–2.0° / 3–16mm 量级吻合;mini 包内部 hash 链 8/8 全对、按发布参数"反投影到世界再投回像素"的闭环误差 ≤6.0e-5 px。**未能独立复算的部分**:论文主打的 IoU 类声明(+0.228/+0.212,97.1%/91.6%/75.5%)依赖 SAM3 掩码与真实视频帧,mini 包两者都没有;153,666 集的完整标定发布在 ModelScope 仓库中尚不存在(GitHub README 标 "Coming soon");**标定优化器本身未开源**——calibration 子包只有消费侧(条件渲染器),没有优化侧实现。

---

## 1. 素材与下载清单

| 素材 | 来源 | 大小 | 用途 |
| --- | --- | --- | --- |
| condition-mini.tar.gz | ModelScope 数据集 `huoxingdawang/DreamTrue`,`condition/condition-mini.tar.gz` | 267,999,132 B | 精标定参数 + 8 个渲染条件 h5 |
| reward-data-mini.tar.gz | 同上,`reward/reward-data-mini.tar.gz` | 119,557,886 B | reward 评估小样抽查 |
| 机器人资产(3×yaml+urdf) | ModelScope 数据集 `huoxingdawang/DreamTrueAssets`(即 calibration 仓库的 assets 子模块):agibot/G1_120s、droid/franka、robomind/franka | KB 级 | 固定初始化(init)相机参数 |
| DROID 原始外参 metadata_*.json | 公开 GCS bucket `gs://gresearch/robotics/droid_raw/1.0.1/<lab>/<status>/<date>/<ts>/metadata_*.json`(经 `storage.googleapis.com/storage/v1` JSON API 列目录与下载) | ~1.8KB × 3 | 数据集自带外参(ext1/ext2/wrist) |
| DROID trajectory.h5 × 3 | 同上目录 `trajectory.h5` | 1.1–3.4MB | 逐帧相机外参 + 关节角(wrist FK 用) |
| DROID 官方改进标定 | HF **model** 仓库 `KarlP/droid`(cam2base_extrinsics.json 36,084 集 / intrinsics.json 72,468 集,2025-04 更新) | 16MB / 126MB | 交叉验证原始外参与工厂内参 |
| AgiBot 原始标定 | ModelScope 数据集 `agibot_world/agibot_world_beta`,`parameters/{648533-663145,663146-674519,834407-841619}.tar`(各 ~48GB,HTTP Range 访问) | 传输 ~8.7GB(窗口采样) | 3 集原始内外参 |
| 论文代码 | GitHub 克隆 `code/DreamTrue`(工作区实验台) | — | 实现走查 |

注:HF 的 AgiBotWorld-Beta 数据集是 gated(401),故走 ModelScope 镜像。

## 2. mini 包内容盘点

```
condition/
├── calibration/           # 8 集精标定参数
│   ├── AgibotWorld/param/{380/665152, 410/649000, 599/835856}/optimize/{local_pose_best.json, intrinsic_best.json}
│   ├── droid_raw/param/1.0.1/{AUTOLab/failure/Sat_Oct_21_19:58:12_2023,
│   │                          ILIAD/failure/Thu_Jun_15_13:56:46_2023,
│   │                          ILIAD/success/Thu_Jul_20_11:48:53_2023}/optimize/{同上 2 个}
│   └── robomind_franka/param/{open_drawer_.../0623_161143, place_blue_block_.../0526_161107}/optimize/{同上 2 个 + offset_joints_best.json}
├── conditions/            # 8 个渲染条件 h5(agibot/droid 各 3、robomind 2),约 267MB,zstd 压缩
├── task-lists/            # 3 个 JSON:data_key → calib_key 映射 + calib_hash + 切片
├── data-config/           # 下游训练配置(threeviews_concat、use_plucker 等)
└── merge-config.yaml      # 由原始数据集重建 data-config 的管线配置
```

要点:

- **跨 episode、跨任务复用标定**:task-lists 把数据集 episode 映射到标定集,且不必同任务——AgiBot `764/909599` 用 `599/835856` 的标定、`390/682715` 用 `380/665152` 的。这正是论文附录 A.2.1 的"按采集元数据分组、组内抽标定"机制的发布物形态。
- mini 包**只有 refined(*_best)参数,没有 original 参数对**——"original vs refined" 的比较必须由我们自己去找各数据集的原始标定(下面 §3.1/§3.2)。
- 参数格式:`local_pose_best.json` = {cam: {p, q(wxyz)}}(相机相对 mount link 的 SAPIEN 相机系位姿,x 前 y 左 z 上);`intrinsic_best.json` = {cam: {fx,fy,cx,cy,k1,k2,p1,p2,k3,w,h}}(OpenCV plumb-bob;存储的 rgb 是**畸变后**图像,渲染用从这套内参推出的 rectified 内参 rfx/rfy/rcx/rcy)。渲染分辨率:AgiBot/RoboMIND 640×360,DROID 1280×720(与 h5 内 rgb/depth/mask 形状一致)。
- h5 attrs 带 `calib_key`、`calib_hash`、`cameras_intrinsic`、`cameras_local_pose`、`segmentation_ids`、`parts_config` 等;`start_time` 显示 mini 包渲染于 2026-10-08。

## 3. 复算

所有脚本与结果在 `code/`:`recalc_calibration.py`(DROID 与 h5 一致性)、`agibot_probe_layout.py` / `agibot_probe_order.py` / `agibot_dense_scan.py` / `agibot_dense_scan_ext.py`(AgiBot tar 猎取)、`agibot_orig_compare.py`(坐标系解析与比较)、`make_figures.py`(图)。结果 JSON:`recalc_results.json`、`agibot_orig_vs_refined.json`;猎取的 AgiBot 原始参数原件在 `code/agibot-orig/`。约定与代码侧出处已逐条核对(SAPIEN↔OpenCV 相机系转换 C=[[0,-1,0],[0,0,-1],[1,0,0]]、存储图为畸变图、r\* 的不动点去畸变算法,均对照 `code/DreamTrue/calibration` 源码确认)。

### 3.1 DROID:数据集自带标定 → 精标定(核心独立复算)

DROID 三个发布集都有公开的原始标定:GCS metadata 给外参 `[x,y,z,rx,ry,rz]`(ext1/ext2 为 base 系固定外参,wrist 为 base 系逐帧变化),`KarlP/droid` 的 intrinsics.json 给 ZED 工厂内参。wrist 原始外参借助逐帧关节角 + 纯 numpy 实现的 URDF FK(`FrankaFK`,已对 trajectory.h5 的 cartesian_position 验证到 0.00cm/0.0°)换算成 cam2flange 再与 refined local_pose 比。

两个值得记录的判定:

- **DROID euler 约定是 extrinsic xyz(R=Rz@Ry@Rx)**:若按 intrinsic 解,wrist 的 cam2flange 旋转随时间方差达 73–173°,extrinsic 下逐帧方差为 0.000°(机器精度);外参相机上 intrinsic 解与 refined 差 77–175°,extrinsic 解差 1.2–6.6°。注意 `KarlP/droid` README 的 scipy 示例写的是 `from_euler("xyz")`(intrinsic 语义),与数据实际约定相反——直接照抄会把外参解错到上百度。
- **相机对应关系**:DreamTrue 的 `left`/`right` = DROID 的 ext1/ext2(外参平移直接吻合到 1–2cm;若交换则差 >30cm)。

结果(`recalc_results.json` 的 `droid_original_vs_refined`,单位 cm/°,fx 为 refined vs 工厂值):

| episode | left ΔT/ΔR/Δfx | right ΔT/ΔR/Δfx | wrist ΔT/ΔR/Δfx |
| --- | --- | --- | --- |
| AUTOLab Oct 21 | 0.51 / 1.22 / −0.15% | 1.62 / 2.75 / +0.96% | 1.95 / 2.44 / +3.64% |
| ILIAD Jul 20 | 2.45 / 2.83 / +1.11% | 1.85 / 2.42 / +0.44% | 2.50 / 1.78 / +3.21% |
| ILIAD Jun 15 | 3.83 / 5.17 / +4.24% | 6.15 / 6.62 / +1.28% | 2.60 / 1.71 / +6.92% |

结论:精标定是对数据集标定的**小幅修正**——量级与论文 Table 4 的"DROID dataset-provided 已经不错、refined 再进一步"自洽(Sync error 11.78→8.06px)。一个系统性信号:3 集的 wrist fx 都被精标定从工厂值 732–733 拉到 757–784(+3.2/+3.2/+6.9%),方向一致,提示 ZED wrist 相机的工厂内参偏小是真实偏差而非噪声。畸变:数据集侧 distCoeffs 全 0(等于没给),refined 给了小的非零 plumb-bob 系数。

### 3.2 AgiBot:48GB tar 猎取 + 未文档化坐标系解析

AgiBot 的原始相机参数不在 mini 包里,而在 AgiBotWorld-Beta 的 `parameters/*.tar`(每个 ~48GB)。HF gated,ModelScope 可匿名 range 读取。 tar 内部结构(探针实测,`agibot_probe_layout.py`):条目按 `{task}/{episode}/parameters/camera/*.json` 组织,**task 占 GB 级连续块,块内 episode 顺序无规律**(既不能二分也不能插值),每集约 3.3–16MB(大头是每相机 ~0.5–2.2MB 的 `*_aligned.json`,6 个目标小 JSON 只有 300–700B)。因此采用稠密扫描:1MB 窗口 / 2MB 步长(每集 ≥3.3MB,保证必被命中),12 路 curl 并行,命中后 ±7MB 提取。合计约 8,678 个窗口 / ~8.7GB 传输,3 集全部拿到 6 文件全套(`code/agibot-orig/*.json`,含扫描地图 scanmap_*.json)。

拿到后发现原始外参的父坐标系**没有任何文档**。用 3 集联立解出:

- 原始外参按相机跨集稳定(head 最大两两差 2.8°/2.0cm,两手 ≤1.0°/1.2cm)→ 父系是机器人上的固定系;
- 存在**固定旋转 A** 使 `A·R_orig ≈ R_refined` 到噪声底(残差:head 1.97°、hand_left 1.43°、hand_right 1.07°,与原始/精标定各自的跨集波动同量级);head 的 A 近似一个轴置换,两手的 A 是一般旋转——即原始手眼外参挂的父 link 与 DreamTrue 的 mount link(gripper_*_base_link)不同。右侧(相机系)不需要修正:B=I 已最优。
- 平移:`t_refined − A·t_orig` 跨集恒定(残差即下表),再次支持同一固定父系。

吸收这个固定坐标系差异后,**每集的"精标定修正量"**(见 `agibot_orig_vs_refined.json`):

| episode | head ΔR/ΔT | hand_left ΔR/ΔT | hand_right ΔR/ΔT |
| --- | --- | --- | --- |
| 380/665152 | 1.18° / 9.5mm | 0.87° / 5.7mm | 0.98° / 4.9mm |
| 410/649000 | 1.14° / 15.8mm | 1.43° / 3.3mm | 1.07° / 3.5mm |
| 599/835856 | 1.97° / 10.9mm | 0.61° / 8.5mm | 0.57° / 5.9mm |

(注意:A 与平移偏移是用这同 3 集拟合的,所以这是"除掉坐标系约定后"的残差口径,不是无偏的修正量估计;但跨集一致性与残差量级本身说明原始与精标定在 ~1–2° / ~1cm 内吻合。)

内参更有故事性:原始参数 fy/fx=1.323–1.332(9 组全部如此)——对正常相机这是异常的非方像素;但若原始内参对应 **640×480** 帧(ppx≈320、ppy≈244 也吻合),而 DreamTrue 管线工作在 640×360(高度 ×0.75),则 fy×0.75 后与 fx 相称,异常消失。按此口径:refined fx 与原始差 −4.1%~+4.4%,fy(×0.75) 差 −2.1%~+2.7%,主点差 cx −15.9~+10.3px、cy(×0.75) −12.3~+4.0px。畸变:原始 k1≈−0.05(轻度桶形),refined k1≈0(±0.005)——精标定几乎不保留畸变。("640×480→640×360"是从参数数值反推的假设,未被数据方文档确认;三个集 9 组内参在该假设下全部自洽。)

### 3.3 init(机器人 YAML)→ 精标定:固定初始化离真值有多远

论文说 AgiBot/RoboMIND 用固定初始化、DROID 用 32 个初始化。我们把机器人 YAML 里的 init 参数与 refined 比(全部 8 集 × 全部相机,`init_vs_refined`):

- **AgiBot head 的固定 init 系统性偏高**:ΔR 11.4°/11.4°/12.9°、ΔT 2.5–3.6cm、fx −4.3~−6.4%(3 集方向一致);手部 init 已经很好(ΔT 0.2–0.9cm、ΔR 0.2–2.6°)。
- **DROID 外参相机:固定 init 完全不靠谱**(ΔT 19–70cm、ΔR 10–43°)——DROID 是众包采集,相机摆设每家不同,这组数字直接解释了论文为什么要给 DROID 准备 32 个初始化;wrist init(fx=700)与 refined(757–784)差 +8~+12%。
- **RoboMIND 多数相机 init 差 1.4–15° / 1.2–5.8cm**,另有一个 `camera_right`(place_blue_block 集)差 83cm/83° 的硬离群——固定 init 对多工位采集同样不够。
- 安装偏置(offset_joints)只在 RoboMIND 两集出现:仅左臂 base,两次精标定分别为 y=0.4548m 与 0.4390m(**相差 15.8mm**,旋转分量是固定的 180° 结构量)——论文"为 RoboMIND 优化臂安装偏置"在发布物中落到了具体数字上。

### 3.4 发布物内部一致性(8/8 全过)

对 8 个条件 h5 逐一核验(`h5_consistency`、`projection_consistency`):

1. **hash 链**:`calib_hash` = sha256(local_pose_best.json + intrinsic_best.json [+ offset_joints_best.json]) 重算,8/8 匹配——h5 与标定文件的绑定没有被篡改或错配;
2. **attrs vs JSON**:`cameras_intrinsic` attr 与 JSON 完全相等(max abs 0),`cameras_local_pose` ≤2.96e-8(float32 量化级);
3. **渲染内参 r\***:按发布侧 `OpenCVRenderIntrinsic.from_opencv` 的不动点去畸变算法从 k1..k3 重算 rfx/rfy/rcx/rcy,与 attr 差 ≤3.0e-5;
4. **投影闭环**:mask 像素 + 深度(mm)反投影到世界,再按发布参数投回像素——全部 8 个 h5 × 全部相机 roundtrip 最大误差 **6.0e-5 px**(均值 ≤3.2e-5),整条 pose/内参/畸变约定链在数值上完全自洽;link 原点投影落在自身 segmentation id 上的比例低的个案(如 wrist 相机视角下小 link)经核查均为相邻 link 遮挡,非参数错误。

## 4. 与论文声明逐条对照

| 论文声明(出处) | 核验结果 | 判定 |
| --- | --- | --- |
| 每条记录含 intrinsics/extrinsics/distortion(附录 A.2.1 "Released calibration") | mini 8 集格式一致:intrinsic_best(fx,fy,cx,cy,k1,k2,p1,p2,k3,w,h)+ local_pose_best(p,q) | **一致** |
| DROID 与 RoboMIND 记录另含 mounting offsets 与质量指标(同上) | RoboMIND 有 offset_joints_best.json(左臂 base y 偏置,两集差 15.8mm)✓;但 mini 的 **DROID 记录没有 offset 文件**,且**任何记录都没有质量指标**(质量分只在代码仓库测试 fixture 里见过,如 {"head":0.9}) | **mini 与描述部分不符**(可能完整版才有;完整版未发布,见 §7) |
| 发布规模 153,666 集 / 1,660.31h(同上) | ModelScope 仓库现仅有 mini(8 标定集 + 8 h5)+ reward 包 + 两个 checkpoint;GitHub README 自述 "Full calibration dataset: Coming soon" | **无法验证(发布未到)** |
| 组内抽标定、跨集复用、差的补标(附录 A.2.1) | task-lists 的 data_key→calib_key 跨任务复用(764/909599←599/835856 等)与分组机制自洽;"随机选取/补标"过程本身无发布物可验 | **机制一致,过程不可验** |
| AgiBot/RoboMIND 固定 init、DROID 32 个 init(附录 A.2.1) | 三个机器人 YAML 各只有一套固定 init;实测 DROID 固定 init 离 refined 19–70cm/10–43°,印证"单固定 init 对 DROID 不可行";32-init 过程的代码未发布 | **动机被独立数据支持,过程不可验** |
| IoU:AgiBot +0.228(130,182 集)、DROID +0.212(63,061 集)、改善率 97.1%/91.6%、胜 PointWorld 75.5%(§4.5) | 需要 SAM3 掩码 + 真实视频帧;mini 包两者皆无,SAM3 fine-tuned checkpoint 标 "Coming soon" | **无法独立复算(缺环明确)** |
| Table 4 标定消融(EWMScore-P/PSNR/SSIM/nDTW/Sync error)(§4.5) | 需要训练/推理世界模型与全量条件包 | **超出发布物可及范围** |
| DROID 原始标定质量(隐含:dataset-provided 是可用基线) | 独立复算:原始 vs 精标定 0.5–6.2cm / 1.2–6.6°(§3.1) | **与论文基调一致** |

## 5. 实现走查:标定优化器没有开源

`code/DreamTrue/calibration` 子包里**只有消费侧**:`calibrate/condition.py` 是 SAPIEN 条件渲染器(读 `*_best.json` 渲染 rgb/depth/mask/segmentation),外加数据集读取器(AgibotDataset/DroidDataset/RobomindFrankaDataset)与任务/合并脚本。全文搜索没有 least_squares/ceres/对应点求解/共面约束/P_align/P_time/P_view 的任何实现——§3.2 的联合优化管线(论文核心方法之一)不在发布代码里,GitHub README 的 "Coming soon" 清单也明确列着 "The full calibration pipeline"。

但论文机制在发布代码里留下了配置级痕迹(均未被发布代码调用,属于"证据不是实现"):

- `pyproject.toml` 依赖 fork 版 `romav2`(brave-eai/RoMaV2,对应论文的稠密匹配)与 fork 版 ultralytics(SAM3);
- `SapienEnv`:`sam3_prompt`(默认 ['robot'])、`cam_use_bg_mask`(真机对真机匹配时用 SAM3 掩码去掉机器人本体、只匹配背景——对应时序项 P_time 的静态背景假设)、`part2gripper`(sim-real 匹配时跳过活动夹爪)、`cam2gripper`(temporal 匹配:相机所在臂夹爪两帧间非全闭合则跳过)、`offset_joints`(把可学习 SE(3) 安装偏置挂到 URDF fixed joint;robomind 配置为 left 臂 base joint);
- `load_calibration`(sim/SapienEnv.py:792)按 postfix='_best' 读三个 JSON 并返回其 sha256——即 h5 里 `calib_hash` 的来源,与我们重算的 hash 链一致;
- `AgibotDataset._sam3_path` 读 `{root}/sam3/{key}` 掩码 → SAM3 掩码按数据集 episode 存储,不在 mini 包。

## 6. reward-data-mini 抽查

`reward-data-mini.tar.gz`(119.6MB):34 条 sharegpt 风格**评估**样本(`split/test_mini.json`),视频是自家模型 demo rollout 的预测 mp4(320×240×3 cell 横拼,含 retreat 反事实变体),meta 带 nfr/fps/cell/ncell;**assistant content 为空——L1/L2/L3 人工标注不在 mini 包里**,所以它是纯推理演示包,不能用来数值复算 reward 与人工打分的相关表。reward 模型本身已发布(ModelScope `reward/checkpoint`,~18.8GB,4 个 safetensors 分片,Qwen3.5 系 VLM,按 label token 概率打 L1 embodiment / L2 object / L3 interaction 三轴分);world model 视频 checkpoint(`wmvideo/checkpoint-wmvideo.safetensors`,1.48GB)也在同一仓库。

## 7. 障碍与边界

1. **IoU 类声明无法独立复算**:缺 SAM3 掩码(未发布,checkpoint coming soon)与真实视频帧(需 AgiBotWorld-Beta/DROID/RoboMIND 全量,数百 GB 级,且 HF 侧 gated)。缺的环节是明确的:优化器输出(*_best.json)已发布并可验证,验证 IoU 需要的输入(掩码+帧)未发布。
2. **完整标定集未发布**:153,666 集规模无从核对;mini 只有 8+8。
3. **AgiBot 原始参数获取成本高**:48GB tar、块内无序、每集 3.3–16MB,我们按 1MB/2MB 窗口稠密扫描(~8.7GB 传输)猎得 3 集;扫描保证命中(集长 ≥ 步长+窗口),但不排除块内还有我们没扫到的区段边界。原始外参父坐标系无文档,§3.2 的固定旋转 A 是 3 集拟合所得——若官方文档给出真实父系,结论应以官方为准。
4. **AgiBot 内参的 640×480→640×360(×0.75)是数值反推假设**,未被数据方文档确认;在此假设下 3 集 6 组内参全部自洽(§3.2)。
5. reward 与 world model 的数值行为未在本线程验证(无人工标注、无 GPU 推理);§4.5 的 Table 4 消融同理。
6. DROID 复算只用 3 集(mini 包的全部);Conclusion 对全集的外推由"数据集标定本身是成熟基线"支撑,但 3 集样本量小是事实。

## 8. 产物清单

- `code/recalc_calibration.py` — DROID 原始 vs 精标定 + init vs 精标定 + h5 一致性 + 投影闭环(主复算)
- `code/agibot_probe_layout.py` / `agibot_probe_order.py` — tar 结构与 episode 序探针(否决二分方案的证据)
- `code/agibot_dense_scan.py` / `agibot_dense_scan_ext.py` / `code/tarhunt.py` — 48GB tar 稠密 range 扫描与提取
- `code/agibot_orig_compare.py` — AgiBot 原始 vs 精标定(坐标系解析 + 内参比较)
- `code/make_figures.py` — 误差分布图
- `code/recalc_results.json` — DROID/init/h5 全部数值结果
- `code/agibot_orig_vs_refined.json` — AgiBot 比较全部数值结果
- `code/agibot-orig/{410_649000,380_665152,599_835856}.json` — 猎取的原始参数原件(6 文件/集)
- `code/agibot-orig/scanmap_*.json` — 扫描地图(块结构、命中点)
- `code/dt-recalc-calib-deltas.png` — 三联图:外参平移/旋转/焦距差(init→refined 实心,dataset-provided→refined 斜纹;另拷一份至 `../assets/dt-recalc-calib-deltas.png`)

复现方式:脚本均按自身位置相对寻址,数据需按 §1 的 URL 重新下载放入对应目录(ModelScope/GCS 均匿名可读);Python 依赖 h5py + hdf5plugin + numpy + matplotlib + pyyaml。
