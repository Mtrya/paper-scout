# Video to Data：公开工作流实跑

## 重写包的证据边界

本目录继承 `runs/2026-11-15-human2sim2robot-v2d/` 的实跑脚本、补丁与精选结果，供 2026-11-17 报告重写核查。此次重写未新增实验；下文“本轮”均指 2026-11-15 完成的实跑。旧运行包与旧报告保持不变。

## 调查对象与证据边界

上游为 [NVIDIA-Isaac/video_to_data](https://github.com/NVIDIA-Isaac/video_to_data)，检出 `1e862a8984bf28474c324c1b49db99b2da7ece80`。本目录保存实跑脚本、兼容补丁、配置和汇总证据；模型权重、原始视频帧、HDF5、episode 与 LeRobot 原始数据不进入运行包。

两条证据链严格分开：

- `code/results/reference/`：官方附带、经人工检查修正的纸巾参考轨迹。它验证 Vega 物理专家、Isaac Sim 三路录制、LeRobot 转换和逐帧审计。32 条 pilot 与 12 条正式 rollout 中选取 10 条，每条 519 帧、20 Hz；10/10 完成抬升、保持和放回。四组初始状态噪声均为零。`dataset_audit.json` 的最终结果为 `ok=true`：10 个 Parquet episode、5,190 行、30 个视频，每个视频恰好 519 个编码帧。
- `code/results/automatic/`：原始 13 秒、390 帧纸巾视频。完整自动链已走过掩码与几何重建、重定向、物理专家、三路录制及 LeRobot 导出。32 条 pilot 与 12 条正式 rollout 中，12 条正式轨迹全部完成抬升、保持和放回，选取其中 10 条导出。`dataset_audit.json` 的最终结果为 `ok=true`：10 个 Parquet episode、5,190 行、30 个视频，每个视频恰好 519 个编码帧。

## 主要结果

`code/results/reference/source_task_metrics.json` 记录源专家的真实任务指标：最大抬升 0.144602–0.158690 m，中位数 0.151996 m，保持 9.15–10.10 秒，最终回桌面。这里评价物理任务，不是策略存活到超时，也不是闭环视觉策略成功率。

`code/results/reference/render_throughput.json` 记录单张 RTX 4090 的成功轨迹相机重放：587.149 秒生成 259.5 秒三路 320×240、20 Hz 数据，即 2.262616 GPU 秒/数据秒。HDF5 为 1,414,558,935 字节，LeRobot 为 38,117,933 字节。大型中间文件已在审计后删除，元数据和审计结果保留在 `code/results/reference/gr00t_dataset/meta/`。

原片自动重建已经完成，`reconstruction_metrics.json` 核验最终 `result.npz`、`mesh.obj` 和 `manifest.json`：物体网格含 6,043 个顶点与 11,204 个面，相机、物体和双手在全部 390 帧中均有效，全部变换为有限值。物体在相机系中的相邻帧平移量中位数为 1.55 mm、95% 分位为 3.97 mm，相邻帧旋转量中位数为 0.104°。这些指标证明几何束完整且连续；表面接触与物理可执行性由后续阶段验收。最终四视图已经亲眼检查，保存为 `assets/tissue-refined-preview.gif`。

`reconstruction_stage_times.json` 将成本分口径保存：到手物对齐的首次成功生产累计 1,272.199 秒；后处理插值 1.6 秒，可选 2DGS 精炼 519.677 秒，最终打包 13.646 秒；四次诊断预览合计 1,611.733 秒，全部排除在核心生产成本之外。`code/results/scale_envelope.json` 是显式假设下的量纲演算，不是生产报价；其感知外推只采用到手物对齐的 1,272.199 秒口径。

`code/results/vlm/` 保存 DeepSeek 对参考帧、任务语义和仿真图像的实际输出。它支持任务分类与复核队列，不代替几何和物理状态验收。

## 代码与补丁

`code/setup_native_recon.sh` 安装原生重建环境，使用系统 TensorRT 10.0.1 与同版本 `trtexec`，并固定 FoundationPose 要求的 Warp 1.0.2。`code/setup_native_isaac.sh` 配置已有 Isaac Lab 安装；通过 `ISAACLAB_PATH` 指定位置。大文件下载应在免费远端网络执行。

`code/native_process.py` 将本工作流用到的容器命令映射到共享工作根中的原生 Python 环境，并记录各模块墙钟时间。`code/native_rg.py` 与 `patches/native-e2e-runner.patch` 对机器人数据阶段做同样的路径适配。位姿子进程使用系统 CUDA 用户态库，EGL 配置由运行脚本设置。

其余补丁分别处理 SAM2 原生提示路径、SAM3D 的 ModelScope 预置权重、gsplat 1.4 的背景维度和 TensorRT 10.0 运行时。它们属于执行接口兼容，不改变模型算法。MANO 使用用户许可资产，SAM3D 和其他模型沿用各自许可证。

典型执行顺序如下，`W` 是包含 `video_to_data/`、`Isaac-GR00T/` 和 `mano/` 的绝对工作根：

```bash
mkdir -p "$W/native"
cp code/*.py code/*.sh "$W/native/"
bash "$W/native/setup_native_recon.sh" "$W"
ISAACLAB_PATH=/workspace/isaaclab bash "$W/native/setup_native_isaac.sh" "$W"
bash "$W/native/run_tissue_reconstruct.sh" "$W"
"$W/native-venv/bin/python" "$W/native/inspect_reconstruction.py" \
  --bundle "$W/runs/tissue_box/reconstruction/result_slam_gravity_aligned" \
  --output "$W/runs/tissue_box/automatic/reconstruction_metrics.json"
bash "$W/native/run_tissue_retarget.sh" "$W"
bash "$W/native/run_tissue_collect.sh" "$W"
```

公开 ONNX 专家已经在本轮适配两条参考：人工修正参考的 10 条入选轨迹全部完成物理任务，自动重建参考的 12 条正式轨迹也全部完成物理任务。这是同一纸巾任务、同一机器人上的复用证据；跨任务能力须另行训练和测量。`automatic/source_task_metrics.json` 记录自动链最大抬升 0.175908–0.214715 m、保持 7.55–9.85 秒。`automatic/retarget_report.json` 的 390 个源帧均达到 IK 100 次上限、内部收敛计数为 0，说明运动学收敛门与物理任务门必须分开解释。

自动链 HDF5 为 1,522,468,222 字节，LeRobot 目录为 43,700,672 字节；`automatic/recording_summary.json` 与 `automatic/gr00t_dataset/meta/` 保留结构和导出元数据。两条链的大型 HDF5、episode 和视频均在审计后删除。本轮未微调 GR00T。

`code/results/automatic/e2e_config.json` 与 `code/results/reference/e2e_config.json` 将实际工作根替换为 `${V2D_WORKROOT}`，保留任务合同、关节定义、相机、帧率和采集参数。`inspect_recording.py`、`evaluate_source_tasks.py`、`export_lerobot.py` 和 `inspect_reconstruction.py` 分别核验录制、物理任务、训练数据和重建数组。`plot_reference_episode.py`、`render_hdf5.py` 与 `inspect_masks.py` 生成报告中的精选可视化。

## 报告资产

重写报告保留 `tissue-source-preview.gif`、`tissue-refined-preview.gif`、`tissue-retarget-preview.gif` 和 `automatic-replay-preview.gif`，依次展示原片、重建四视图、运动学参考与自动参考驱动物理专家后的三路仿真录制；`reconstruction-cost.png` 展示感知耗时，`v2d-official-overview.png` 展示官方模块组织。其余继承素材仍保留在旧运行包，本包只收录新稿实际使用的图像。
