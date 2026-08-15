# EVOKE 线程:无尽世界审计笔记(2608.13546)

代码仓库:github.com/SII-YuanyangYin/Evoke(Apache-2.0,权重全链放出)。远端推理验证进行中,此文件先记录代码审计结论,推理结果回填后定稿。

## 审计结论(代码 vs 论文)

1. **三阶段训练链全部放出**:stage1_camera_control → stage2_few_step_training → stage3_long_distillation → stage3_post_distillation,教师(high/low noise 专家各 5 shard)也放出。post_distill 学生 12 shard ≈57GB(bf16),单进程推理峰值 50.4GB VRAM(launcher 注释,2 进程可同驻 H200 139.8GB)。
2. **"每步调用有界"是代码里真机制**:FrameBank(frame_bank.py)= append-only keyframe 存储,8-bit 量化帧 + c2w 位姿,FIFO 驱逐;检索 = nearby_k(按 pixel_idx 最近)+ select_k(按共可见性 metric v1/v2/v3 贪心选最多 8 源)+ anchor。pipeline 里每 chunk:深度估计 → unproject → append → 按当前相机位姿检索 → batched projection z-buffer 渲染(warp 帧 + 可见性 mask)→ 只注入最粗 pyramid stage(--geo_warp_stage0_only)。可见性剪枝删掉无支撑 token,几何成本随覆盖而非会话时长。
3. **相机是 warp 的唯一通道**:infer_single.py 注释明说 "warp is this model's only camera channel"——chunk0 之后相机控制完全经几何 warp 注入(use_raw_sink_frames 只带历史帧不含相机)。这意味着状态库双重身份:持久记忆 + 控制通道。**这是论文没明说的耦合**:关掉状态库就同时失去相机控制——"记忆"与"控制"在 EVOKE 里不可分。
4. **监督时域与 chunk 结构**:训练=1 个真值前缀 chunk + 自迫 rollout(3 步/chunk,30s≈20 chunks),教师+critic 对全部 latent 帧打分(全窗 DMD),逐帧归一化。学生 3-step 金字塔(1+1+1),无 CFG,persistent decode(跨 chunk 连续帧分布,边界无 flicker,0.98-1.13× |dframe|)。
5. **诚实性细节**:i2v/t2v/segment 对该 ckpt 是 **zero-shot**(训练只有 v2v 条件,geo_condition_i2v_t2v_ratio=0.0)——launcher 显式警告。论文主图的 i2v 演示即零样本外推,这比论文正文更克制。
6. **"无限"的边界**:GEO_HIST_MAX_FRAMES 约束 DA3 点云(小时级要设),状态库保留有限时间窗几何;论文 4.4 的 recall 实验承认"可辨识而非像素保真重建"(X-Y dB 平台)。"endless" 的准确含义 = 每步成本有界 + 光度稳定,而非场景身份永久保持(4.2 明确:内容描述符 decorrelate 与真实视频对照同速)。

## 与 CMD/DreamX-Phi 的交叉点

- CMD 修"信息边界"(未来帧/未来控制),EVOKE 修"监督时域"(30s 全窗)与"条件调度"(每 chunk 文本)——EVOKE 引用 Zhu et al. 2026(双向教师 vs 因果学生错配)后自称聚焦互补轴。
- EVOKE 教师的 chunk-wise 稀疏注意力仍有"limited non-causal paths"(局部 chunk 内双向)——不是严格因果,与 CMD 的严格因果教师不同;3 步学生是 chunk 级 AR。
- DreamX-Phi 用 DMD2 蒸馏但学生一次性生成完整未来(离线完整动作轨迹)——双向教师与部署信息集一致,合法;CMD 语境是逐帧在线控制。三篇拼出规则:**教师的信息集必须匹配部署时的信息集**(CMD 时间边界、EVOKE 时域+调度+空间状态、DreamX-Phi 离线合法性)。

## 远端实跑计划(进行中)

- 环境:下载器(4090,可上网区)下载权重(57GB 学生+22.7GB text encoder+0.5GB VAE+5GB ViGeo)到共享 GPFS;venv torch 2.4+cu124+flash-attn(8.9;9.0 双 arch)也存 GPFS;H200(143GB,分布式训练空间,evoke-probe 实例)直读同 GPFS 跑推理。
- 实验:①冒烟 i2v 2 chunks;②主实验 i2v 22 chunks(~33s)计时 vs 论文 2.11s/chunk;③segment 6 chunks prompt 切换;④warp off 对照(--geo_drop_warp,同 seed)——测"状态库=唯一相机通道"的因果性。

## 远端环境搭建记录(可复用经验)

- 平台:下载用可上网区 4090(evoke-downloader),推理用分布式训练空间 H200(evoke-probe);两 workspace 共享同一 GPFS workroot,权重/venv 传一次两边可见。
- 分布式训练空间的 notebook 网关域名 ai-notebook-inspire.sii.edu.cn 公网/校内 DNS 均不解析;SNI 扫描定位 10.252.252.20,本机 127.0.0.1:7899 起 CONNECT 转发代理 + HTTPS_PROXY 环境变量打通 CLI(详见 INSPIRE.md 本次更新)。
- 权重下载:hf-mirror 单连接 0.6MB/s,16 线程 range 下载 12-16MB/s;requests 线程会挂死(慢读无超时),必须 socket.setdefaulttimeout + 每文件 wall-clock 预算(600s)+ daemon 线程,超时弃文件断点续传。
- NGC 镜像 pip 有来源不明的 ngc extra-index(改完所有 pip.conf 仍出现),每个包解析先对 pypi.ngc.nvidia.com 重试 5 次;不阻塞但拖慢。装依赖要用 PIP_CONFIG_FILE 显式最小配置;第二条 install 会把 torch 升级到 2.13(cpu)——装完必须 `--force-reinstall --no-deps` 重装本地 cu124 wheel。
- flash-attn 2.8.3 在 4090 下载器编译需 TORCH_CUDA_ARCH_LIST="8.9;9.0"(否则 H200 sm90 无 kernel)。

## 实跑结果(2026-08-15 完成)

- 环境:H200(143GB,分布式训练空间),venv 继承 NGC torch 2.7(torch 2.4+cuDNN 与 H200 驱动不兼容,CUDNN_STATUS_NOT_INITIALIZED;NGC 镜像 UCX 信号处理在长序列触发 segfault,sitecustomize 屏蔽 SIGFPE 绕过),diffusers 0.37+两处补丁(onnx stub、attention_dispatch 守卫强制 no-op)。flash-attn 2.8.3 与 diffusers 0.39 签名不兼容、torchvision→torch.onnx 链断裂,均经补丁绕过。
- 计时:diffusion 2.01s/chunk(warp 0.80 + 3-step 2.01),**论文口径 2.11s 复现**;完整管线 10.2s/chunk(decode+dump 7.4s);22 chunks=789 帧=32.9s,总 5:39,显存 44.7GB。
- segment:chunk3 prompt 切换实测生效(天空绿像素占比 0.15→0.81,地面结构保持)。
- warp-off 对照(同 seed 同轨迹,--geo_drop_warp):相机轨迹不再驱动画面(唯一相机通道证实);但零样本 i2v 上 warp on 反而色彩漂移(红通道 59→14、饱和度 141→233),warp off 稳定(56→51)——warp 的收益是训练域(v2v)内的,论文未覆盖此盲区。限制:pose 前 15s 几乎静止,相机运动检验强度有限;单 seed 单场景。
- 产物:assets/ 下 evoke_*_strip.png、evoke_warp_ablation.png、evoke_color_trends.png、evoke_captions.json、evoke_frame_stats.json(Florence-2 图说 + 光流/配准/颜色统计);视频 drafts/evoke-results/(不进 git)。
