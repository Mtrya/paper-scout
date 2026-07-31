# Thread: ABot-World-0 (2607.19191) — 交互世界模型的代码与权重核验

## 研究问题

论文声称在单张 RTX 5090 上实现 720p/16 FPS/1.2s 延迟的"无限"交互世界 rollout。代码和权重已开源(github.com/amap-cvlab/ABot-World,HF acvlab/ABot-World-0-5B-LF),这次巡航直接核验:动作如何注入、"无限"靠什么撑住、16 FPS 的代价是什么。

## 尝试了什么

1. 端到端追踪推理代码(pipeline/causal_inference.py, wan/modules/causal_model.py, web_client/),覆盖动作注入、causal chunking、KV cache、LongForcing 的推理侧形态、实时工程栈。
2. 只拉 safetensors 头(约 8 MiB)枚举全部 831 个张量,对比 Wan2.2-TI2V-5B stock 架构,量化动作条件化的权重增量 —— 见 `code/inspect_abotworld0_weights.py`(可重跑,无需 torch/GPU)。
3. 对照论文全文(arXiv HTML v1,无附录)核验代码与声称的一致性。
4. **真实验(猜想检验,启智 4090 实跑)**:论文展示了 60s 无视觉伪影,但从未测长程 action-following。猜想:视觉质量不崩时,听话度可能已先衰减。在 4090(48GB)上跑通完整推理栈(int8-torchao + flash-attn 2.8.2 + sdpa;lightx2v_kernel 用 stub 绕过),640 block(≈10.7 分钟 @12fps)rollout,动作协议 J×8s/W×8s/L×8s/W×8s 循环,对输出视频逐 block 做光流分析(yaw 段:全局水平流方向 vs 指令;walk 段:径向膨胀量)—— `code/abotworld_flow_probe.py` + `code/abot-640-probe.{csv,json}`。

## 保留的证据

- `code/inspect_abotworld0_weights.py` — 权重头探针。2026-07-31 重跑结果:总参数 5,270,329,536;唯一非 stock 张量是 6 个 `act_control_adapter.*`,合计 270,541,824 参数(5.1%):conv [3072,8192,2,2] + 两层 residual 3×3 conv [3072,3072,3,3]。
- `code/abotworld_flow_probe.py` + `code/abot-640-probe.{csv,json}` — 640-block 可控性分析(639 个有效 block)。`code/frames/` — block 100(健康)/165(崩溃窗)/630(末尾)抽帧。
- 4090 实测(int8-torchao):峰值显存 31.2GB allocated / 39GB reserved(论文 5090 FP8 口径 19.3GiB);640 blocks 生成 48 分钟(≈4.5s/block,含加载);16-block smoke 画面完好。

## 实验:可控性不是衰减,是阵发性崩溃(episodic collapse)

**猜想被部分证伪,真实现象更有意思**:yaw 指令跟随率(16-block 滑窗)在 10.7 分钟里**没有长期衰减**(前 1/4 段 0.96 vs 后 1/4 段 0.94),但出现两次**持续数十秒的全面失控**:block 130-180 跟随率塌到 0.25,block 460-500 塌到 0.74-0.82,之后都自行恢复到 ~1.0。崩溃窗不与协议相位转换对齐。抽帧显示崩溃窗(block 165)同时伴随画面涂抹伪影与角色静止,健康窗与末尾帧画面完好——视觉质量与可控性**同步阵发性恶化、同步恢复**,而非我们猜想的"听话度先于画质衰减"。walk 膨胀指标被场景纹理混淆(末段反而 2.16×),不作结论。

**解读**:滚动窗 + 5 张参考图 sink 的机制在统计意义上让 rollout 无限续命(无长期漂移),但模型会反复进入"退化态"(画面涂抹 + 拒绝指令)并靠 sink 重新锚定恢复。这恰好印证了线程对"无限"机制的分析——超过 ~7 秒的一致性只活在 5 张参考图里,当滚动窗内容与 sink 冲突大到临界点,可控性与画质会一起暂时让位。

**局限**:N=1 rollout、单 prompt、int8-torchao + sdpa 路径(非论文 5090 FP8 + SageAttention 配置);光流指标本身有噪;崩溃的触发条件(场景内容?窗口状态?)未分离,是明确的后续问题。


## 关键发现(报告用)

- **动作注入**:8 键 multi-hot(WASD+IJKL)广播成 32 通道全分辨率常量图,经 PixelUnshuffle(16)+Conv(8192→3072)+1 个 residual block,在 patch-embedding 输出处加性注入(causal_model.py:1990-2009)。一个 8-bit 按键向量配了 270M 参数的 adapter(约占模型 5%)。控制粒度是每 block 一次采样 = 每 12 输出帧,实际控制带宽 ~1.3 Hz。
- **"无限"的真相**:21 个 latent frame 的滚动 KV 窗(local_attn_size: 21,约 7 秒记忆)+ 5 张离线预生成的场景 canonical 视图(head/left/right/front/back,仓库自带 outputs/ref_image_cache/),以负 temporal RoPE 位置钉为永不被驱逐的 attention sink。超过 ~7 秒的世界一致性只活在这 5 张图里;web demo 还硬上限 600 blocks(~10 分钟)。
- **16 FPS 的代价**:4 步 DMD 蒸馏学生 + FP8 per-token 动态量化(AngelSlim 衍生 quantizer)+ SageAttention3 + Helios Triton kernels + TAEHV 轻量解码器;论文自己的 Table 2 显示 16 FPS 对应 MXFP4 激进配置,而 FP8 配置 VRAM 反而最低(15.93 GiB)。仓库 demo 实际以 12 FPS 录制/播放(web_client/config.py VIDEO_FPS=12)。
- **不一致点**:offline inference() 路径 act_context_scale 默认 0.0(动作被静默关闭)vs streaming 路径 1.0;adapter 标记 requires_grad_(False) 却带训练好的权重 —— 训练管线未开源的佐证。
- **论文侧补充**:动作 token 与 VAE temporal patch=4 严格对齐(4 帧 × 8 键 = 32 维);caption 刻意剔除相机运动描述以防模态泄漏;WorldRoamBench 上几乎全面第二(输给未开源的阿里 HappyOyster 5/7 项);LongForcing 消融只到 60s 且只有曲线,day-scale 声明缺量化支撑,且从未测长程 action-following 保持率。

## 如何重跑

```bash
python3 code/inspect_abotworld0_weights.py   # 权重头分析,CPU 即可
python3 code/abotworld_flow_probe.py <video.mp4> <out_prefix>   # 可控性光流分析,需 cv2/matplotlib
```

代码追踪基于 `code/abot-world/`(已清理的浅克隆);关键引用见上文 file:line。

## 对报告意味着什么

ABot-World-0 的真实贡献是一套完整可跑的实时交互世界模型工程栈(蒸馏+量化+kernel+轻量 VAE),但"无限 rollout"是滚动窗 + 5 张参考图的务实妥协,而非持久世界记忆;动作通道是重参数、低带宽的 roaming 级控制。报告应把卖点从"infinite"重新定位到"single-GPU real-time engineering + 诚实的 drift 缓解边界"。
