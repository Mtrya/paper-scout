# SimWAM (2608.07468) — "推理时丢弃视频塔"审计

论文:华科+东风,Wan2.2-5B 视频专家 + 1.02B action DiT,MoT 联合流匹配 + isolated attention mask,NAVSIM 91.5 PDMS。有代码有权重。

## 我们的研究动作

**代码审计**(官方仓库快照):论文/README 称"只有 action DiT 被保留用于推理和 RL"。审计结论:**不符**。eval 走 action_only → `SimWAM.infer_action`(src/simwam/models/wan22/simwam.py:1074-1098)仍跑 `video_expert.pre_dit` + `mot.prefill_video_cache`(mot.py:257 起,30 层视频 DiT block 全跑,产 per-layer K/V cache),action DiT 去噪循环每层 cross-attention 读这份 cache。被丢弃的只是未来帧生成支路。isolated mask 属实(simwam.py:441),`infer_joint` 内置 joint vs action-only 一致性自检(atol 1e-2)。配置:视频 DiT 30 层 hidden 3072,action DiT 30 层 hidden 1024(configs/model/simwam_navsim.yaml)。

**权重静态分析**(ckpt_analysis.py,本地 CPU mmap):census + IL/RL delta,均完成,见 `code/results/census_delta.json`。census:IL payload(step 44400,bf16)= mot(视频 4.978B/30 blocks + 动作 1.021B + 双塔 text_embedding 22M)+ proprio_encoder(8→4096 线性);权重内无 VAE/T5(运行时从 Wan2.2-TI2V-5B 基座取)。delta:mot 1649 张量中 1551 有数值差,但视频塔 825 张量全在 bf16 重存噪声地板(rel 中位 1.5e-3),action 的 FFN/embedding/head 同样在地板(8e-4);真实 RL 信号只在 action 注意力矩阵——self_attn q/k/v/o rel 中位 1.3%-3%(o 最强,max 30%),cross_attn 中位 0.35%(max 4.5%)。"RL 只更新 action 专家注意力 rank-32 LoRA"声明在权重层面成立。

**推理探针**(probe_prefill.py,启智 4090 实例,官方 IL 权重 step 44400 真实加载 + Wan2.2 VAE/umt5):prefill 截断敏感性(k=0..30 冻结深度扫描,12 组输入:4 场景图 × 3 驾驶指令)+ prefill/去噪延迟分解,结果 `code/results/prefill_probe.json`,图 `../assets/simwam-prefill-cliff.png`(make_probe_fig.py)。三个发现:①k≤10 时相对 L2 中位 0.32-0.39 ≈ k=0(无视频塔)的 0.39,误差在 k=10→15 崩塌,k=15→0.05、k=20→0.01,12 组一致——动作专家读的表征在第 ~15 层收敛,后半座塔(~2.5B)对动作零贡献;②k=0 轨迹仍执行指令(最大逐点偏差 6.6m、航向 25°),驾驶常识在动作头+proprio/文本条件,视频流供场景几何;③完整 prefill 38ms vs 20 步去噪 750ms(4.5%),延迟税小,真实账单是 12GB+ 常驻显存。边界:相对 L2 是对全塔输出的保真度,非闭环 PDMS。

## 核心发现

- 推理形态不是"1B 动作头单飞",而是"冻结的 5B 视频世界模型当感知骨干 + 小动作头"——脚手架没有内化进权重,被留在权重里当眼睛。
- 低延迟的正确解释是视频塔开销为一次性 prefill;实测 prefill 仅占单次决策 4.5%(单前帧输入),真正的部署瓶颈是常驻显存(6B 双塔+VAE+umt5)。
- 动作头实际只消费视频塔前 ~15 层的表征;后半座塔服务的是被丢弃的未来帧生成——部署裁剪有实证抓手(相对 L2 口径,闭环未验)。
- 方法论:训练期特权组件部署时是否还在,必须逐案审计推理代码。

## 环境备忘(复跑用)

- SimWAM.pt 只在 HF;实例单连接 ~1MB/s,用 `parallel_download_simwam.sh`(16 分块,~8.5MB/s)。VAE/umt5/tokenizer 走 loader 默认 modelscope 源(~25MB/s),或 `pull_base.py` 预拉。
- venv 最小依赖在 remote_setup.sh;探针额外需要 accelerate/imageio/boto3/rich/termcolor(runtime.py 训练侧 import 链)。
- 本机 `code/simwam/checkpoints/` 持有 12GB IL 权重(完整性已校验)。

## 内容

- `code/ckpt_analysis.py`:census + delta 分析脚本
- `code/probe_prefill.py` + `remote_*.sh` + `parallel_download_simwam.sh` + `pull_base.py`:GPU 探针全链路
- `code/make_probe_fig.py`:探针绘图;`code/results/census_delta.json`、`code/results/prefill_probe.json`
