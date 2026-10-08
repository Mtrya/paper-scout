# 代码追踪:Long-WAM 发布资产核验与"访问≠使用"的代码对应物(arXiv 2610.10528)

本轮线程 A 的执行记录。研究动作:对 NVlabs/LongLive monorepo 的 `Long-WAM/` 子项目(HEAD `4ce82ab`,克隆于实验台 `code/LongLive/`)做只读代码追踪,回答两个问题:① 论文的中心对照"AR 预训练底座吃长上下文、双向不吃"在代码里对应什么;② 论文报告的数字有多少能从发布物复现。

一句话结论:**机制层全部可查(causal mask + streaming 路径 + 初始化单键切换),但中心对照实验本身不可复现——AR 初始化资产 `longlive_ar_video_expert.pt` 标注 "download URLs are TBD",双向对照的训练配方被明确排除在发布之外;13 个策略 checkpoint 已发,而论文的成功率数字全仓无一出现。这个 release 自带一份诚实的验证台账(`docs/VERIFICATION.md`),引用其数字时应连同限定一起引用。**

---

## 1. 发布清单:放了什么,没放什么

| 类别 | 状态 | 证据 |
|---|---|---|
| 训练代码 | 全放 | `scripts/train.py` + 5 个 benchmark 的 `train.sh`;`src/longwam/trainer.py`(1747 行);Hydra 配置树 |
| 评测脚本 | 全放(5 benchmark) | `scripts/{libero,robotwin2,domino,robocasa_gr1,robocasa365}/eval.sh` |
| 策略权重 | 13 个 HF bundle,revision 钉死 | `configs/checkpoints.yaml`:LIBERO/RoboTwin2 各两种去噪口径、GR1 上下文系列 0/2.4/4.8/9.6/19.2 s、RoboCasa365、YAM、Robot-S/M |
| 未放权重 | Domino(`repo_id: null`)、GR1 p768(只有配方) | `configs/checkpoints.yaml:66-69` |
| **AR 初始化资产** | **未放**,标注 "download URLs are TBD" | `docs/TRAINING.md:98-105`;`longlive_ar_video_expert.pt` 与 `ActionDiT_from_longlive_ar_1024hdim.pt` 只给转换脚本(`tools/weights/`) |
| AR 预训练代码 | 放(`stage1/`,LongLive 2.0 Robot S/M/L 课程) | `stage1/configs/train_{s,m,l}.yaml` |
| AR 预训练数据 | 没放,清单/receipt "download links TBD" | `stage1/README.md:29-32` |
| 论文数字 | **只有 107.4 ms 出现在仓库**;63.3%/78.7% 等成功率全仓 grep 无命中 | `infra/README.md:220` |
| 对照实验配方 | **明确排除**:"No ... historical imagination-ablation recipes" | `docs/MERGE.md:106` |

## 2. "访问≠使用"的代码对应物

AR 与双向的差别在代码里只有两个旋钮,训练目标与噪声调度完全共用:

1. **初始化权重**:`configs/model/long_wam.yaml:32` 的 `model.longlive_video_weights` 一个键切换;`docs/TRAINING.md:76-87` 明说 Robot 初始化对照只改了这一个键。
2. **注意力 mask 模式**:`wan_video_dit.py:995-1012` 三态(bidirectional / per_frame_causal / first_frame_causal),Long-WAM 配方写死 `per_frame_causal`。

代码层面"必须 AR"的硬证据:流式长上下文推理入口 `infer_joint_ar` 直接拒收非因果 mask(`longwam_base.py:1785-1789`)。双向模型仍可训练,但没有可用的长上下文流式路径。这意味着论文的"双向初始化无净收益"对照在发布物中**没有可执行的复现路径**(缺初始化资产 + 缺对照配方),结论只能以论文数字为准。

## 3. 架构与接口追踪(报告"是什么"一节的代码依据)

- 骨干 = Wan2.2-TI2V-5B 的 DiT 几何(hidden 3072 / 30 层 / 24 heads / head_dim 128),权重从 LongLive 2.0 Robot 的 AR checkpoint 初始化;转换脚本确认子模块名逐字节同名(`tools/weights/map_longlive_ar_to_video_expert.py:48-70`)。
- 动作专家 = 独立 ActionDiT(hidden 1024,层数/头数强制与视频专家一致),MoT 拼接注意力:两个专家各自算 q/k/v 后沿序列维 concat,一次 flash-attention 加联合 bool mask 再切开(`mot.py:592-701`)。
- 联合 mask 语义(`longwam_base.py:657-710`):clean 帧块因果自注意;noisy→clean 读 0..i;action→全 clean + 前 k 个想象未来帧;**video 永不看 action,action 永不看噪声**。
- IDM 接口:视频专家 4 步部分去噪到 σ=0.9(`configs/denoising/idm.yaml`),Prefill 成 KV cache 后 detach,动作专家带梯度读——视频专家拿不到 action 梯度(`trainer.py` P4 想象分支在 `torch.no_grad()` 下预填)。
- 上下文长度:`num_clean_frames = past_obs_size // 16 + 1`(16 = stride 4 × VAE temporal 4);GR1 的 417 帧 = 384 + 2×16 + 1。

## 4. 异步执行与 107.4 ms 的口径

- **没有 chunk 内 pipeline**:单次推理严格串行(想象未来 → prefill → 动作去噪)。并发在三层:chunk 级 async(`Schedule(H,R,S)`,约束 R/2 ≤ S < R ≤ H,推理在独立进程)、观测编码重叠(StreamingVAE 逐帧喂 + next 窗口预编码)、设备加速(NVFP4 仅量化视频权重,动作与 KV 保 BF16;Triton 分段动作注意力按视频 KV 长度特化;5090/Spark/Thor 三档 capability)。
- **107.4 ms 出处**:`infra/README.md:218-222`(RTX 5090:BF16 eager 356.0 → 共享优化 126.9 → 设备调优 107.4 ms,3.3×)。口径 = IDM V4/A4 含完整 observation VAE 与 video/action 推理,排除预处理/文本编码/控制器/IPC。**关键限定**:数字来自被导入的 `infra` 分支(`fbe5e55`),作者注明 "they were not remeasured during this integration"。

## 5. 真机与台账

- 真机部分放的是通用接口与配置模板(`runtime/robots/{g1,yam,franka}.py`、`deploy.py` 双重安全门禁),**权重/任务管线/成功率均未放**;G1 桥依赖外部受限仓库 `kaiknower/Long-WAM-G1-Dynamic-Task-Deploy`。`docs/VERIFICATION.md:125-155` 明说机器人部分 "all use mocks … no new task success rates"。
- 元事实:release 自带 `docs/VERIFICATION.md`,逐条声明哪些只跑了 CPU 测试、哪些 GPU 路径没跑、哪些评测被环境卡住。这是本期三个仓库里唯一的诚实台账,值得作为发布实践范式记录。

## 6. 复现障碍的精确说明(报告引用)

要复现论文的中心对照(AR vs 双向初始化 × 上下文长度),缺三样东西,均在仓库中被显式标注为未发布:

1. `longlive_ar_video_expert.pt`(AR 视频专家初始化)— "URLs are TBD"(`docs/TRAINING.md:104`);
2. 双向对照的训练配方与数据选择 — 明确排除(`docs/MERGE.md:106`);
3. AR 预训练数据(receipt)— "download links TBD"(`stage1/README.md:29-32`)。

可行的替代路径:用 `tools/weights/` 的转换脚本自行从 LongLive 2.0 Robot 公开权重构造初始化,再按 `configs/context/*` 的配方续训——但这需要 GB200 级训练资源,超出本轮巡航预算,故本线程止步于发布资产核验。

## 7. 产物

本线程的研究动作是只读代码追踪与发布资产核验;核验脚本与原始输出在本目录 `code/`:

- `code/audit_release.sh` — 七项核验查询(checkpoint 计数、TBD 标注、论文数字入库检查、107.4ms 出处、对照配方排除声明、因果 mask 强制点、FastWAM 继承计数),可对实验台克隆重跑。
- `code/audit_output.txt` — 本轮实际输出(HEAD `4ce82ab`)。要点:14 个 repo_id 条目(13 有效 + Domino 为 null);成功率数字零入库;107.4ms 仅在 infra/README.md 延迟表;FastWAM 引用宽口径 88 文件、含 provenance 标注窄口径 49 个(src/ 范围内;正文第 1 节的 55 为全仓口径)。

实验台克隆:`code/LongLive/`(gitignored,可再克隆)。
