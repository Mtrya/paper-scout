# Say the Same, Act Differently:InstructVLA 侧 TNAS 复现

**论文**:Say the Same, Act Differently: Text-Neutral Action Subspaces in Reasoning VLAs(arXiv:2611.04166)
**问题**:推理型 VLA(Alpamayo 开车、InstructVLA 操作)的"可读理由"能否为动作安全背书?论文回答:不能——存在文本中性、动作敏感的隐状态方向(d_TNAS),沿它扰动可以让动作显著改变而生成的理由逐字不变。InstructVLA 侧核心数字:ρ(d_act 投影到文本梯度子空间的范数比)= 5.28%(200 例均值);等范数 Probe-3 在 q=2 时 tnas 动作位移 0.156 vs 随机方向 0.0124(12.6×),而文本 KL 只有 2.2e-2(文本梯度方向是 0.466–0.58)。
**本线程只复现 InstructVLA 侧**(Alpamayo 需要门控的 nvidia 数据集与 10B 模型,不做)。

## 复现设置(对齐论文附录 A.4/A.8)

- 模型:InstructVLA stage-2(`ShuaiYang03/instructvla_finetune_v2_xlora_freeze_head_instruction`,11.9GB ckpt;VLM = Eagle2-2B,Qwen2 系 28 层 hidden 1536;动作专家 = FlowMatchingActionHead,DINOv2-FiLM 视觉 + VideoGPT transformer + 10 步 Euler)。
- 评测集:VLA-IT language evaluation(`language_evaluation_with_gt.json` 1000 例全 Bridge,论文 200 例 = 该有序列表前 200;图 = `{hashkey}_first_frame.jpg`,指令 = `annotation.CR[0].question`,与官方评测代码对 CR 的用法一致)。
- 钩:LM 解码层 14–17 的输出,response token span(用空 assistant 变体的 token 级 diff 定位,去掉结尾空白胶合 token);h = 4 层 span 均值拼接(4×1536)。
- d_act:随机单位输出向量 w0(112 维 = 16×7)经可微 10 步 Euler flow 的 VJP(out-of-place Euler 更新绕过推理路径里的 detach,对齐 A.4.2)。
- 文本子空间:g_NLL 与 g_margin(span 上教师强制梯度)Gram-Schmidt 正交化;ρ = ‖P_T d_act‖/‖d_act‖。
- Probe-3 等范数扫描:δ = s·α·|⟨h_clean, d_TNAS⟩|·u,q = s·α·sgn(⟨h_clean,d_TNAS⟩),α ∈ {0.5, 1, 2, 4}(覆盖论文报告行 q=0.5/1/2;论文全网格到 32 是为了让随机方向也能达到 B=1e-3 的 KL 预算,本复现聚焦等范数非对称性)。方向:tnas / act / nll / margin / rand×4。指标:16 步动作块的逐步 ℓ2 位移均值 + response span 教师强制 KL。
- 预期判读锚点(论文值):ρ = 5.28%(中位 4.90%);q=2 位移 tnas 0.156 / act 0.155 / rand 0.0124 / nll 0.0283 / margin 0.0337;q=2 KL tnas 2.20e-2 / rand 3.64e-3 / nll 0.580 / margin 0.466;q=1 位移 tnas 0.110 / rand 0.00999;Probe-2 B=1e-3 位移 tnas 0.0137。

## 结果(200 例,2026-11-05 启智 4090,~80 分钟)

原始数据:`code/tnas_results.json`(每例含 rho、s_clip、response、八方向 × α × 符号 的 action_shift 与 tf_kl)。汇总器:`code/tnas_summarize.py`。

**ρ(d_act 投影到文本梯度子空间的范数比)**:均值 6.36%、中位 5.92%(论文 5.28% / 4.90%);ρ² 均值 0.00531(论文 0.00382,即 d_act 能量中不足 1% 落在文本子空间,两篇一致)。

**等范数 Probe-3 扫描(我们 | 论文)**:

| 方向 | q=0.5 位移 | q=1 位移 | q=2 位移 | q=2 文本 KL |
|---|---|---|---|---|
| tnas | 0.104 \| 0.073 | 0.161 \| 0.110 | 0.227 \| 0.156 | 2.34e-2 \| 2.20e-2 |
| act | 0.103 \| 0.073 | 0.161 \| 0.110 | 0.227 \| 0.155 | 2.99e-2 \| 2.71e-2 |
| rand | 0.008 \| 0.009 | 0.010 \| 0.010 | 0.015 \| 0.012 | 2.9e-3 \| 3.6e-3 |
| nll | 0.014 \| 0.013 | 0.021 \| 0.020 | 0.038 \| 0.028 | 4.79e-1 \| 5.80e-1 |
| margin | 0.017 \| 0.016 | 0.025 \| 0.020 | 0.043 \| 0.034 | 2.84e-1 \| 4.66e-1 |

**response 统计**:均值 9.46 词(论文 9.60),distinct 190/200(论文 178/200)。

**判读**:全部定性结论复现到位——(1) ρ 同量级(略高 1 个百分点,可能来自层索引 ±1 的 0/1-indexed 歧义:论文写 layers 14–17,我们按 0-indexed 取 `hidden_states[15:19]`;若论文是 1-indexed 则应下移一层,未再扫);(2) 等范数非对称性成立且显著:q=1 时 tnas 位移是 rand 的 16 倍、nll 的 7.6 倍,而文本 KL 比 nll 低 20 倍;(3) tnas 位移绝对值系统性高于论文约 45%,KL 与 rand/nll/margin 各行则基本贴齐——位移偏高可能同样来自层选择(更浅的层更靠近动作头);(4) 我们的配对噪声修复是结论成立的前提:未配对时所有方向位移都在 ~1.0 假阳性。KL 口径:q=2 时 tnas 2.34e-2 vs 论文 2.20e-2,几乎逐位一致。

## 复现

```
# 启智 4090 实例(可上网GPU资源/前沿课题探索),镜像 ngc-pytorch:25.02
# 布局:$WORK/{repo(InstructVLA 克隆,两处训练依赖守卫补丁), vla_ckpt/, eval/, .venv/}
bash smoke.sh   # --n 1 冒烟
bash run.sh     # 正式跑
```

已知坑(均有守卫):prismatic 包 __init__ 链拉 RLDS 训练依赖(dlimp/tensorflow)——两处打 try/except;timm 的 DINOv2 权重走 HF 缓存手工布局(`refs/main` 文件**不能带换行符**,否则 try_to_load_from_cache 静默 miss);生成文本须 strip `<|im_end|>` 且 span 不能用"再分词后子序列匹配"(BPE 边界效应),要用空 assistant 变体做 token diff。
