# DA-LeWM(2608.18746)深挖线程

潜空间世界模型的「决策度量对齐」诊断框架:规划需要的是潜空间代价排序与真实代价排序一致(序数性质),与状态可解码性(信息充分性)逻辑独立。DA-LeWM = LeWM + inverse-dynamics 头 + goal-action 头。

## 研究动作与发现

用论文框架对自家 08-17 复训的一对模型(同种子同数据的 LeWM baseline 与 PSG 接地头变体,cube-single 检查点)做可证伪预言检验:**PSG 接地头修的是信息充分性,按定义不该修决策对齐。**

- 候选分布绕开论文 Cube 缺口(随机候选不接触方块→真实代价并列):32 条 GT 动作段加噪扰动(σ∈{0.05,0.1,0.2,0.4}×8)+ 16 条跨集 + 16 条高斯 = 64 条。
- 真实侧用 ogbench set_state 精确复位,30 对起点/目标(目标=同集 30 原始步之后);潜侧 3 帧上下文 + 6 模型步 rollout。
- **结果:预言成立。** Plan-Real Spearman 0.665→0.670,τ_a 0.515→0.506,soft-margin p 0.362→0.372,lat_range 13.0→12.6,全在噪声内;CEM elite 崩塌双模型复现(random 0.14-0.15 → elite 转负)。唯一位移:Claim-1 签名 PSG 0.722 vs baseline 0.583。
- 结论:信息充分性与决策对齐是两根独立的轴;下一步明确动作是给自家检查点加 inverse/goal-action 头。

## 文件

- `code/probe_align.py`:全套诊断协议复刻(Plan-Real / CEM-stage / Claim-1)。
- `code/align_results.json`:完整数值结果。

论文缓存:`papers/world-model/dalewm-2608.18746.md`。
