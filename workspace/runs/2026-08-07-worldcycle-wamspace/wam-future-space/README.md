# 线程 B:WAM 未来监督空间之争——ST-WAM 三元组诊断独立复现

论文:ST-WAM(2607.28993,DINOv3 语义空间,DSFE+CAIR)vs SG-WAM(2608.01397,策略自身 EMA 空间 + VGGT 几何)。LIBERO-Plus 零样本 72.8% vs 73.0%(后者 0.9B)。

## 代码审计(已完成,结论)

ST-WAM 三机制(三分支 MoT 结构化掩码、τ_v=τ_s 同步 + τ_a 独立、CAIR 只注入 action context)与代码一致(`src/fastwam/models/wan22/fastwam_vae_dino_mot.py:1151-1197`)。出入:推理"action-only"被夸大(VAE/DINO 专家仍做干净首帧 KV prefill,1.24× 延迟来源);RoboTwin 全局 batch 论文 1024 vs 仓库 128;**三元组诊断(290 组)与 180 例幻觉审计的代码/数据均未放出**。SG-WAM 仓库只有 "Code is coming soon"。

## 复现实验

数据:physical-intelligence/libero(无视频,PNG bytes 嵌 parquet)↔ lerobot/libero_plus(AV1 mp4,用 imageio_ffmpeg 读)。LIBERO-Plus 复用 LIBERO 初始状态只换视觉扰动。抽帧脚本 `code/extract2.py`;人工按任务+初始布局配对(code/montages/ 为逐任务拼图),`code/pairs.json` 99 组三元组(9 任务 × 13 扰动类型,含 16 组 clean 对照)+ 45 组控制。

特征(论文未说明,ours):DINOv3-vits16 patch token 均值余弦;Wan2.2 VAE 单帧 latent 展平余弦。脚本 `code/triplet_diagnosis.py`,结果 `code/triplet_results.json`。

## 关键数字(我们 vs 论文)

| 量 | DINOv3(我们/论文) | Wan VAE(我们/论文) |
|---|---|---|
| S_same(扰动保持) | 0.862 / 0.904 | 0.617 / 0.686 |
| 判别率 | 94.9% / 95.2% | **98.0% / 60.0%(翻转)** |
| 同 episode 初始↔终帧余弦(控制) | 0.952 | 0.806 |
| 跨任务同场景余弦(控制) | 0.862 | 0.467 |

## 结论

1. 核心主张成立:语义空间扰动保持度全面领先,结构有理(纹理 0.83 vs 0.50;相机 0.86 vs 0.41;极限烟雾两者皆失守)。
2. "VAE 分不清扰动初始与干净终帧"(60%)在 our 协议下不成立(98%)——该数字协议敏感。
3. 新发现:DINOv3 单帧特征不变性过剩,几乎编码不了任务进展(0.952)——ST-WAM 路线未被讨论的软肋,也是 SG-WAM 押注内生空间的合理性所在。
4. 下一步:把三元组协议跑到两个模型自己的 rollout 上,测未来预测的表征漂移(训练分布幻觉真正发作处)。
