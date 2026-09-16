# SyncWorld:标定 episode 作为动作-视觉映射的上下文接口

- 论文:Visual Calibration Enables World Models as Zero-Shot Simulators(arXiv 2609.09155),papers/world-model/syncworld-2609.09155.md
- 研究动作:启智 4090 实例复现官方评测,再做"可观性探针"——残缺标定下缺失自由度是否恰好失控

## 是什么

动作条件世界模型 W_θ(·|C^s, H_t, A_t):标定上下文 C^s(每个可控自由度一段配对帧+动作)作为 prompt 指定 setup 特定的 Action–Visual Mapping,单 checkpoint 零样本跨相机视角/本体。训练:RLBench+RoboCasa+RoboMimic 随机相机回放 + 反事实扰动 rollout + DROID;坐标增强(符号翻转/轴置换/平移缩放,视频不变)强制从 C^s 读语义;标定蒸馏让模型无标定时回退交互历史。全开源:代码+权重(yyuncong/SyncWorld)+评测集。

发布版与论文的差异(docs/paper_differences.md):骨干 Wan2.2→Cosmos3-Nano(MoT,动作进联合自注意力);动作单位 m→cm;**标定 12 段→6 段**(每自由度一段而非双向各一段)。

## 我们的问题(可观性探针)

与我们 8 月 EKF 外参标定线的核心发现同构:滤波里没被充分激励的方向不可观。在学习系统里对应的猜想:**标定 episode 没覆盖的自由度,在世界模型里应当失控**——而且退化应当选择性地出现在缺失自由度上,而不是均匀变差。这是从我们自己工作导出的、对 SyncWorld 机制边界的独立预测。

## 实验设计

- 实例 syncworld-probe(1×4090 48G,可上网GPU资源);workroot /inspire/qb-ilm2/.../syncworld(幂等 setup.sh)。
- 复现:examples/eval_gripperhead_fdm_rollout.py,maniskill/libero 各 50 episode,PSNR/SSIM/LPIPS,含 --calib-null 对照。
- 探针:破坏标定(去掉旋转段/单轴段),按未来动作块的主导自由度分桶测退化。

## 结果(2026-09-16 闭环)

**复现**:发布权重首次真跑。evaluation_libero 一集 272 帧、论文口径(35 步/CFG 5.0、单 4090 43 分钟):PSNR 27.78 / SSIM 0.954 / LPIPS 0.022。逐段 23.8–31.3 振荡、无单调衰减,接触密集段下探后回锚——闭环回写不丢锚。发布权重能跑且质量合理(论文主表数字因四点发布差异本不可复现,不矛盾)。

**可观性探针:预测被证伪**。先给全部 50 集做 DoF 画像(转动/平移总量比 0.53–4.24),选强对比对 ep14 开抽屉(4.24× 转动主导)与 ep26 捡字母汤(0.53× 平移主导),三条件(完整标定 / 全零 --calib-null / 抹旋转三槽 --calib-drop-slots 3,4,5)× 双视角,快速采样(20 步/CFG 1.0)。结果:抹旋转槽在两集上都与完整标定无差(27.4 vs 27.5、29.6 vs 29.4,±0.1dB 内符号翻转);全零标定在全部四组 episode×视角一致变差(-0.35~-0.93dB)。段级相关 4 组仅 1 组 -0.56、其余≈0 且符号跨视角不稳。结论:标定收益真实存在(full>null 一致),但不按论文宣称的逐自由度协议分解——"有标定"是一个比特,协议结构在发布权重上名存实亡。诚实标注:2 集×2 视角×单 rollout、快速采样,±0.2dB 在种子噪声内;"零效应"限于该灵敏度;12 槽论文版无法用发布权重检验。

产物:assets/sw_probe_result.png(左:复现逐段曲线;右:三条件对照);分析脚本 drafts/syncworld_analyze_probe.py + syncworld_plot.py(本地);数据 sw_{full,null,droprot}_{agentview,side}_rgb.csv、probe_analysis.json(drafts/)。

## 环境备忘(2026-09-16 实战版)

- **uv.lock 钉死的 URL 绕过索引镜像**:files.pythonhosted.org 直连 flaky(大 nvidia 包反复失败);`sed 's|https://files.pythonhosted.org/packages|https://pypi.tuna.tsinghua.edu.cn/packages|g' uv.lock`(同字节 hash 照过),0.6→9.7MB/s。
- **网络存储(qb-ilm2)是写瓶颈**:uv cache+venv 放实例本地盘,网络 cache `cp -a` 回灌(11G 读 2 分钟)。
- **空闲停机**:下载期 GPU 0% 必触发;stop 的 save-image 会 COMMITTING 近 1 小时且大概率失败,`inspire notebook cancel-save-image` 中止但 /root 全丢——HF_HOME 放网络存储、环境重建幂等(rebuild_env.sh,15 分钟)。
- **dl .done 标记不可靠**:以文件大小为准(本轮 shard3 缺 3.5G);模型加载报裸 FileNotFoundError 时看 "direct cause" 链上方拿真实文件名。
- **uvx 不在 PATH** 同样报裸 FileNotFoundError(checkpoint_db 用 `uvx hf download` 拉 Cosmos3-Nano 主干 33G,不在发布 checkpoint 内);run 脚本 export PATH=$WR/bin:$PATH。
- **eval 输出全缓冲**:整集 rollout 完才写文件,监控靠 nvidia-smi + py-spy dump。
- **评测集前 N 集常同任务**:先扫全集 DoF 画像,再用 --episode-indices 补丁(add_epidx.py)选强对比集;输出 episode_k 是过滤后位置,分析要映射回全局序号。
