# τ₀-VLA(2608.16885)深挖线程

阶跃星辰分层机器人基础模型。低层 VLA(Qwen3.5-2B 混合线性注意力骨干 + 0.23B MoT 流匹配动作专家)权重代码全开源;高层四件套(proposal/WM/value/reflective + 束搜索)未放出。

## 研究动作与发现

在启智 48GB 单卡(tau0vla-probe)上实跑官方 AgiBot World「敲锣」子集 openloop 评估(100 锚点 × H=30,G1 关节路线):

1. **零样本画像**:mean MSE 0.01126 / L1 0.0759;失败模式为夹爪常数偏移(预测 0.80 vs GT 恒 1.00),臂关节跟形状但漂移。
2. **微调动力学**:冻骨干只训 286M 动作专家,500 步 batch 1 仅 202 秒;MSE 0.01126→0.00264(4.3×),250 步已拿大头(0.00330)。
3. **FM 步数平台期**:K=1/2/4/10 → MSE 0.00255/0.00237/0.00248/0.00264,1 步去噪几乎不损失——与 ForgeWM 的 test-time scaling 结论跨域互证。
4. **静默解冻 bug**:config 的 freeze 标志在 from_pretrained 时被 Parameter 替换静默丢弃,2.5B 参数全变可训练→单卡 OOM。补丁在 model_builder.set_training_parameters 加载后重放冻结,补丁后 286M 可训练,已验证。

## 文件

- `code/tau0_policy_stepoverride.py`:openloop 评估脚本补丁,支持 num_steps 覆盖(FM 步数扫描用)。
- `code/tau0_exp2.sh`:FT 250/500 步微调 + 评估流水线。
- `code/tau0_fix.sh`:静默解冻 bug 修复补丁的应用脚本(补丁本体为对 model_builder.py 的改动,见脚本内 diff)。

论文缓存:`papers/vla/tau0vla-2608.16885.md`。
