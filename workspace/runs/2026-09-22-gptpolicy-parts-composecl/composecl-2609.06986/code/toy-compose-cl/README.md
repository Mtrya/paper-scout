# toy-compose-cl

玩具尺度复现 ComposeCL(arXiv:2609.06986)的核心因子交互,并对组合机制做解剖。
论文全文:`papers/llm-agents/compose-cl-2609.06986.md`;官方参考实现:`code/compose-cl/`(本目录不依赖、不修改它)。

## 设置

- **模型**:从零训练的 2 层 decoder-only transformer(d_model=128,4 头,FF 512),字符级词表 42(a-z、0-9、`\n`、空格、`:` + PAD/EOS/replay token)。LoRA r=8 alpha=16(ρ=2)挂在全部 12 个 attn/MLP 投影上;持续学习阶段只有 LoRA 参数可训练。
- **预训练**:50k 字符的随机 QA 填充文本(与评测流完全不重叠的随机 key/value 对,同格式),2000 步。给 base 一个 QA 格式与有置信度的输出头。
- **任务流**:20 个 QA 任务 × 100 条 key(6 字符)→value(4 字符)关联,格式 `key: xxxxxx\nvalue: xxxx\n`,key 全局唯一(Symbol-QA 风格)。逐任务顺序 SFT,评测无任务 id、不许回看原始数据。
- **机制**(语义对齐官方 `core/`):
  - replay(`mechanisms.py: generate_replay / replay_kl_loss`,对齐 `core/generative_replay.py` 与论文 B.3):任务边界冻结上一模型为 teacher,从单个 replay token 无条件采样 300 条(τ_G=1.0,top-p 0.9);训练中当前 minibatch 配一个 replay minibatch,L = 0.25·CE + 0.75·τ_D²·KL(teacher‖student),τ_D=2(w=0.75 是论文在 Symbol-QA 上的注册值)。
  - merged LoRA(`model.py: LoRALinear.merge_adapter`,对齐论文 B.6.2):任务结束把 ρB\*A\* 折进 dense W,A 重新 Kaiming、B 归零,optimizer 每任务新建。shared LoRA 则同一对 A,B 贯穿。
  - SI(`mechanisms.py: ToySI`,对齐 `core/si.py` 与论文 B.5.1):在线路径积分 ω=−Σg_fit·Δθ,任务末 Ω+=max(0,ω)/(Δ²+ξ)(ξ=0.1),惩罚 λ·ΣΩ(θ−θ\*)²(λ=0.1,标定见下),只作用于 LoRA 参数,**按参数名绑定坐标**——merge 重初始化后旧 Ω/θ\* 仍施加在新坐标上,忠实复刻论文 B.5.4 的坐标错配。
- **臂**:A vanilla(shared)/ B merge / C replay(shared)/ D replay+merge / E SI+merge / F SI+replay+merge,2 种子(seed 0/1;任务数据跨种子相同,种子只控制初始化和训练随机性)。
- **评测**:学完任务 i 后评全部 j≤i(时序准确率矩阵 M,论文 §4.1),exact match(贪心解码);另记录 R(age) 衰减曲线。

## 机制解剖

- **merge 保护了什么**:对比 D 与 C 的逐任务衰减(每任务学完时准确率 → 最终准确率,R(age) 斜率与半衰期)。见 `results/fig2_decay.png`、`fig3_trajectories.png`。
- **replay 约束了什么**:D 臂第 20 个任务训练时抽样 50 step,统计 ∇L_replay 与 ∇L_SFT 在 LoRA 参数上的余弦相似度、replay 梯度的逐步自一致性、梯度范数比,区分"拉住旧函数"与"加噪声"。见 `results/fig4_grad_cosine.png`。
- **SI×merge 坐标错配**:E/F 臂记录每任务开始时(merge+重初始化之后、更新之前)的 SI 惩罚值(共享 LoRA 下应为 0,merge 下 >0 即错配的直接证据),以及旧 Ω 与新坐标实际重要性 ω_t 的相关性。见 `results/fig5_si_mismatch.png`。

## 运行

```bash
../toy-parts/.venv/bin/python run.py --smoke --out results_smoke   # 冒烟:3 任务 × 40 epoch
../toy-parts/.venv/bin/python run.py --out results                 # 全量:20 任务 × 2 种子 × 6 臂
../toy-parts/.venv/bin/python plots.py results                     # 聚合 + 图 + aggregates.json
```

## 与任务规格的偏差(都有实测依据)

1. **SFT lr 5e-4 → 2e-3,10 epochs → 100 epochs**:玩具尺度下 rank-8 LoRA 驱动冻结输出头,memorize 100 条任意关联需要 ~700 步;5e-4×10ep(70 步)习得率恰为 0,信号全无。2e-3×100ep 时 Diag≈1.0,对齐论文"即时习得近完美"的状态(App. E.2)。200ep 与 100ep 的保持率几乎相同(0.280 vs 0.274),不必更久。
2. **预训练文本:纯随机字符 → 同格式随机 QA 对,且 key 取自 300 个固定 key 的池**(与评测流不相交)。从零预训于均匀随机文本时输出头范数 ~0.2,softmax 近似均匀,LoRA 任何预算下都推不动分布(习得率恒 0);key 无上限随机时,base 学到"key 均匀分布于 36^6",SFT 后 p(·|s) 从不命中已训 key(实测 300 条 replay 样本中真实 key 命中 0 条)→ replay 锚完全失效。300-key 池让 base 学会格式、有置信度的头、以及"key 来自一个可枚举集合"的集中先验——SFT 后无条件生成在 τ_G=1.5 时 60% 精确复现已训关联、τ_G=1.0 时 100%,这正是论文 replay 机制依赖的前提。
3. **replay 权重 w 0.5 → 0.75、N_R 150 → 300、τ_G 1.5 → 1.0**:w=0.75 是论文 App. B.8 在 SYMBOL-QA 上的注册值(我们的玩具正是 Symbol-QA 类似物),N_R=300 是论文值,τ_G=1.0 是论文搜索集 {1.0, 1.5} 内的候选(玩具尺度下 1.5 把 ~40% 样本腐蚀成非真实 key)。规格值(w=0.5/150/1.5)下 D 臂增益仅 +4.5pp;注册值下 +15pp 且超可加。
4. **SI λ 1 → 0.1**:λ=1 在玩具尺度使惩罚(merge 重初始化后作用于新坐标)到第 5 个任务就压垮可塑性(acq→0);λ=0.1 时 SI 明显激活(任务初始惩罚随任务数持续增长)但习得存活。ξ=0.1 不变。
5. **25 任务 → 20 任务**:CPU 预算(全网格 ~90 分钟)。2 种子保留。
6. batch 16、LoRA r=8/alpha=16、τ_D=2、top-p 0.9、SI 路径积分与 ξ、merge/reinit 规则、评测协议均按规格与论文实现。merge 正确性已数值验证(merge 前后 logits 差 ~1e-5,fp32 舍入级)。

另:论文 B.8 的 5% 学习率 warmup 与梯度裁剪未实现(玩具训练稳定,影响不大);评测每任务取前 50 条(降 O(T²) 评测成本)。

## 产物

- `results/run_<arm>_seed<s>.json`:时序准确率矩阵、final/acq/forget、梯度探针与 SI 日志。
- `results/aggregates.json` + `fig1`–`fig5`:聚合指标与曲线。
- `results/summary.md`:结论汇总。
