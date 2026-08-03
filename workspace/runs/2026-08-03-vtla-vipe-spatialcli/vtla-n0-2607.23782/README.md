# N₀-VTLA (arXiv:2607.23782) — 线程记录

## 这条线做了什么

论文把触觉当作预测目标(latent tactile tokens z 预测未来 50 步动作 chunk 的净
接触变化),并声称放出的 arch C 权重在架构上堵死了 vision-language shortcut。
repo(neoteai/N0-VTLA)给了完整模型代码、7.9GB base 权重与探针方法论
(docs/TACTILE_CAUSAL_PROBE.md),但官方探针依赖未放出的 NeoData 数据加载器。
本线程用合成批次在真实放出权重上独立复刻了 z 敏感性探针。

## 代码解剖发现

- 论文 §2.2 写的架构(z ∈ R^{10×d},kv=[VL;g],"joint_kv")与放出权重不符:
  `NeoteAI/n0-vtla-base` 的 config.json 是 n_latent=5、predictor_arch="tactile_kv"
  (arch C:触觉是 z 唯一 K/V,VL 只调制 query),另有 z_gate 零初始化 +
  vl_dropout 0.15。权重逐键吻合(missing=0 unexpected=0)。

## 实验:合成批次 z 敏感性探针

- 脚本:`code/probe_z_synthetic.py`(复刻官方五组扰动:z_real/z_null/z_shuffle/
  z_vlswap/z_padpert;centered cosine;R = 触觉/VL 敏感度)。
- 批次:8 样本,Physics-IQ 条件帧作 base RGB + 8 条不同 prompt + 合成凝胶读数
  (暗基线 + 样本特定亮斑)。两个独立批次 seed 1000/5000。
- 运行:启智 4090 实例 wan22-vipe,`$W/n0-vtla/`(PYTHONPATH=repo),
  venv `$W/vtla-venv`(torch 2.7.1, transformers 4.53.2, jaxtyping==0.2.36)。
- 结果:`code/probe_synth_s1000.json` / `code/probe_synth_s5000.json`。
  R = 0.41 / 0.28(远小于官方参考值 A 架构 3.9);z_null 0.76/0.74(触觉存在性
  被读取);z_vlswap 0.84/0.94 与 z_padpert 0.80/0.82(VL/指令主导 z);
  z_xsample −0.11/−0.16(无坍缩);z_gate = −0.0116(base 权重动作专家尚未消费 z)。
- 置信度边界:合成触觉相对 NeoData 真实分布 OOD;结论否决的是"架构本身保证
  z 由触觉驱动",不否决论文训练管线在真实数据上的表现。
- 出图:`code/make_probe_figure.py` → `../../assets/vtla-probe-bars.png`。

## 踩坑记录(复用价值)

- 不可对模型全局 `.to(bf16)`:PaliGemmaWithExpertModel 内部按 config.dtype
  处理精度,tactile predictor 必须保持 fp32,否则 LayerNorm 报
  "expected scalar type Float but found BFloat16"。
- jaxtyping 必须 0.2.36(高版本报 `_check_dataclass_annotations`)。
- safetensors `load_model` 返回的 missing/unexpected 是 set,JSON 序列化前转 list。
