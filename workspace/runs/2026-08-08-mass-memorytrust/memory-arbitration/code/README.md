# memory-arbitration — When Memory Lies (2608.04574) 复现 + 扩展探针

对 SpatialSTALE 测试床的独立重建与三个探针实验:

- **探针 1**:模态鸿沟独立复现(文本 vs 视觉 × 两个开源 VLM 的 staleness 检测 P/R/F1)。
- **探针 2**:图像消融(正确 / 全空白 / 错配图像),直接量化"记忆主导审计"。
- **探针 3**:仲裁加权曲线(文本模式),受控记忆-观测冲突下四个因素(置信措辞 / 记忆年龄 / 环境先验 / 观测强调)对 stale 判定的影响。

## 布局

- `testbed.py` — 测试床生成器:8×8 FrozenLake(25% 洞,S=(0,0),G=(7,7),BFS 有解拒绝采样)、
  64 条记忆快照(行主序 `mem_000..mem_063`)、L1(均匀 5–7 处翻转)/L2(1–2 个曼哈顿半径 2 簇、
  12–16 次有放回抽样)变更、文本观测与 384×384 渲染(F=#A8C8E8 / H=#202020 / S=#2E8B57 / G=#DAA520,
  48px 格,坐标标签,红色三角 agent)。生成校验:L1 stale 比 0.094±0.013(论文 0.094±0.011),
  L2 0.138±0.026(论文 0.141±0.014)。
- `prompts.py` — OMCD 式审计 prompt(B=10 批,JSON 输出;论文未公布原文,按 §3.5/§4.2/附录 B 重建)
  与探针 3 单条审计 prompt。
- `build_inputs.py` — 生成全部输入到 `data/`(master seed 2024,30 个实例 seed,全部确定性)。
- `remote_runner.py` — 远端 vLLM 批量推理,原始输出逐条落盘 `results/raw/<model>.jsonl`,幂等续跑。
- `analyze.py` — 解析与统计,写 `results/tables/RESULTS.md`(与原始输出分离)。
- `extract_cases.py` — 探针 2 verbatim 案例抽取。
- `remote_setup.sh` — 启智实例部署(venv + vLLM + ModelScope 权重下载)。

## 复跑

```bash
# 本地(生成输入,纯 CPU)
uv venv .venv --python 3.11 && uv pip install --python .venv/bin/python numpy pillow
.venv/bin/python build_inputs.py     # -> data/tasks.jsonl, data/gold.json, data/images/

# 启智实例(<workroot>/memory-arbitration/)
bash remote_setup.sh                 # venv + vllm + 下载两个模型
setsid nohup .venv/bin/python code/remote_runner.py \
  --model-path models/Qwen3-VL-8B-Instruct --model-name qwen3vl8b \
  --data code/data --out results/raw/qwen3vl8b.jsonl > logs/qwen.log 2>&1 < /dev/null &
setsid nohup .venv/bin/python code/remote_runner.py \
  --model-path models/GLM-4.1V-9B-Thinking --model-name glm41v9b \
  --data code/data --out results/raw/glm41v9b.jsonl > logs/glm.log 2>&1 < /dev/null &

# 本地(统计,与原始输出分离)
.venv/bin/python analyze.py          # -> results/tables/RESULTS.md
.venv/bin/python extract_cases.py glm41v9b --condition blank --n 5
```

## 模型版本与来源

| 模型 | ModelScope repo | 权重 | 采样参数(模型 generation_config) |
|---|---|---|---|
| Qwen3-VL-8B-Instruct | `Qwen/Qwen3-VL-8B-Instruct` | bf16 safetensors ×4 (~17.5GB) | T=0.7, top_p=0.8, top_k=20 |
| GLM-4.1V-9B-Thinking | `ZhipuAI/GLM-4.1V-9B-Thinking` | bf16 safetensors ×4 (~20.6GB) | T=0.8, top_p=0.6, top_k=2 |

下载:`remote_setup.sh` 走 ModelScope `resolve/master` 直链,>200MB 文件用 `pget.py`(24 线程 range 下载),
逐文件校验 Content-Length 并留 `.done` 标记。

## 结果数字位置

- 探针 1 P/R/F1 表、探针 2 翻转率/相似度表、探针 3 逐因素表:`results/tables/RESULTS.md`
- 原始模型输出:`results/raw/<model>.jsonl`(每行 task_id + response,可复核)
- 探针 2 典型案例:`results/tables/cases_<model>_<condition>.txt`
