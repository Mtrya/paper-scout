# dash-real-probe 代码

实验的全部脚本。设计、判据与结果见上级 `README.md`；这里只写怎么重跑。

## 环境

单卡 4090 足够（特征提取与训练都不超过 6 GB 显存）。启智平台上的实测配置：

- Workspace `可上网GPU资源`，Project `前沿课题探索`，Compute Group `4090-2-cuda13.2`，quota `1,10,100`。
- 镜像 `ngc-pytorch:25.02-cuda12.8.0-py3`（自带 torch 2.7 + cu128）。在 GPFS 上建 venv：
  `python3 -m venv --system-site-packages venv`，然后
  `PIP_CONFIG_FILE=/dev/null venv/bin/python -m pip install "transformers>=4.57" tokenizers accelerate pillow "numpy==1.26.4" pyarrow hf_transfer "jinja2>=3.1.6" -i https://mirrors.aliyun.com/pypi/simple/`
  （镜像内的 `/etc/pip.conf` 指向 pypi.org 与 pypi.ngc.nvidia.com，不覆盖会卡住；装完必须验证 `torch.cuda.is_available()`）。
- 权重走 ModelScope（`scripts/pget2.py` 并行 Range 下载 4.26 GB 单文件，约 20 MB/s），数据走 HF 直连或 hf-mirror；hf-mirror 的 resolver 有共享 IP 限流（429），并发不要超过 6。

## 阶段

所有阶段幂等：已存在的特征文件、checkpoint、评测 summary 会被跳过。

```bash
export PYTHONPATH=$PWD
W=<项目共享 workroot>/dash-probe    # 工作根：放代码、权重、数据与输出

# 1. 数据：只下载 episode 隔离划分用到的 88 条 + index.jsonl
python3 src/fetch_data.py 32 12 6          # (n_train, n_val, workers); data/seeds.json 定义全集

# 2. 冻结 VLM 特征（两相机各 16 token × 4 个路由层 + 指令嵌入 + 原始帧）
python3 src/extract_features.py --data-root data --split train --out feats \
        --model models/Qwen3-VL-2B-Instruct --batch 16
python3 src/extract_features.py --data-root data --split val   --out feats \
        --model models/Qwen3-VL-2B-Instruct --batch 16
python3 src/extract_text_pure.py models/Qwen3-VL-2B-Instruct feats/text_pure.npz

# 3. 拼 memmap、算训练集统计量
python3 src/pack_cli.py --ep-dir feats --out packed --split train
python3 src/pack_cli.py --ep-dir feats --out packed --split val
python3 src/make_norms.py --packed packed --out packed/norm.json --scene

# 4. 语义断言（新视觉取 t、慢分支取 t-Δ；见 README 的修正说明）
python3 src/check_semantics.py --packed packed

# 5. 训练（单进程、共享 RAM 缓存；24 次 run 约 1.4 GPU·h）
python3 src/train_many.py --packed packed --out out --epochs 10 --batch 128 \
  --arms sync_nofresh sync_fresh desync8_nofresh desync8_fresh \
         desync4_nofresh desync4_fresh leak_nofresh leak_fresh --seeds 0 1 2

# 6. 评测（固定 Δ 网格 + 通道消融，配对噪声）
bash scripts/run.sh eval        # 或 src/evaluate.py 单臂调用

# 7. 汇总、配对 CI、图
python3 src/analyze.py --out out --packed packed
```

## 文件

| 文件 | 作用 |
|---|---|
| `fetch_data.py` | 校验式下载（size + zip 结构），hf-mirror 与 huggingface.co 双源重试 |
| `pget2.py` | 并行 Range 下载器；ModelScope 的 HEAD 没有 Content-Length，改用 `Range: bytes=0-0` 探总长 |
| `dashdata.py` | NPZ 读取、JPEG 解码（PIL，与官方 cv2 路径逐像素一致）、11 维本体感受、13 维动作、episode 划分 |
| `extract_features.py` | Qwen3-VL 前向，存 4 个路由层的视觉 token 与指令均值；批处理预热模板，9.1 ms/帧 |
| `extract_text_pure.py` | 两张灰图 + 指令的前向，得到只含语言内容的嵌入，供 `other_txt` 消融 |
| `packed.py` | 每帧数组拼成 memmap；`gather()` 定义两路视觉的时间语义（关键：`img` 默认取 `idx`，`img_delta` 仅用于 `stalefresh`） |
| `check_semantics.py` | 对 `gather()` 的逐元素断言 |
| `model.py` | 4 层 cross-only flow-matching 头 + 从零 ResNet-18 |
| `train.py` / `train_many.py` | 单臂训练 / 单进程多臂训练（28 GB 特征缓存只加载一次） |
| `evaluate.py` | 固定 Δ 网格、通道消融、常数预测器 |
| `analyze.py` | 汇总 CSV、跨 episode 自助重采样配对 CI、工作版三联图（含补充臂） |
| `scripts/make_report_figure.py` | 报告版主图：只画核心四臂、通道消融删去文本两组、保留特征漂移；自带文本碰撞与裁切检查 |

## 图

```bash
# 报告版主图（读 results/*.csv 与 drift.json，不重跑训练）
uv run --with numpy --with matplotlib python code/scripts/make_report_figure.py
# -> code/results/fig_dash_probe_report.png，并打印文本碰撞/裁切检查结果
```

工作版图（含 `desync4_*`、`leak_*` 与文本消融）由 `src/analyze.py` 生成 `results/fig_dash_probe.png`。

## 配置（写死在代码里的常量）

`train_many.py`：10 epoch、batch 128、AdamW lr 3e-4 / wd 0.01、cosine、3 seeds、`H=8`、4 步 Euler。
`model.py`：4 个 cross-only block、宽度 512、8 头、FFN 1024；ResNet-18 通道 64/128/256/512。
`evaluate.py`：Δ ∈ {0,1,2,4,8}、stride 4、噪声种子 20261119。

## 远端清理

实验结束后删除实例（delete，不是 stop）与工作目录下的 `models/`、`data/`、`feats/`、`packed/`、`out/`、`venv/`；运行包只保留数值结果、小图与这些脚本。共享盘上不属于本实验的目录不要动。
