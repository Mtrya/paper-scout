---
name: remote-compute
description: "当 `full` 流的深度调查需要超出本机的算力(GPU 探针、复现、消融、批量推理)时,按 本机 → Colab → 启智(Inspire) 的顺序选择执行后端,并把证据带回线程包。"
user-invocable: false
---

# remote-compute

本技能回答一个问题:当一个值得做的研究动作超出本机算力时,后端怎么选。答案是按代价从低到高依次尝试,而不是一律推给最重的平台,也不是放弃或降级成纯阅读。

## 何时使用

只在 `full` 流的研究动作需要 GPU、大内存或长时间运行时使用:复现关键实验、跑消融、对模型做探针、批量推理、评价基准子集。MinerU 解析、小规模 CPU 探针和代码阅读留在本机。

`lite` 流不向启智平台提交重型 GPU 训练或长时间作业;若仅需轻量级 GPU 验证或绘图,可使用轻量 Colab(如可用)或本机检查,超出该边界的设想记入报告的开放问题或日期记忆。

## 后端顺序

1. **本机**。计算密集但不需要大量下载的动作留在本机:小模型推理、轻量复现、绘图、CPU 探针。本机存储有限,先把数据需求算清楚,不要为了跑一个探针把几十 GB 权重落到本地盘。
2. **Colab**。需要单卡 GPU、但下载量与运行时长都可控的动作走 Colab。它是临时的、用完即弃的机器,不必担心磁盘被撑爆,适合探针、小规模复现、4B 级模型推理与绘图。月度计算单元有限(约 200 CU,大致相当于 100 小时 T4 或十几到几十小时 A100/H100),开销敏感的作业优先。
3. **启智(Inspire)**。多卡、长任务、需要持久 Workspace、镜像或数据集驻留的动作才上启智。本机与启智都要留意磁盘压力,小模型的大规模作业不要丢给启智——它比 Colab 更重,清理成本也更高。

上一档不可用、配额耗尽或确实不适合任务形态时再下沉一档,不要跳过 Colab 直接上启智。

## Colab

环境里有 `colab` CLI(`google-colab-cli`)时,headless 用法:

```bash
colab usage                                        # 先看计算单元余额与速率
colab run --gpu T4 --timeout 0 train.py --steps 200  # 一次性:建机、执行、取回产物、删机
```

`colab run` 的脚本参数按 `sys.argv[1:]` 原样转发。`--timeout` 计的是"多久没有输出",默认 30 秒;安静的训练步骤要么定期打印进度,要么显式设 `--timeout 0`。

需要多次交互时用会话:

```bash
colab new -s probe --gpu A100
colab install -s probe numpy pandas
colab exec -s probe -f probe.py --output-image drafts/probe.png
colab download -s probe /content/results.json drafts/results.json
colab stop -s probe
```

约定:

- 会话会被回收。任务要能被打断后重跑,不要把唯一副本留在远端;结果先用 `colab download` 取回,再 `colab stop`。
- GPU 型号写进线程包(例如 T4、A100、H100),重跑的结论不能跨型号外推。
- 认证配置与会话状态在 `~/.colab-cli-oauth-config.json`、`~/.config/colab-cli/` 下,属于本机环境;不要把 token 或配置副本写进工作区、运行包或任何被追踪的文件。

## 启智(Inspire)

1. 先读 `INSPIRE.md`(工作区根目录),获取本项目的启智上下文:默认 Workspace、资源组、路径约定、进行中的工作负载。
2. 加载 harness 级的 `inspire` 技能获取平台操作模型和命令细节;如果当前 harness 没有该技能,以 `inspire --help` / `inspire <group> --help` 为准。
3. 用 Live 查询确认资源(配额、空闲、镜像),不要凭记忆假设。提交后观察 events / logs / metrics 判断排队、失败与空转;如果一直排不到卡,检查项目配额与点券,按 `INSPIRE.md` 的约定换项目重试。
4. 工作负载用完即删(delete,不是 stop);临时镜像同样清理。

与启智的网络连接可能不稳定。命令超时、连接中断或响应缓慢时耐心重试,拉长等待、多试几次;不要因为一两次网络失败就判定平台不可用或放弃远程动作。

## 工具

- `scripts/pget.py`:并行 HTTP range 下载器(`pget.py <url> <out> [threads]`),hf-mirror 等慢源大文件(>10GB)必备——单连接可能只有 0.6MB/s,16-24 线程可到 8-15MB/s。自带 206 校验与分块重试;下完仍须做完整性抽查(safetensors header + safe_open 抽样;pth 用 `zipfile.testzip`)。

## 证据规则不变

远程执行不改变 `paper-deep-dive` 的证据契约:

- 探针、脚本、复现的持久副本晋升到 `runs/<run-id>/<thread-id>/code/`,面向报告的结果产物晋升到 `runs/<run-id>/assets/`。
- 不要让持久结果只存在于远端共享盘、Colab 会话或任务日志里;把关键数字、曲线和产物取回本地线程包。
- `README.md` 里写清后端与型号(本机 / Colab / 启智)、镜像或机型、配额,以及重跑方式。

## 兜底

本机没有 GPU、Colab 与启智又都不可用时,记 `BLOCKER.md`:写明缺的是本机 GPU、Colab 授权还是启智配额,以及要解开它需要什么条件。

- `colab` CLI 不在 PATH 上、未登录或授权失败:记为 blocker,不要把手工在浏览器里跑的过程伪装成自动执行,也不要假设自己能操作网页。
- `full` 流要求执行环境能提供 GPU(本机、Colab、启智任一)。三者都不可用时如实记 blocker 并在报告中说明,不要用玩具实验冒充"真实权重、真实数据"的实验。