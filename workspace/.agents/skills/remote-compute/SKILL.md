---
name: remote-compute
description: "当深度调查需要超出本机的算力(GPU 探针、复现、消融、批量推理)时,把该研究动作路由到启智(Inspire)平台执行,并把证据带回线程包。"
user-invocable: false
---

# remote-compute

本技能回答一个问题:当一个值得做的研究动作超出本机算力时,怎么办。答案是路由到启智平台,而不是放弃或降级成纯阅读。

## 何时使用

只在研究动作需要 GPU、大内存或长时间运行时使用:复现关键实验、跑消融、对模型做探针、批量推理、评价基准子集。MinerU 解析、小规模 CPU 探针和代码阅读留在本机。

## 如何使用

1. 先读 `INSPIRE.md`(工作区根目录),获取本项目的启智上下文:默认 Workspace、资源组、路径约定、进行中的工作负载。
2. 加载 harness 级的 `inspire` 技能获取平台操作模型和命令细节;如果当前 harness 没有该技能,以 `inspire --help` / `inspire <group> --help` 为准。
3. 用 Live 查询确认资源(配额、空闲、镜像),不要凭记忆假设。提交后观察 events / logs / metrics 判断排队、失败与空转;如果一直排不到卡,检查项目配额与点券,按 `INSPIRE.md` 的约定换项目重试。
4. 终态且不再需要的工作负载先 stop 再 delete;临时镜像同样清理。

与启智的网络连接可能不稳定。命令超时、连接中断或响应缓慢时耐心重试,拉长等待、多试几次;不要因为一两次网络失败就判定平台不可用或放弃远程动作。

## 工具

- `scripts/pget.py`:并行 HTTP range 下载器(`pget.py <url> <out> [threads]`),hf-mirror 等慢源大文件(>10GB)必备——单连接可能只有 0.6MB/s,16-24 线程可到 8-15MB/s。自带 206 校验与分块重试;下完仍须做完整性抽查(safetensors header + safe_open 抽样;pth 用 `zipfile.testzip`)。

## 证据规则不变

远程执行不改变 `paper-deep-dive` 的证据契约:

- 探针、脚本、复现的持久副本晋升到 `runs/<run-id>/<thread-id>/code/`,面向报告的结果产物晋升到 `runs/<run-id>/assets/`。
- 不要让持久结果只存在于远端共享盘或任务日志里;把关键数字、曲线和产物取回本地线程包。
- `README.md` 里写清远程环境(Workspace、镜像、配额)和重跑方式。

## 兜底

如果 `inspire` CLI 不在 PATH 上或账号未配置,不要停滞:说明精确的障碍,退回本机可行的最强动作;若本机也无有意义的动作,按 `paper-deep-dive` 保留 `BLOCKER.md`。
