---
name: paper-source
description: "从配置的来源发现并拉取宽泛的近期论文池。默认来源是 Hugging Face Papers;随着工作流演进,更多来源会加入本技能。"
user-invocable: true
---

# paper-source

本技能负责论文的获取方式。

## 要收集什么

每个候选论文保留足够支撑后续分诊的元数据:

- 标题
- 论文 id 及其他标识符
- 作者
- 摘要或概要片段
- 项目主页、代码仓库、模型卡、数据集卡等链接(如有)

如果将候选池落盘,写入 `drafts/`(暂存区,可被覆盖)——绝不写入 `runs/`,那里只放持久笔记。

## 来源 A:Hugging Face Papers

用 `hf` CLI 做快速的论文侦查、扫读与筛选,不用于精读。

列出论文:

```bash
hf papers ls --sort trending --limit N
hf papers ls --date YYYY-MM-DD --limit N
hf papers ls --week YYYY-Www --limit N
```

获取论文元数据:

```bash
hf papers info <paper-id>
```

按关键词搜索:

```bash
hf papers search "<query>" --limit N
```

结构化输出更易处理时,使用 `--format json`。

## 其他来源

HF 是默认池,但不是唯一的网。主动检查当前环境里可用的其他论文来源与数据工具(搜索插件、arXiv 接口、订阅源等),在 HF 覆盖不足时用起来——尤其是按领域和日期做补充扫荡。

## 来源 B:arXiv PDF + 本地 MinerU

精读论文用 arXiv PDF + 本地 MinerU,全文与图表抽取更可靠。

端到端脚本:

```bash
.agents/skills/paper-source/scripts/arxiv-mineru-parse.sh <paper-id> <area> <slug>
```

脚本将 PDF 下载到 `drafts/`,调用本地 `mineru` 解析,把产出的 Markdown 复制到 `papers/<area>/<slug>-<paper-id>.md`,然后清理临时 PDF 与解析输出。依赖 `curl` 和本地 `mineru`(经 `uv tool install 'mineru[all]'` 安装)。首次运行需要模型权重;若自动下载失败,先执行 `mineru-models-download -s modelscope -m pipeline`。

常用选项:

```bash
.agents/skills/paper-source/scripts/arxiv-mineru-parse.sh <paper-id> <area> <slug> --backend pipeline
.agents/skills/paper-source/scripts/arxiv-mineru-parse.sh <paper-id> <area> <slug> --copy-images
```

`--copy-images` 会把抽取的图片一并复制到 `drafts/images/<slug>-<paper-id>/`,便于深挖与写作时引用;进入报告的图片再按 `paper-deep-dive` 的证据规则晋升到运行包。

只需要 PDF 审计产物时:

```bash
.agents/skills/paper-source/scripts/fetch-arxiv-pdf.sh <paper-id> <slug>
```
