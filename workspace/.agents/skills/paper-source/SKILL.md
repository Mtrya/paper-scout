---
name: paper-source
description: "从配置的来源发现近期论文与研究文章候选,并获取精读文本。默认论文源是 Hugging Face Papers,科学空间是独立研究线索源。"
user-invocable: true
---

# paper-source

本技能负责研究候选的发现与全文获取。

## 要收集什么

每个候选条目保留足够支撑后续分诊的元数据:

- 标题
- 来源与类型(`paper` 或 `blog_post`)
- 论文 id、文章 id 及其他标识符
- 作者与发布日期
- 摘要或概要片段
- 项目主页、代码仓库、模型卡、数据集卡、引用论文与系列文章等链接(如有)

如果将候选池落盘,写入 `drafts/`(暂存区,可被覆盖)——绝不写入 `runs/`,那里只放持久笔记。

## 通用 RSS/Atom 获取

`scripts/fetch-rss.py` 是所有订阅源共用的零依赖拉取器,支持 RSS 2.0 与 Atom,向标准输出写候选 JSON。`--feed-url` 与 `--source` 必填;`--kind` 默认为 `blog_post`,`--since YYYY-MM-DD` 与 `--limit` 控制时间窗和条数。同一站点有多个官方域名时,用 `--canonical-host` 统一文章 URL 与 id。新网站只需提供这些参数及其筛选、精读规则,不必新增解析器。

## 发现源 A:Hugging Face Papers

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

## 发现源 B:科学空间 RSS

科学空间的近期文章由官方 RSS 获取:

```bash
python3 .agents/skills/paper-source/scripts/fetch-rss.py \
  --source scientific_spaces \
  --feed-url https://spaces.ac.cn/feed \
  --canonical-host spaces.ac.cn \
  --limit 10 > drafts/scientific-spaces-rss.json
```

按上次巡航日期向前留少量重叠时使用 `--since YYYY-MM-DD`,再按规范 URL 去重。输出保留文章 URL、发布日期、摘要和正文开头引用的链接;RSS 正文有截断,只用于发现与分诊,不用于精读。`spaces.ac.cn` 不可达时改用 `--feed-url https://kexue.fm/feed` 并保留相同的 canonical host,仍失败则跳过该来源并继续其他侦查。

评论某篇论文的文章作为该论文的外部信号,相关论文按 arXiv id 去重;原创推导或系列文章可以独立成为研究线程,获取全文后遵循 `paper-deep-dive`。作者身份不替代质量筛选,也不给该来源设置每轮入选配额。

### 全文与历史检索

选中科学空间文章后,按 archive id 从非官方镜像 [`spaces-index`](https://github.com/caojiaolong/spaces-index) 获取 Markdown:

```bash
curl -fL "https://raw.githubusercontent.com/caojiaolong/spaces-index/main/data/articles/<archive-id>/article.md" -o "papers/<area>/<slug>-spaces-<archive-id>.md"
```

镜像的 `data/posts_classified.json` 可按需下载到 `drafts/`,用于关键词、引用论文与系列上下文检索;不要每轮克隆全站正文或重跑镜像抓取器。镜像尚未收录、正文校验失败或公式可疑时回科学空间原页。报告引用仍链接原站,不把镜像全文复制进报告或运行包。

## 其他发现源

HF 是默认论文池,但不是唯一的网。主动检查当前环境里可用的其他论文来源与数据工具(搜索插件、arXiv 接口、订阅源等),在 HF 覆盖不足时用起来——尤其是按领域和日期做补充扫荡。

## 精读载体:arXiv PDF + 本地 MinerU

精读论文用 arXiv PDF + 本地 MinerU 4.x,全文与图表抽取更可靠。

端到端脚本:

```bash
.agents/skills/paper-source/scripts/arxiv-mineru-parse.sh <paper-id> <area> <slug>
```

脚本将 PDF 下载到 `drafts/`,调用本地 `mineru` 解析,把产出的 Markdown 复制到 `papers/<area>/<slug>-<paper-id>.md`,然后清理临时 PDF。依赖 `curl` 和本地 `mineru`(经 `uv tool install 'mineru[all]'` 安装)。

MinerU 4.x 是客户端-服务架构,一次性配置:

```bash
mineru-models-download --tier basic -s modelscope   # standard/advanced 需要带 CUDA 工具链的 VLM 引擎,本机没有
mineru config set parse_server.local.mode self_hosted
mineru config set parse_server.local.self_hosted_url http://127.0.0.1:8000
```

每次使用前要起两个常驻服务(脚本会检查健康端点,不通则报错并提示):

```bash
mineru server start
mineru-api --port 8000 --tier basic --no-advanced --no-flash   # 用后台任务起,不要用 shell 内 &
```

常用选项:

```bash
.agents/skills/paper-source/scripts/arxiv-mineru-parse.sh <paper-id> <area> <slug> --tier basic
.agents/skills/paper-source/scripts/arxiv-mineru-parse.sh <paper-id> <area> <slug> --copy-images
```

`--copy-images` 把抽取的图片按 `doc:...` locator 逐张导出到 `drafts/images/<slug>-<paper-id>/`,便于深挖与写作时引用;进入报告的图片再按 `paper-deep-dive` 的证据规则晋升到运行包。

只需要 PDF 审计产物时:

```bash
.agents/skills/paper-source/scripts/fetch-arxiv-pdf.sh <paper-id> <slug>
```
