---
name: report-compose
description: "以 Lark DocxXML 撰写、发布并存档 Paper Scout 研究报告,包括结构、图示产物、本地媒体插入、飞书交付与用户通知。命令细节加载 lark-doc 与 lark-im。"
user-invocable: false
---

# report-compose

本技能负责写作与交付两个步骤。产出是一份权威研究报告:本地的 `runs/<run-id>/report.docxxml` 源文件和发布的飞书文档。论文文本、代码探针、补丁、MinerU 输出和工作笔记是源材料,不是并行的用户交付物。

起草或发布前加载 `lark-doc`,遵循其 DocxXML 参考文档的语法,尤其是 `references/lark-doc-xml.md`。通知前加载 `lark-im`。本地图片插入遵循 `references/figure-embedding.md`。

## 报告契约

近期论文是研究的种子,不是报告的边界。从新东西出发,沿着最强的线索走:相关论文、代码、产物、诊断、实验、公式、可建造的问题。报告前置的是从论文加外部信号中学到的东西,而不是论文文本的重组。

图示产物是一等公民:公式、代码片段、伪代码、论文插图、结果图、策展图表、产出物、真实表格。它们的目的不是装饰报告,而是比文字更快地展示机制、结果、对比或失败模式。

好的报告洞见密集且可扫读。平实的文字负责定位读者、讲清该看什么、保持逻辑流畅。不要交付一堆只是把论文文本重新排列的密实文字。

报告以中文撰写。

## 格式规则

- 以 DocxXML 写作,v2 API。不用 `--title` 标志——在开头放一个 `<title>` 元素,正文不再重复。
- 使用合理的 `<h1>` / `<h2>` 层级。飞书会自动生成目录。
- callout 的子元素必须是块级元素(`<p>`、标题、列表)。不要裸文本,不要在 callout 内放表格或代码块。
- 文本和代码中的 `<`、`>`、`&` 要转义为 `&lt;`、`&gt;`、`&amp;`。
- DocxXML 是片段格式,有多个顶层块。不要用标准的单根 XML 解析器校验它。
- 临时媒体锚点必须是独立的顶层段落,在文档内唯一,且在媒体插入后易于删除,例如 `<p>[[figure-anchor:paper-slug:overview]]</p>`。
- 本地 `report.docxxml` 中至少包含两个不同的 `[[figure-anchor:...]]` 锚点。交付的飞书文档在媒体插入后不应留下可见的占位锚点。
- 在 `<latex>` 块内,LaTeX 命令用单个反斜杠(`\pi`、`\mathcal{L}`);双反斜杠(`\\`)是换行命令,会把每个符号渲染到新行。

## 结构

按主题组织,不要平铺成列表。

```xml
<title>Paper Scout 研究报告 - YYYY-MM-DD</title>
开篇综述(2-4 句)
<h1>主题 A</h1>
  小综述(1-3 句)
  短名单表(仅轻量留意的论文;没有则省略)
  <h2>深度线程</h2> + 研究叙事 + 图示产物
  <hr/>
  <h2>深度线程</h2> + 研究叙事 + 图示产物
<h1>主题 B</h1>
  ...
可选:<h1>跨主题观察</h1>
```

### 开篇综述

回答:*这次巡航发现了什么?*点出报告其余部分将使用的主题。

如果有一篇论文明显出众:

```xml
<callout emoji="✅" background-color="light-green">
  <p><b>亮点:</b>[标题]是[研究线程]最清晰的种子。[原因。]</p>
</callout>
```

如果这一时期整体平淡,直说。不要硬凑。

### 主题小节

每个真实主题一个 `<h1>`。主题从研究线程中生长出来,不套固定分类法。

规则:

- 不要重复论文摘要。一篇轻量留意的论文只出现在一个短名单表里。一个深度线程有一个主要叙事位置,但可以在能讲清线程的地方引用其他论文。
- 深度线程作为 `<h2>` 研究小节放进所属主题。
- 轻量留意的论文作为每个主题短名单表的行。
- 只有一篇孤文的主题,并入更大的主题或兜底主题。
- 如果论文池无法聚类,用单个 `<h1>精选</h1>`。

#### 短名单表

四列:**论文**、**核心贡献**、**为什么重要**、**链接**。

```xml
<table>
  <colgroup>
    <col width="200"/>
    <col width="230"/>
    <col width="230"/>
    <col width="90"/>
  </colgroup>
  <thead>
    <tr>
      <th background-color="light-gray">论文</th>
      <th background-color="light-gray">核心贡献</th>
      <th background-color="light-gray">为什么重要</th>
      <th background-color="light-gray">链接</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>标题</b>(作者等,2026)</td>
      <td>他们做了什么。</td>
      <td>这位用户为什么该关心。</td>
      <td><a href="url">HF</a> / <a href="url">PDF</a></td>
    </tr>
  </tbody>
</table>
```

贡献与相关性要分开写。如果主题里每篇论文都被深度线程覆盖了,就省略表格。单元格塞满段落的表格不算图示产物。

### 深度线程小节

每个深度线程是所属主题下的一个 `<h2>`,用 `<hr/>` 分隔。线程可以围绕一篇论文、一次比较、一个方法问题、一个代码/探针结果,或一个由论文激发的可建造想法。

头部格式:

```xml
<h2>论文标题</h2>
<p>作者等(2026)· <a href="url">HF</a> · <a href="url">PDF</a></p>
```

写成流畅的叙事,不要套固定模板。每个深度线程都应讲清:问题是什么、核心机制、调查发现了什么、哪些外部信号支撑、还有什么不确定。

在能承载理解的地方使用 DocxXML 特性:

- `<latex>` 用于讲清机制的公式。
- `<pre lang="python" caption="..."><code>...</code></pre>` 用于紧凑的代码或伪代码片段。
- `<grid>` 用于两方对比。
- `<table>` 用于结构化的多方对比、指标或消融。
- 本地图片锚点用于论文插图、定性示例、失败案例,或应出现在相关文字附近的策展图表。

选择最服务于解释的产物:

- 讲论文本身时,用论文的主图
- 解释方法、回路、算法或推导时,用代码片段、伪代码或公式——当它们比文字更清楚
- 覆盖报告的结果时,用论文的实验图、结果图或表格
- 展示本次巡航的发现时,用产出物和表格

把 MinerU 图片文件映射到图号时,要看图注和正文的引用,不要从文件名猜。

### 图示计划

写最终稿之前,为每个深度线程留一份临时的图示计划:

- 产物类型:公式 / 代码 / 伪代码 / 论文插图 / 结果图 / 策展图表 / 产出物 / 真实表格
- 来源:论文资产、代码库、相关论文、诊断、实验,或代理自绘图表
- 目的:这个产物帮助读者理解什么
- 位置:附近的段落或临时媒体锚点

在能让报告更清楚、更易扫读且不破坏叙事逻辑的地方使用产物。如果某个深度线程没有合适的产物,文字应讲清原因。

### 跨主题观察

只当存在横跨主题的模式时,才加收尾的 `<h1>`:一次趋同、一个空缺、一次转变、一个可复用的方法想法,或一个开放研究问题。没什么可说就跳过。

## 交付

所有 `docs` 命令都带 `--api-version v2`。以 bot 身份创建和更新。本地源文件是 `runs/<run-id>/report.docxxml`。

创建真实文档之前,确保运行清单已完成,并运行:

```bash
python .agents/skills/workspace-manage/scripts/verify_run.py runs/<run-id> --mode prepublish
```

### 飞书交付(需要 lark-cli)

如果 PATH 上有 `lark-cli`(或可通过 `npx --yes @larksuite/cli` 拉起),走下面的完整飞书流程。如果**没有**,直接跳到**仅本地兜底**小节。

首选方式是运行本技能的推送脚本,它封装了分段、插图与验证的全部细节:

```bash
python .agents/skills/report-compose/scripts/push_report.py <run-id> \
  --user-id <ou_xxx>            # 不给 --user-id 则只建文档不通知
  # --dry-run 只分段并打印计划  # --cli 覆盖 lark-cli 调用前缀
  # --report/--figures 改报告源与配图清单路径(相对运行包目录)
  # --existing-doc <URL|token> 追加到既有文档而非新建(wiki 链接直接传;
  #   丢弃 <title>、全部段落 append 到文末;bot 需已有该文档编辑权限)
```

脚本做的事(手工流程等价物,需要手工时按此执行):

1. **分段推送**。单次 `--content` 超过 ~10KB 会被静默截断,脚本按顶层块边界把 `report.docxxml` 切成 ≤5.8KB 的段:首段 `docs +create`(从响应捕获 `data.document.document_id` 与 `data.document.url`),其余 `docs +update --command append`。**每步必须检查返回 JSON 的 `ok` 字段与 `data.result`**——`ok:true` 只代表传输层成功,权限不足时 `data.result=failed` 而真正的错误只出现在 `warnings`(2026-08-24 实测:无权限 append 用户 wiki 文档,17 段全部静默失败);"Command executed successfully" 字样不代表成功。
2. **锚点插图**。`[[figure-anchor:...]]` 独占段落在导入时**不会**被飞书丢弃(历史上曾假设会,导致锚点文本残留文档),因此脚本在全部插图完成后显式抓取锚点段落块 id 并 `block_delete`,再核验无 `figure-anchor` 残留。插图定位用块位置法:脚本从源文件取锚点的前一个正文块,在带块 id 的抓取结果中定位(块内去标签匹配),`docs +media-insert` 插图到末尾,再 `docs +update --command block_move_after` 移到该块之后;同一锚点多张图按 `runs/<run-id>/assets/figures.json` 的顺序链式移动。图清单格式见脚本 docstring。
3. **白边机械检查**。插图前脚本对每张图跑 `check_image_whitespace.py` 的检查:单边空白占比 >8% 自动裁剪到临时副本再上传(不动运行包原件),整图 >60% 空白直接拒绝推送。该脚本也可独立用于任何素材的预检(`--crop` 原处裁剪)。
4. **验证**。重新抓取核对 `<img>` 数量(防止孤儿图)、锚点残留与渲染宽度异常。

bot 拥有产出的文档。**没有 `--parent-token`**,也没有配置文件夹/知识库目的地——不要自己加。内容里的 `<title>` 元素设置文档标题。

然后加载 `lark-im`,给用户发一条包含文档 `url` 的私信。用 `--text`(不是 `--markdown` 或 `--content`),这样飞书会把文档 URL 自动展开成富预览卡片:

```bash
lark-cli im +messages-send --as bot --user-id <ou_xxx> \
  --text "https://<tenant>.feishu.cn/docx/<doc_id>"
```

只有当这条私信发出并确认后,一次巡航才算完成。如果收件人解析或发送失败,停下并报告。

私信确认之后:

1. 确保已交付的 DocxXML 源文件保存在 `runs/<run-id>/report.docxxml`。
2. 保存文档 `url`,交给 `workspace-manage` 记入 `runs/INDEX.md`。

### 仅本地兜底(无 lark-cli)

当 `lark-cli` 未安装时,优雅地跳过飞书交付。不要视为错误。改为:

1. 确保 DocxXML 源文件保存在 `runs/<run-id>/report.docxxml`。
2. 用工作区绝对路径告诉用户报告保存在哪里,例如:

   > 飞书交付已跳过(未找到 lark-cli)。报告已保存到:
   > `/absolute/path/to/workspace/runs/<run-id>/report.docxxml`

3. 照常进入 `workspace-manage` 收尾流程。在 `runs/INDEX.md` 中用本地文件路径代替飞书 URL 记录本次巡航。

## 不要做什么

- 不要省略或重复 `<title>`。
- 不要用 `overwrite` 在巡航中途修报告。用 `append`,或 `lark-doc` 中文档化的块级编辑命令。
- 不要在 `<callout>` 里放裸文本、表格或代码块。
- 不要写"取得了 SOTA"。写数字、基准和比较对象。
- 不要整份报告都用平铺的要点列表。
- 不要给每个深度线程盖上同样的五个小标题。
- 不要把深度线程又复制成短名单的一行。
- 不要把整个论文池做成一张巨大的短名单表。
- 不要硬编码主题列表。
- 不要交付只是复述摘要的深度线程。
- 不要在交付的文档里留下可见的临时媒体锚点。
- 不要跳过私信就认为交付完成(除非 lark-cli 不可用且你在走仅本地兜底)。
- 通知 URL 不要用 `--markdown` 或 `post` 内容;那会渲染成朴素的蓝色链接,而不是飞书文档卡片。
- 不要发明收件人解析规则;遵循 `lark-im`。

## 清单

- [ ] 已加载 `lark-doc`
- [ ] 通知前已加载 `lark-im`
- [ ] DocxXML 源文件在 `runs/<run-id>/report.docxxml`
- [ ] `<title>` 存在且不重复
- [ ] 开篇综述点出了研究主题
- [ ] 短名单表按主题组织且只含轻量留意的论文
- [ ] 每个深度线程有图示计划
- [ ] 在能讲清发现的地方使用了图示产物
- [ ] 本地报告源文件中至少有两个不同的图锚点
- [ ] 运行清单已完成
- [ ] 工作区校验器 prepublish 模式通过
- [ ] 本地媒体锚点已插入、解析、删除并验证
- [ ] callout 只用在真正值得注意的地方
- [ ] `<`、`>`、`&` 已按 `lark-doc` 转义
- [ ] `<latex>` 内的 LaTeX 命令使用单反斜杠
- [ ] 创建/更新命令在真实写入前做过 dry-run
- [ ] 飞书 URL 已捕获
- [ ] 用户私信已确认
- [ ] URL 已交给 `workspace-manage` 记入 `runs/INDEX.md`
