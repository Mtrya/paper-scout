# 格式细则

DocxXML 语法以 `lark-doc` 的 `references/lark-doc-xml.md` 为准。本文件收 Paper Scout 的格式规则、排版细则与图示计划模板——写作线程片段前必读。

## DocxXML 机械规则

- 以 DocxXML 写作,v2 API。不用 `--title` 标志——在开头放一个 `<title>` 元素,正文不再重复。
- 使用合理的 `<h1>` / `<h2>` 层级。飞书会自动生成目录。
- callout 的子元素必须是块级元素(`<p>`、标题、列表)。不要裸文本,不要在 callout 内放表格或代码块。
- 文本和代码中的 `<`、`>`、`&` 要转义为 `&lt;`、`&gt;`、`&amp;`。
- DocxXML 是片段格式,有多个顶层块。不要用标准的单根 XML 解析器校验它。
- 临时媒体锚点必须是**裸行**(整行只有锚点本身,不包 `<p>` 标签),在文档内唯一,且在媒体插入后易于删除,例如单独一行 `[[figure-anchor:paper-slug:overview]]`。`push_report.py` 用整行精确匹配来定位锚点,写成 `<p>[[figure-anchor:...]]</p>` 会导致找不到锚点。
- 本地 `report.docxxml` 中至少包含两个不同的 `[[figure-anchor:...]]` 锚点。交付的飞书文档在媒体插入后不应留下可见的占位锚点。
- 在 `<latex>` 块内,LaTeX 命令用单个反斜杠(`\pi`、`\mathcal{L}`);双反斜杠(`\\`)是换行命令,会把每个符号渲染到新行。
- `<latex>` 同样支持行内用法。任何含上下标、分式或 LaTeX 命令的记号一律放进 `<latex>`(行内或块级);纯文本段落只允许平坦记号(如 m=127)。`verify_run.py` 会扫描 `<latex>` 块之外的 LaTeX 语法并拒绝发布。
- 图注不渲染 `<latex>`:图注里只用平坦记号,允许 Unicode 下标(如 σₑ、δₜ)。verify_run 一并扫描 `figures.json` 的 caption。

## 排版细则

排版服务叙事,不为装饰;颜色保持中性色默认,彩色只承担语义。复杂结构的首选媒介是示意图,不是段落复述。

载体按叙事功能选择:

- `<latex>`(块级或行内):讲清机制的公式与符号。
- `<pre lang="python" caption="..."><code>...</code></pre>`:紧凑的代码或伪代码片段。
- `<whiteboard type="mermaid|plantuml|svg">` 或自绘图片:结构示意。
- `<grid>`:对比,也用于同类多张相关小图的横排(不要竖着排队)。
- `<callout>`:亮点、关键结论与警告。
- `<table>`:结构化的多方对比、指标或消融;彩色单元格只表达状态或分类。
- 本地图片锚点:论文插图、定性示例、失败案例,或应出现在相关文字附近的策展图表。

选择最服务于解释的产物:

- 讲论文本身时,用论文的主图
- 解释方法、回路、算法或推导时,用代码片段、伪代码或公式——当它们比文字更清楚
- 覆盖论文的结果时,用论文的实验图、结果图或表格
- 展示本次巡航的发现时,用产出物和表格

把 MinerU 图片文件映射到图号时,要看图注和正文的引用,不要从文件名猜。

亮点 callout 示例:

```xml
<callout emoji="✅" background-color="light-green">
  <p><b>亮点:</b>[标题]是[研究线程]最清晰的种子。[原因。]</p>
</callout>
```

短名单表模板(四列:论文、核心贡献、为什么重要、链接):

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

## 图示计划模板

每个深度线程一份,写进写作简报:

- 产物类型:公式 / 代码 / 伪代码 / 论文插图 / 结果图 / 策展图表 / 产出物 / 真实表格
- 来源:论文资产、代码库、相关论文、诊断、实验,或代理自绘图表
- 目的:这个产物帮助读者理解什么
- 位置:附近的段落或临时媒体锚点

如果某个深度线程没有合适的产物,文字应讲清原因。
