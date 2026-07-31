# 媒体嵌入工作流

当 Paper Scout 报告需要包含从论文中抽取的本地插图时使用——例如 `drafts/` 下的 MinerU 图片或用 `--copy-images` 复制的图片。飞书本地媒体无法只靠写 DocxXML 草稿文件嵌入;先创建或追加文字,再把媒体插入显式锚点。

## 什么时候用媒体

当图片是理解机制、结果、对比或失败模式的最快方式时,使用本地媒体。好的候选是:架构图、主要结果图、定性示例、失败模式图、策展图表。不要加装饰性图片。

## 锚点模式

在图片应该落下的位置,放一个唯一的独立段落:

```xml
<p>图片前的正文。</p>
<p>[[figure-anchor:paper-a:overview]]</p>
<p>图片后的正文。</p>
```

规则:

- 锚点段落必须是顶层块,不能在 callout、表格、grid 列、嵌套列表或表格单元格内。
- 每个锚点必须唯一。用论文 slug 加短图名。
- 起草时维护一份小的媒体计划:锚点、本地图片路径、显示宽高、图注。
- 插入媒体后删除锚点。

## 插入媒体

文档存在之后,在图片路径可以是相对路径的目录里运行 `docs +media-insert`。绝对 `--file` 路径会被拒绝。

```bash
cd drafts/<paper-slug>-<paper-id>-mineru
lark-cli docs +media-insert --as bot \
  --doc "<document_id>" \
  --file images/<figure>.jpg \
  --selection-with-ellipsis '[[figure-anchor:paper-a:overview]]' \
  --width 800 --height 449 \
  --align center \
  --caption "论文 A 概览"
```

默认媒体插入到匹配的锚点之后。图片应出现在锚点之前时,加 `--before`。`--width` 和 `--height` 两个都传以保证可靠;旧版 `lark-cli` 可能无法自动探测抽取的论文图片尺寸。

## 删除锚点

带块 id 抓取文档,找到锚点段落,删除该块:

```bash
lark-cli docs +fetch --api-version v2 --as bot --doc "<document_id>" --detail full
lark-cli docs +update --api-version v2 --as bot \
  --doc "<document_id>" \
  --command block_delete \
  --block-id "<anchor_block_id>"
```

再抓取一次验证最终顺序。预期的模式是:相关正文、插入的 `<img>`、后续正文,然后是下一篇论文或下一节。

## 实践要点

- `docs +media-insert` 适用于本地图片,返回图片块 id 和文件 token。
- 报告文档为 bot 所有时,使用 `--as bot`。
- 对 `paper-source` 的 `--copy-images` 复制的图片,从 `drafts/images/<slug>-<paper-id>/` 目录运行,传相对文件名。
- 对原始 MinerU 输出,从解出目录运行,传 `images/<figure>.jpg`。
- 如果选择文本出现多次,使用 `--selection-with-ellipsis` 支持的 `start...end` 形式,或把锚点写得更具体。
