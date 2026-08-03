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

**实测(2026-08-02/08-03 两轮巡航):`[[figure-anchor:...]]` 独占段落在 DocxXML 导入时会被飞书整体丢弃**,无法用 `--selection-with-ellipsis` 匹配。可用的块位置法如下(`scripts/push_report.py` 已封装,含图清单 `assets/figures.json` 驱动):

1. 带块 id 抓取:`docs +fetch --doc <id> --as bot --detail with-ids`。
2. 在源文件里取锚点的前一个正文块(跳过连续锚点行),用其段尾文本在抓取结果中定位块 id——块内要先去标签再匹配,因为抓取结果会在 `<b>` 等内联标签周围插入额外空格。
3. 插图(默认落到文档末尾):`docs +media-insert --doc <id> --as bot --file <相对路径> --caption "<图注>"`,从返回 JSON 取 `data.block_id`。
4. 移位:`docs +update --doc <id> --as bot --command block_move_after --block-id <目标块id> --src-block-ids <图块id>`。同一位置多张图时链式移动(后一张移到前一张之后)保持顺序。
5. 重新抓取核对 `<img>` 数量,防孤儿图。

`--file` 只接受 cwd 相对路径,绝对路径报 `unsafe file path`。

## 实践要点

- `docs +media-insert` 适用于本地图片,返回图片块 id 和文件 token;已上传图的 file token 可在 overwrite 后以 `<img src="TOKEN">` 直接引用。
- 报告文档为 bot 所有时,使用 `--as bot`。
- 每步检查返回 JSON 的 `ok` 字段;错误可能只走 stderr,"Command executed successfully" 字样不代表成功。
- 本地 `report.docxxml` 源文件中保留锚点;交付的飞书文档里锚点已被导入丢弃,无需删除。
