# 交付细则

首选方式是 `scripts/push_report.py`。本文件记录完整选项、手工等价流程与故障排查,仅在需要手工操作或排查脚本问题时阅读。

## push_report.py 完整选项

```bash
python3 .agents/skills/report-compose/scripts/push_report.py <run-id> \
  --user-id <ou_xxx>            # 不给 --user-id 则只建文档不通知
  # --dry-run 只分段并打印计划  # --cli 覆盖 lark-cli 调用前缀
  # --report/--figures 改报告源与配图清单路径(相对运行包目录)
  # --existing-doc <URL|token> 追加到既有文档而非新建(wiki 链接直接传;
  #   丢弃 <title>、全部段落 append 到文末;bot 需已有该文档编辑权限,
  #   未授权时加 --as user 改用以用户身份操作)
```

## 手工流程等价物

1. **分段推送**。单次 `--content` 超过 ~10KB 会被静默截断。按顶层块边界把 `report.docxxml` 切成 ≤5.8KB 的段:首段 `docs +create`(从响应捕获 `data.document.document_id` 与 `data.document.url`),其余 `docs +update --command append`。**每步必须检查返回 JSON 的 `ok` 字段与 `data.result`**——`ok:true` 只代表传输层成功,权限不足时 `data.result=failed` 而真正的错误只出现在 `warnings`(无权限 append 用户 wiki 文档时整批段落会静默失败);"Command executed successfully" 字样不代表成功。
2. **锚点插图**。`[[figure-anchor:...]]` 独占段落在导入时**不会**被飞书丢弃,因此全部插图完成后要显式抓取锚点段落块 id 并 `block_delete`,再核验无 `figure-anchor` 残留。插图定位用块位置法:从源文件取锚点的前一个正文块,在带块 id 的抓取结果中定位(块内去标签匹配),`docs +media-insert` 插图到末尾,再 `docs +update --command block_move_after` 移到该块之后;同一锚点多张图按 `runs/<run-id>/assets/figures.json` 的顺序链式移动。图清单格式见 `push_report.py` docstring。
3. **白边机械检查**。插图前对每张图跑 `check_image_whitespace.py` 的检查:单边空白占比 >8% 自动裁剪到临时副本再上传(不动运行包原件),整图 >60% 空白直接拒绝推送。该脚本也可独立用于任何素材的预检(`--crop` 原处裁剪)。
4. **验证**。重新抓取核对 `<img>` 数量(防止孤儿图)、锚点残留与渲染宽度异常。

## 故障排查

- 整批段落静默失败:查响应里的 `warnings`,九成是目标文档的编辑权限问题;wiki 文档用 `--as user` 或先授权。
- 插图找不到锚点:确认锚点是裸行且全文唯一;`<p>[[figure-anchor:...]]</p>` 形式不被匹配。
- 图片在文档里显示过小:白边没裁干净,用 `check_image_whitespace.py --crop` 预检素材。
