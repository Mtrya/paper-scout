# UniMem 线程:openpi fork 代码审计

论文:UniMem (arXiv 2608.22869,Stanford)。官方放出 openpi fork:
github.com/losterberg3/unimem-vla-openpi(**有代码无权重**)。本线程为只读审计,
无训练/推理实验。

## 审计结论

实现与论文描述一致:

- `event_head.py`:事件分类器 MLP 挂在骨干最后层 latent 上;事件触发才写两路记忆
  (文本事件 append 进 prompt + 关键帧入集);null 类权重 0.02 而非 mask——mask 会让
  分类器在"无事帧"上无监督,部署时学不会不触发。append-only 记忆的不对称:
  假阴性下一秒可补,假阳性污染全程。
- `siglip_hidden_cache.py`:SigLIP 每隔 4 层插因果时序注意力,复用同层 QKV/LN
  权重;缓存 pre-LN/pre-位置编码的 hidden state;新帧只平移 PE 不重算空间注意力,
  当前帧单独作 query。**作者自己在注释里文档化了一个 bug**:seeding 阶段(首批
  关键帧进缓存)位置编码放置与 rollout 期不一致,引入约 2-3% 表示分歧;选择保留
  现状并注释说明。
- `label_dataset_libero.py`:事件标签由 Claude Sonnet 5.0 生成的脚本从动作签名
  检测(夹爪翻转=scooping 等),非事件帧标 null(-1);文本记忆在事件窗口整段过后
  才更新,防训练泄漏。

## 与主线的连接

大小脑接口投票:UniMem 投"不要独立大脑",但把上行通道内化进同一骨干——事件分类器
= 小脑的自我监测上行;append 的自然语言事件 = 写回的工作记忆。对照组 MemER 的失败
模式(BeanScoop 高层发错子任务→低层不可恢复)实证了分层接口的脆性。

## 置信度边界

无权重,性能数字(仿真 93.4 / 真机 80.0 / 90ms 恒定)自报且无法第三方复现;
事件词表手工预定义,标签经人工抽检。

## 文件

- `code/` 三个被审计文件的原样拷贝(供引用,非本仓产物)。
