# Paper Scout 巡航清单

## 运行

- 运行 id：2026-08-05-reactivity-state-commitment
- 覆盖时段：2026-08-01 ~ 2026-08-05
- 论文池：Hugging Face Papers，67 篇去重条目
- 报告：`runs/2026-08-05-reactivity-state-commitment/report.docxxml`
- 飞书文档：https://fudan-nlp.feishu.cn/docx/Z4PgduLiLo64z3xTLzEc1JSinih（已回读核验并 IM 通知用户）

## 研究契约

- [x] 深挖占主要精力：三篇全文审读 + 三个官方仓库逐路径核验 + 一次真实权重、
  真实 Waymo 数据、真实 4090 的配对因果干预。
- [x] 报告交付的是论文外洞见：WorldExam 的接口带宽混杂；WCM 的预测目标是
  state-identification regularizer 而非长程 rollout；DEFT 的 HLD 是软脚手架而非
  语义绑定承诺。
- [x] 至少一次真实实验：Qwen3-VL-8B-Instruct bf16，6 个 AD-MCQ 场景 ×
  original / HLD 删除 / HLD 反向覆写，greedy，Turn-2 上限 1,200 tokens。
- [x] 预试验污染被识别并纠正：第一版误插 `[withheld]` 且 500-token 截断，未用于
  报告；最终版真正删除 HLD，并给三条件相同的 1,200-token 上限。
- [x] negative result 如实写入：删除 HLD 3/6 改变；反向 HLD 1/6 改变；由干预
  新跟随反向目标 0/6。
- [x] 精确记录复现障碍：DEFT-RLVR checkpoint 页面存在但大分片分发路径当前不可达；
  实验降级为论文明确报告的 training-free base，未外推到训练后模型。

## 报告契约

- [x] 开篇有 67 篇池子的筛选综述，深度线程与轻量短名单清楚分开。
- [x] 4 个图锚点：三张人工检查过的论文主图 + 一张自研因果干预结果图。
- [x] 公式先定义后使用；WCM 联合 loss 与三个“预测债务”层级都给出明确接口含义。
- [x] 数字有可追溯出处；小样本实验不冒充 benchmark 准确率。
- [x] 每个线程都有 README 与非空 code 证据目录；原始 JSONL、汇总 JSON、分析脚本
  与图都进入运行包。

## 交付

- [x] `push_report.py --dry-run` 通过。
- [x] 飞书文档创建、分段写入、4 张图按锚点插入并回读核验。
- [x] 用户 IM 通知成功。
- [ ] `verify_run.py --mode final` 通过。
- [ ] 运行包 PR 的 CI 通过并按授权合入。
