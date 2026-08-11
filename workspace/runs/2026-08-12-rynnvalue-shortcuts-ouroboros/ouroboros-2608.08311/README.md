# Ouroboros (2608.08311) — 自我演化 agent 的 git 考古与治理审计

论文:Samsung AI Center Moscow 等。被授权改写自身全部代码的 coding agent,两种演化模式(recursive free / experience-driven),reviewed-commit 门禁(静态检查+测试+三模型评审团+objective 指纹门),宪法 BIBLE.md 可经门禁自修订。自测 Terminal-Bench 2.0 86.74%。

## 我们的研究动作

论文快照无法证伪,证据在 git 历史里。完整克隆(unshallow,947 commits,2026-04-22→08-11):

- **署名人普查**:699/947(74%)署名 "Ouroboros",持续三个半月至论文挂出当周。
- **BIBLE.md 修订史**:共 16 次修改,12 次为 agent 在初版之后经门禁完成的自修订——含"重写为 pragmatic self-governance"与"修复门禁抓出的宪法自相矛盾"等实质修订。
- **保护机制审计**:`ouroboros/runtime_mode_policy.py` 的 SAFETY_CRITICAL_PATHS(BIBLE.md/safety.py/自身)触发更严评审;代码注释记录大小写绕过漏洞的真实修复。`triad_review.py` 三模型 quorum、`evolution_fingerprint.py` 的 BUG3(重复目标误放行)修复均有实证痕迹。
- **门禁真实发威**:commit message 中有 agent 修复被拦后重提、owner revert agent 改动(Hermes cell)的序列。
- **体量核验**:py 文件 313→770,核心代码(ouroboros/+supervisor/,不含 tests/devtools)现 18.8 万行,论文截稿口径 175,755,一致。
- **benchmark 成色**:Terminal-Bench 86.74% 为 self-reported,官方收录走 harbor-framework/terminal-bench-2-1 PR 流程,Ouroboros 的 PR #175 仍 open 未合入;官方榜当前第一 Claude Code(Fable 5)83.8%。

## 核心发现

- 自我演化的治理工程真实可考古:宪法被系统按自己的流程迭代 12 次未崩坏,保护路径与门禁有真实拦截记录。
- 唯一需打折:benchmark 数字尚未经官方榜收录。
- 开放问题:reviewed-commit 门禁在 19 万行体量兜得住,更长自主时间尺度上是否仍兜得住未知。

## 内容

- `code/plot_ouroboros.py`:考古图绘制(commit 累积曲线 + BIBLE 修订标记)
- 仓库克隆在 `code/ouroboros/`(workspace lab bench,gitignored)
- 未实际运行该 agent(运行自修改系统超出本轮授权范围)
