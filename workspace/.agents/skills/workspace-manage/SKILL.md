---
name: workspace-manage
description: "Paper Scout 的工作区布局、命名规则、产物追踪与清理策略、运行包校验器、分支/PR 收尾,以及 INDEX.md 覆盖日志。"
user-invocable: false
---

# workspace-manage

## 布局

```text
.
├── papers/<area>/<slug>-<id>.md          # 下载的论文 Markdown
├── code/                                 # 忽略的实验台,用于外部信号工作
├── drafts/                               # 忽略的 DocxXML、媒体与暂存产物
└── runs/
    ├── INDEX.md                          # 覆盖日志 / 去重事实源
    └── <run-id>/                         # 持久运行包
        ├── report.docxxml                # 已交付报告的源文件
        ├── checklist.md                  # 人工完成闸门
        ├── assets/                       # 面向报告的资产与小型结果产物
        └── <thread-id>/                  # 论文或跨论文研究线程
            ├── README.md                 # 保留证据时必需
            ├── code/                     # 可选保留的探针、脚本、复现
            └── patches/                  # 可选保留的针对外部代码的补丁
```

一个线程是一个持久的调查单元。它可以是一篇论文、一次跨论文比较、一个方法问题、一条代码路径,或一个由论文池激发的可建造想法。当没有任何有意义的外部信号可以保留时,线程目录也可以只含 `BLOCKER.md`。

## 命名

运行 id 以日期开头、尽量可读,例如 `2026-06-07-cosmos3-grail-qwenvla`。论文集没有现成名字时,不要硬造花哨的 slug。

论文缓存文件以标题 slug 开头、论文 id 结尾:

- `papers/vla/robosemanticbench-2606.02277.md`
- `papers/world-models/cosmos3-2606.02800.md`

线程 id 要可读且稳定:

- `runs/2026-06-07-cosmos3-grail-qwenvla/cosmos3-2606.02800/`
- `runs/2026-06-07-cosmos3-grail-qwenvla/action-tokenization/`

## 目录规则

- `papers/`:被追踪的持久论文文本缓存。
- `code/`:忽略的实验台。在这里克隆仓库、创建虚拟环境、跑实验、给上游代码打补丁、写临时探针。不要把有用工作的唯一副本留在这里。
- `drafts/`:忽略的暂存区。随意覆盖。绝不把持久内容放在这里。
- `runs/<run-id>/report.docxxml`:被追踪的已交付报告源文件。
- `runs/<run-id>/checklist.md`:被追踪的人工完成闸门。
- `runs/<run-id>/assets/`:被追踪的扁平目录,放面向报告的资产与小型结果产物。
- `runs/<run-id>/<thread-id>/README.md`:被追踪的说明,解释保留的代码或补丁。
- `runs/<run-id>/<thread-id>/code/`:被追踪的、值得保留的精选代码。
- `runs/<run-id>/<thread-id>/patches/`:被追踪的、值得保留的精选补丁。
- `runs/<run-id>/<thread-id>/BLOCKER.md`:无法保留代码或补丁证据时,被追踪的障碍说明。
- 工作区根目录:不放散落的运行脚本或临时输出。

## 产物策略

暂存之前,检查 `.gitignore` 与实际状态:

```bash
git status --short --ignored
git check-ignore -v <path>
```

绝不强制添加被忽略的文件。如果某个被忽略的文件应当持久化,先把精选的副本、补丁、结果或 README 移入正确的被追踪运行包。

需要追踪:

- `papers/` 下的论文 Markdown
- `runs/<run-id>/report.docxxml` 已交付报告源文件
- `runs/<run-id>/checklist.md` 运行清单
- `runs/<run-id>/assets/` 下面向报告的资产
- `runs/<run-id>/<thread-id>/` 下的线程证据
- `runs/INDEX.md`

确认交付后清理:

- `code/` 回到只剩 `README.md`
- `drafts/` 回到只剩 `README.md`
- 其他被忽略的暂存,仅在持久证据已晋升之后清理

## 校验器

把校验器当作运行契约中可机器检查的子集。清单仍然是更广的人工契约。

发布前运行:

```bash
python .agents/skills/workspace-manage/scripts/verify_run.py runs/<run-id> --mode prepublish
```

交付、清理与索引更新之后运行:

```bash
python .agents/skills/workspace-manage/scripts/verify_run.py runs/<run-id> --mode final
```

校验器检查:

- `report.docxxml`、`checklist.md`、`assets/` 存在。
- `report.docxxml` 含有至少两个不同的 `[[figure-anchor:...]]` 锚点。
- 至少存在一个线程目录。
- 每个非保留的运行级目录都是合法线程。
- 线程要么是 `BLOCKER.md`,要么是 `README.md` 加 `code/`、`patches/` 或两者兼有。
- 存在的 `code/` 或 `patches/` 目录内至少有一个非空文件。
- final 模式下 `code/` 和 `drafts/` 只剩各自的 README 标记。
- final 模式要求 `runs/INDEX.md` 提到该运行 id。

## 清单模板

`runs/<run-id>/checklist.md` 是本次巡航的人工完成契约。发布前它必须存在且全部为真,但它不被机器解析。

```md
# Paper Scout 巡航清单

## 运行
- 运行 id:
- 覆盖时段:
- 报告:
- 飞书文档:

## 研究契约
- [ ] 报告前置的是从论文加外部信号中赢得的洞见,而不是论文内容的重组。
- [ ] 每个深度线程都有建设性的研究动作,或一个精确的障碍说明。
- [ ] 关键论断有代码、探针、补丁、相关工作、数据样本、推导、产出物支撑,或明确陈述的障碍。
- [ ] 报告讲清了这次巡航学到的、仅靠重读论文文本无法看出的东西。

## 报告契约
- [ ] 报告可扫读:清晰的开篇综述、有力的分节、有用的图/表/公式/代码片段、流畅的逻辑。
- [ ] `report.docxxml` 中至少有两个图锚点。
- [ ] 深度线程读起来像研究叙事,而不是填模板的摘要。
- [ ] 轻量留意的论文与深度线程被干净地区分开。

## 保存契约
- [ ] 持久证据在线程目录中。
- [ ] 面向报告的资产在 `assets/` 中。
- [ ] `code/` 和 `drafts/` 不持有持久工作的唯一副本。
- [ ] 工作区校验器 prepublish 模式通过。

## 发布契约
- [ ] 报告已发布。
- [ ] 用户通知已确认。
- [ ] `runs/INDEX.md` 已更新。
- [ ] 工作区校验器 final 模式通过。
```

## INDEX.md

持久去重日志。只追加,新的在前。

**侦查之前先读。**不要重复已覆盖的论文或研究线程,除非被明确推翻。短名单中的论文如仍相关,可以再次出现。

**确认交付后追加。**记录:

- 运行日期/时间
- 覆盖时段
- 飞书文档 URL
- 运行包路径
- 深度线程
- 已覆盖论文
- 短名单论文

条目保持简洁。不要改写历史。

## 预检

1. 确保 `papers/`、`code/`、`runs/`、`drafts/` 存在,缺失则创建。
2. 读 `runs/INDEX.md`(如存在)。
3. 运行前检查 git 状态。如果工作区有无关改动,停下并报告。
4. 在分支上开始本次巡航,而不是 `main`/`master`。使用形如 `scout/YYYY-MM-DD` 或 `scout/YYYY-MM-DD-<topic>` 的分支,撞名时加后缀。
5. 绝不把候选池或暂存写进 `runs/`。

## 收尾与 PR

在飞书文档已创建、媒体已插入、用户私信已确认、`runs/INDEX.md` 已更新之后:

1. 按产物策略把所有持久材料移入被追踪的位置。
2. 清理被忽略的暂存,使 `code/` 和 `drafts/` 只剩各自的 README 标记。
3. 运行最终校验:

```bash
python .agents/skills/workspace-manage/scripts/verify_run.py runs/<run-id> --mode final
```

4. 检查 `git status --short`。它应该只显示持久的论文缓存、运行包和 `runs/INDEX.md`。
5. 只暂存持久产出:

```bash
git add papers runs
```

6. 用以运行为中心的信息提交,例如 `Add 2026-06-07 paper scout report`。
7. 推送分支并创建 ready-to-review 的 PR。创建 ready-to-review PR，等待 CI 通过，然后合并。
8. 拉取
