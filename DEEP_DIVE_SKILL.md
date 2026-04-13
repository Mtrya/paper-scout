---
name: read-paper-phd
description: "给定 arxiv ID，下载 LaTeX 源码，解析论文内容与图表（含附录），生成面向 junior PhD 组会准备的中文精读报告（HTML）。报告以可复现为标准，深度解析方法细节，并将附录内容系统整合入正文分析。"
argument-hint: <arxiv_id> [输出目录路径]
user-invocable: true
---

# read-paper-phd: 组会论文精读报告生成器（junior PhD 版）

给定一个 arxiv ID，下载论文 LaTeX 源码，解析文字与图表（**含附录**），生成一份面向组会准备的中文 HTML 精读报告。

**目标读者**：正在准备组会汇报或深度阅读某篇具体论文的 junior PhD。报告读完后，读者应能做到：

1. 清楚说明这篇论文的动机与贡献
2. 向别人完整复述方法的每一个关键模块
3. 理解关键数学推导而不是"记住公式"
4. 复现主要实验（知道用什么数据、什么代码、什么超参数）
5. 在组会上回答导师和同学关于方法细节的追问

**与通用版的核心差异：**

- 方法部分深度优先，追求"看完能复现"而非"看完能描述"
- 附录（Appendix）的内容系统整合进各对应章节，而不是单独列出或略去
- 相关工作以论文 Related Work 章节内容为主，不做大规模外部检索
- 去掉联动章节、角色扮演和 next actions

---

## 输入

- `$ARGUMENTS` 格式：`<arxiv_id> [输出目录路径]`
- arxiv_id 示例：`2301.00234`、`2301.00234v2`
- 输出目录未指定时，默认为当前工作目录下的 `Report/` 目录
- 本 skill 依赖联网；如果当前环境无法访问外网，应直接停止并说明无法执行

## 输出规范

```
<输出目录>/<arxiv_id>/
├── index.html              # 主报告文件
└── figures/                # 论文图表（从源码提取的实际图像文件）
    ├── fig1.png
    └── ...
```

- `index.html` 允许依赖在线资源，不要求离线自包含
- 图表从 LaTeX 源码提取，PDF 格式的图转换为 PNG，移入 `figures/`
- 若论文使用 TikZ/PGFPlots 内联绘图（无独立图像文件），figures/ 目录可为空，报告中需注明
- 公式使用在线数学渲染器渲染，优先保证 LaTeX 兼容性
- **临时源码工件在用户确认后清理**：解析完成、报告生成后，临时解压目录、源码压缩包、PDF 等文件暂时保留，等用户审阅报告并明确确认不需要补充后，执行 Phase 7 清理。最终输出目录只保留报告所需文件。

---

## 执行前检查（必须在所有 Phase 之前完成）

### 检查 1：输出目录是否已存在

```
若 <output>/<arxiv_id>/ 目录已存在：
  → 询问用户："该论文的输出目录已存在（<路径>），是否覆盖重新生成？[y/N]"
  → 若用户选择 N，停止执行
  → 若用户选择 Y，继续（原有文件将被覆盖）
```

### 检查 2：Python 环境（仅在流程中需要运行 Python 脚本时触发）

若执行流程中需要调用 Python（如 EPS 转 PNG、批量图像处理等脚本），需提前询问：

```
询问用户："执行过程中可能需要运行 Python 脚本，请问使用哪个环境？（如 conda 环境名、venv 路径，或直接回复 'system' 使用系统默认）"
```

收到回答后，在所有 Python 调用前加上对应的环境激活命令（如 `conda run -n <env_name> python ...`）。

**Python 文件存放规则：执行流程中如需创建任何 `.py` 脚本，必须将其保存在 `<output>/<arxiv_id>/` 目录下，而不是 `/tmp/` 或其他位置。**

---

## 执行流程

### Phase 1: 获取论文源码

```yaml
步骤:
  1. 解析 arxiv_id，构建 URL
  2. 下载 LaTeX 源码包：https://arxiv.org/e-print/{arxiv_id}
  3. 解压到临时目录 /tmp/arxiv_source_{arxiv_id}/
  4. 识别主 .tex 文件（查找 \documentclass 或 \begin{document}）
  5. 解析所有 \input{} 和 \include{} 引用的子文件，包括附录子文件
  6. 检测图表类型（见下方「图表处理策略」），按对应策略处理
```

**下载源码：**

```bash
curl -L -o /tmp/arxiv_source_{arxiv_id}.tar.gz "https://arxiv.org/e-print/{arxiv_id}"
mkdir -p /tmp/arxiv_source_{arxiv_id}
tar -xzf /tmp/arxiv_source_{arxiv_id}.tar.gz -C /tmp/arxiv_source_{arxiv_id}/
# 如果不是 tar.gz（有些是单个 .tex 文件），尝试直接作为 tex 处理
```

**图表处理策略：**

**情况 A：存在独立图像文件（.pdf、.png、.jpg、.eps）**

```bash
mkdir -p <output>/<arxiv_id>/figures

# PDF 转 PNG
if command -v pdftoppm &>/dev/null; then
    pdftoppm -r 200 -png -cropbox input.pdf figures/output
else
    sips -s format png input.pdf --out figures/output.png -Z 2000
fi
```

**情况 B：论文使用 TikZ / PGFPlots 内联绘图**

1. 提取关键数据，在报告中重建为 HTML 表格或 inline SVG 折线图
2. 用文字精确描述图表要展示的对比关系和数值
3. 在对应位置加注 `[此图为 TikZ 代码生成，见原文 Figure X]`，并附上 arxiv PDF 链接

---

### Phase 2: 解析论文内容（**含附录全文**）

**超大 .tex 文件的处理规则：**

- 若文件 **< 80,000 字符**：直接全量读取
- 若文件 **≥ 80,000 字符**：分段读取，优先保证 Abstract + Introduction + Method + Experiment + Appendix 完整

**附录解析要求（本版本新增）：**

附录通常包含被正文引用但未展开的内容，必须在解析阶段识别并记录以下类型的附录信息：

```yaml
附录内容分类:
  - 完整证明 / 定理推导（正文只给结论，附录给完整证明）
  - 超参数全表（正文只列关键参数，附录列完整配置）
  - 额外实验 / 消融变体（正文放核心结果，附录放全量结果）
  - 实现细节（代码结构、训练 trick、边界条件处理）
  - 数据集细节（统计信息、预处理脚本、数据划分）
  - 局限性 / 失败案例（作者自己放进附录的诚实讨论）
```

在报告中，附录信息**不单独列章节**，而是整合进对应的正文分析模块：

- 附录里的证明 → 整合进「方法详解 > 核心设计与数学推导」
- 附录里的超参数 → 整合进「实验 > 实验设置」
- 附录里的额外实验 → 整合进「实验 > 补充结果」
- 附录里的实现细节 → 整合进「可复现性审计」

每处引用附录内容时，标注来源：`[附录 A.3]`

---

**从 LaTeX 源码中提取：**

```yaml
必须提取:
  - 标题、作者、摘要
  - 章节结构（含附录章节）
  - 正文全部内容
  - 数学公式（行内和行间）
  - 图表及其标题
  - 表格内容
  - 参考文献（用于还原 Related Work 中的引用关系）
  - 附录全文（按上述分类）
```

---

### Phase 2.5: 论文章节盘点（**必须完成，生成报告前的强制检查**）

解析完成后，在开始写报告之前，**必须先做一次完整的章节盘点**，建立"论文结构 → 报告覆盖位置"的映射表。

这一步的目的是防止报告按自己的逻辑结构推进时，遗漏论文中实际存在的内容。

**执行方式：**

1. 列出论文的所有章节和子章节（包括附录），按原文顺序排列
2. 对每一节，用 1-3 句话记录它的核心内容
3. 标注该内容将被整合进报告的哪个位置
4. 标注该内容是否有对应的图表或表格

**输出格式（内部工作文档，不出现在最终报告中）：**

```
论文章节盘点
═══════════════════════════════════════════════════════

§ Abstract
  内容摘要：[1-2句]
  → 报告位置：§1 论文速览

§ 1. Introduction
  §1.1 [子节标题]
    内容摘要：[1-2句]
    关键论点/分析：[是否有定量分析、失败案例、对比？]
    → 报告位置：§2 动机 / §1 论文速览
  §1.2 ...

§ 2. Related Work
  §2.1 [子节标题，通常按技术线划分]
    内容摘要：[覆盖了哪些工作，作者如何定位本文]
    → 报告位置：§8 相关工作梳理

§ 3. Method（或 Approach / Model）
  §3.1 ...
    内容摘要：[核心内容]
    涉及公式：[列出关键公式编号]
    涉及图表：[Figure X / Table X]
    → 报告位置：§4.X [对应子节]
  ...

§ 4. Experiments
  §4.1 Setup
    → 报告位置：§5.1
  §4.2 Main Results
    涉及表格：[Table X]
    → 报告位置：§5.2
  §4.3 Ablation
    → 报告位置：§5.3
  §4.4 [其他子节，如 Analysis / Visualization]
    → 报告位置：§5.4 / §6

§ 5. Conclusion
  内容摘要：[作者总结了什么，提到了什么局限，展望了什么]
  → 报告位置：§6 分析与讨论

§ Appendix A / B / C ...
  [按上述分类记录]
  → 报告位置：[整合进对应正文章节]

═══════════════════════════════════════════════════════
未覆盖内容检查：
  [ ] 所有节是否都有对应的报告位置？
  [ ] 是否有节被标注为"→ 报告位置：待确认"？（需要在报告中找到合适位置）
  [ ] Introduction 中是否有不属于动机的内容（如对 contribution 的详细展开、对实验结果的提前预告）？
  [ ] Related Work 中是否有超出 §8 能覆盖的技术背景内容（可能需要补充进 §3 前置知识）？
```

**盘点后必须解决的问题：**

- 若某节内容没有对应报告位置 → 在报告相应位置新建子节，或归入"分析与讨论"
- 若 Introduction 包含大量技术分析（如证明前作方法的理论缺陷） → 这些内容必须进入 §2.2，不能丢
- 若 Conclusion 包含作者未在实验中展示但提到的局限 → 必须进入 §6
- 若论文有 Discussion 或 Analysis 独立章节 → 内容进入 §6，不能忽略

---

### Phase 3: 检索相关论文（**精简版，以论文内容为主**）

本版本不做大规模外部检索，相关工作分析以论文自身 Related Work 和 Introduction 中的引用为主。

仅做以下有限检索，用于补充论文中提及但未给出足够上下文的重要前作：

```
# 仅在以下情况才搜索
WebSearch: "{前作方法名}" arxiv
  → 适用于：正文明确 "building on X" / "different from Y" 但未给原文链接的情况

WebSearch: "{本文方法名}" code github
  → 查找官方代码仓库，用于可复现性审计
```

所有外部检索结果仅用于补充可复现性信息和前作链接，不用于扩充"研究脉络"章节的论文列表。

---

### Phase 4: 生成报告

报告为单个 HTML 文件 `index.html`，公式渲染使用在线 MathJax。

---

### Phase 5: 初步整理（**不删除源码，等待用户确认**）

```yaml
步骤:
  1. 确认所有需要的图表已复制到 <o>/<arxiv_id>/figures/
  2. 确认 index.html 可以正常引用 figures/ 中的图像
  3. 保留所有临时文件，不做删除：
       /tmp/arxiv_source_{arxiv_id}/    ← 保留，用户可能需要查原文某节
       /tmp/arxiv_source_{arxiv_id}.tar.gz  ← 保留
       任何下载的 PDF                   ← 保留
```

**这一步结束后不进行任何删除。** 源码和 PDF 要等用户审阅报告、确认不需要补充内容之后再清理。

---

### Phase 6: 自验收 + 向用户报告状态

生成报告后，逐条检查验收清单（见文末）。

验收完成后，向用户输出以下状态摘要：

```
报告已生成：<o>/<arxiv_id>/index.html

临时文件状态（尚未删除）：
  源码目录：/tmp/arxiv_source_{arxiv_id}/
  源码压缩包：/tmp/arxiv_source_{arxiv_id}.tar.gz
  [PDF（如有）：路径]

请审阅报告。确认无需补充后，回复"清理"或"clean up"，
我将删除上述临时文件。
```

---

### Phase 7: 清理临时文件（**等用户确认后执行**）

**触发条件：** 用户审阅报告后明确表示不需要修改，或明确说"清理"/"clean up"/"删了吧"等。

**不要在用户未明确确认前自动执行此 Phase。**

```yaml
步骤:
  1. 删除临时源码目录：rm -rf /tmp/arxiv_source_{arxiv_id}/
  2. 删除下载的源码归档：rm -f /tmp/arxiv_source_{arxiv_id}.tar.gz
  3. 删除仅用于提取图表或兜底解析的 PDF（如有）
  4. 保留：<o>/<arxiv_id>/index.html、figures/、.py 脚本（如有）
  5. 向用户确认清理完成
```

---

## 报告结构

### 0. 论文元信息

```html
<header>
  <h1>论文标题</h1>
  <div class="meta">
    <p>作者：Author1, Author2, ...</p>
    <p>机构：Institution1, Institution2</p>
    <p>发表：Conference/Journal, Year</p>
    <p>arxiv：<a href="https://arxiv.org/abs/{id}">arxiv_id</a> |
       PDF：<a href="https://arxiv.org/pdf/{id}">下载</a></p>
  </div>
</header>
```

### 1. 论文速览

- **一句话总结**：一句中文概括这篇论文做了什么、解决了什么问题
- **难度评级**：★☆☆☆☆ 到 ★★★★★，附一句说明（如"需要熟悉 Transformer 架构和变分推断"）
- **关键词**：3-5 个关键词
- **核心贡献清单**：直接摘录并整理论文 Contribution 部分，用 bullet list 列出 3-5 条，每条附一句"这意味着什么"的解释

这部分不做价值判断，只做准确转述。

---

### 2. 动机

回答三个问题（严格基于 Introduction 和 Related Work 内容）：

**2.1 要解决什么问题**

- 现实中遇到的问题或理论上的缺口是什么
- 先给一个让人好奇的具体场景或失败案例（尽量用论文自己举的例子）
- 如果问题涉及专有背景，用类比或生活化例子解释

**2.2 已有方法的局限**

- 前作具体卡在哪里。不说"效果不好"，要说清楚在什么条件下、为什么不行
- 如果论文在 Introduction 里做了某种分析或证明来说明前作的问题，复述该分析
- 若附录有更详细的前作分析，在此整合 `[附录 X.X]`

**2.3 本文的解决思路（高层次）**

- 核心 insight 是什么：论文观察到了什么、利用了什么来解决上述问题
- 不要在这里解释方法细节，只说大方向

---

### 3. 前置知识

把理解本文所需的关键概念逐个拆解。**只收录对理解本文逻辑链必需的概念**（跳过"神经网络"、"梯度下降"等常识级别内容，拿不准时宁可多收录）。

每个概念：

- **通俗解释**：大白话说清楚这是什么、干什么用
- **较严谨的定义**：一两句更正式的描述（不照抄教科书）
- **出处**：最初提出这个概念的论文（标题 + 作者 + 年份 + arxiv 链接，如有）
- **与本文的关系**：本文用这个概念做了什么

---

### 4. 方法详解（核心章节，要求达到"可复现"深度）

**目标：一个从未接触过该方法的 PhD 读完后，能写出伪代码，并开始动手实现。**

#### 4.1 方法概览

- 整体 pipeline：输入 → 每个模块做什么 → 输出，用文字清楚描述数据流向
- 若论文有 overview figure（独立图像文件），直接嵌入，并在图下方用中文逐模块标注
- 若图为 TikZ 绘图（无独立图像文件），用文字描述图的结构和各模块关系，标注 `[见原文 Figure X]`
- 标注论文哪些部分在附录中有详细展开（例如 "完整证明见 [附录 A.2]，已整合至 4.3 节"）

#### 4.2 方法演变脉络

- 这个方法从什么方法演变而来（以论文 Related Work 和 Introduction 的引用为依据）
- 与直接前作方法的关键区别（最好用对比说明）
- 用"A → B → 本文方法"的文字形式展示演变关系，每步标注改进动机

#### 4.3 核心设计与数学推导

**这是报告中最重要的部分。** 对每个重要公式，必须做到：

```
1. 先用一句大白话说这个公式在干什么
2. 给出公式本身（在线数学渲染）
3. 逐项解释每个符号的含义（包括维度）
4. 解释公式的直觉：为什么要这样设计？有无替代方案？
5. 如果论文跳过了推导步骤，补充中间推导细节
6. 如果有近似或假设，明确指出并讨论其合理性
7. 如果附录有对应的完整证明，直接整合进来（标注 [附录 X.X]）
```

补全论文跳过的推导步骤时：
- 用 `<details>` 折叠标签放详细推导，主线保持流畅
- 明确标注哪些是原文的，哪些是补充的

#### 4.4 算法流程

- 如果论文有算法伪代码，重新排版并逐行注释
- 如果论文没有伪代码但方法足够复杂，**自行根据正文归纳**一份伪代码
- 标注每一步的输入输出和关键维度变化
- 如果算法区分 Training 和 Inference 两条路径，分别写出，不要混在一起

#### 4.5 实现要点（面向复现）

提炼正文和附录中所有与实现相关的细节，包括但不限于：

- 数值稳定性处理（如 log-sum-exp trick、gradient clipping 等）
- 初始化方式（若论文有特殊说明）
- 与直觉不一致的实现细节（常见坑点）
- 并行化 / 向量化的实现思路（若论文或附录有说明）
- 边界条件和特殊 case 的处理

若附录有实现细节章节，全部整合进来 `[附录 X.X]`。

---

### 5. 实验

#### 5.1 实验设置

**以下信息必须完整给出，正文+附录合并整理：**

- **数据集**：名称、规模（样本数、类别数、分辨率等）、划分方式、预处理方法（附录中的细节必须整合）
- **Baselines**：每个 baseline 一句话说明是什么方法，为什么选这些 baseline
- **评价指标**：每个指标的定义（尤其非标准指标），公式（如有）
- **训练配置**：完整超参数表（正文+附录合并）

```html
<!-- 超参数完整表格示例 -->
<table>
  <tr><th>超参数</th><th>值</th><th>来源</th></tr>
  <tr><td>学习率</td><td>1e-4</td><td>正文 §4.1</td></tr>
  <tr><td>Batch size</td><td>256</td><td>[附录 B.1]</td></tr>
  <!-- ... -->
</table>
```

- **硬件配置**：GPU 类型和数量，训练时长（若论文提到）
- **代码仓库**：若论文提供，给出链接

#### 5.2 主要结果

- 嵌入关键结果表格和图表（若图表来自附录，标注 `[附录 X.X]`）
- 若图表为 TikZ 生成，用 HTML 表格重建关键数值
- 逐行/逐列解读主结果表：哪些比较最关键，差距有多大，有无异常值
- 指出论文自己解释了哪些 surprising 的结果

#### 5.3 消融实验

- 每个消融实验的目的：在验证哪个设计选择
- 结论：这个组件去掉后影响多大（用具体数值）
- 若附录有额外消融，整合进来 `[附录 X.X]`

#### 5.4 补充实验（来自附录）

整理附录中所有额外实验，包括：
- 在更多数据集上的结果
- 超参数敏感性分析
- 可视化和定性分析
- 计算效率分析

每个补充实验说明目的和结论。

---

### 6. 分析与讨论

**严格基于论文内容，不做过度推断。**

- **反直觉的发现**：有什么实验结果或结论出乎意料？论文自己如何解释？
- **作者自陈的局限性**：论文在 Conclusion 或附录中明确提到的局限是什么？
- **未解释的现象**：你在读论文时发现的、论文没有解释清楚的地方（如某个结果的原因不明）
- **方法的适用边界**：根据实验设置和消融，这个方法在什么条件下有效？在什么条件下可能失效？

---

### 7. 关键术语表

| 术语（英文） | 中文翻译 | 通俗解释 | 首次出处 |
|---|---|---|---|
| Term | 翻译 | 一句话解释 | 论文 §X / Appendix A |

---

### 8. 相关工作梳理

**本节以论文自身的 Related Work 章节内容为主，不做大规模外部检索。**

#### 8.1 论文自述的相关工作

按论文 Related Work 章节组织，还原论文作者如何定位自己的工作：

- 论文划分了哪几条相关工作线
- 每条线的核心工作是什么（列出论文提到的关键 paper，每篇 1-2 句话）
- 每条线如何与本文区别

目的是帮助读者理解论文在自己定义的问题空间中的位置，以及作者选择对比方法的逻辑。

#### 8.2 直接前作对比

基于 Introduction 和 Related Work 中的内容，整理出论文与其 2-4 个最重要前作的关键差异：

| 维度 | 前作 A | 前作 B | **本文** |
|------|-------|-------|---------|
| 核心思路 | ... | ... | ... |
| 关键假设 | ... | ... | ... |
| 适用场景 | ... | ... | ... |
| 实验性能 | ... | ... | ... |

---

### 9. 可复现性审计

**核心目标：读完这节，你应该知道复现这篇论文需要做什么，会卡在哪里。**

#### 9.1 数据审计

```
✅ / ⚠️ / ❌  数据集是否公开可获取？（给出下载链接或说明）
✅ / ⚠️ / ❌  数据预处理流程是否完整描述？（整合附录中的数据处理细节）
✅ / ⚠️ / ❌  有没有使用私有数据或特殊权限数据？
✅ / ⚠️ / ❌  数据规模是否明确（样本数、特征维度、分割比例）？
✅ / ⚠️ / ❌  是否存在数据泄露风险？
```

#### 9.2 代码审计

```
✅ / ⚠️ / ❌  代码是否开源？（给出仓库链接）
✅ / ⚠️ / ❌  核心算法的伪代码 / 流程图是否完整？
✅ / ⚠️ / ❌  是否依赖特定框架 / 版本？
✅ / ⚠️ / ❌  有无非标准操作在正文中未提及？（从附录实现细节中找线索）
```

#### 9.3 实验审计

```
✅ / ⚠️ / ❌  完整超参数是否全部给出？（正文 + 附录合并检查）
✅ / ⚠️ / ❌  模型初始化方式是否说明？
✅ / ⚠️ / ❌  是否报告了多次运行的均值和方差？
✅ / ⚠️ / ❌  Baseline 对比是否公平（相同数据、相同评测协议）？
✅ / ⚠️ / ❌  消融实验是否充分（每个关键组件都被测试）？
```

#### 9.4 资源审计

```
✅ / ⚠️ / ❌  训练需要什么硬件（GPU 类型和数量）？
✅ / ⚠️ / ❌  训练时间是多少？
✅ / ⚠️ / ❌  一个独立研究者（如 1-2 张 A100）能否复现？
```

#### 9.5 复现性评级与行动建议

| 维度 | 评级 | 说明 |
|------|------|------|
| 数据可获取性 | ✅/⚠️/❌ | ... |
| 代码可用性 | ✅/⚠️/❌ | ... |
| 实验细节完整度 | ✅/⚠️/❌ | ... |
| 资源可达性 | ✅/⚠️/❌ | ... |
| **综合复现性** | **高/中/低** | ... |

**如果要复现，建议从哪里开始：**（具体的第一步，例如"先跑作者代码在数据集 X 上的复现，对齐 Table 1 第三行的数字"）

**预计会遇到的障碍：**（例如"附录 B 的数据预处理没有代码，需要自行实现"）

---

## 在线公式渲染配置

在 HTML 的 `<head>` 中添加：

```html
<script>
  window.MathJax = {
    tex: {
      inlineMath: [['$', '$'], ['\\(', '\\)']],
      displayMath: [['$$', '$$'], ['\\[', '\\]']]
    },
    svg: { fontCache: 'global' }
  };
</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
```

---

## HTML 报告模板要点

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{论文标题} - 精读报告</title>
  <script>
    window.MathJax = {
      tex: {
        inlineMath: [['$', '$'], ['\\(', '\\)']],
        displayMath: [['$$', '$$'], ['\\[', '\\]']]
      },
      svg: { fontCache: 'global' }
    };
  </script>
  <script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
  <style>
    body { max-width: 960px; margin: 0 auto; padding: 2rem; font-family: -apple-system, "Noto Sans SC", sans-serif; line-height: 1.8; color: #333; }
    h1 { font-size: 1.8rem; border-bottom: 2px solid #2563eb; padding-bottom: 0.5rem; }
    h2 { font-size: 1.4rem; color: #1e40af; margin-top: 2.5rem; border-left: 4px solid #2563eb; padding-left: 0.8rem; }
    h3 { font-size: 1.2rem; color: #1e3a5f; }
    h4 { font-size: 1.05rem; color: #334155; }

    /* 元信息 */
    .meta { background: #f0f4ff; padding: 1.2rem; border-radius: 8px; margin: 1rem 0; }
    .meta p { margin: 0.3rem 0; }

    /* 速览 */
    .tldr { background: #fef3c7; padding: 1rem 1.2rem; border-radius: 8px; border-left: 4px solid #f59e0b; font-size: 1.05rem; margin: 1.5rem 0; }
    .contrib-list li { margin: 0.4rem 0; }
    .contrib-explain { font-size: 0.9rem; color: #6b7280; margin-left: 1rem; }

    /* 附录引用标注 */
    .appendix-ref { font-size: 0.85rem; color: #7c3aed; background: #f5f3ff; padding: 0.1rem 0.4rem; border-radius: 3px; font-family: monospace; }

    /* 图表 */
    .figure { text-align: center; margin: 1.5rem 0; }
    .figure img { max-width: 100%; border: 1px solid #e5e7eb; border-radius: 4px; }
    .figure .caption { font-size: 0.9rem; color: #6b7280; margin-top: 0.5rem; font-style: italic; }
    .figure-tikz { background: #f8fafc; border: 2px dashed #cbd5e1; border-radius: 8px; padding: 1.5rem; text-align: center; color: #64748b; margin: 1.5rem 0; }

    /* 公式解释块 */
    .formula-block { background: #f8fafc; padding: 1.2rem; border-radius: 8px; margin: 1rem 0; border: 1px solid #e2e8f0; }
    .formula-block .intuition { font-size: 1rem; color: #1e3a5f; font-weight: bold; margin-bottom: 0.5rem; }
    .formula-block .explain { font-size: 0.95rem; color: #475569; margin-top: 0.8rem; }
    .symbol-table { width: 100%; border-collapse: collapse; margin: 0.8rem 0; }
    .symbol-table td { padding: 0.4rem 0.8rem; border-bottom: 1px solid #e5e7eb; font-size: 0.95rem; }
    .symbol-table td:first-child { font-family: monospace; color: #1e40af; width: 30%; }

    /* 实现要点 */
    .impl-note { background: #fff7ed; border-left: 4px solid #f97316; padding: 0.8rem 1rem; border-radius: 0 6px 6px 0; margin: 0.8rem 0; font-size: 0.95rem; }
    .impl-note::before { content: "⚙ 实现要点："; font-weight: bold; color: #c2410c; }

    /* 伪代码 */
    .pseudocode { background: #1e2433; color: #e2e8f0; padding: 1.2rem; border-radius: 8px; font-family: monospace; font-size: 0.9rem; line-height: 1.6; overflow-x: auto; margin: 1rem 0; }
    .pseudocode .comment { color: #64748b; }
    .pseudocode .keyword { color: #93c5fd; font-weight: bold; }
    .pseudocode .dim { color: #86efac; }

    /* 超参数表格 */
    table { width: 100%; border-collapse: collapse; margin: 1rem 0; }
    th { background: #f1f5f9; text-align: left; font-weight: 600; }
    th, td { padding: 0.6rem 0.8rem; border: 1px solid #e2e8f0; }
    .from-appendix { color: #7c3aed; font-size: 0.85rem; }

    /* 折叠推导 */
    details { margin: 0.8rem 0; }
    details summary { cursor: pointer; font-weight: bold; color: #2563eb; padding: 0.4rem 0; }
    details > div { padding: 1rem; background: #f8fafc; border-radius: 4px; margin-top: 0.5rem; border-left: 3px solid #93c5fd; }

    /* 可复现性审计 */
    .audit-pass { color: #16a34a; font-weight: bold; }
    .audit-warn { color: #d97706; font-weight: bold; }
    .audit-fail { color: #dc2626; font-weight: bold; }
    .audit-card { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1rem 1.2rem; margin: 0.8rem 0; }
    .audit-card h4 { margin-top: 0; color: #334155; }
    .action-box { background: #f0fdf4; border: 1px solid #86efac; border-radius: 8px; padding: 1rem 1.2rem; margin: 1rem 0; }
    .action-box h4 { color: #166534; margin-top: 0; }
    .obstacle-box { background: #fef2f2; border: 1px solid #fca5a5; border-radius: 8px; padding: 1rem 1.2rem; margin: 1rem 0; }
    .obstacle-box h4 { color: #991b1b; margin-top: 0; }

    /* 时间线（方法演变） */
    .timeline { margin: 1rem 0; padding: 0.5rem 0; }
    .tl-item { display: flex; align-items: baseline; gap: 0.8rem; padding: 0.6rem 1rem; background: #f8fafc; border-radius: 8px; margin: 0.3rem 0; border-left: 3px solid #93c5fd; }
    .tl-item.tl-current { background: #eff6ff; border-left-color: #2563eb; }
    .tl-year { font-family: monospace; color: #64748b; min-width: 3rem; font-size: 0.9rem; }
    .tl-name { font-weight: bold; color: #1e3a5f; min-width: 8rem; }
    .tl-note { font-size: 0.9rem; color: #475569; }
    .tl-arrow { text-align: center; color: #94a3b8; font-size: 0.85rem; padding: 0.2rem 0; }

    /* 结构图说明标注 */
    .diagram-note { font-size: 0.85rem; color: #7c3aed; background: #f5f3ff; border: 1px dashed #c4b5fd; border-radius: 6px; padding: 0.5rem 0.8rem; margin: 0.5rem 0; }
    .toc { background: #f8fafc; padding: 1.2rem; border-radius: 8px; margin: 1.5rem 0; }
    .toc ul { list-style: none; padding-left: 1.2rem; margin: 0.3rem 0; }
    .toc li { margin: 0.2rem 0; }
    .toc a { text-decoration: none; color: #2563eb; font-size: 0.95rem; }
    .toc a:hover { text-decoration: underline; }

    /* 打印优化 */
    @media print {
      body { max-width: none; }
      details { open; }
      .no-print { display: none; }
    }
  </style>
</head>
<body>
  <!-- 报告内容 -->
</body>
</html>
```

---

## 写作风格指南

### 核心原则

1. **方法深度第一** — 这份报告的读者是要准备组会的 PhD，他们需要理解到能回答"为什么不这样做"的程度
2. **附录必须整合** — 附录信息不是可选项，是正文必要的补充。每处引用附录内容用 `[附录 X.X]` 标注
3. **公式必须接地气** — 每个重要公式先给直觉，再给形式化，再逐符号解释。不允许只有公式没有解释
4. **具体而不笼统** — 不说"效果更好"，说"在 ImageNet 上 Top-1 准确率提升了 2.3%"
5. **推导要完整** — 论文跳步的推导要补完。不确定的推导要标注"以下为推测补充，原文未给出"
6. **实现导向** — 在可能的地方，把方法描述转化为实现语言（"这等价于一个 masked attention"、"可以用 einsum 表达为..."）
7. **不填充** — 删掉所有"近年来随着...的发展"、"值得注意的是"之类的学术套话

### 公式解释模板

```html
<div class="formula-block">
  <p class="intuition">这个公式在计算：每个 token 应该关注其他哪些 token，权重由它们的相似度决定。</p>

  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

  <table class="symbol-table">
    <tr><td>$Q$</td><td>Query 矩阵，维度 $n \times d_k$，代表"我在找什么"</td></tr>
    <tr><td>$K$</td><td>Key 矩阵，维度 $m \times d_k$，代表"我有什么"</td></tr>
    <tr><td>$V$</td><td>Value 矩阵，维度 $m \times d_v$，代表"我能提供什么信息"</td></tr>
    <tr><td>$\sqrt{d_k}$</td><td>缩放因子，防止点积值过大导致 softmax 梯度消失</td></tr>
  </table>

  <div class="explain">
    <strong>为什么除以 $\sqrt{d_k}$？</strong>假设 $Q$ 和 $K$ 的每个元素独立采样自均值为 0、方差为 1 的分布，则点积 $q \cdot k$ 的方差为 $d_k$，标准差为 $\sqrt{d_k}$。不缩放时，高维向量的点积值偏大，softmax 饱和，梯度接近零。
  </div>

  <details>
    <summary>完整推导（点击展开）</summary>
    <div>
      <p>设 $q_i, k_j \sim \mathcal{N}(0, 1)$ 独立同分布，则 $q \cdot k = \sum_{l=1}^{d_k} q_l k_l$...</p>
    </div>
  </details>
</div>
```

### 伪代码模板

```html
<div class="pseudocode">
<span class="keyword">Algorithm</span>: Forward pass of {方法名}
<span class="comment"># 输入</span>
Input: x <span class="dim">[B, T, d_model]</span>  <span class="comment"># batch, sequence length, hidden dim</span>

<span class="comment"># Step 1: 投影到 Q, K, V</span>
Q = Linear(x)   <span class="dim">[B, T, d_k]</span>
K = Linear(x)   <span class="dim">[B, T, d_k]</span>
V = Linear(x)   <span class="dim">[B, T, d_v]</span>

<span class="comment"># Step 2: 计算注意力权重</span>
scores = Q @ K.transpose(-1, -2) / sqrt(d_k)  <span class="dim">[B, T, T]</span>
weights = softmax(scores, dim=-1)

<span class="comment"># Step 3: 加权聚合</span>
out = weights @ V   <span class="dim">[B, T, d_v]</span>
</div>
```

---

## 验收清单

生成报告后，逐条自查：

**论文内容覆盖（首要检查）**
- [ ] Phase 2.5 章节盘点是否完成？每一个原文章节是否都有对应的报告位置？
- [ ] Introduction 中的所有论点和分析是否都已进入报告（特别是技术性分析、失败案例、对比数据）？
- [ ] Related Work 的所有技术线是否都已在 §8 中覆盖？
- [ ] Conclusion / Discussion 中的局限性和展望是否进入了 §6？
- [ ] 论文如有独立的 Analysis 或 Discussion 章节，是否完整整合进 §6？
- [ ] 所有正文中的图表（Figure / Table）是否都在报告中出现（嵌入或说明）？没有任何图表被静默跳过？

**动机与贡献**
- [ ] 动机是否基于论文 Intro 的具体分析，而不是泛化描述？
- [ ] 核心贡献列表是否准确还原了论文自己的 Contribution 部分？

**前置知识**
- [ ] 是否只收录了真正必要的概念？是否附有出处论文？

**方法详解**
- [ ] §4.1 是否用文字清楚描述了整体 pipeline？论文有独立 overview figure 的是否已嵌入并标注？
- [ ] 每个重要公式是否有直觉说明 + 逐符号解释 + 推导（含补充）？
- [ ] 是否归纳了伪代码（论文有则重排注释，论文无则自行补充）？Training / Inference 是否分开写？
- [ ] 实现要点是否覆盖了正文 + 附录中所有与复现相关的细节？
- [ ] 附录中的完整证明 / 定理是否已整合进对应推导位置，并标注 `[附录 X.X]`？

**实验**
- [ ] 超参数是否给出完整表格（正文 + 附录合并）？
- [ ] 主结果表是否逐行/逐列解读，而不只是转述？
- [ ] 附录中的补充实验（额外数据集、超参数敏感性等）是否整合进 5.4 节？

**可复现性**
- [ ] 四个维度（数据/代码/实验/资源）是否都逐条检查？
- [ ] 是否给出了具体的"第一步该做什么"和"预计障碍"？

**相关工作**
- [ ] 相关工作部分是否以论文自身 Related Work 为主，没有做大规模外部检索？
- [ ] 前作对比表是否基于论文内容填写？

**写作质量**
- [ ] 全文是否消除了学术套话？
- [ ] 所有附录引用是否都标注了 `[附录 X.X]`？
- [ ] 联网时 HTML 能否正常显示公式和图表？

---

## 错误处理

| 错误类型 | 处理方式 |
|----------|----------|
| e-print 下载失败 | 尝试用 WebFetch 获取 arxiv HTML 页面内容作为备选 |
| 源码非 tar.gz 格式 | 可能是单个 .tex 文件，直接重命名处理 |
| 图表 PDF 转换失败 | 记录失败文件，在报告中标注"图表无法显示"；优先用 pdftoppm 重试 |
| 仅有 TikZ 绘图，无图像文件 | 提取数据建表格，描述图表内容，figures/ 保持为空 |
| 论文无 Appendix | 在各章节中注明"本文无附录"，按正文内容正常生成 |
| 附录过长（>50 页） | 按分类（证明/超参/实验/实现）提取关键内容，略去低优先级部分，注明"附录 X 章节已简略" |
| .tex 文件超大 | 优先保证 Abstract + Method + Experiment + Appendix 完整 |
| 在线数学渲染器加载失败 | 提示公式可能无法渲染，保留原始 LaTeX |

## 约束

1. **报告语言为中文**，数学符号和专有名词保留英文原文
2. **图表资源使用相对路径引用**，在线数学渲染脚本可使用外部 URL
3. **图表从源码提取**（独立图像文件），TikZ 绘图不使用外部图床
4. **公式在线渲染**，默认使用 MathJax
5. **不省略数学细节**，宁可啰嗦也不跳步，跳步时补充中间推导
6. **附录内容必须整合**，不单独列章节，不略去
7. **相关工作以论文自身为准**，不做大规模外部检索
8. **去掉联动章节、角色扮演和 next actions**