# ACE-Data-0 (arXiv:2607.28625) — BLOCKER

**障碍**:数据不可得。截至 2026-08-02,HuggingFace 仓库 `ACE-Robotics/ACE-Data-0`
处于 gated 状态且页面明示 "Data files are not published yet"(API 实测返回授权
错误)。论文承诺的 150h/17M 帧/75k episodes 无法获取,任何基于该数据的动手
实验都无法开展。本线程因此没有 code/ 或 patches/ 证据,只保留此障碍说明与
围绕它的生态核验。

## 障碍之外仍完成的工作

ACE-Data-0 是精读线程，没有配独立实验——实验预算给了 PhiZero(Wan2.2 物理探针,
`../phizero-2607.28624/`)与 ShadowDancer(sprites cross-shadow 探针,
`../shadowdancer-2607.28362/`)。本线的"调查"性质是生态核验与三角验证:

1. **数据可得性核验(2026-08-02)**:HuggingFace 仓库 `ACE-Robotics/ACE-Data-0`
   为 gated,且页面明示 "Data files are not published yet"。论文承诺的
   150h/17M 帧/75k episodes 当前完全无法外部复核,benchmark 只有论文自报数字。
   这与它在报告中的定位直接相关:它现在是一份"测量引擎的规格说明书",
   而不是一个可用的数据源。
2. **与 Data Pyramid (arXiv:2607.24744) 的三角验证**:两篇同期论文独立指出
   同一缺口——failure/recovery 数据。ACE-Data-0 的 goal-level 指令设计
   (不纠正操作者的犹豫与恢复动作)是供给侧的回答;Data Pyramid 的五层
   金字塔把 "collecting failure and recovery data" 列为六大挑战之一,是
   需求侧的确认。接上次巡航的 UMI recovery 猜想。
3. **benchmark 数字的内部解读**(全部来自论文自报,无法独立复现):
   - 世界系轨迹误差(WA-MPJPE 180-256mm)远大于局部姿态误差
     (PA-MPJPE 55-70mm)——全局定位比姿态难一个量级;
   - ego 手部轨迹误差 98-102mm vs exo 63mm——主误差源是 egomotion 估计
     而非手指关节。对"ego 视频预训练"路线是个值得记住的警告。

## 复现方式

无需环境。精读笔记在 `papers/embodied-data/ace-data-0-2607.28625.md`;
HF 状态用浏览器或 `curl -sI https://huggingface.co/datasets/ACE-Robotics/ACE-Data-0`
复核即可(登录墙即 gated)。
