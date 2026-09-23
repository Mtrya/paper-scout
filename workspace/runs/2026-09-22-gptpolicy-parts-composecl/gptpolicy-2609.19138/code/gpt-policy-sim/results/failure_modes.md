以下结论全部来自对 28 个失败 episode 的 `events.jsonl` 逐格人工归因(transcript、决策序列、执行反馈、world 事件均已核对)。类别定义见上表。

### 首步幻觉完成(A 类,9 格)——真实模型行为,非 harness 伪影

9 个 episode 在第一步就输出 `done` 并编造完成摘要,一次动作都没执行。逐格核对原始 transcript 确认:对话干净(系统提示 + 一条观测),模型原始输出就是 `done`,harness 按协议正常终止——**不是解析失败、不是截断、不是 API 错误**。例:T2 none s0 首步即声称"插柱已夹取并垂直插入盒子中心……目标状态已完成",而地面真值显示柱子离盒口 41cm,根本未动。

分布:T1 demo-action s0/s1、T1 demo-video s2、T3 demo-video s2、T3 demo-action s0/s2(demo 臂共 6 格);T2 none s0/s1、T3 none s1(none 臂 3 格);target 臂 0 格。demo 臂浓度(6/9)提示:demonstration 以一段名为 "done" 的完成场景收尾,抬高了模型直接宣布完成的先验——尽管输入里有醒目的 "HISTORICAL DEMONSTRATION … not the current scene" 包裹与逐图 "Historical" 标注,模型仍把示范叙述成了自己的成果。但 none 臂也出现 3 格,说明这是模型的固有行为倾向,demo 只是放大器。

### B 类(上下文/prompt 伪影):0 格

全部 36 格日志扫描:无截断、无 usage-limit、无解析失败循环。仅有 9 次瞬时 "Model overloaded" 重试(全部 1–2 次后恢复)与 1 次非法四元数工具拒绝(模型下一步行恢复)。demo 臂输入为 20–34 张图像引用 + 20–30KB JSON,未触发任何溢出。**demo 臂的反常成绩不能用 prompt 伪影解释。**

### 真失败(G 类,17 格)的三种机制

**1. 坐标轴交换 + 自陷(T1 demo-video s0/s1)。** 模型把按钮的坐标读成 x/y 互换后的位置(真值 ≈(0.28,0.72),在门近侧;模型读出 (0.72,0.28),在门远侧),首次移动从 home 角(low-y 缺口)滑过了全高闸门,此后所有返回按钮的直线路径都被门挡住。s1 第 10 步模型自己写下"之前坐标轴误换导致撞门",却仍回到错误坐标反复空按按钮 6 次,最终 give_up。作为对照,none/target 6/6 全部在首步就正确定位按钮(目标坐标 (0.28–0.33, 0.69–0.74))并完成任务。轴交换只在带 demo 图像的臂出现(n=2,小样本,但 trace 里有模型自己的文字证据)。

**2. 抓取精度窗口打不中(T2 全部臂、T3 demo-video)。** T2 的柱子只有在推到红色对齐标记 ±1.5cm 窗口内才可抓(`plug_off_alignment_mark`);T3 钩子的抓取窗口是 2.5cm(xy)/ 3cm(z)。模型从这些 matplotlib 渲染图里读出的度量坐标误差普遍 2–5cm:T2 none s2 闭合 15 次全部抓空(反馈原因对模型不可见,只能靠扭矩 0.25Nm 自己推断);T2 demo-video s1 把柱子推过标记窗口(0.5058 → 0.4726,推出窗口)后 12 次抓空;T3 demo-video s0/s1 明知要用钩子(决策备注 32/19 次提到钩子)却在离钩子 5cm 处反复闭合。相比之下 T3 demo-action s1 在第 23 步闭合于 (0.346,0.265)——与钩子真值 (0.345,0.264) 误差 1.4mm——直接抓中。**demo-action 的动作坐标提供了 demo-video 纯图像给不了的度量锚点。**

**3. 策略缺口(T3 none/target)。** none 与 target 臂 6/6 从未想到用钩子(决策备注 0 次提到 hook),只会反复把 EE 往管里伸,被管壁挡住(z≈0.07 处持续 blocked),give_up 理由都是"无法形成可靠夹持"。只有看了示范的臂才知道钩子是关键道具——demo 在策略层面的迁移是真实的。

### 假阳性 done(真失败的尾声,6 格)

T1 demo-action s2 完成了"按按钮→开门→抓方块→释放"全流程,但释放点离碗心 7cm(碗半径 6cm),随后宣布"方块仍稳定可见于碗中"。T2 target s2 / demo-video s0 / demo-action s2 分别抓起柱子后释放在离插口 10.4cm / 4.3cm / 2.5cm 处(成功阈值 2cm),都声称"插入完成"。T2 demo-action s1 连抓都没抓上就宣布完成。唯一例外是 T3 demo-video s1:用 `done` 结束但摘要如实写明"任务尚未完成"——协议使用错误但自我评估诚实。

### demo 到底帮了还是害了

分任务方向相反。**T1(简单任务):demo 纯害。** none/target 6/6 全胜,demo 臂 0/6——3 格首步幻觉、2 格轴交换自陷、1 格释放偏差+假阳性。任务简单到不需要示范时,demo 只注入失败模式。**T2/T3(难任务):demo 是唯一亮点。** T2+T3 合计:none 0/6、target 0/6、demo-video 0/6、demo-action 2/6——仅有的两个非 T1 成功都来自 demo-action(T2 s0 推到标记→抓取→插入;T3 s1 用钩子拖出方块→抓取入碗,world 事件链完整:38 次 cube_dragged → cube_left_tube → released:hook → grasp_attached:cube)。策略层面 demo-video 也迁移成功(钩子、推拨对齐的概念都出现在了决策里),只是败在度量定位精度。
