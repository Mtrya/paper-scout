# gpt-policy-sim — GPT-Policy 模拟机器人后端

在没有真机的情况下,用 GPT-Policy 官方的 `run_loop` + 官方 Codex harness(`codex app-server`,模型 gpt-6-astra)跑闭环实验。官方仓库(`../gpt-policy/`)保持干净克隆,未做任何修改;本目录通过 `.venv` 里的 `.pth` 文件直接引用其 `src/`(editable install 因官方 `pyproject.toml` 的 yam extra 直接引用 git URL 而无法构建,故采用 `.pth`,效果等同)。

## 组成

| 文件 | 作用 |
|---|---|
| `sim_world.py` | 3D 桌面世界:点状末端执行器(EE)+ 固定朝向四元数,工作区 x,y∈[0.15,0.85]、z∈[0,0.45],桌面 z=0。`SimRobot` 暴露官方协议:`state()`(全字段,关节为平滑 mock 运动学 + 微小噪声)、`execute("move_to"/"move_eef_chunk"/"check_path"/"set_gripper")`、`return_home()`。EE 以 0.1 m/s 走直线密化路径;越界目标抛官方 `TrajectoryIKError`(可恢复,与真机 IK 拒绝一致);路径撞上障碍在接触点停下并在 `execution_feedback.blocked` 里报告。抓取:闭合时 EE 距物体中心 xy≤2.5cm 且 z 窗口内即吸附;张开时落到支撑面(桌面/容器)。 |
| `sim_render.py` | matplotlib(Agg)渲染两个视角为 JPEG `CapturedImage`:`top`(俯视;图像上=+x 前、左=+y)与 `front`(从 -y 侧视;右=+x、上=+z)。标注工作区、物体名、EE 位姿/夹爪状态、门/按钮/槽/管状态。含 `top_pixel_to_world` 供定位工具用。 |
| `tasks.py` | T1 gate-button(透明挡板挡住红方块,按蓝色按钮滑开;指令只说"把红方块放进绿碗")、T2 align-insert(槽内柱子偏离对齐标记,直接抓因 1.5cm 容差失败,须先推正;指令只说"把柱子插到盒子里")、T3 hook-retrieve(方块在单侧开口管内,EE 进不去;用钩子拖出再抓;指令同 T1)。每个任务:seed 化布局(物体 xy ±5cm 抖动)、ground-truth 成功判据、脚本 oracle 专家(三任务 24/24 rollout 全部成功)。 |
| `demo_record.py` | 用脚本专家在不同 seed 布局下录制 episode,产出官方 reviewed-bundle 格式的 `demo.json`(schema_version 1,keyframes + 图像 + 可选 state/action 段),再调官方 `prepare_demonstration` 编译成 prepared input 目录。`--mode video`(数值字段被官方剥离,对应论文 Human Video 条件)或 `video+action`(保留 EE/关节/夹爪数值轨迹,对应 Robot Video+Action)。 |
| `run_episode.py` | CLI 接线:RunRecorder + ToolExecutor + 官方 `instructions()`/`observation()` + `run_loop`,agent 为真实 `CodexSession` 或 `--mock` 的 MockAgent(内部调脚本专家,走同一 `decide()` 接口)。 |
| `tools.sim.json` | 官方 `configs/tools.json` 的原样拷贝(全部 7 个工具)。 |

## 运行

```bash
# 环境(已完成):uv venv .venv && uv pip install matplotlib pillow numpy rich ruckig
#   + echo <gpt-policy>/src > .venv/lib/python3.13/site-packages/gpt_policy_src.pth

# 单 episode(真实 codex):
.venv/bin/python run_episode.py --task T1 --context none --seed 0
.venv/bin/python run_episode.py --task T3 --context demo-action --seed 5 --demo-seed 1042

# 冒烟(脚本专家,无模型调用):
.venv/bin/python run_episode.py --task T1 --context none --seed 0 --mock

# 录制 demo 包:
.venv/bin/python demo_record.py --task T1 --seed 1042 --mode video+action
```

`--context`:`none`(纯指令)/ `target`(目标状态参考图)/ `demo-video` / `demo-action`(后者自动在缺省时先录制)。运行包落盘 `runs/<TASK>_<context>_s<seed>_<ts>_<outcome>/`,与官方真机运行包同构(events.jsonl、frames/、protocol.json、transcript.json、status.json……),外加 `sim_evaluation` 事件记录 ground-truth 成败(目录后缀的 success/failed 由它决定,不由模型的 done 决定)。

## 与真机协议的偏差(已知)

1. **系统提示词**:`harness/protocol.py` 的 `_robot_calibration_notes` 写死了 ARX X5 与 "left/right wrist D405 + top D405" 相机布局,无法通过 settings 覆盖。我们的做法:`settings` 提供 `runtime.robot_model="SIM-X5"`、`interface="sim0"` 与 `scene.safety_notes`(声明模拟桌面与真实相机集合),并在官方 `instructions()` 输出后追加一段 `SIMULATION BACKEND NOTES`,明确:关节是 mock 运动学、只有 top/front 两相机(无腕部相机)、工作区硬边界、blocked 语义、夹爪负载读数含义。写死的 D405/腕机句子因此与模拟场景不符,但以追加说明的方式覆盖,未 patch 官方文件。
2. **check_path**:真机只查 IK/关节限位/时序(`collision_checked: False`);模拟版额外做静态障碍相交检查并在 `path_check.obstacle_contacts` 里报告(字段结构与官方 `path_check_result` 对齐,`collision_checked: True`)。这是有意的仿真增强。
3. **locate_point**:相机枚举仍是官方模板的 `["left","top"]`;`left`(腕机)在模拟里不存在,调用会像真机缺内参一样被可恢复地拒绝。`top` 返回与真机同构的单目射线结果(`metric_position_available: False`),射线由渲染器的像素→世界仿射逆映射精确给出,另附 `sim_table_intersection_xyz`(射线与桌面交点,模拟专有字段)。
4. **姿态**:EE 抽象为点,测量朝向跟随指令朝向(永远不报告旋转误差);渲染不画朝向。对需要侧向姿态才能完成的操作(如伸进水平管)建模为管壁碰撞阻挡,而不是姿态不可达。
5. **关节遥测**:`joint_positions_rad` 是平滑 mock 映射 + σ=0.0015 rad 噪声,只做协议填充;模型应看 `tcp_pose`(协议本身也这么要求)。
6. **视频录像**:真机 `RunVideo` 写 mp4;模拟 `SimVideo` 无后台线程,每步按需渲染,帧落盘在 recorder 的 `frames/`(step-XXXXX-top/front.jpg),不写 mp4。
7. **时间**:世界有确定性模拟时钟(`sim_time_s`,运动 0.1 m/s、夹爪 1s),但执行不占用真实时间;`captured_at`/`timestamp_s` 仍用 wall clock 以满足 recorder 的新鲜度检查。
8. **物理简化**:只有注册的障碍(门/管壁)阻挡 EE;一般物体不挡 EE(无全身碰撞);推动/拖动是接触即随动的简化模型;抓取无滑脱概率。T2 直抓必然失败是任务设计(布局 jitter 使偏移恒大于 1.5cm 容差),不是噪声结果。

## 已验证

- 脚本专家 oracle:T1/T2/T3 × seeds 0-7 共 24 次 rollout 全部成功;三个任务的反事实均成立(T1 关门直穿被挡、T2 未对齐直抓失败、T3 EE 进管被挡/管内直抓失败)。
- MockAgent 冒烟(--mock):T1/T2/T3 各 ≥1 episode 全程无异常,recorder 落盘,成功判据触发。
- 真实 codex 冒烟:T1 + none + seed 0,`codex app-server` 握手成功,模型 5 次决策全部合法(单次 6.6–17.8s 真实推理),并在 5 步预算内完整解出任务(按按钮开门→抓块→入碗→释放,ground-truth success=True);进程干净退出无残留。

## 消融实验(2026-09-22)

- `ablate.py`:3 任务 × 4 上下文臂 × 3 seeds = 36 格真实 codex episode,3 并发,单格 1800s 超时;每格一行追加到 `results/cells.jsonl`(权威数据源),可断点续跑。
- `summarize.py`:读 `results/cells.jsonl` 生成 `results/summary.md`(成功率网格 + 失败归因分解 + 逐格统计 + 失败 trace 附录)与 `results/ablation_grid.png`(分组柱状图);失败模式定性叙述手写在 `results/failure_modes.md`,重新生成时自动嵌入。
- 结果(36/36,8 成功):T1 none/target 3/3、demo 臂 0/3;T2 仅 demo-action 1/3;T3 仅 demo-action 1/3。失败归因:A 类首步幻觉 done 9 格(demo 臂 6/9)、B 类上下文伪影 0 格、真失败 17 格、超时 2 格。详见 `results/summary.md`。
