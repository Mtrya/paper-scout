# OC-VLA(2508.13103)深挖线程:「外参给定」路线的代表

- 论文:Grounding Actions in Camera Space: Observation-Centric Vision-Language-Action Policy(ZJU + 上海 AI Lab + SenseTime + 南大 + 清华,AAAI 2026,2025-08 上 arXiv)
- 论文 markdown:`papers/vla/ocvla-2508.13103.md`;图:`drafts/images/ocvla-2508.13103/`
- 代码:**OC-VLA 官方仓库(`ZTY0213/OC-VLA`)是空壳**,README 指向 Dita;OC-VLA 实现实际位于 Dita 仓库(RoboDita/Dita)的 `use_baseframe_action=False` 分支。已克隆:`code/oc-vla/`(空壳)、`code/dita/`(稀疏克隆;关键文件另行抓到本地核验)

## 问题与动机

VLA 的观测-动作空间错位:视觉编码器(DINOv2/CLIP)在图像/相机坐标系里被监督,动作却几乎总定义在**机器人基座系**。单目/双目第三人称输入下,要模型从 2D 观测隐式重建「相机↔基座」的 3D 变换再去预测基座系动作,是病态的——尤其当数据跨异构视角(DROID 1417 个相机位姿)预训练时,同一动作在不同视角被同一个基座系监督信号硬约束,产生学习冲突。OC-VLA 的立场:预测目标不应该是「基座系里的动作」,而应该是「相机系里的动作」——**给定外参**,把动作目标先变换进相机系再监督,让感知与动作共享一个坐标系。这是与 CamVLA「自估外参」相对的另一条路线:外参不是被学出来的,是被**给定**的,训练与部署双端都硬依赖它。

## 方法

记基座(论文称 world)系两个相邻末端位姿为 $P_{w1}, P_{w2} \in SE(3)$(4×4),外参 $T \in SE(3)$ 为世界→相机变换(OpenCV 约定,4×4,含旋转 $R$ 与平移 $t$)。

**论文宣称的目标**(式 1–4):相机系位姿 $P_{ci}=T P_{wi}$,相对动作取群 delta:

$$A_{cam}=P_{c2}P_{c1}^{-1}=T\,A_{world}\,T^{-1},\quad A_{world}=P_{w2}P_{w1}^{-1}$$

再转成 7 维 $\langle x,y,z,\text{roll},\text{pitch},\text{yaw},g\rangle$ 作为监督。推理时把预测的相机系动作经 $T^{-1}$ 变回基座系执行。整体是数据集端预处理,模型结构(Dita 的 334M:CLIP 文本编码 + DINOv2 + 4 层 Q-Former/FiLM + 12 层 LLaMA2 风格因果 Transformer)零改动;连续(扩散)与离散(量化 token)两种动作空间各跑一个变体。

**代码实际实现**(`utils/data_utils.py::process_traj_v3`,连续/离散共用,默认 `use_euler=0`):

- 平移 delta:$\Delta p = t_{c2} - t_{c1} = R(p_2 - p_1)$ —— **朴素坐标差,不是群 delta 的平移分量**(后者应为 $t_{c2} - R_{c2}R_{c1}^\top t_{c1}$;代码里该行被注释掉,`# translation_delta = pose1_to_pose2.get_matrix()[0, -1, :3]`);
- 旋转 delta:相对旋转 $R_{c2}R_{c1}^\top = R(R_{b2}R_{b1}^\top)R^\top$ 的四元数(带 `w-1` 编码,推理时 `w+1` 还原;欧拉路径可选);
- 基座→世界仅对 x 减 0.615m(`pose1[0] -= 0.615`),即曼尼希尔基座与世界原点差。

**两个直接推论(论文均未指出)**:
1. **训练目标只依赖 $T$ 的旋转部分 $R$**——平移 $t$ 在坐标差与共轭中都严格相消;
2. 部署端却**全 $T$ 参与**:`cal_action`/`cal_action_from_pose`(来自 OpenVLA 的 robot_utils)把当前位姿经 $\tilde T$ 转进相机系、再把目标经 $\tilde T^{-1}$ 转回基座系。外参平移误差 $\delta t$ 以常值偏置 $R^\top\delta t$ 泄漏进每一步执行(绝对位姿模式是直接平移偏置,delta 模式经「当前位姿变换」泄漏)。**这恰是 CamVLA 附录 B 论证并消除的东西**:CamVLA 执行端只用估计的 $R$ 旋转自由向量 delta、当前位姿始终留在基座系,平移从不进入管线,故 $\tau$ 误差严格相消。

## 证据链(关键数字与协议,含弱证据)

**仿真(ManiSkill2,从零训练,表 I)**——5 任务,30 万相机视角池、每轨迹随机 20 视角渲染、~4 万轨迹、19:1 划分、每任务 100 条闭环评测:

| 动作空间 | 目标系 | All | PickCube | StackCube | PickSingleYCB | PickClutterYCB | PickSingleEGAD |
|---|---|---|---|---|---|---|---|
| 连续 | 基座 | 45.2% | 71.0% | 62.0% | 30.0% | 15.0% | 48.0% |
| 连续 | 相机 | 53.2% | 88.0% | 65.0% | 46.0% | 19.0% | 48.0% |
| 离散 | 基座 | 38.6% | 61.0% | 51.0% | 28.0% | 8.0% | 45.0% |
| 离散 | 相机 | 52.4% | 80.0% | 65.0% | 48.0% | 19.0% | 50.0% |

相机系目标全面占优,离散空间差距最大(+13.8pp,论文称 "about 14%")。**但**:单 seed、无方差;硬任务仍是单位数/十位数(ClutterYCB 19% vs 8%);基线即同一架构仅换监督坐标系——对照干净,但结论的规模意义有限。

**真机(Franka + Robotiq,10-shot,每任务 10 次,表 II/III)**:固定视角 OC-VLA 68.0% vs 基座系 58.0%(+10pp),OpenVLA-OFT 63.3%、π0 50.7%;新颖相机零样本 OC-VLA 54.0%(−14pp)vs OFT 42.0%(−21.3pp)、基座系 41.3%;相机微扰(Camera 2)下相机系 73.8% vs 基座系 61.3%(+12.5pp)。**关键协议事实:新颖视角与相机微扰两个设置都要求「重标定拿到新外参」再评估**——robustness 属于动作表示(策略无需重训),不是对外参不确定性的鲁棒;外参误差本身从未被注入测试。真机数字无 error bar,100% 即 10/10;15 任务为私有数据集,不可复现对照。

**弱点汇总**:无多 seed/方差报告;DROID 预训练只用于初始化、收益未做有无对照;消融(表 IV)只在离散变体上;手腕相机、多相机融合、跨本体均未覆盖;论文表述的「群 delta + 欧拉 7 维」与代码的「平移差 + 四元数(w-1)」不一致(代码默认 `use_euler=0`);无官方 checkpoint 与真机数据。

## 代码核验结果

- `github.com/ZTY0213/OC-VLA`:`LICENSE` + `README.md` 共 2 文件、单 commit(2025-11-15),正文即「Code 见 Dita 仓库」+ 引用。论文 "code will be publicly available" 只以**指向基座架构的指针**形式兑现,OC-VLA 特有部分(变换、配置)未单独发布。
- Dita README 白纸黑字:"This Codebase can also be used to run OC-VLA, set the `use_baseframe_action=False`... 训练于第三人称相机坐标"。即 **OC-VLA = Dita + 一个 flag**;贡献主体在经验故事(坐标系选择)而非新结构——论文自己的定位(lightweight, plug-and-play)与代码一致。
- 变换实现位置:`Dataset_Sim/SimDataset.py`(连续)/`SimDataset_discrete.py`(离散)→ `utils/data_utils.py::process_traj_v3/get_pose_cam`;外参来自每 episode 的 `camera_extrinsic_cv`(4×4 OpenCV 约定),`use_baseframe_action=True` 时换成 `torch.eye(4)`(退化为基座系)。
- 数据侧与论文吻合:`ManiSkill2/openx_utils/generate_camera_pool.py`(NUM_CAMERAS=300000)+ `configs/camera_pool_300k.npz`;`perturb_extrinsic`(5% 乘性扰动)存在但在主路径被注释掉——**仿真全程无外参噪声**。
- 部署端:`scripts/close_loop_eval_diffusion.py` 里 `model.inference(extrinsic_cv if not use_baseframe_action else eye(4))` 把外参传给推理类,`cal_action_from_pose`/`cal_action` 做 $\tilde T^{-1}$ 逆变换;`deploy_diffusion.py`(真机服务端)传 `torch.eye(4)` 并把逆变换留给调用方——真机端外参由机器人侧持有,论文没有发布这段完整闭环代码。
- 结论:动作重参数化的**存在形式**与论文一致(数据集端预处理、零结构改动);**数学细节与论文公式有出入**(见「方法」两条推论),且推理端把「当前位姿经外参往返」的路径写进了执行,外参误差不会像 CamVLA 那样被 delta 结构消掉。

## 与本轮两条主线的关系

- **主线一(外参在线/自标定)**:OC-VLA 是「外参给定」分支的典型消费者——训练时 $T$ 决定目标,部署时 $\tilde T$ 决定执行,中间没有估计环节。滤波式在线标定(EKF 手眼标定等)的输出**正是它部署管线缺的那一块**:相机被碰/漂移后由标定器重估 $T$ 喂回去。反过来,OC-VLA 也是评测在线标定价值的理想平台:注入外参误差测端到端成功率退化曲线,把「标定好」的收益量化出来。它与 CamVLA 构成自标定线的两极:OC-VLA 假设 $T$ 已知(误差不注入),CamVLA 从 RGB 每步自估(误差逐帧独立);两者都未处理「静态/慢变外参失准」——OC-VLA 是测这个问题的更自然对象,因为它真的用 $T$。
- **主线二(相机系、跨本体、VLM 友好的动作表示)**:OC-VLA 与 CamVLA 独立得出同一结论——相机系目标优于基座系目标,支持「动作应落在观测空间」这一表示层面的共识。但 OC-VLA 的「相机系」仍是 3D SE(3) delta(平移差 + 四元数),不是像素/UV 系动作;且其跨视角统一性依赖「每视角都给外参」这一数据条件(DROID 恰好满足,一般数据集不满足)——这正是它作为跨本体通用表示的软肋:表示本身对 VLM 友好,但**进入与退出相机系都需要校准服务**,把最脆弱的环节留在部署时。
- **与 CamVLA 的本质差异一句话**:外参的归属。OC-VLA 在训练与部署两端都**给定并应用** $T$(训练端因 delta 结构实际只用 $R$);CamVLA 只在训练端把 $T$ 当监督、部署端零外参、执行只经估计的 $R$ 旋转自由向量。OC-VLA 的部署缺口 = 每次相机变动后的重标定 + 外参误差(尤其平移)以常值偏置泄漏进执行。

## 开放问题与可做的研究动作

1. **外参误差敏感度定量评测**(OC-VLA 版「附录 C」):仿真注入 $\delta t,\delta R$(静态/慢变,而非 CamVLA 的每步独立重采样),测 delta 模式与绝对位姿模式下的成功率退化;验证「平移偏置 $R^\top\delta t$」的预测是否成立、误差多大开始致命。这是本轮两条主线交汇处最直接的实验。
2. **在线标定闭环接入**:EKF/滤波式手眼标定 → 给 OC-VLA 部署管线供 $T$,相机被碰后自动恢复,测端到端成功率保持。可直接复用本线已核验的 Kalib/ARC-Calib 代码与 OC-VLA 的 Franka 设置。
3. **训练目标平移的两种选择**:朴素坐标差 vs 真群 delta 平移($t_{c2}-R_{c2}R_{c1}^\top t_{c1}$),在大旋转任务(PickSingleYCB)上对比——代码里作者已写下群 delta 版本又换回朴素差,动机未解释。
4. **欧拉 vs 四元数 delta** 在同一相机系目标下消融(论文声称 7 维欧拉,代码默认四元数 w-1 编码)。
5. **视角密度依赖曲线**:稀疏视角训练(如 CamVLA 的 45° 间隔)下 OC-VLA 的退化,与 CamVLA 同协议直接对比「给定外参 vs 自估外参」在视角稀少时的优劣。
6. **跨本体验证**:只在 Franka + 私有 15 任务上验过;用 DROID 预训练 + 多本体真机/仿真评测「相机系目标」的跨本体收益,检验它作为通用表示的普适性。
7. **复现风险**:无 checkpoint、无 seed 报告、真机数据私有;代码与论文公式不一致处需要先跑通 `train_diffusion_sim.py` 才能确定实际行为——复现成本偏高。
