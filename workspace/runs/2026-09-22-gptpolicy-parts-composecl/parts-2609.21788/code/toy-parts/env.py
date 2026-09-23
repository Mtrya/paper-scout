"""ToyChainEnv: 2D 点质量末端执行器三段任务链环境(σ1 抓取 → σ2 绕障搬运 → σ3 精密插入)。

平面语义:x = 桌面横向,y = 竖直方向。EE 速度控制(30Hz 离散),夹爪一维开合。
被抓住的物块刚性跟随 EE;释放后静止在释放位置(玩具抽象,无重力落体)。

阶段判据 φ_i:
  φ1: 夹爪在物块 2.5cm 内闭合(grip ≥ 0.9 时 |ee−obj| ≤ 0.025)→ 抓住。
  φ2: 保持抓住且物块进入 staging 点 P2=(0.70, 0.30) 半径 5cm 内(必然在墙右侧)。
  φ3: σ3 内发生释放事件(grip ≤ 0.1)或 σ3 超时:成功当且仅当释放时
      |obj_x − slot_x| ≤ 3mm 且 |obj_y − slot_y| ≤ 8mm(槽宽 = 块宽 + 6mm,单侧间隙 3mm)。

阶段推进由事件驱动:φ_i 满足即进入下一段;每段 200 步上限,超时即该段失败。
评估时一次通过(不重试);训练时 PARTS 臂在 σ3 失败后 restage 重试(见 run.py)。
"""

import numpy as np

DT = 1.0 / 30.0
V_MAX = 0.5            # m/s,动作 a[:2] ∈ [-1,1]^2 乘此系数
GRIP_RATE = 3.0        # 1/s,grip 向 a[2] 符号方向线性移动
GRASP_DIST = 0.025     # 2.5cm 抓取半径
P2 = np.array([0.70, 0.30])          # σ2 staging  waypoint
P2_TOL = 0.05
SLOT = np.array([0.70, 0.10])        # 槽中心
SLOT_LAT_TOL = 0.003   # 3mm 横向容差
SLOT_VER_TOL = 0.008   # 8mm 竖向容差(块需落入槽深)
WALL_X = 0.35
WALL_H = 0.35          # 墙段:x=0.35, y ∈ [0, 0.35]
EE_HOME = np.array([0.0, 0.5])
OBJ_SPAWN = (np.array([-0.05, 0.15]), np.array([0.15, 0.30]))  # x/y 矩形
WS_LO = np.array([-0.10, 0.0])   # 工作区边界
WS_HI = np.array([0.90, 0.60])
STAGE_MAX_STEPS = 200
EPISODE_MAX_STEPS = 600

# 观测归一化常数
_OFS = np.array([0.40, 0.30, 0.0, 0.0, 0.40, 0.30, 0.0, 0.0, 0.5, 0.40, 0.30])
_SCL = np.array([0.30, 0.30, 0.5, 0.5, 0.30, 0.30, 0.5, 0.5, 0.5, 0.30, 0.30])

N_STATE = 11   # ee2, ee_v2, obj2, obj_v2, grip1, slot2
N_ACTION = 3
N_OBS = N_STATE + 3  # + 阶段 one-hot


class ToyChainEnv:
    def __init__(self, seed=0, obs_noise=0.002):
        self.rng = np.random.default_rng(seed)
        self.obs_noise = obs_noise   # 策略/专家可见的 EE 与物块位置传感噪声(米)
        self._sensed_ee = None
        self._sensed_obj = None
        self.reset()

    # ---------------- 底层动力学 ----------------
    def reset(self):
        self.ee = EE_HOME + self.rng.uniform(-0.02, 0.02, size=2)
        self.ee_v = np.zeros(2)
        lo, hi = OBJ_SPAWN
        self.obj = self.rng.uniform(lo, hi)
        self.obj_v = np.zeros(2)
        self.grip = 0.0
        self.grasped = False
        self.stage = 1
        self.stage_steps = 0
        self.total_steps = 0
        self.stage_success = [False, False, False]
        self.s3_entry_ee = None      # 本 episode 首次进入 σ3 时的 EE 位置(restage 中心)
        self.done = False
        self.fail_reason = None
        return self.obs()

    def obs(self):
        # 传感噪声只加在位置观测上;判据 φ_i 始终用真实状态(verifier 是几何真值,
        # 其标签噪声由 ε 翻转单独建模)
        ee_o = self.ee + self.rng.normal(0, self.obs_noise, 2)
        obj_o = self.obj + self.rng.normal(0, self.obs_noise, 2)
        self._sensed_ee, self._sensed_obj = ee_o, obj_o
        s = np.concatenate([ee_o, self.ee_v, obj_o, self.obj_v,
                            [self.grip], SLOT])
        return (s - _OFS) / _SCL

    def obs14(self):
        oh = np.zeros(3)
        oh[self.stage - 1] = 1.0
        return np.concatenate([self.obs(), oh])

    def _clamp_wall(self, pos, new):
        # 墙在 x=WALL_X、y<WALL_H 处不可穿越(双向),只对 x 方向做钳制
        if pos[1] < WALL_H and (pos[0] - WALL_X) * (new[0] - WALL_X) < 0:
            new = new.copy()
            new[0] = pos[0]
        return new

    def step(self, a):
        """a ∈ [-1,1]^3。返回 (obs14, events)。events 键:
        stage_advance: int|None; attempt_end: dict|None (σ3 结束,含 true label);
        episode_end: bool; env_steps: 1。"""
        a = np.clip(np.asarray(a, dtype=np.float64), -1.0, 1.0)
        ev = {"stage_advance": None, "attempt_end": None, "episode_end": False}
        if self.done:
            return self.obs14(), ev

        v = a[:2] * V_MAX
        new = self._clamp_wall(self.ee, self.ee + v * DT)
        new = np.clip(new, WS_LO, WS_HI)
        self.ee_v = (new - self.ee) / DT
        self.ee = new

        prev_grip = self.grip
        self.grip = float(np.clip(self.grip + GRIP_RATE * a[2] * DT, 0.0, 1.0))

        if (not self.grasped) and prev_grip < 0.9 <= self.grip and \
                np.linalg.norm(self.ee - self.obj) <= GRASP_DIST:
            self.grasped = True
        released = False
        if self.grasped and prev_grip > 0.1 >= self.grip:
            self.grasped = False
            released = True
        if self.grasped:
            self.obj = self.ee.copy()
            self.obj_v = self.ee_v.copy()
        else:
            self.obj_v = np.zeros(2)

        self.stage_steps += 1
        self.total_steps += 1

        # ---- 阶段判据 ----
        if self.stage == 1 and self.grasped:
            self.stage_success[0] = True
            self.stage, self.stage_steps = 2, 0
            ev["stage_advance"] = 2
        elif self.stage == 2 and self.grasped and \
                np.linalg.norm(self.obj - P2) <= P2_TOL:
            self.stage_success[1] = True
            self.stage, self.stage_steps = 3, 0
            self.s3_entry_ee = self.ee.copy()
            ev["stage_advance"] = 3
        elif self.stage == 3 and (released or self.stage_steps >= STAGE_MAX_STEPS):
            ok = released and abs(self.obj[0] - SLOT[0]) <= SLOT_LAT_TOL and \
                abs(self.obj[1] - SLOT[1]) <= SLOT_VER_TOL
            self.stage_success[2] = bool(ok)
            ev["attempt_end"] = {"true": bool(ok), "timeout": not released,
                                 "obj": self.obj.copy()}
            ev["episode_end"] = True   # 是否重试由训练循环决定(C 臂)
        elif self.stage_steps >= STAGE_MAX_STEPS and self.stage in (1, 2):
            self.done = True
            self.fail_reason = f"s{self.stage}_timeout"
            ev["episode_end"] = True

        if self.total_steps >= EPISODE_MAX_STEPS and not self.done:
            self.done = True
            self.fail_reason = self.fail_reason or "episode_timeout"
            ev["episode_end"] = True
        return self.obs14(), ev

    def end_episode(self):
        self.done = True

    # ---------------- PARTS 训练用 restage ----------------
    def restage_s3(self, jitter=0.008):
        """把 EE 重摆到本 episode σ3 入口附近(窄分布 μ̂_3),物块重新抓住。
        对应论文 reset policy 的"restage 到子任务起始条件";不计环境步数。"""
        center = self.s3_entry_ee if self.s3_entry_ee is not None else P2
        self.ee = center + self.rng.uniform(-jitter, jitter, size=2)
        self.ee = np.clip(self.ee, WS_LO, WS_HI)
        self.ee_v = np.zeros(2)
        self.obj = self.ee.copy()
        self.obj_v = np.zeros(2)
        self.grip = 1.0
        self.grasped = True
        self.stage = 3
        self.stage_steps = 0
        return self.obs14()

    @property
    def full_success(self):
        return all(self.stage_success)
