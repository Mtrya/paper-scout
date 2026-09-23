"""脚本专家与演示生成。

每个阶段一个 P 控制专家;σ3 专家带系统性偏差(槽位标定误差,模拟真机的
mm 级对准误差)与动作噪声,且演示条数稀少(8 条)→ BC base 在 σ3 上不可靠。

演示按阶段段生成(σ1 段、σ2 段、σ3 段),输入为 obs14(含阶段 one-hot),
标签为 chunk:从 t 开始的 C=10 步动作序列(不足 C 步用末动作填充)。
"""

import numpy as np

import env as E

CHUNK = 10


def _p(pos, target, gain=3.0):
    return np.clip(gain * (np.asarray(target) - pos) / E.V_MAX, -1.0, 1.0)


def expert_action(stage, env, rng, noise=0.0, slot_bias=0.0):
    """返回 a ∈ [-1,1]^3。专家只见带噪观测(模拟遥操作者/视觉伺服噪声)。
    调用前须先调 env.obs14()(生成 sensed 状态)。"""
    ee = env._sensed_ee if env._sensed_ee is not None else env.ee
    obj = env._sensed_obj if env._sensed_obj is not None else env.obj
    if stage == 1:
        d = np.linalg.norm(obj - ee)
        a = np.concatenate([_p(ee, obj), [1.0 if d < 0.02 else -1.0]])
    elif stage == 2:
        # 绕行:先升到墙上方,再向右,越过墙顶后降到 staging。
        # 分支顺序即状态机:已过墙(x≥0.68)优先判定下降,避免在墙顶高度附近振荡。
        if ee[0] >= 0.68:
            tgt = E.P2
        elif ee[1] < 0.44:
            tgt = np.array([ee[0], 0.45])
        else:
            tgt = np.array([0.70, 0.45])
        a = np.concatenate([_p(ee, tgt), [1.0]])
    else:
        tgt = E.SLOT + np.array([slot_bias, 0.002])
        d = np.linalg.norm(obj - tgt)
        gain = 3.0 if d > 0.02 else 1.2     # 接近目标时减速,争取 mm 级定位
        a = np.concatenate([_p(ee, tgt, gain=gain), [-1.0 if d < 0.004 else 1.0]])
    if noise > 0.0:
        # 乘性噪声:抖动随动作幅度缩放,精调阶段(速度≈0)不被噪声淹没
        a[:2] = np.clip(a[:2] * (1.0 + rng.normal(0.0, noise, size=2)), -1.0, 1.0)
    return a


def gen_stage_demos(stage, n_demos, seed, noise=0.15, s3_bias=(0.005, 0.0025),
                    s3_noise=0.20):
    """生成单阶段演示段。返回 list[traj],traj = (obs14[T,14], act[T,3])。"""
    rng = np.random.default_rng(seed)
    trajs = []
    attempts = 0
    while len(trajs) < n_demos and attempts < n_demos * 25:
        attempts += 1
        env = E.ToyChainEnv(seed=int(rng.integers(0, 2**31)))
        bias = float(rng.normal(*s3_bias)) if stage == 3 else 0.0
        n = noise if stage != 3 else s3_noise
        if stage == 2:
            # σ2 段起点 = σ1 完成后的状态分布:EE 在物块上方已抓住
            env.ee = env.obj.copy(); env.grip = 1.0; env.grasped = True
            env.stage, env.stage_steps = 2, 0
        elif stage == 3:
            # σ3 段起点 = 全链进入 σ3 的真实入口分布:σ2 从上方逼近 P2,
            # φ2 在容差圆上沿处触发 → 入口集中在 P2 上方 60°–120° 弧线上。
            th = rng.uniform(np.pi / 3, 2 * np.pi / 3)
            r = rng.uniform(0.042, 0.050)
            env.ee = E.P2 + np.array([r * np.cos(th), r * np.sin(th)]) \
                + rng.uniform(-0.005, 0.005, size=2)
            env.obj = env.ee.copy(); env.grip = 1.0; env.grasped = True
            env.stage, env.stage_steps = 3, 0
        obs, acts = [], []
        completed = False
        for _t in range(E.STAGE_MAX_STEPS):
            o = env.obs14()
            a = expert_action(stage, env, rng, noise=n, slot_bias=bias)
            obs.append(o); acts.append(a)
            _, ev = env.step(a)
            if ev["stage_advance"]:
                completed = True
                break
            if ev["attempt_end"] is not None:
                completed = not ev["attempt_end"]["timeout"]  # 演示必须真的释放
                break
            if ev["episode_end"]:
                break
        # 只保留完成的演示(演示者视角:演示必然完成;σ3 的真值成败由偏差决定)
        if completed:
            trajs.append((np.array(obs), np.array(acts)))
    if len(trajs) < n_demos:
        raise RuntimeError(f"stage {stage}: only {len(trajs)}/{n_demos} completed demos")
    return trajs


def trajs_to_chunk_samples(trajs):
    """(obs14, act) 轨迹 → (x[14], chunk[C*3]) 监督样本,stride=1。"""
    xs, ys = [], []
    for obs, acts in trajs:
        T = len(acts)
        for t in range(T):
            chunk = acts[t:t + CHUNK]
            if len(chunk) < CHUNK:
                chunk = np.concatenate(
                    [chunk, np.repeat(chunk[-1:], CHUNK - len(chunk), axis=0)])
            xs.append(obs[t]); ys.append(chunk.reshape(-1))
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def gen_bc_dataset(seed=0, n12=200, n3=8):
    """σ1/σ2 各 n12 条(低噪声),σ3 仅 n3 条(高噪声+标定偏差)。
    返回 X, Y, W(逐样本权重:按阶段平衡,消除段长差异造成的损失淹没)。"""
    s1 = gen_stage_demos(1, n12, seed + 101)
    s2 = gen_stage_demos(2, n12, seed + 202)
    s3 = gen_stage_demos(3, n3, seed + 303)
    x1, y1 = trajs_to_chunk_samples(s1)
    x2, y2 = trajs_to_chunk_samples(s2)
    x3, y3 = trajs_to_chunk_samples(s3)
    X = np.concatenate([x1, x2, x3]); Y = np.concatenate([y1, y2, y3])
    n = np.array([len(x1), len(x2), len(x3)], dtype=np.float32)
    w = (n.sum() / 3.0) / n                     # 每阶段等权
    W = np.concatenate([np.full(len(x1), w[0]), np.full(len(x2), w[1]),
                        np.full(len(x3), w[2])]).astype(np.float32)
    return X, Y, W, {"n_s1": len(x1), "n_s2": len(x2), "n_s3": len(x3)}
