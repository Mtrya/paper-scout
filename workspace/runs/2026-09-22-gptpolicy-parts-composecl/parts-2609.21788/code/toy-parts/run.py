"""单次实验运行:方法臂 × 种子 × verifier 噪声 ε。

方法臂:
  A base-only    : 不训练,只评估(扁平曲线)
  B full-task RL : 冻结 base,全段全维残差,只用 episode 末端稀疏奖励 R(人工标注,无噪声)
  C PARTS        : 仅 σ3 激活残差(规则 selector),局部奖励 r3=φ3(带 verifier 噪声 ε),
                   online 更新 + 每 15k 步 success-reweighted 重训(ρ=0.25)
  D PARTS-noret  : C 去掉周期性重训(消融)

用法:
  python run.py --method C --seed 0 --eps 0.0 --budget 60000 --out results/
  python run.py --method C --seed 0 --smoke     # 2000 步冒烟
"""

import argparse
import json
import os
import time

import numpy as np
import torch

import base as B
import env as E
import rl as RL

torch.set_num_threads(1)


# ---------------- rollout ----------------
class ChunkRunner:
    """驱动一个 episode:base 每 REPLAN 步出 nominal chunk;残差按 selector 激活。"""

    def __init__(self, env, base, learner, method, sigma=0.0, rng=None):
        self.env, self.base, self.learner = env, base, learner
        self.method, self.sigma = method, sigma
        self.rng = rng or np.random.default_rng(0)
        self.nominal = None
        self.ptr = B.REPLAN   # 强制第一步重规划
        self.cur_chunks = []  # 当前 attempt 的 chunk 记录

    def residual_active(self):
        if self.method == "B":
            return self.env.stage >= 1
        return self.env.stage == 3          # C / D:仅瓶颈段

    def plan(self):
        obs = self.env.obs14()
        self.nominal = self.base.chunk(obs)
        self.u = None
        if self.residual_active() and self.learner is not None:
            x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            n = torch.as_tensor(self.nominal.reshape(-1), dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                f = self.learner.actor(x, n).squeeze(0).numpy()
            if self.sigma > 0:
                # chunk 级探索:一个 chunk 共享一个偏移 δ(探索"持续性纠正",
                # 即可发现的系统对象),叠加极小逐行抖动保持平滑
                delta = self.rng.normal(0, self.sigma, size=E.N_ACTION)
                jitter = self.rng.normal(0, 0.02, size=(B.CHUNK, E.N_ACTION))
                u = f.reshape(B.CHUNK, E.N_ACTION) + delta + jitter
                u = u.reshape(-1)
            else:
                u = f
            self.u = np.clip(u, -1.0, 1.0)
            if self.sigma > 0:  # 仅训练时记录
                self.cur_chunks.append((obs, self.nominal.reshape(-1).copy(),
                                        self.u.copy(), self.sigma))
        self.ptr = 0

    def act(self):
        if self.ptr >= B.REPLAN:
            self.plan()
        i = self.ptr
        if self.u is None:
            a = self.nominal[i]
        else:
            a = RL.combine_chunk(self.nominal, self.u.reshape(B.CHUNK, E.N_ACTION))[i]
        self.ptr += 1
        return a

    def notify_event(self, ev):
        if ev["stage_advance"] or ev["attempt_end"]:
            self.ptr = B.REPLAN   # 中断当前 chunk,下步重规划


def eval_episodes(base, learner, method, n_eps, seed):
    succ = np.zeros((n_eps, 4))   # full, s1, s2, s3
    for i in range(n_eps):
        env = E.ToyChainEnv(seed=seed * 100003 + i)
        runner = ChunkRunner(env, base, learner, method, sigma=0.0)
        while not env.done:
            _, ev = env.step(runner.act())
            runner.notify_event(ev)
            if ev["episode_end"]:
                env.end_episode()
        succ[i] = [env.full_success, *env.stage_success]
    r = succ.mean(axis=0)
    return {"full": float(r[0]), "s1": float(r[1]), "s2": float(r[2]),
            "s3": float(r[3])}


# ---------------- 训练 ----------------
def train(method, seed, eps, budget, eval_every, eval_n, update_every,
          retrain_every, rho, out_dir):
    t0 = time.time()
    base = B.load_or_train_base(seed=seed)
    learner = RL.ResidualLearner(seed=seed * 1000 + 7)
    rng = np.random.default_rng(seed * 1000 + 11)

    snapshots = [dict(step=0, **eval_episodes(base, learner, method, eval_n, seed + 900))]
    attempts_log = []     # (env_step, label_true, label_used)
    retrain_log = []
    env_steps = 0
    pending = []          # 待更新 attempt
    next_retrain = retrain_every
    next_eval = eval_every

    while env_steps < budget:
        env = E.ToyChainEnv(seed=int(rng.integers(0, 2**31)))
        sigma = 0.05 - 0.02 * (env_steps / budget)   # 0.05 → 0.03 线性衰减
        runner = ChunkRunner(env, base, learner, method, sigma=sigma, rng=rng)
        while not env.done and env_steps < budget:
            _, ev = env.step(runner.act())
            env_steps += 1
            runner.notify_event(ev)

            attempt_done = False
            if method == "B":
                if ev["episode_end"]:
                    label = float(env.full_success)   # 人工标注:无噪声
                    learner.add_attempt({"chunks": runner.cur_chunks,
                                         "label": label, "label_true": label})
                    attempts_log.append((env_steps, label, label))
                    pending.append(learner.buffer[-1])
                    attempt_done = True
                    env.end_episode()
            else:  # C / D:σ3 尝试级
                if ev["attempt_end"] is not None:
                    true = float(ev["attempt_end"]["true"])
                    noisy = true if rng.random() >= eps else 1.0 - true
                    learner.add_attempt({"chunks": runner.cur_chunks,
                                         "label": noisy, "label_true": true})
                    attempts_log.append((env_steps, true, noisy))
                    pending.append(learner.buffer[-1])
                    runner.cur_chunks = []
                    attempt_done = True
                    if noisy > 0.5 or env.total_steps > E.EPISODE_MAX_STEPS - 40:
                        env.end_episode()      # verifier 判成功 → 交接;或步数耗尽
                    else:
                        env.restage_s3()       # 判失败 → 重摆重试(0 环境步)
                elif ev["episode_end"]:
                    env.end_episode()

            if attempt_done and len(pending) >= update_every:
                learner.update(pending)
                pending = []

            if method == "C" and env_steps >= next_retrain:
                info = learner.retrain(rho=rho)
                info["step"] = env_steps
                retrain_log.append(info)
                next_retrain += retrain_every

            if env_steps >= next_eval or env_steps >= budget:
                snapshots.append(dict(
                    step=env_steps,
                    **eval_episodes(base, learner, method, eval_n,
                                    seed + 900 + len(snapshots))))
                next_eval += eval_every

    if pending:
        learner.update(pending)

    result = {
        "method": method, "seed": seed, "eps": eps, "budget": budget,
        "snapshots": snapshots, "attempts": attempts_log,
        "retrains": retrain_log, "wall_time_s": round(time.time() - t0, 1),
        "config": {"eval_n": eval_n, "eval_every": eval_every,
                   "update_every": update_every, "retrain_every": retrain_every,
                   "rho": rho, "lam_pos": learner.lam_pos,
                   "lam_neg": learner.lam_neg, "lr": learner.lr},
    }
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{method}_s{seed}_e{eps}.json")
    with open(path, "w") as f:
        json.dump(result, f)
    print(f"[done] {path}  wall={result['wall_time_s']}s  "
          f"final full={snapshots[-1]['full']:.2f} s3={snapshots[-1]['s3']:.2f}")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True, choices=["A", "B", "C", "D"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eps", type=float, default=0.0)
    p.add_argument("--budget", type=int, default=60000)
    p.add_argument("--eval-every", type=int, default=10000)
    p.add_argument("--eval-n", type=int, default=100)
    p.add_argument("--update-every", type=int, default=16)
    p.add_argument("--retrain-every", type=int, default=15000)
    p.add_argument("--rho", type=float, default=0.25)
    p.add_argument("--out", default="results")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    if args.smoke:
        args.budget, args.eval_n, args.eval_every = 2000, 20, 1000

    base = B.load_or_train_base(seed=args.seed)
    if args.method == "A":
        m = eval_episodes(base, None, "A", args.eval_n, args.seed + 900)
        os.makedirs(args.out, exist_ok=True)
        path = os.path.join(args.out, f"A_s{args.seed}_e{args.eps}.json")
        with open(path, "w") as f:
            json.dump({"method": "A", "seed": args.seed, "eps": args.eps,
                       "snapshots": [dict(step=0, **m)]}, f)
        print(f"[done] {path}  base full={m['full']:.2f} s1={m['s1']:.2f} "
              f"s2={m['s2']:.2f} s3={m['s3']:.2f}")
        return

    train(args.method, args.seed, args.eps, args.budget, args.eval_every,
          args.eval_n, args.update_every, args.retrain_every, args.rho, args.out)


if __name__ == "__main__":
    main()
