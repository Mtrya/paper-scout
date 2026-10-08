"""Teacher-Anchored Consistency Distillation (TACD) at toy scale.

Paper: RealtimeWAM (arXiv 2610.06617), Sec. 3.2 + 4.1.

Coordinates.  The paper uses t = 0 data, t = 1 noise.  This bench keeps the toy
convention of the previous round: tau = 0 noise, tau = 1 data, so
tau = 1 - t and every "adjacent step" flips sign.

    x_tau = (1 - tau) eps + tau a          (identical to the paper's a_t)

The policy is x-parameterised (see lowrank-flow README decision 1), so the
endpoint map f_theta(x, tau) = x + (1 - tau) * v_theta(x, tau) is exactly the
network's clean-action head; it is the toy image of the paper's
f_theta(a_t, t) = a_t - t * v_theta(a_t, t).

Losses (paper Eq. 5 and Eq. 8), written in tau:

  neighbour step   x' = x + (1/K) * v_T(x, tau),            s -> t flips to tau -> tau + 1/K
  L_CD = E || f_S(x, tau) - sg[ f_ema(x', tau + 1/K) ] ||^2
  L_TA = E || v_S(x, tau) - sg[ u_T(x, tau) ] ||^2
       = E || f_S(x, tau) - sg[ a_0^T ] ||^2 / (1 - tau)^2      (paper App. C.2)
  L    = L_CD + lambda * L_TA,   lambda = 0.2 (paper Sec. 5.1)

with a_0^T = Solver(x, tau, 1; theta_T) = K uniform Euler steps of the frozen
teacher from tau to 1.

Arms:  bare (no training, teacher 1-step), cd, ta, cdta (lambda 0.2), cdta1
(lambda 1.0).
"""
import argparse
import copy
import json
import os
import time

import torch

from data import make_demos
from model import FlowPolicy
from rollout import run_episodes

K_STEP = 10                      # denoising steps (teacher rollout + deployment)
DELTA = 1.0 / K_STEP             # spacing of the CD neighbour
TAU_MAX = 1.0 - DELTA            # CD needs tau + DELTA <= 1
TAU_FLOOR = 0.05                 # guard for the 1/(1-tau)^2 weight
EMA_DECAY = 0.995                # paper Sec. 5.1
DIAG_SEED = 424242              # fixed held-out diagnostic set, shared by all arms
EVAL_SEED = 4242                 # fixed eval environments, shared by all arms


# ---------------------------------------------------------------- primitives
def endpoint(policy, x, obs, tau):
    """f_theta(x, tau): clean-action estimate, in either parameterisation."""
    v = policy.velocity(x, obs, tau)
    return x + (1.0 - tau).view(-1, 1, 1) * v


@torch.no_grad()
def teacher_rollout(teacher, x0, obs, tau0, K=K_STEP, keep_path=False):
    """K uniform Euler steps of the frozen teacher from tau0 to 1.

    tau0 may be a scalar or a (B,) tensor.  Returns the endpoint a_0^T, or
    (endpoint, [x at tau0, ..., x at 1]) when keep_path is set.
    """
    B = x0.shape[0]
    if not torch.is_tensor(tau0):
        tau0 = torch.full((B,), float(tau0))
    h = (1.0 - tau0) / K
    x = x0
    path = [x0.clone()] if keep_path else None
    for k in range(K):
        v = teacher.velocity(x, obs, tau0 + h * k)
        x = x + h.view(-1, 1, 1) * v
        if keep_path:
            path.append(x.clone())
    return (x, path) if keep_path else x


def make_diag_set(teacher, n=512, demo_noise=0.15, n_demos=400, max_steps=60,
                  tang_sign="fixed"):
    """Held-out (obs, clean action, eps) triples, identical across arms/seeds."""
    obs, act, _ = make_demos(n_demos, DIAG_SEED, env_kwargs={"max_steps": max_steps},
                             noise=demo_noise, tang_sign=tang_sign)
    g = torch.Generator().manual_seed(DIAG_SEED + 1)
    idx = torch.randperm(obs.shape[0], generator=g)[:n]
    obs_d, act_d = obs[idx], act[idx]
    eps_d = torch.randn(act_d.shape, generator=torch.Generator().manual_seed(DIAG_SEED + 2))
    return obs_d, act_d, eps_d


@torch.no_grad()
def diagnostics(policy, teacher, obs_d, act_d, eps_d):
    """Arm-independent, cheap, deterministic diagnostics on the fixed set."""
    B = obs_d.shape[0]
    z = torch.zeros(B)
    out = {}

    # --- global endpoint error at tau = 0: one-step student vs 10-step teacher
    aT = teacher_rollout(teacher, eps_d, obs_d, z, K=K_STEP)
    fS = endpoint(policy, eps_d, obs_d, z)
    out["global_err"] = float(((fS - aT) ** 2).mean())
    out["teacher_1step_err"] = float(((endpoint(teacher, eps_d, obs_d, z) - aT) ** 2).mean())
    out["global_norm_aT"] = float((aT ** 2).mean())

    # --- local consistency: student's endpoint prediction along the teacher's
    #     own denoising path from pure noise (frozen trajectory, tau = k/K)
    _, path = teacher_rollout(teacher, eps_d, obs_d, z, K=K_STEP, keep_path=True)
    loc_traj = []
    for k in range(K_STEP):
        tau_k = torch.full((B,), k / K_STEP)
        tau_n = torch.full((B,), (k + 1) / K_STEP)
        f_0 = endpoint(policy, path[k], obs_d, tau_k)
        f_1 = endpoint(policy, path[k + 1], obs_d, tau_n)
        loc_traj.append(float(((f_0 - f_1) ** 2).mean()))
    out["local_err_traj"] = sum(loc_traj) / len(loc_traj)
    out["local_traj_by_k"] = loc_traj

    # --- local consistency on the training distribution: x_tau from data,
    #     neighbour produced by one teacher Euler step (paper Eq. 4/5 sampling)
    loc_data = []
    for k in range(K_STEP):
        tau_k = torch.full((B,), k / K_STEP)
        x = (1 - tau_k).view(-1, 1, 1) * eps_d + tau_k.view(-1, 1, 1) * act_d
        x_nb = x + DELTA * teacher.velocity(x, obs_d, tau_k)
        f_0 = endpoint(policy, x, obs_d, tau_k)
        f_1 = endpoint(policy, x_nb, obs_d, tau_k + DELTA)
        loc_data.append(float(((f_0 - f_1) ** 2).mean()))
    out["local_err_data"] = sum(loc_data) / len(loc_data)
    out["local_data_by_k"] = loc_data

    # --- global target error averaged over the same tau grid the distill
    #     losses sample (paper's e_global, student in place of the EMA target)
    glob_data = []
    for k in range(K_STEP):
        tau_k = torch.full((B,), k / K_STEP)
        x = (1 - tau_k).view(-1, 1, 1) * eps_d + tau_k.view(-1, 1, 1) * act_d
        a0T = teacher_rollout(teacher, x, obs_d, tau_k, K=K_STEP)
        glob_data.append(float(((endpoint(policy, x, obs_d, tau_k) - a0T) ** 2).mean()))
    out["global_err_avg"] = sum(glob_data) / len(glob_data)
    out["global_data_by_k"] = glob_data
    return out


# ------------------------------------------------------------------- training
def train_arm(teacher, student, arm, obs, act, steps, batch, lr, lam, seed,
              eval_every, obs_d, act_d, eps_d, eval_episodes, log):
    use_cd = arm in ("cd", "cdta", "cdta1")
    use_ta = arm in ("ta", "cdta", "cdta1")
    ema = copy.deepcopy(student) if use_cd else None
    if ema is not None:
        for p in ema.parameters():
            p.requires_grad_(False)

    opt = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps,
                                                       eta_min=lr * 0.02)
    gen = torch.Generator().manual_seed(seed * 7717 + 3)
    M = obs.shape[0]
    hist = []
    t0 = time.time()
    acc = {}          # running means of the training losses since the last snapshot
    n_acc = 0

    def snapshot(step):
        rec = {"step": step, "t": time.time() - t0, "lr": sched.get_last_lr()[0]}
        rec.update(diagnostics(student, teacher, obs_d, act_d, eps_d))
        for k, v in acc.items():
            rec[k + "_train"] = v / max(1, n_acc)
        if not acc and hist:            # final snapshot without new updates
            for k, v in hist[-1].items():
                if k.endswith("_train"):
                    rec[k] = v
        if eval_episodes:
            rec["success_1step"] = float(run_episodes(
                student, eval_episodes, EVAL_SEED, K=1,
                env_kwargs={"max_steps": 60})["success"].float().mean())
        hist.append(rec)
        log(f"  step {step:5d} local {rec['local_err_data']:.5f} "
            f"global {rec['global_err']:.5f} "
            f"succ {rec.get('success_1step', float('nan')):.3f} "
            f"({rec['t']:.0f}s)")

    snapshot(0)
    for step in range(steps):
        idx = torch.randint(0, M, (batch,), generator=gen)
        ob, ac = obs[idx], act[idx]
        tau = torch.rand(batch, generator=gen) * TAU_MAX
        eps = torch.randn(ac.shape, generator=gen)
        x = (1 - tau).view(-1, 1, 1) * eps + tau.view(-1, 1, 1) * ac
        a_hat = endpoint(student, x, ob, tau)

        loss = torch.zeros(())
        parts = {}
        if use_cd:
            with torch.no_grad():
                x_nb = x + DELTA * teacher.velocity(x, ob, tau)
                tgt = endpoint(ema, x_nb, ob, tau + DELTA)
            parts["loss_cd"] = ((a_hat - tgt) ** 2).mean()
            loss = loss + parts["loss_cd"]
        if use_ta:
            with torch.no_grad():
                a0T = teacher_rollout(teacher, x, ob, tau, K=K_STEP)
            resid = (a_hat - a0T) / (1.0 - tau).clamp_min(TAU_FLOOR).view(-1, 1, 1)
            parts["loss_ta"] = (resid ** 2).mean()
            loss = loss + lam * parts["loss_ta"]

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        sched.step()
        for k, v in parts.items():
            acc[k] = acc.get(k, 0.0) + float(v)
        n_acc += 1
        if ema is not None:
            with torch.no_grad():
                for pe, ps in zip(ema.parameters(), student.parameters()):
                    pe.mul_(EMA_DECAY).add_(ps, alpha=1 - EMA_DECAY)
                for be, bs in zip(ema.buffers(), student.buffers()):
                    be.copy_(bs)

        if (step + 1) % max(1, steps // 40) == 0:
            log(f"  step {step+1}/{steps} " +
                " ".join(f"{k} {float(v):.5f}" for k, v in parts.items()) +
                f" ({time.time()-t0:.0f}s)")

        if eval_every and (step + 1) % eval_every == 0:
            snapshot(step + 1)
            acc, n_acc = {}, 0

    snapshot(steps)
    return hist


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=["bare", "cd", "ta", "cdta", "cdta1"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--teacher", type=str, default="out/teacher0/M_bc.pt")
    ap.add_argument("--out", type=str, default="out")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lam", type=float, default=0.2)
    ap.add_argument("--n-demos", type=int, default=2500)
    ap.add_argument("--demo-noise", type=float, default=0.15)
    ap.add_argument("--tang-sign", type=str, default="fixed",
                    help="'fixed' (single circling direction) or 'random'")
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-episodes", type=int, default=200)
    ap.add_argument("--eval-teacher-steps", type=int, default=200)
    ap.add_argument("--diag-batch", type=int, default=512)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)

    env_kwargs = {"max_steps": args.max_steps}
    teacher = FlowPolicy(pred_mode="x")
    teacher.load_state_dict(torch.load(args.teacher, map_location="cpu")["model"])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    obs_d, act_d, eps_d = make_diag_set(teacher, n=args.diag_batch,
                                        demo_noise=args.demo_noise,
                                        max_steps=args.max_steps,
                                        tang_sign=args.tang_sign)

    logf = open(os.path.join(args.out, f"{args.arm}{args.tag}.log"), "w")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    log(f"=== arm {args.arm} seed {args.seed} steps {args.steps} lr {args.lr} "
        f"lam {args.lam} threads {args.threads} {time.strftime('%H:%M:%S')}")

    # teacher references (identical for every arm of this seed)
    res = {"arm": args.arm, "seed": args.seed, "steps": args.steps, "lr": args.lr,
           "lam": args.lam, "batch": args.batch,
           "teacher": args.teacher, "n_demos": args.n_demos,
           "demo_noise": args.demo_noise, "tang_sign": args.tang_sign}
    t0 = time.time()
    res["teacher_success_10step"] = float(run_episodes(
        teacher, args.eval_teacher_steps, EVAL_SEED, K=K_STEP,
        env_kwargs=env_kwargs)["success"].float().mean())
    res["teacher_success_1step"] = float(run_episodes(
        teacher, args.eval_teacher_steps, EVAL_SEED, K=1,
        env_kwargs=env_kwargs)["success"].float().mean())
    log(f"teacher 10-step {res['teacher_success_10step']:.3f} "
        f"1-step {res['teacher_success_1step']:.3f} ({time.time()-t0:.0f}s)")

    if args.arm == "bare":
        res["success_1step"] = res["teacher_success_1step"]
        res["success_10step"] = res["teacher_success_10step"]
        res["diag"] = diagnostics(teacher, teacher, obs_d, act_d, eps_d)
        res["history"] = []
    else:
        obs, act, _ = make_demos(args.n_demos, args.seed, env_kwargs=env_kwargs,
                                 noise=args.demo_noise, tang_sign=args.tang_sign)
        student = copy.deepcopy(teacher)
        for p in student.parameters():          # teacher was frozen in place
            p.requires_grad_(True)
        res["history"] = train_arm(teacher, student, args.arm, obs, act,
                                   args.steps, args.batch, args.lr, args.lam,
                                   args.seed, args.eval_every, obs_d, act_d,
                                   eps_d, args.eval_episodes, log)
        res["diag"] = {k: v for k, v in res["history"][-1].items()
                       if not k.startswith("success")}
        res["success_1step"] = float(run_episodes(
            student, args.eval_episodes, EVAL_SEED, K=1,
            env_kwargs=env_kwargs)["success"].float().mean())
        res["success_2step"] = float(run_episodes(
            student, args.eval_episodes, EVAL_SEED, K=2,
            env_kwargs=env_kwargs)["success"].float().mean())
        res["success_5step"] = float(run_episodes(
            student, args.eval_episodes, EVAL_SEED, K=5,
            env_kwargs=env_kwargs)["success"].float().mean())
        res["success_10step"] = float(run_episodes(
            student, args.eval_episodes, EVAL_SEED, K=K_STEP,
            env_kwargs=env_kwargs)["success"].float().mean())
        torch.save({"model": student.state_dict()},
                   os.path.join(args.out, f"{args.arm}{args.tag}.pt"))
    res["wall_s"] = time.time() - t0
    with open(os.path.join(args.out, f"{args.arm}{args.tag}.json"), "w") as f:
        json.dump(res, f, indent=1)
    log(f"=== done arm {args.arm} seed {args.seed}: success1 {res['success_1step']:.3f} "
        f"global {res['diag']['global_err']:.5f} "
        f"local {res['diag']['local_err_data']:.5f} ({res['wall_s']:.0f}s)")


if __name__ == "__main__":
    main()
