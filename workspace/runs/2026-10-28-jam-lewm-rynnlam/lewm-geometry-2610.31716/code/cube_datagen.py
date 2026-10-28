#!/usr/bin/env python3
"""Generate cube-single-v0 anchor episodes with a scripted pick-and-place expert.

Records qpos/qvel/actions/224x224 frames/privileged info in the same layout the
Gram probe consumes (no h5 required; saved as a torch list of dicts).
Action convention is probed and printed at startup.
"""
import argparse
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=120)
    ap.add_argument("--out", default="cube_anchors.pt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    import gymnasium as gym
    import ogbench  # noqa: F401

    env = gym.make("cube-single-v0", render_mode="rgb_array", width=224, height=224)
    rng = np.random.default_rng(args.seed)
    eps = []
    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed * 1000 + ep)
        if ep == 0:
            print("action_space", env.action_space)
            print("obs keys", list(obs.keys()) if isinstance(obs, dict) else type(obs))
            print("info keys", list(info.keys()))
        u = env.unwrapped
        ee0 = info["proprio/effector_pos"].copy()
        b0 = info["privileged/block_0_pos"].copy()
        target = b0 + np.array([rng.uniform(-0.25, 0.25), rng.uniform(-0.25, 0.25), 0.0])
        if np.linalg.norm(target[:2] - b0[:2]) < 0.15:
            target[:2] += 0.2
        yaw_target = rng.uniform(-np.pi, np.pi)
        ou = np.zeros(5)
        S, A, I = [], [], []
        phase = "approach"
        grip = -1.0
        for t in range(160):
            info = u.get_step_info()
            ee = info["proprio/effector_pos"]
            b = info["privileged/block_0_pos"]
            S.append({"qpos": u.data.qpos.copy(), "qvel": u.data.qvel.copy(),
                      "pixels": env.render(), "info": {k: v.copy() if hasattr(v, "copy") else v
                                                       for k, v in info.items()}})
            d_xy = b[:2] - ee[:2]
            d_z = b[2] - ee[2]
            if phase == "approach":
                act = np.zeros(5)
                act[:2] = np.clip(4 * d_xy, -1, 1)
                act[2] = np.clip(2 * (b[2] + 0.12 - ee[2]), -1, 1)
                if np.linalg.norm(d_xy) < 0.015:
                    phase = "descend"
            elif phase == "descend":
                act = np.zeros(5)
                act[:2] = np.clip(4 * d_xy, -1, 1)
                act[2] = np.clip(3 * d_z, -1, 1)
                if abs(d_z) < 0.012:
                    phase = "grasp"
                    t_grasp = t
            elif phase == "grasp":
                act = np.zeros(5)
                grip = 1.0
                if t - t_grasp > 6:
                    phase = "lift"
            elif phase == "lift":
                act = np.zeros(5)
                act[2] = 0.6
                if ee[2] > b[2] + 0.12:
                    phase = "transport"
            elif phase == "transport":
                act = np.zeros(5)
                act[:2] = np.clip(3 * (target[:2] - b[:2]), -1, 1)
                act[2] = np.clip(2 * (b[2] + 0.10 - ee[2]), -1, 1)
                if np.linalg.norm(target[:2] - b[:2]) < 0.02:
                    phase = "place"
            elif phase == "place":
                act = np.zeros(5)
                act[2] = np.clip(2 * (target[2] - b[2]), -1, 1)
                if abs(target[2] - b[2]) < 0.02:
                    phase = "release"
                    t_rel = t
            else:
                act = np.zeros(5)
                grip = -1.0
                if t - t_rel > 5:
                    A.append(np.r_[np.zeros(4), grip])
                    break
            act[4] = grip
            yaw_cur = float(info["proprio/effector_yaw"][0])
            dyaw = (yaw_target - yaw_cur + np.pi) % (2 * np.pi) - np.pi
            act[3] = np.clip(0.8 * dyaw, -1, 1)
            ou = 0.9 * ou + 0.15 * rng.normal(size=5)
            act[:4] = np.clip(act[:4] + ou[:4], -1, 1)
            A.append(act)
            env.step(act)
        if len(A) >= 30:
            eps.append({"states": S, "actions": np.array(A), "target": target,
                        "success": bool(phase == "release")})
        if (ep + 1) % 10 == 0:
            succ = np.mean([e["success"] for e in eps]) if eps else 0
            print(f"ep {ep+1}/{args.episodes} kept {len(eps)} succ {succ:.2f}", flush=True)
    torch.save(eps, args.out)
    print("saved", len(eps), args.out)


if __name__ == "__main__":
    main()
