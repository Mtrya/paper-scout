"""World-model generalization probe: nse1/nse8 on train vs held-out motion windows.

For each clip, roll the policy in the real simulator (Appendix F protocol) and log
(states, actions, xi, done). Then replay 8-step windows through the learned world
model and compute the same normalized squared error used in the training logs,
so the WM's prediction error can be compared across splits (policy generalization
vs world-model generalization). Policy return and failure are recorded on the
identical rollout, never joined from an independent evaluation trajectory.
"""
from pathlib import Path

import torch
import tyro

from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.eval.benchmark import load_policy_from_checkpoint, reset_at_forced_frame
from supermimic.data.motion_convert import load_g1_model
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.world_model.heads import AnalyticHeads
from supermimic.envs.tracking_env_cfg import TRACKED_BODIES, ANCHOR_BODY, ROOT_BODY
from supermimic.world_model.world_model import WorldModel, WorldModelCfg
from supermimic.world_model.state_spec import StateSpecCfg
from supermimic.envs.state_io import StateIO
from supermimic.envs.sim_context import SimContext


@torch.no_grad()
def main(ckpt: Path, npz_dir: Path = Path("data/npz/lafan1"), clips: list[str] | None = None,
         num_envs: int = 64, window: int = 8, stride: int = 8, device: str = "cuda:0",
         out: Path | None = None) -> None:
    files = [str(f) for f in discover_npz(npz_dir, names=clips)]
    policy, cfg = load_policy_from_checkpoint(ckpt, device)
    args = SuperMimicEnvCfgArgs(
        npz_files=files, num_envs=min(num_envs, len(files)), future_frames=cfg["env"]["future_frames"],
        episode_length_s=1.0e5, action_noise_std=0.0, obs_noise=False, reset_noise=False,
        domain_randomization=False, sampling_mode="start", record_transitions=False,
        seed=0, anchor_obs=cfg["env"].get("anchor_obs", False),
    )
    env = make_env(args, device=device)
    env.cfg.auto_reset = False
    robot = env.scene["robot"]
    fk = G1Kinematics(load_g1_model(), device=device)
    heads = AnalyticHeads(fk, TRACKED_BODIES, ANCHOR_BODY, ROOT_BODY,
                          robot.data.soft_joint_pos_limits[0].clone(), robot.data.default_joint_pos[0].clone())
    wm_cfg = dict(cfg["world_model"])
    wm_cfg["state_spec"] = StateSpecCfg(**wm_cfg["state_spec"])
    wm = WorldModel(WorldModelCfg(**wm_cfg), heads, device)
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    wm.load_state_dict(ck["wm"])
    wm.eval()
    state_io = StateIO(env)
    ctx = SimContext(env, ".*foot.*collision.*")
    cmd = env.command_manager.get_term("motion")
    lib = cmd.library
    n_clips = lib.num_clips
    current = torch.arange(env.num_envs, device=device)
    nxt = current + env.num_envs
    cmd.forced_frame[:] = lib.clip_start[current]
    obs = reset_at_forced_frame(env, torch.arange(env.num_envs, device=device))
    cmd.forced_frame[:] = lib.clip_start[torch.clamp(nxt, max=n_clips - 1)]
    active = torch.ones(env.num_envs, dtype=torch.bool, device=device)
    ret_sum = torch.zeros(env.num_envs, device=device)
    policy_steps = torch.zeros(env.num_envs, dtype=torch.long, device=device)
    policy_results = {}
    S = [[] for _ in range(env.num_envs)]
    A = [[] for _ in range(env.num_envs)]
    X = [[] for _ in range(env.num_envs)]
    D = [[] for _ in range(env.num_envs)]
    results = {}
    while len(results) < n_clips:
        s_now = state_io.read().clone()
        xi_now = ctx.read().clone()
        act = policy(obs["policy"], obs["goal"])
        obs, rew, term, tout, _ = env.step(act)
        ret_sum += rew
        policy_steps += 1
        for e in range(env.num_envs):
            if active[e]:
                S[e].append(s_now[e].cpu())
                A[e].append(act[e].cpu())
                X[e].append(xi_now[e].cpu())
                D[e].append(bool(term[e] | tout[e]))
        done = term | tout | (cmd.frame >= lib.clip_last[current])
        if done.any():
            ids = done.nonzero().squeeze(-1)
            for e in ids.tolist():
                ci = int(current[e])
                if active[e] and ci < n_clips and ci not in results:
                    results[ci] = e
                    policy_results[ci] = {
                        "policy_return_full_clip": float(ret_sum[e]) / (0.1 * max(lib.clips[ci].length - 1, 1)),
                        "policy_failed": bool(term[e]),
                        "policy_steps": int(policy_steps[e]),
                    }
            advance = ids[active[ids]]
            current[advance] = nxt[advance]
            nxt[advance] = nxt[advance] + env.num_envs
            finished = current[advance] >= n_clips
            active[advance[finished]] = False
            current[advance[finished]] = n_clips - 1
            cmd.forced_frame[advance] = lib.clip_start[torch.clamp(nxt[advance], max=n_clips - 1)]
            obs = reset_at_forced_frame(env, ids)
            ret_sum[ids] = 0.0
            policy_steps[ids] = 0
    env.close()

    rows = []
    for ci in range(n_clips):
        e = results[ci]
        s = torch.stack(S[e]).to(device)
        a = torch.stack(A[e]).to(device)
        xi = torch.stack(X[e]).to(device)
        d = torch.tensor(D[e], device=device)
        n = s.shape[0]
        errs1, errs8 = [], []
        sq_err1 = torch.zeros(70, device=device)
        sq_err8 = torch.zeros(70, device=device)
        for start in range(0, n - window - 1, stride):
            pred = s[start]
            for j in range(window):
                pred = wm.step(pred, a[start + j], xi[start + j])
                true = s[start + j + 1]
                err = wm.spec.normalized_sq_error(pred, true)
                if j == 0:
                    errs1.append(float(err))
                    sq_err1 += wm.spec.error_vector(pred, true).square()
                if j == window - 1:
                    errs8.append(float(err))
                    sq_err8 += wm.spec.error_vector(pred, true).square()
        count = max(len(errs8), 1)
        groups = {"root_pos_m": slice(0, 3), "root_ori_rad": slice(3, 6),
                  "root_lin_vel_mps": slice(6, 9), "root_ang_vel_radps": slice(9, 12),
                  "joint_pos_rad": slice(12, 41), "joint_vel_radps": slice(41, 70)}
        def group_rms(normalized_sq):
            raw_sq = normalized_sq * wm.spec.err_scale.square()
            return {name: float((raw_sq[sl].sum() / count).sqrt()) for name, sl in groups.items()}
        rows.append({"clip": lib.clips[ci].name, "steps": n, "windows": len(errs8),
                     "nse1": sum(errs1) / max(len(errs1), 1),
                     "nse8": sum(errs8) / max(len(errs8), 1),
                     "rms1": group_rms(sq_err1), "rms8": group_rms(sq_err8),
                     **policy_results[ci]})
        print(f"{lib.clips[ci].name:32s} nse1 {rows[-1]['nse1']:.5f} nse8 {rows[-1]['nse8']:.5f}")
    import json
    out = out or ckpt.parent / f"wm_nse_{ckpt.stem}.json"
    out.write_text(json.dumps({"ckpt": str(ckpt), "window": window, "rows": rows}, indent=1))
    print("saved", out)


if __name__ == "__main__":
    tyro.cli(main)
