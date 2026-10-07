"""Vectorized episode runner (used for evaluation, probing and steering curves)."""
import torch

from env import PushEnv


def run_episodes(policy, n_episodes, seed, K=10, exec_h=2, sigma=0.0,
                 record_hidden=False, dist_lo=0.6, dist_hi=1.0, max_batch=150,
                 env_kwargs=None):
    env_kwargs = dict(env_kwargs or {})
    nb = min(n_episodes, max_batch)
    env = PushEnv(n_envs=nb, seed=seed, **env_kwargs)
    L, d = policy.n_blocks, policy.d
    gen_env = torch.Generator().manual_seed(seed * 7919 + 13)
    gen_pol = torch.Generator().manual_seed(seed * 7919 + 29)
    succ_out = torch.zeros(n_episodes, dtype=torch.bool)
    ret_out = torch.zeros(n_episodes)
    steps_out = torch.zeros(n_episodes)
    hid_out = torch.zeros(n_episodes, K, L, d) if record_hidden else None
    idx = 0
    while idx < n_episodes:
        n = min(nb, n_episodes - idx)
        env.n = n
        obs = env.reset(gen_env, dist_lo, dist_hi)
        active = torch.ones(n, dtype=torch.bool)
        ret = torch.zeros(n)
        acc = torch.zeros(n, K, L, d) if record_hidden else None
        cnt = torch.zeros(n)
        succ_final = torch.zeros(n, dtype=torch.bool)
        nsteps = torch.zeros(n)
        while bool(active.any()):
            chunk, _, _, hid = policy.sample(obs, K=K, sigma=sigma, gen=gen_pol,
                                             hidden=record_hidden)
            if record_hidden:
                acc += hid * active.view(-1, 1, 1, 1)
                cnt += active.float()
            for hh in range(exec_h):
                obs, r, done, succ, dist = env.step(chunk[:, hh])
                ret += r * active.float()
                nsteps += active.float()
                newly = done & active
                succ_final = torch.where(newly, succ, succ_final)
                active = active & ~done
        succ_out[idx:idx + n] = succ_final
        ret_out[idx:idx + n] = ret
        steps_out[idx:idx + n] = nsteps
        if record_hidden:
            hid_out[idx:idx + n] = acc / cnt.clamp_min(1).view(-1, 1, 1, 1)
        idx += n
    out = {"success": succ_out, "ret": ret_out, "steps": steps_out}
    if record_hidden:
        out["hidden"] = hid_out
    return out
