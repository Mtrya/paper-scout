"""Expert demonstrations -> (obs, action-chunk) pairs."""
import torch

from env import PushEnv


def expert_action(state, tang_sign=None):
    """Scripted pushing controller. Smooth (soft-blended phases), so it is
    approximable by a small MLP:

      attraction  a_att  -> head for the stand point behind the block
      circling    a_circ -> tangential motion at ring radius (fixed orientation)
      pushing     a_push -> along block -> target once aligned behind

    Phases are blended with sigmoids, so the field is continuous.

    tang_sign (N,1) flips the circling direction per episode.  Both directions
    are valid strategies, so the demo distribution stays expert-level while
    p(action | obs) becomes bimodal wherever the hand still has to go around the
    block -- the structural property that makes one-step conditional-mean
    predictions unusable (paper's premise).
    """
    hand = state[:, 0:2]
    block = state[:, 4:6]
    target = state[:, 8:10]
    to_t = target - block
    gd = to_t / to_t.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    stand = block - gd * 0.14
    to_b = block - hand
    db = to_b.norm(dim=-1, keepdim=True)
    n = to_b / db.clamp_min(1e-6)
    behind = (-((hand - block) / db.clamp_min(1e-6)) * gd).sum(-1, keepdim=True)
    s = stand - hand
    ds = s.norm(dim=-1, keepdim=True)
    att = s / ds.clamp_min(1e-6) * torch.clamp(ds * 4.0, max=1.0)
    tang = torch.stack([-n[:, 1], n[:, 0]], -1)
    if tang_sign is not None:
        tang = tang * tang_sign
    radial = -n * torch.clamp((0.20 - db) / 0.20, 0.0, 1.0)
    circ = tang + 0.7 * radial
    circ = circ / circ.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    w_near = torch.sigmoid((0.45 - db) / 0.06)
    w_push = torch.sigmoid((behind - 0.85) / 0.06)
    a = (1 - w_push) * ((1 - w_near) * att + w_near * circ) + w_push * gd
    return a.clamp(-1.0, 1.0)


def sample_tang_sign(kind, n, gen):
    """None / 'random': per-episode circling direction for the demo expert."""
    if kind in (None, "", "none", "fixed"):
        return None
    if kind == "random":
        return torch.where(torch.rand(n, 1, generator=gen) < 0.5, -1.0, 1.0)
    raise ValueError(kind)


def _roll_expert(env, gen, n, max_steps=None, noise=0.0, tang_sign=None):
    env.n = n
    obs = env.reset(gen)
    sgn = sample_tang_sign(tang_sign, n, gen)
    active = torch.ones(n, dtype=torch.bool)
    succ_final = torch.zeros(n, dtype=torch.bool)
    ep_obs, ep_act = [], []
    for _ in range(max_steps or env.max_steps):
        if not bool(active.any()):
            break
        a = expert_action(env.state, sgn)
        if noise > 0:
            a = (a + noise * torch.randn(a.shape, generator=gen)).clamp(-1, 1)
        ep_obs.append(obs.clone())
        ep_act.append(a.clone())
        obs, r, done, succ, dist = env.step(a)
        newly = done & active
        succ_final = torch.where(newly, succ, succ_final)
        active = active & ~done
    return succ_final, torch.stack(ep_obs, 0), torch.stack(ep_act, 0)


def expert_success_rate(n_episodes, seed, max_batch=150, env_kwargs=None, noise=0.0,
                        tang_sign=None):
    env = PushEnv(n_envs=min(n_episodes, max_batch), seed=seed, **(env_kwargs or {}))
    gen = torch.Generator().manual_seed(seed * 104729 + 11)
    out = []
    idx = 0
    while idx < n_episodes:
        n = min(max_batch, n_episodes - idx)
        env.n = n
        env.reset(gen)
        sgn = sample_tang_sign(tang_sign, n, gen)
        active = torch.ones(n, dtype=torch.bool)
        succ_final = torch.zeros(n, dtype=torch.bool)
        for _ in range(env.max_steps):
            if not bool(active.any()):
                break
            a = expert_action(env.state, sgn)
            if noise > 0:
                a = (a + noise * torch.randn(a.shape, generator=gen)).clamp(-1, 1)
            _, _, done, s, _ = env.step(a)
            newly = done & active
            succ_final = torch.where(newly, s, succ_final)
            active = active & ~done
        out.append(succ_final)
        idx += n
    return torch.cat(out).float().mean().item()


def make_demos(n_demos, seed, chunk=4, exec_h=2, max_batch=150, env_kwargs=None,
               noise=0.0, tang_sign=None):
    env = PushEnv(n_envs=min(n_demos, max_batch), seed=seed, **(env_kwargs or {}))
    gen = torch.Generator().manual_seed(seed * 104729 + 7)
    obs_all, act_all, succ_all = [], [], []
    idx = 0
    while idx < n_demos:
        n = min(max_batch, n_demos - idx)
        env.n = n
        succ, O, A = _roll_expert(env, gen, n, noise=noise, tang_sign=tang_sign)
        T = O.shape[0]
        for t in range(0, T, exec_h):
            seg = A[t:t + chunk]
            if seg.shape[0] < chunk:
                pad = seg[-1:].repeat(chunk - seg.shape[0], 1, 1)
                seg = torch.cat([seg, pad], 0)
            obs_all.append(O[t])
            act_all.append(seg)
        succ_all.append(succ)
        idx += n
    return (torch.cat(obs_all, 0),
            torch.cat([a.permute(1, 0, 2) for a in act_all], 0),
            torch.cat(succ_all))


if __name__ == "__main__":
    import time
    import sys
    t0 = time.time()
    kw = {}
    if len(sys.argv) > 1:
        kw = dict(max_steps=int(sys.argv[1]))
    for seed in range(3):
        r = expert_success_rate(200, seed, env_kwargs=kw)
        rm = expert_success_rate(200, seed, env_kwargs=kw, tang_sign="random")
        print(f"expert seed={seed} success={r:.3f}  random-sign={rm:.3f}  cfg={kw}")
    print("%.1fs" % (time.time() - t0))
    o, a, s = make_demos(200, 0)
    print("demo pairs", o.shape, a.shape, "demo-episode expert success",
          float(s.float().mean()))
    om, am, sm = make_demos(200, 0, tang_sign="random")
    print("random-sign demo pairs", om.shape, "expert success",
          float(sm.float().mean()), "action std", float(am.std()))
    print("obs range", float(o.min()), float(o.max()))
