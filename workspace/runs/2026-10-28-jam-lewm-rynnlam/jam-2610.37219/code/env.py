"""2D point-mass pushing task, vectorized in torch.

State (10): [hand(2), hand_vel(2), block(2), block_vel(2), target(2)]
Action (2): hand velocity command in [-1, 1] (scaled by vmax), dt = 0.1.

Dynamics: quasi-static pushing. If the hand overlaps the block, the block takes
the normal component of the hand velocity; otherwise it decelerates.
Reward: -shape_coef * ||block - target||  +  success_bonus * [success]; episode
terminates early on success.
"""
import math

import torch


class PushEnv:
    obs_dim = 10
    act_dim = 2

    def __init__(self, n_envs=64, seed=0, dt=0.1, vmax=1.0, r_hand=0.05,
                 r_block=0.09, success_radius=0.15, max_steps=60, arena=1.0,
                 friction=0.5, start_range=1.0, success_bonus=5.0,
                 shape_coef=0.2, min_sep=0.35, min_target_dist=0.5):
        self.n = n_envs
        self.dt = dt
        self.vmax = vmax
        self.r_hand = r_hand
        self.r_block = r_block
        self.contact_r = r_hand + r_block
        self.success_radius = success_radius
        self.max_steps = max_steps
        self.arena = arena
        self.friction = friction
        self.start_range = start_range
        self.success_bonus = success_bonus
        self.shape_coef = shape_coef
        self.min_sep = min_sep
        self.min_target_dist = min_target_dist
        self.gen = torch.Generator().manual_seed(seed)
        self.state = torch.zeros(n_envs, 10)
        self.t = torch.zeros(n_envs, dtype=torch.long)
        self._scale = torch.tensor([1 / arena, 1 / arena, 1 / vmax, 1 / vmax,
                                    1 / arena, 1 / arena, 1 / vmax, 1 / vmax,
                                    1 / arena, 1 / arena])

    def reset(self, gen=None, dist_lo=0.6, dist_hi=1.0, mask=None):
        g = self.gen if gen is None else gen
        self._n_active = self.n
        n = self.n
        st = self.sample_init(g, n, dist_lo, dist_hi)
        if mask is None:
            self.state = st
            self.t = torch.zeros(n, dtype=torch.long)
        else:
            self.state = torch.where(mask.unsqueeze(-1), st, self.state)
            self.t = torch.where(mask, torch.zeros(n, dtype=torch.long), self.t)
        return self.obs()

    def sample_init(self, gen, n, dist_lo=0.6, dist_hi=1.0):
        g = gen
        hand = (torch.rand(n, 2, generator=g) * 2 - 1) * self.start_range
        block = (torch.rand(n, 2, generator=g) * 2 - 1) * self.start_range
        for _ in range(6):
            sep = (block - hand).norm(dim=-1)
            bad = sep < self.min_sep
            if not bad.any():
                break
            fresh = (torch.rand(n, 2, generator=g) * 2 - 1) * self.start_range
            block = torch.where(bad.unsqueeze(-1), fresh, block)
        theta = torch.rand(n, generator=g) * 2 * math.pi
        dist = dist_lo + torch.rand(n, generator=g) * (dist_hi - dist_lo)
        dirv = torch.stack([theta.cos(), theta.sin()], -1)
        lim = self.arena * 0.95
        target = block + dirv * dist.unsqueeze(-1)
        for _ in range(6):
            target_c = target.clamp(-lim, lim)
            too_close = (target_c - block).norm(dim=-1) < self.min_target_dist
            if not too_close.any():
                break
            theta2 = torch.rand(n, generator=g) * 2 * math.pi
            dirv2 = torch.stack([theta2.cos(), theta2.sin()], -1)
            target = torch.where(too_close.unsqueeze(-1),
                                 block + dirv2 * dist.unsqueeze(-1), target)
        target = target.clamp(-lim, lim)
        block = block.clamp(-lim, lim)
        hand = hand.clamp(-lim, lim)
        z = torch.zeros(n, 2)
        return torch.cat([hand, z, block, z, target], dim=-1)

    def obs(self):
        return self.state * self._scale

    def step(self, action):
        a = action.clamp(-1.0, 1.0)
        s = self.state
        hand = s[:, 0:2]
        hand_vel = a * self.vmax
        hand = hand + hand_vel * self.dt
        block = s[:, 4:6]
        dvec = block - hand
        d = dvec.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        nrm = dvec / d
        contact = (d.squeeze(-1) < self.contact_r).unsqueeze(-1)
        overlap = (self.contact_r - d).clamp_min(0.0)
        hand = hand - nrm * overlap
        push = (hand_vel * nrm).sum(-1, keepdim=True) * nrm
        block_vel = torch.where(contact, push, s[:, 6:8] * (1 - self.friction))
        block = block + block_vel * self.dt
        lim = self.arena
        hand = hand.clamp(-lim, lim)
        block = block.clamp(-lim, lim)
        target = s[:, 8:10]
        self.state = torch.cat([hand, hand_vel, block, block_vel, target], dim=-1)
        self.t = self.t + 1
        dist = (block - target).norm(dim=-1)
        success = dist < self.success_radius
        reward = -self.shape_coef * dist + self.success_bonus * success.float()
        done = success | (self.t >= self.max_steps)
        return self.obs(), reward, done, success, dist


def expert_action(state):
    """Scripted pushing controller. state: (N, 10) raw (unnormalized)."""
    hand = state[:, 0:2]
    block = state[:, 4:6]
    target = state[:, 8:10]
    to_t = target - block
    gd = to_t / to_t.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    stand = block - gd * (0.05 + 0.09 + 0.005)
    to_stand = stand - hand
    ds = to_stand.norm(dim=-1, keepdim=True)
    approach = to_stand / ds.clamp_min(1e-6) * torch.clamp(ds * 4.0, max=1.0)
    push = gd
    a = torch.where((ds > 0.05), approach, push)
    return a.clamp(-1.0, 1.0)
