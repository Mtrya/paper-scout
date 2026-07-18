"""Synthetic visuomotor in-context task family (RoboTTT mechanism probe).

Each episode is a rollout of a 2D point-mass system driven by an *expert*
controller whose parameters are latent and vary per episode:

    target  g  ~ U(-0.8, 0.8)^2          (fixed within an episode)
    gain    k  ~ logU(0.5, 2.5)          (fixed within an episode)
    expert  a_t = clip( k*(g - x_t)*(1 + 0.5*||g - x_t||), -1, 1 )  (nonlinear P-control)
    dynamics x_{t+1} = x_t + dt*a_t,  dt = 0.15
    observation o_t = x_t + N(0, sigma^2 I),  sigma = 0.15  (heavy)

dt = 0.15 is deliberately large: the per-step contraction factor
(1 - dt*k*(1+0.5*||err||)) varies strongly with the latent gain k, so even
short-horizon extrapolation of the action stream requires identifying k
from context. (With dt = 0.05, a_t ~ 0.925*a_{t-1} regardless of the task,
and a no-context model reaches R^2 ~ 0.95 -- a trivial local-extrapolation
shortcut that drowns the in-context-identification signal.)

Multi-phase structure ("repeated reach trials", the toy analogue of the
paper's multi-stage assembly with state aliasing): every P steps the point
is teleported to a fresh random position while (g, k) stay fixed. Actions
therefore stay alive through the whole episode, and each phase is a new
attempt at the SAME latent task: the (o, a) history of earlier phases
sharpens the posterior over (g, k) for later phases.

The policy sees tokens [o_t, a_{t-1}] and must predict a_t at every step.
More context => tighter task identification + better denoising of x_t =>
better action prediction. That is the mechanism-level claim we probe.
"""
import math
import torch

PHASE = 48  # steps per reach trial


def sample_episode_batch(batch, T, device="cpu", gen=None,
                         obs_noise=0.15, dt=0.15, phase=PHASE):
    g = torch.rand(batch, 2, device=device, generator=gen) * 1.6 - 0.8
    lo, hi = math.log(0.5), math.log(2.5)
    k = (torch.rand(batch, 1, device=device, generator=gen) * (hi - lo) + lo).exp()
    x = torch.rand(batch, 2, device=device, generator=gen) * 1.6 - 0.8

    obs, act = [], []
    for t in range(T):
        if t > 0 and t % phase == 0:  # teleport: new attempt, same latent task
            x = torch.rand(batch, 2, device=device, generator=gen) * 1.6 - 0.8
        err = g - x
        a = k * err * (1.0 + 0.5 * err.norm(dim=-1, keepdim=True))
        a = a.clamp(-1.0, 1.0)
        o = x + obs_noise * torch.randn(batch, 2, device=device, generator=gen)
        obs.append(o)
        act.append(a)
        x = x + dt * a

    obs = torch.stack(obs, dim=1)            # (B, T, 2)
    act = torch.stack(act, dim=1)            # (B, T, 2)
    a_prev = torch.cat([torch.zeros(batch, 1, 2, device=device), act[:, :-1]], dim=1)
    tokens = torch.cat([obs, a_prev], dim=-1)  # (B, T, 4): [o_t, a_{t-1}]
    return tokens, act
