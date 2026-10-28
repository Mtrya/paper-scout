"""Visual push-block: same dynamics as lowrank-flow PushEnv, plus 48x48 RGB rendering.

Rendering is vectorized torch (GPU-friendly): hand = white dot, block = blue disc,
goal = green ring, on dark background. State layout matches env.py:
[hand(2), hand_vel(2), block(2), block_vel(2), target(2)].
"""
import torch
from env import PushEnv


class VisualPushEnv(PushEnv):
    img_size = 48

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        s = self.img_size
        ys, xs = torch.meshgrid(torch.linspace(-1, 1, s), torch.linspace(-1, 1, s), indexing="ij")
        self.registered_grid = torch.stack([xs, ys], dim=-1)  # [S,S,2] in arena coords

    def render(self):
        """Return [n,3,S,S] float images in [0,1]."""
        g = self.registered_grid.to(self.state.device)
        n = self.state.shape[0]
        hand = self.state[:, 0:2] / self.arena
        block = self.state[:, 4:6] / self.arena
        goal = self.state[:, 8:10] / self.arena

        def dist2(center):
            # center [n,2] -> [n,S,S]
            d = g.unsqueeze(0) - center[:, None, None, :]
            return (d ** 2).sum(-1)

        r_hand = self.r_hand / self.arena
        r_block = self.r_block / self.arena
        r_goal = self.success_radius / self.arena
        hand_m = (dist2(hand) < r_hand ** 2).float()
        block_m = (dist2(block) < r_block ** 2).float()
        dg = dist2(goal) ** 0.5
        ring_w = 0.03
        goal_m = ((dg < r_goal) & (dg > r_goal - ring_w)).float()

        img = torch.zeros(n, 3, self.img_size, self.img_size, device=self.state.device)
        img += 0.08  # background
        # goal ring green (draw first, under others)
        img[:, 1] = img[:, 1] + goal_m * 0.8
        # block blue
        img[:, 2] = torch.where(block_m > 0, torch.ones_like(block_m) * 0.9, img[:, 2])
        img[:, 0] = torch.where(block_m > 0, torch.ones_like(block_m) * 0.15, img[:, 0])
        img[:, 1] = torch.where(block_m > 0, torch.ones_like(block_m) * 0.35, img[:, 1])
        # hand white on top
        for c in range(3):
            img[:, c] = torch.where(hand_m > 0, torch.ones_like(hand_m), img[:, c])
        return img.clamp(0, 1)
