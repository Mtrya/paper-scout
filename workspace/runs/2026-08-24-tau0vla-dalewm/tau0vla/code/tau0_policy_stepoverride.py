"""Step-count override wrapper for FM denoising ablation.

openloop.py takes any policy class via --policy-module/--policy-class;
this wrapper sets model.config.num_steps from env TAU0_NUM_STEPS.
"""

from __future__ import annotations

import os

from deploy.policy import Tau0VLAPolicy


class Tau0VLAPolicyStepOverride(Tau0VLAPolicy):
    @classmethod
    def from_checkpoint(cls, ckpt_dir, *, route=None, device=None):
        policy = super().from_checkpoint(ckpt_dir, route=route, device=device)
        k = int(os.environ.get("TAU0_NUM_STEPS", "10"))
        policy.model.config.num_steps = k
        return policy
