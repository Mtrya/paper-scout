"""
Preserved snippet: agents/edp.py — Efficient Diffusion Policy (Kang et al., NeurIPS 2023).

EDP is the closest training-time baseline. It uses the same single Euler-step
approximation of the clean action, but bakes Q-maximization into the actor
training loss rather than applying it at test time:

  a_eval = clip(x_t + (1-t)*v_pred, -1, 1)   # same approx as QGF
  q_loss = -Q(s, a_eval).mean()
  actor_loss = q_loss + bc_weight * bc_loss

At inference, EDP does standard unguided flow denoising.
The paper finds QGF (test-time guidance) is competitive with EDP and sometimes
better, without needing to tune bc_weight during training.
"""

# ------------------------------------------------------------------
# Extracted from https://github.com/zhouzypaul/qgf/blob/main/agents/edp.py
# ------------------------------------------------------------------

# In policy_loss(...):

v_pred = self.policy(observations, x_t, t, params=policy_params)
bc_loss = jnp.mean((vel - v_pred) ** 2)

# One-step denoised action — gradient flows through v_pred to policy_params
a_eval = jnp.clip(x_t + (1 - tv) * v_pred, -1, 1)
qs = self.target_critic(observations, a_eval)
q = self._aggregate_q(qs)
q_loss = -q.mean()

actor_loss = q_loss + self.config["bc_weight"] * bc_loss
