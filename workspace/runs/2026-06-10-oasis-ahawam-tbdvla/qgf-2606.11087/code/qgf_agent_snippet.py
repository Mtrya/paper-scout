"""
Preserved snippet: agents/qgf.py — the core QGF inference loop.
This is the canonical implementation of Algorithm 1 from the paper.

Key lines:
  - Line 199: a_approx = clip(a + (1-t)*v_bc_sg, -1, 1)   # single Euler step
  - Line 209: qgrad = jax.grad(q_fn)(stop_gradient(a_approx))  # grad at clean approx
  - Line 214-223: optional Jacobian application (QGF-Jacobian variant)
  - Line 225: a_{t+dt} = a_t + dt*(v_bc + guidance_weight*qgrad)  # guided step

The default config has apply_jacobian=False and denoised_action_approx=
"one_euler_step_approx", which is the QGF estimator in Eq. (9).
"""

# ------------------------------------------------------------------
# Extracted from https://github.com/zhouzypaul/qgf/blob/main/agents/qgf.py
# ------------------------------------------------------------------

# In sample_actions(...):

def step(a, t_idx):
    ti = jnp.ones((a.shape[0],)) * (t_idx / self.config["denoise_steps"])
    tv = ti[..., None]

    v_bc = self.policy(observations, a, ti)

    """
    Getting the approximated clean action given a_t
    """
    if denoised_action_approx == "noisy":
        a_approx = a
    elif denoised_action_approx == "one_euler_step_approx":
        v_bc_sg = jax.lax.stop_gradient(v_bc)
        a_approx = jnp.clip(a + (1 - tv) * v_bc_sg, -1, 1)
    else:
        raise ValueError(
            f"denoised_action_approx '{denoised_action_approx}' is not supported at inference"
        )

    def q_fn(a):
        return self._aggregate_q(self.target_critic(observations, a)).sum()

    qgrad = jax.grad(q_fn)(jax.lax.stop_gradient(a_approx))

    """
    Applying the Jacobian of d a_approx / d a_t
    """
    if apply_jacobian:
        assert denoised_action_approx == "one_euler_step_approx"

        def map_single(a_i, obs_i, tv_i):
            v = self.policy(obs_i[None], a_i[None], tv_i)[0]
            return jnp.clip(a_i + (1 - tv_i[0]) * v, -1, 1)

        jac_per_batch = jax.vmap(jax.jacrev(map_single, argnums=0))
        jac = jac_per_batch(a, observations, tv)
        qgrad = jnp.einsum("bi,bij->bj", qgrad, jac)

    return a + (v_bc + guidance_weight * qgrad) * dt, None
