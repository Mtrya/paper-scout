"""Lorenz-63 regimes for the round-trip blind-spot probe.

One equation, one parameter r sweeps the dissipative structure:
  r = 0.5  -> origin globally stable ("world death": all trajectories die)
  r = 3    -> weakly damped stable fixed points C+/C-
  r = 10   -> stable spiral fixed points
  r = 20   -> stable fixed points, weaker damping (near Hopf ~24.74)
  r = 28   -> strange attractor (chaotic, positive Lyapunov exponent)
"""
import numpy as np

SIGMA = 10.0
BETA = 8.0 / 3.0

REGIMES = [0.5, 3.0, 10.0, 20.0, 28.0]
# Burn-in puts ICs onto the attractor; for dissipative regimes that would make
# every trajectory a constant sitting at the fixed point (degenerate). There
# we want transient trajectories that die into the attractor, so no burn-in.
BURN_IN = {0.5: 0, 3.0: 0, 10.0: 0, 20.0: 400, 28.0: 400}

# Positive control: undamped pendulum (r = -1), divergence-free Hamiltonian
# dynamics -- phase-volume preserving, so the true inverse is as stable as the
# forward map. This is where the paper's Assumption 1 should hold.
PENDULUM = -1.0
BURN_IN[PENDULUM] = 0


def lorenz_rhs(x, r):
    dx = SIGMA * (x[1] - x[0])
    dy = x[0] * (r - x[2]) - x[1]
    dz = x[0] * x[1] - BETA * x[2]
    return np.array([dx, dy, dz])


def pendulum_rhs(x):
    return np.array([x[1], -np.sin(x[0])])


def rhs(x, r):
    if r == PENDULUM:
        return pendulum_rhs(x)
    return lorenz_rhs(x, r)


def rk4_step(x, r, dt):
    k1 = rhs(x, r)
    k2 = rhs(x + 0.5 * dt * k1, r)
    k3 = rhs(x + 0.5 * dt * k2, r)
    k4 = rhs(x + dt * k3, r)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate(r, ic, n_steps, dt=0.02, burn_in=0):
    """Return trajectory [n_steps+1, dim] after burn-in from ic."""
    x = np.asarray(ic, dtype=np.float64)
    for _ in range(burn_in):
        x = rk4_step(x, r, dt)
    traj = np.empty((n_steps + 1, x.shape[0]), dtype=np.float64)
    traj[0] = x
    for i in range(n_steps):
        x = rk4_step(x, r, dt)
        traj[i + 1] = x
    return traj


def sample_ics(r, n, rng, burn_in=400, dt=0.02):
    """Sample initial conditions on/near the attractor.

    Seed uniformly in a generous box, burn in onto the attractor, return the
    post-burn-in state as the IC.
    """
    ics = []
    for _ in range(n):
        if r == PENDULUM:
            ic = rng.uniform([-3.0, -2.0], [3.0, 2.0])
        else:
            ic = rng.uniform([-20, -25, 0], [20, 25, 50])
        x = np.asarray(ic, dtype=np.float64)
        for _ in range(burn_in):
            x = rk4_step(x, r, dt)
        ics.append(x)
    return np.array(ics)


def gen_dataset(r, n_traj, n_steps, rng, dt=0.02, burn_in=400):
    """[n_traj, n_steps+1, 3] trajectories from on-attractor ICs."""
    ics = sample_ics(r, n_traj, rng, burn_in=burn_in, dt=dt)
    return np.stack([simulate(r, ic, n_steps, dt=dt) for ic in ics])
