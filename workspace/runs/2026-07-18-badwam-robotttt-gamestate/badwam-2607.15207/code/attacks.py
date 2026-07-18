"""BadWAM attacks, faithfully re-implemented in numpy against the toy WAM.

`query_search_attack` replicates src/attackwam/attacks.py::QuerySearchAttack
(the paper's black-box optimizer) exactly for the `full_linf` space used in
the paper-grade experiments:

    init:      delta ~ U(-eps, eps)                       [random init]
    per iter:  u ~ Rademacher
               s+ = J(clip(delta + sigma*u)), s- = J(clip(delta - sigma*u))
               g  = (s+ - s-) / (2*sigma) * u             [1-direction SPSA]
               delta <- clip(delta + step * sign(g))      [signed PGD step]
    track the best-scoring of the 2*budget evaluated candidates
    objective J(delta) = ||a(o+d)-a(o)||_2 - lam*||z(o+d)-z(o)||_2
                         (- perturb_weight * mean|delta|, off by default)

Query accounting matches the paper: 1 clean + 2*iters attack queries
(default iters=8 -> 17 forward queries per replan, eps=0.06, sigma=0.02,
step=0.02 -- the paper's defaults from appendix A.3).

Also included:
  * `random_attack`     -- the paper's random-perturbation baseline
  * `whitebox_pgd`      -- the appendix-A.4 gradient-access reference
                           (random init, 16 steps, alpha=0.01, L-inf clip)
"""

from __future__ import annotations

import numpy as np


def _clip01(x):
    return np.clip(x, -1.0, 1.0)


def d_act(a_adv, a_clean):
    return float(np.linalg.norm((a_adv - a_clean).reshape(-1)))


def d_img(z_adv, z_clean):
    if z_adv is None or z_clean is None:
        return 0.0
    return float(np.linalg.norm((z_adv - z_clean).reshape(-1)))


class QueryFn:
    """Wraps the toy WAM as the paper's black-box query interface + counting."""

    def __init__(self, model):
        self.model = model
        self.n_queries = 0

    def __call__(self, o_flat):
        self.n_queries += 1
        return self.model.predict(o_flat)


def _objective(model, o_flat, delta, clean_a, clean_z, lam, perturb_weight, query_fn):
    adv_img = _clip01(o_flat + delta)
    a_adv, z_adv = query_fn(adv_img)
    da = d_act(a_adv, clean_a)
    dz = d_img(z_adv, clean_z)
    score = da - lam * dz - perturb_weight * float(np.mean(np.abs(delta)))
    return score, da, dz, adv_img


def query_search_attack(model, o_img, eps=0.06, sigma=0.02, step=0.02, iters=8,
                        lam=0.0, perturb_weight=0.0, seed=0, query_fn=None, n_dir=1):
    """BadWAM query attack (sign-SPSA, full-image L-inf). Returns best delta.

    n_dir=1 reproduces the paper's implementation exactly (one Rademacher
    direction per paired finite-difference update).  n_dir>1 averages several
    directions per update (paper Eq. 11's general form) -- used to probe how
    the attack scales with query budget."""
    rng = np.random.default_rng(seed)
    o_flat = o_img.reshape(-1)
    qf = query_fn or QueryFn(model)
    clean_a, clean_z = qf(o_flat)  # the 1 clean reference query

    delta = rng.uniform(-eps, eps, o_flat.shape)
    best = {"score": -np.inf, "delta": None, "d_act": 0.0, "d_img": 0.0}
    traj = []
    for _ in range(iters):
        grad_est = np.zeros_like(o_flat)
        for _ in range(n_dir):
            u = rng.choice([-1.0, 1.0], size=o_flat.shape)
            d_plus = np.clip(delta + sigma * u, -eps, eps)
            d_minus = np.clip(delta - sigma * u, -eps, eps)
            s_plus, da_p, dz_p, _ = _objective(model, o_flat, d_plus, clean_a, clean_z,
                                               lam, perturb_weight, qf)
            s_minus, da_m, dz_m, _ = _objective(model, o_flat, d_minus, clean_a, clean_z,
                                                lam, perturb_weight, qf)
            for s, d, da, dz in ((s_plus, d_plus, da_p, dz_p), (s_minus, d_minus, da_m, dz_m)):
                if s > best["score"]:
                    best = {"score": s, "delta": d.copy(), "d_act": da, "d_img": dz}
            grad_est += (s_plus - s_minus) / (2.0 * sigma) * u
        grad_est /= n_dir
        delta = np.clip(delta + step * np.sign(grad_est), -eps, eps)
        traj.append({"best_d_act": best["d_act"], "best_d_img": best["d_img"]})
    return best["delta"], {"clean_a": clean_a, "clean_z": clean_z,
                           "traj": traj, "n_queries": qf.n_queries,
                           "best_d_act": best["d_act"], "best_d_img": best["d_img"]}


def random_attack(model, o_img, eps=0.06, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-eps, eps, o_img.reshape(-1).shape)


def whitebox_pgd(model, o_img, eps=0.06, step=0.01, iters=16, lam=0.0, seed=0):
    """Gradient-access reference (paper appendix A.4). Action-only by default."""
    rng = np.random.default_rng(seed)
    o_flat = o_img.reshape(-1)
    clean_a, clean_z = model.predict(o_flat)
    delta = rng.uniform(-eps, eps, o_flat.shape)
    for _ in range(iters):
        adv = _clip01(o_flat + delta)
        g = model.input_grad_objective(adv, clean_a, clean_z, lam)
        delta = np.clip(delta + step * np.sign(g), -eps, eps)
    a_adv, z_adv = model.predict(_clip01(o_flat + delta))
    return delta, {"d_act": d_act(a_adv, clean_a), "d_img": d_img(z_adv, clean_z)}
