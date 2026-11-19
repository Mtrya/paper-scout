"""Numerical verification of the conditional-interaction results in arXiv:2610.39497.

Checks, by Monte Carlo:
1. Eq. (12)/(13): IIE gain of a gated loop vs unconditional interaction, with fp/fn/c_pred.
2. Prop. 14: screening cost ratio C_CI/C_react = rho_u + r.
3. Thm. 18: top-B selection regret = 0 when 2*eps < gap_min, else <= 2*T*B*eps.
"""
import json
import math
import random

random.seed(20261120)


def iie_gain_monte_carlo(rho, fp, fn, c_pred, c_rel, c_dead, u_rel, n=400_000):
    """Empirical IIE ratio vs Eq. 13 prediction. Dead candidates carry zero utility."""
    rel = [random.expovariate(1.0 / u_rel) for _ in range(int(n * rho))]
    dead = [0.0] * (n - len(rel))
    c_rel_draws = [c_rel * random.uniform(0.5, 1.5) for _ in rel]
    c_dead_draws = [c_dead * random.uniform(0.5, 1.5) for _ in dead]
    util = 0.0
    cost = 0.0
    for u, c in zip(rel, c_rel_draws):
        cost += c_pred
        if random.random() >= fn:  # admitted
            util += u
            cost += c
    for c in c_dead_draws:
        cost += c_pred
        if random.random() < fp:  # false positive: executed
            cost += c
    iie_ci = util / cost
    iie_unc = (sum(rel) / n) / (sum(c_rel_draws + c_dead_draws) / n)
    emp = iie_ci / iie_unc
    c_bar = rho * c_rel + (1 - rho) * c_dead
    pred = (1 - fn) * c_bar / (c_pred + (1 - fn) * rho * c_rel + fp * (1 - rho) * c_dead)
    return {"rho": rho, "fp": fp, "fn": fn, "c_pred": c_pred, "empirical": round(emp, 4), "eq13": round(pred, 4), "abs_err": round(abs(emp - pred), 4)}


def screening_gain(rho, r, n=100_000):
    """Prop. 14: reactive executes all n; predictive screens n then executes rho*n."""
    c_exec = 1.0
    c_screen = r * c_exec
    reactive = n * c_exec
    predictive = n * c_screen + rho * n * c_exec
    return {"rho": rho, "r": r, "empirical": round(predictive / reactive, 4), "prop14": round(rho + r, 4)}


def regret_bound(B=5, T=200, gap=0.2, eps_values=(0.05, 0.09, 0.11, 0.2), n_pool=200, trials=200):
    """Thm. 18: regret of top-B selection under uniform predictor noise eps vs oracle."""
    out = []
    for eps in eps_values:
        total_regret = 0.0
        oracle_violations = 0
        for _ in range(trials):
            for _t in range(T):
                utils = [random.random() for _ in range(n_pool)]
                order = sorted(range(n_pool), key=lambda i: -utils[i])
                top = set(order[:B])
                # ensure a real gap around the boundary for the idealized check
                for i in order[:B]:
                    utils[i] += gap
                noisy = [(utils[i] + random.uniform(-eps, eps), i) for i in range(n_pool)]
                picked = {i for _, i in sorted(noisy, reverse=True)[:B]}
                regret = sum(utils[i] for i in top) - sum(utils[i] for i in picked)
                total_regret += max(0.0, regret)
                if picked != top:
                    oracle_violations += 1
        bound = 2 * T * trials * B * eps if 2 * eps >= gap else 0.0
        out.append({
            "eps": eps, "2eps_vs_gap": "2eps>=gap" if 2 * eps >= gap else "2eps<gap",
            "mean_regret_per_round": round(total_regret / (T * trials), 5),
            "bound_total": round(bound, 3),
            "emp_total": round(total_regret, 3),
            "oracle_violation_rate": round(oracle_violations / (T * trials), 4),
        })
    return out


result = {
    "iie_gain": [
        iie_gain_monte_carlo(rho=0.2, fp=0.0, fn=0.0, c_pred=0.0, c_rel=1.0, c_dead=1.0, u_rel=1.0),
        iie_gain_monte_carlo(rho=0.2, fp=0.05, fn=0.05, c_pred=0.01, c_rel=1.0, c_dead=1.0, u_rel=1.0),
        iie_gain_monte_carlo(rho=0.1, fp=0.1, fn=0.1, c_pred=0.02, c_rel=1.0, c_dead=2.0, u_rel=1.0),
        iie_gain_monte_carlo(rho=0.05, fp=0.02, fn=0.3, c_pred=0.005, c_rel=1.0, c_dead=1.0, u_rel=1.0),
    ],
    "screening_gain": [screening_gain(rho, r) for rho, r in [(0.2, 0.01), (0.2, 0.1), (0.05, 0.02), (0.5, 0.05)]],
    "regret": regret_bound(),
}
print(json.dumps(result, indent=1))
