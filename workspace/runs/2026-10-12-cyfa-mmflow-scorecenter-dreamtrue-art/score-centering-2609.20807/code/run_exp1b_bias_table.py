"""EXPERIMENT 1b -- how much drift does each correction rule leave behind?

Exactly computable: with a constant reward the expected update of a rule is a
function of (p, q, w) only, so the residual drift of every rule can be read off
analytically.  The identity  grad(official loss) = -(E_q[w s] - centering term)
is certified against the official autodiff in run_exp1_official.py; here it is
used over a severity sweep that would be too slow to autodiff token by token.

Reported per lane: ||residual drift|| / ||q - p|| (vanilla drift), where the
residual drift is the expected update of the rule under a constant reward --
i.e. the part of the bias that survives the correction.  Score centering makes
it exactly zero by construction (Eq. (4)/(9)); the interest is in how much the
IS-based rules leave behind, and how that grows with the mismatch.

Usage: uv run --with numpy python3 run_exp1b_bias_table.py
"""

import json
import os

import numpy as np

FIGDIR = os.environ.get("SC_FIGDIR", "figs")
METHODS = {"vanilla": (None, False), "SC": (None, True),
           "TIS": ("tis", False), "SC+TIS": ("tis", True),
           "MIS": ("mis", False), "SC+MIS": ("mis", True)}


def weight(name, r):
    if name is None:
        return np.ones_like(r)
    if name == "tis":
        return np.minimum(r, 2.0)
    if name == "mis":
        return np.where((r >= 0.5) & (r <= 5.0), r, 0.0)
    raise ValueError(name)


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def residual(p, q, name):
    """Expected update of `name` under a constant reward = Rvec 1 (up to the
    factor E_q[R]); the drift part of Eq. (3)."""
    w = weight(name, p / q)
    qw = q * w
    upd = qw[:, None] * (np.eye(len(p)) - p[None, :])
    upd = upd.sum(0)
    return upd


def main():
    rng = np.random.default_rng(0)
    lanes = []
    for scale in (0.1, 0.3, 0.6, 1.2):
        lanes.append((f"gauss sigma={scale}", "gauss", scale))
    for scale in (0.1, 0.3, 0.6, 1.2):
        lanes.append((f"cauchy scale={scale}", "cauchy", scale))
    for bits in (6, 4, 3, 2):
        lanes.append((f"logit quant {bits}b", "quant", bits))
    for bits in (6, 4, 3):
        lanes.append((f"prob quant {bits}b", "quant_prob", bits))

    out = []
    V = 64
    for rep in range(4):
        theta = 0.8 * rng.standard_normal(V)
        for label, kind, prm in lanes:
            if kind == "gauss":
                d = prm * rng.standard_normal(V)
                d -= d.mean()
                q = softmax(theta + d)
            elif kind == "cauchy":
                d = prm * rng.standard_cauchy(V) / 3.0
                d -= d.mean()
                q = softmax(theta + d)
            elif kind == "quant":
                s = np.abs(theta).max() / (2 ** (prm - 1) - 1)
                q = softmax(np.round(theta / s) * s)
            elif kind == "quant_prob":
                p0 = softmax(theta)
                s = p0.max() / (2 ** prm - 1)
                ph = np.round(p0 / s) * s
                q = ph / ph.sum()
            p = softmax(theta)
            row = {"lane": label, "rep": rep,
                   "tv_pq": float(0.5 * np.abs(p - q).sum())}
            van = np.linalg.norm(residual(p, q, None))
            for m, (name, sc) in METHODS.items():
                upd = residual(p, q, name)
                if sc:
                    w = weight(name, p / q)
                    cbar = q * w - (q * w).sum() * p
                    upd = upd - cbar
                row[m] = float(np.linalg.norm(upd))
                row[f"{m}_frac"] = float(np.linalg.norm(upd) / van)
            out.append(row)

    print(f"{'lane':22s} {'TV(p,q)':>8s} " + " ".join(
        f"{m:>9s}" for m in METHODS))
    for label, _, _ in lanes:
        sub = [r for r in out if r["lane"] == label]
        cells = []
        for m in METHODS:
            frac = np.mean([r[f"{m}_frac"] for r in sub])
            cells.append(f"{frac * 100:8.2f}%")
        print(f"{label:22s} {np.mean([r['tv_pq'] for r in sub]):8.4f} "
              + " ".join(cells))
    print("\n(cells are ||rule residual|| / ||q - p||; SC rows are exactly 0)")

    with open(os.path.join(FIGDIR, "exp1b_bias_table.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {FIGDIR}/exp1b_bias_table.json")


if __name__ == "__main__":
    main()
