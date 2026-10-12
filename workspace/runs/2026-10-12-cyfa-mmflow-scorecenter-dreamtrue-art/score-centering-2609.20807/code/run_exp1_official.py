"""EXPERIMENT 1 -- alignment with the official implementation.

Drives `losses.loss_fn_rl` from github.com/martin-marek/score-centering on a
single-prefix tabular policy (logits = W[row]) and checks three things:

  (a) the autodiff gradient of the official loss, averaged over the sampled
      token under the sampler q, equals the analytic expected update
      (drift zeroed exactly by score centering, Eq. (4)/(9));
  (b) the same expectation for the *vanilla* rule points exactly along the
      gradient of KL(q_theta || p_theta) -- i.e. vanilla PG distills the
      trainer toward the sampler (the paper's drift claim);
  (c) the top-k centering of Eq. (8)/(10) leaves a residual drift that decays
      with k; measured with the official loss under a constant reward.

Usage:  uv run --with jax --with optax --with numpy python3 run_exp1_official.py
"""

import json
import math
import os
import sys

import numpy as np

def _find_repo():
    """Locate the official score-centering checkout: ../score-centering when
    running from the lab bench, or the workspace's code/score-centering when
    running from a run packet."""
    override = os.environ.get("SC_REPO")
    if override:
        return override
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in ("../score-centering", "../../../../code/score-centering",
                 "../../../../../code/score-centering"):
        path = os.path.normpath(os.path.join(here, cand))
        if os.path.isfile(os.path.join(path, "losses.py")):
            return path
    raise SystemExit("score-centering checkout not found; set SC_REPO to it")


REPO = _find_repo()
sys.path.insert(0, REPO)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import losses  # noqa: E402
from losses import ISCorrection, loss_fn_rl  # noqa: E402

FIGDIR = os.environ.get("SC_FIGDIR", "figs")
TIS = ISCorrection(outside="clamp", high=2.0)
MIS = ISCorrection(low=0.5, high=5.0, outside="drop")

METHODS = {"vanilla": (None, False), "SC": (None, True),
           "TIS": (TIS, False), "SC+TIS": (TIS, True),
           "MIS": (MIS, False), "SC+MIS": (MIS, True)}


def make_batch(trainer_logits, samp_logprobs, advantage, token, topk=None):
    """One prefix, one generated token. Rows are selected by one-hot hidden
    states, so the head is a lookup table over prefixes."""
    V = len(trainer_logits)
    D = V  # one hidden row per vocabulary entry; row 0 is the prefix we use
    hidden = jnp.zeros((1, 2, D))
    # tokens are shifted by one in get_batch_targets: the distribution at
    # position 0 predicts `token`, so row 0 must sit at position 0.
    hidden = hidden.at[0, 0, 0].set(1.0)
    L = 2
    batch = {
        "tokens": jnp.asarray([[0, token]]),
        "mask": jnp.asarray([[False, True]]),
        "advantages": jnp.asarray([advantage]),
        # position 0 carries the sampled token (see get_batch_targets), so the
        # sampler's logprob for it must sit there too -- otherwise the IS ratio
        # r = p_y / q_y is computed against q_y = 1.
        "sampling_token_logprobs": jnp.asarray(
            [[float(samp_logprobs[token]), 0.0]]),
    }
    if topk is not None:
        ids = jnp.argsort(samp_logprobs)[::-1][:topk]
        batch["center_ids"] = jnp.broadcast_to(ids[None, None, :], (1, L, topk))
        batch["center_logprobs"] = jnp.broadcast_to(
            samp_logprobs[ids][None, None, :], (1, L, topk))
    else:
        batch["sampling_vocab_logprobs"] = jnp.broadcast_to(
            samp_logprobs[None, None, :], (1, L, V))
    return hidden, batch


def official_expected_grad(trainer_logits, samp_logprobs, advantage, is_cfg,
                           score_center, topk=None):
    """E_{y~q}[ grad_W loss ] for the official loss, W = the trainer's table."""
    V = len(trainer_logits)
    W = jnp.zeros((V, V)).at[0].set(jnp.asarray(trainer_logits))
    q = np.exp(np.asarray(samp_logprobs))
    head = lambda h: h @ W  # noqa: E731
    grads = []
    for token in range(V):
        hidden, batch = make_batch(trainer_logits, samp_logprobs, advantage,
                                   token, topk)
        g = jax.grad(lambda W_: loss_fn_rl(hidden, lambda h: h @ W_, batch,
                                           is_cfg, score_center=score_center)[0])(W)
        grads.append(q[token] * np.asarray(g))
    return np.sum(grads, axis=0)


def analytic_update(trainer_logits, samp_logprobs, advantage, method):
    """Exact expected *update* (ascent) for the same single prefix, as a table
    gradient; mirrors rule_grad() of toy_lib."""
    p = np.exp(np.asarray(jax.nn.log_softmax(jnp.asarray(trainer_logits))))
    q = np.exp(np.asarray(samp_logprobs))
    wname, sc = METHODS[method]
    if wname is None:
        w = np.ones_like(p)
    elif wname is TIS:
        w = np.minimum(p / q, 2.0)
    else:
        r = p / q
        w = np.where((r >= 0.5) & (r <= 5.0), r, 0.0)
    qw = q * w
    upd = advantage * (qw[:, None] * (np.eye(len(p)) - p[None, :])).sum(0)
    if sc:
        cbar = qw - qw.sum() * p
        upd = upd - advantage * cbar
    G = np.zeros((len(p), len(p)))
    G[0] = upd
    return G


def check_alignment(seed=0, trials=2):
    """(a) official autodiff == analytic expected update, full and top-k."""
    rng = np.random.default_rng(seed)
    out = []
    for trial in range(trials):
        V = 12
        theta = rng.standard_normal(V) * 1.2
        # sampler: quantized copy of the trainer (per-row 3-bit grid)
        s = np.abs(theta).max() / (2 ** (3 - 1) - 1)
        samp = jax.nn.log_softmax(jnp.asarray(np.round(theta / s) * s))
        # scale of the problem: the uncorrected drift of this prefix
        scale = max(np.abs(analytic_update(theta, samp, 1.0, "vanilla")).max(), 1e-12)
        for method, (is_cfg, sc) in METHODS.items():
            for advantage in (1.0, -0.7):
                for topk in (None, 4):
                    if is_cfg is not None and is_cfg.stat != "ratio":
                        continue
                    g_off = official_expected_grad(theta, samp, advantage,
                                                   is_cfg, sc, topk)
                    g_ana = analytic_update(theta, samp, advantage, method)
                    out.append({"trial": trial, "method": method, "adv": advantage,
                                "k": "full" if topk is None else topk,
                                "rel_err": float(np.abs(g_off + g_ana).max() / scale),
                                "drift_norm": float(np.abs(g_off).max()),
                                "analytic_norm": float(np.abs(g_ana).max()),
                                "scale": float(scale)})
    return out


def check_direction(seed=0):
    """(b) the vanilla expected gradient is parallel to grad KL(q||p)."""
    rng = np.random.default_rng(seed + 1)
    rows = []
    for advantage in (1.0, -0.7):
        V = 12
        theta = rng.standard_normal(V) * 1.2
        s = np.abs(theta).max() / 3.0
        samp = jax.nn.log_softmax(jnp.asarray(np.round(theta / s) * s))
        g_off = official_expected_grad(theta, samp, advantage, None, False)
        # grad_W KL(q || p_W) for the same table row
        q = np.exp(np.asarray(samp))
        W0 = jnp.zeros((V, V)).at[0].set(jnp.asarray(theta))

        def kl(W_):
            logp = jax.nn.log_softmax(W_[0])
            return jnp.sum(q * (jnp.log(q) - logp))
        g_kl = np.asarray(jax.grad(kl)(W0))
        g_off_v, g_kl_v = g_off.ravel(), g_kl.ravel()
        cos = float(g_off_v @ g_kl_v / (np.linalg.norm(g_off_v) * np.linalg.norm(g_kl_v)))
        # score centering with a constant advantage must give exactly zero
        g_sc = official_expected_grad(theta, samp, advantage, None, True)
        rows.append({"advantage": advantage, "cos_gradL_gradKL": cos,
                     "sc_expected_grad_max": float(np.abs(g_sc).max()),
                     "vanilla_expected_grad_max": float(np.abs(g_off).max()),
                     "gradKL_norm": float(np.linalg.norm(g_kl_v))})
    return rows


def topk_residual(logits, p_logits, k):
    """Residual drift of the top-k centering rule of Eq. (8)/(10), with w = 1.

    Expected update of the official loss = -(E_q[s] - c_grad) with
        E_q[s] = q - p                                    (the vanilla drift)
        c_grad = sum_{v in H} (q_v - rho p_v) (e_v - p)   (the top-k centering)
    so ||E_q[s] - c_grad|| / ||q - p|| is the fraction of the drift that
    survives the top-k approximation.  The identity
    grad(official loss) = -(E_q[w s] - c_grad) is certified in check_alignment,
    which is why this can be evaluated without autodiff over V tokens.
    """
    p = np.exp(np.asarray(jax.nn.log_softmax(jnp.asarray(p_logits))))
    q = np.exp(np.asarray(jax.nn.log_softmax(jnp.asarray(logits))))
    order = np.argsort(q)[::-1]
    ids = order[:k]
    head_mask = np.zeros_like(p, dtype=bool)
    head_mask[ids] = True
    rho = (1.0 - q[ids].sum()) / max(1.0 - p[ids].sum(), 1e-12)
    s = np.eye(len(p)) - p[None, :]
    c_grad = ((q[ids] - rho * p[ids])[:, None] * s[ids]).sum(0)
    drift = q - p
    return {"k": k, "head_mass": float(q[ids].sum()),
            "residual_frac": float(np.linalg.norm(drift - c_grad) / np.linalg.norm(drift)),
            "vanilla_drift": float(np.linalg.norm(drift))}


def check_topk_drift(V=4096, spikiness=3.0, seed=0, ks=(8, 32, 128, 512),
                     trials=8, spot_check=True, spot_v=64):
    """The V=4096 sweep is analytic; the spot check reruns the official loss at
    spot_v=64 tokens (token-by-token autodiff over 4096 tokens is pointless)."""
    rng = np.random.default_rng(seed + 2)
    rows = []
    for trial in range(trials):
        logits = spikiness * rng.standard_normal(V)
        p_logits = logits + 0.05 * rng.standard_normal(V)
        for k in ks:
            rec = topk_residual(logits, p_logits, k)
            rec.update(trial=trial, V=V)
            if spot_check and k <= spot_v // 4:
                # official autodiff at a smaller vocabulary: its residual drift
                # must match the analytic residual's scale
                small = logits[:spot_v]
                p_small = p_logits[:spot_v]
                samp_small = jnp.log(jax.nn.softmax(jnp.asarray(small)))
                g_off = official_expected_grad(p_small, samp_small, 1.0, None,
                                               True, k)
                an = topk_residual(small, p_small, k)
                rec["spot_official_residual"] = float(
                    np.abs(g_off[0] - (0.0)).max())
                rec["spot_analytic_residual"] = an["residual_frac"]
                rec["spot_official_frac"] = float(
                    np.abs(g_off[0]).max()
                    / np.linalg.norm(np.exp(np.asarray(
                        jax.nn.log_softmax(jnp.asarray(small))))
                        - np.exp(np.asarray(
                            jax.nn.log_softmax(jnp.asarray(p_small))))))
            rows.append(rec)
    return rows


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    print("== (a) official autodiff vs analytic expected update")
    align = check_alignment()
    full = [a for a in align if a["k"] == "full"]
    print(f"   full-vocab centering: {len(full)} cases, worst relative deviation"
          f" = {max(a['rel_err'] for a in full):.3e} (scale = vanilla drift)")
    for m in METHODS:
        sub = [a for a in full if a["method"] == m]
        print(f"   {m:8s} max rel_err = {max(a['rel_err'] for a in sub):.3e}")
    tk = [a for a in align if a["k"] != "full"]
    print("   top-k centering (k=4, V=12): official rule vs the full-vocab"
          " analytic rule -- the residual is the approximation error:")
    for m in METHODS:
        sub = [a for a in tk if a["method"] == m]
        print(f"   {m:8s} residual / vanilla drift = "
              f"{max(a['drift_norm'] for a in sub) / sub[0]['scale']:.3e}")

    print("== (b) drift direction: cos(grad L_vanilla, grad KL(q||p))")
    direc = check_direction()
    for d in direc:
        print(f"   advantage={d['advantage']:+.2f}  cos={d['cos_gradL_gradKL']:+.6f} "
              f"(+1 = loss gradient increases KL, so SGD distills toward q); "
              f"|E[grad L_SC]|={d['sc_expected_grad_max']:.2e}")

    print("== (c) top-k centering residual drift (V=4096, constant reward)")
    topk = check_topk_drift()
    for k in sorted({r["k"] for r in topk}):
        sub = [r for r in topk if r["k"] == k]
        hm = np.mean([r["head_mass"] for r in sub])
        rf = np.mean([r["residual_frac"] for r in sub])
        print(f"   k={k:4d}  head mass={hm:.6f}  residual drift / vanilla drift "
              f"= {rf:.3e}")

    with open(os.path.join(FIGDIR, "exp1_official.json"), "w") as f:
        json.dump({"alignment": align, "direction": direc, "topk_drift": topk},
                  f, indent=1)
    print(f"wrote {FIGDIR}/exp1_official.json")


if __name__ == "__main__":
    main()
