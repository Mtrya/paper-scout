"""Exact-expectation toy lab for Score Centering (arXiv 2609.20807).

Setting: an autoregressive chain of depth 2 over a vocabulary of V "tokens".
A "prefix" is either the root context c (depth 1) or the pair (c, a1) (depth 2).
Every prefix owns its own softmax table row, so the score of token a at a
prefix with distribution p is s_a = e_a - p (exact, no chain rule through a
network), and the expected score under the sampler is exactly

    s_bar = sum_a q_a s_a = q - p .

All updates are computed as exact expectations under the sampler, so the drift /
signal decomposition of Eq. (3) of the paper is available in closed form and no
gradient noise enters the main dynamics (a sampled control is run separately).

Methods (all read off the official loss, code/score-centering/losses.py):
  vanilla : update = A * s_y
  SC      : update = A * (s_y - s_bar)                       [Eq. (4)]
  TIS     : update = A * (w_y s_y), w = min(r, 2)
  MIS     : update = A * (w_y s_y), w = r on 0.5 <= r <= 5 else 0
  SC+TIS  : update = A * (w_y s_y - E_q[w s])                 [Eq. (9)]
  SC+MIS  : same with the masked weight
"""

import math

import numpy as np

# ---------------------------------------------------------------- basics


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def log_softmax(z, axis=-1):
    m = z.max(axis=axis, keepdims=True)
    e = np.exp(z - m)
    return z - m - np.log(e.sum(axis=axis, keepdims=True))


def wrap_angle(d):
    return (d + np.pi) % (2.0 * np.pi) - np.pi


def tv(p, q):
    return 0.5 * np.abs(p - q).sum(-1)


def kl(p, q):
    """KL(p || q), elementwise over the last axis."""
    return (p * (np.log(np.maximum(p, 1e-300)) - np.log(np.maximum(q, 1e-300)))).sum(-1)


# ---------------------------------------------------------------- the task

def make_task(V=64, C=4, sigma_r=0.35, phase=0.13, sigma_1=None):
    """Two-step chain with signal at both prefixes.

    Token a sits at angle phi_a on a circle; context c has target angle psi_c.
    A rollout rewards *both* tokens being close to the target,

        R(c, a1, a2) = exp(-d(phi_a1, psi_c)^2 / 2 sigma_1^2)
                     * exp(-d(phi_a2, psi_c)^2 / 2 sigma_r^2),

    so the first token changes the achievable reward (the root prefix carries
    learning signal, and so does every depth-2 prefix) -- which is what makes
    the per-prefix drift anatomy of the paper testable here.
    """
    phi = 2.0 * np.pi * np.arange(V) / V
    psi = 2.0 * np.pi * np.arange(C) / C + phase
    sigma_1 = sigma_r if sigma_1 is None else sigma_1
    d1 = wrap_angle(phi[None, :] - psi[:, None])                 # [C, V]
    d2 = wrap_angle(phi[None, None, :] - psi[:, None, None])     # [C, 1, V]
    R = (np.exp(-(d1 ** 2) / (2.0 * sigma_1 ** 2))[:, :, None]
         * np.exp(-(d2 ** 2) / (2.0 * sigma_r ** 2)))
    return {"V": V, "C": C, "phi": phi, "psi": psi, "R": R}


# ---------------------------------------------------------------- TIM models

class TIM:
    """Sampler-side corruption of the trainer.

    kind='none'  : q = p (on-policy control)
    kind='noise' : q = softmax(theta + dtheta), dtheta fixed  (paper Sec. 5.2)
    kind='quant' : q = softmax(round(theta/s)*s), per-row b-bit grid
    kind='stale' : q = the trainer's distribution k steps ago
    """

    def __init__(self, kind="none", V=64, C=4, seed=0, sigma=0.0, bits=4,
                 stale_k=16, scale="row"):
        self.kind = kind
        self.bits = bits
        self.stale_k = stale_k
        self.scale = scale
        rng = np.random.default_rng(seed)
        if kind == "noise":
            self.d1 = sigma * rng.standard_normal((C, V))
            self.d2 = sigma * rng.standard_normal((C, V, V))
            # remove the per-row mean: a constant shift is invisible to softmax
            self.d1 -= self.d1.mean(-1, keepdims=True)
            self.d2 -= self.d2.mean(-1, keepdims=True)
        self.hist = []

    def quant(self, theta):
        levels = 2 ** (self.bits - 1) - 1
        if self.scale == "row":
            s = np.abs(theta).max(-1, keepdims=True) / levels
        else:  # one scale for the whole tensor (per-tensor INT quantization)
            s = np.abs(theta).max() / levels
        s = np.maximum(s, 1e-12)
        return np.round(theta / s) * s

    def quant_prob(self, p):
        """Quantize the sampler's *probabilities* onto a b-bit grid spanning
        [0, max p], then renormalize. Coarse grids zero the whole tail: the
        sampler is a sharper, truncated copy of the trainer."""
        levels = 2 ** self.bits - 1
        s = np.maximum(p.max(-1, keepdims=True) / levels, 1e-12)
        phat = np.round(p / s) * s
        return phat / phat.sum(-1, keepdims=True)

    def sampler_logits(self, t1, t2):
        if self.kind == "none":
            return t1, t2
        if self.kind == "noise":
            return t1 + self.d1, t2 + self.d2
        if self.kind == "quant":
            return self.quant(t1), self.quant(t2)
        if self.kind == "quant_prob":
            return (np.log(self.quant_prob(softmax(t1))),
                    np.log(self.quant_prob(softmax(t2))))
        if self.kind == "stale":
            if not self.hist:
                return t1, t2
            return self.hist[max(0, len(self.hist) - self.stale_k)]
        raise ValueError(self.kind)

    def push(self, t1, t2):
        if self.kind == "stale":
            self.hist.append((t1.copy(), t2.copy()))


# ---------------------------------------------------------------- update rules

def wfun(name):
    if name is None:
        return None
    if name == "tis":
        return lambda r: np.minimum(r, 2.0)
    if name == "mis":
        return lambda r: np.where((r >= 0.5) & (r <= 5.0), r, 0.0)
    raise ValueError(name)


def rule_grad(p, q, Rvec, wname=None, sc=False):
    """Exact expected gradient of the (negative) loss at one prefix, i.e. the
    parameter update for that prefix's row, as a vector over the vocabulary.

    p, q, Rvec: [..., V].  Rvec is the expected advantage of the next token
    (for the root prefix it already marginalises over the rest of the rollout).
    Returns [..., V]: the update to add to that row's logits (ascent).
    """
    f = wfun(wname)
    w = f(p / np.maximum(q, 1e-300)) if f is not None else np.ones_like(p)
    qw = q * w
    qwR = qw * Rvec
    upd = qwR - qwR.sum(-1, keepdims=True) * p
    if sc:
        mu = qw.sum(-1, keepdims=True)
        cbar = qw - mu * p                       # E_q[w s]
        upd = upd - (q * Rvec).sum(-1, keepdims=True) * cbar
    return upd


def decompose(p, q, Rvec, wname=None):
    """Eq. (3) split of the un-centered rule into drift and signal."""
    f = wfun(wname)
    w = f(p / np.maximum(q, 1e-300)) if f is not None else np.ones_like(p)
    qw = q * w
    qwR = qw * Rvec
    Rbar = (q * Rvec).sum(-1, keepdims=True)
    cbar = qw - qw.sum(-1, keepdims=True) * p     # E_q[w s]
    drift = Rbar * cbar
    signal = qwR - qwR.sum(-1, keepdims=True) * p - drift
    return drift, signal


def make_weight_batch(p, q, name):
    """Sampled-rule weights: per-token w = f(p_y/q_y) with p_y, q_y the trainer /
    sampler logprobs of the *sampled* token (detached, as in the official loss)."""
    if name is None:
        return 1.0
    f = wfun(name)
    return f(p / np.maximum(q, 1e-300))


# ---------------------------------------------------------------- the trainer

# ---------------------------------------------------- policy parameterisation

class TableParams:
    """One free logit row per prefix (the default toy)."""

    label = "table"

    def __init__(self, rng, C, V, scale, rank=None):
        self.t1 = scale * rng.standard_normal((C, V))
        self.t2 = scale * rng.standard_normal((C, V, V))

    def logits(self):
        return self.t1, self.t2

    def grad_apply(self, g1, g2, lr):
        self.t1 = self.t1 + lr * g1
        self.t2 = self.t2 + lr * g2


class LowRankParams:
    """logits(row) = U[row] @ W.T with one shared W over all prefixes, so a
    drift vector produced at one prefix reaches every other prefix through W --
    the cheap stand-in for representation-level coupling in a real network."""

    label = "lowrank"

    def __init__(self, rng, C, V, scale, rank=4):
        self.rank = rank
        self.U1 = (scale / math.sqrt(rank)) * rng.standard_normal((C, rank))
        self.U2 = (scale / math.sqrt(rank)) * rng.standard_normal((C, V, rank))
        self.W = rng.standard_normal((V, rank))

    def logits(self):
        return self.U1 @ self.W.T, self.U2 @ self.W.T

    def grad_apply(self, g1, g2, lr):
        dU1 = g1 @ self.W
        dU2 = g2 @ self.W
        dW = (self.U1.T @ g1 + np.einsum("cak,cav->kv", self.U2, g2)).T
        self.U1 = self.U1 + lr * dU1
        self.U2 = self.U2 + lr * dU2
        self.W = self.W + lr * dW


def make_params(cfg, rng, C, V):
    kind = cfg.get("param", "table")
    scale = cfg.get("init_scale", 0.5)
    if kind == "table":
        return TableParams(rng, C, V, scale)
    if kind == "lowrank":
        return LowRankParams(rng, C, V, scale, cfg.get("rank", 4))
    raise ValueError(kind)


class Toy:
    """Exact-expectation player: one context batch per step, lr as in the paper
    (SGD, constant learning rate)."""

    def __init__(self, cfg):
        self.cfg = cfg
        V, C = cfg["V"], cfg["C"]
        task = make_task(V, C, cfg["sigma_r"], cfg.get("phase", 0.13),
                         cfg.get("sigma_1"))
        self.task = task
        rng = np.random.default_rng(cfg.get("seed", 0))
        self.params = make_params(cfg, rng, C, V)
        self.logits_0 = tuple(x.copy() for x in self.params.logits())
        self.tim = TIM(cfg["tim"], V, C, seed=cfg.get("seed", 0),
                       sigma=cfg.get("sigma", 0.0), bits=cfg.get("bits", 4),
                       stale_k=cfg.get("stale_k", 16),
                       scale=cfg.get("quant_scale", "row"))
        self.method = cfg["method"]
        self.reward_mode = cfg.get("reward_mode", "group_centered")
        self.wname, self.sc = {"vanilla": (None, False), "SC": (None, True),
                               "TIS": ("tis", False), "MIS": ("mis", False),
                               "SC+TIS": ("tis", True), "SC+MIS": ("mis", True),
                               }[self.method]

    # -- distributions
    def dists(self):
        t1, t2 = self.params.logits()
        s1, s2 = self.tim.sampler_logits(t1, t2)
        return softmax(t1), softmax(t2), softmax(s1), softmax(s2)

    # -- expected advantages of the next token, per prefix
    def advantages(self, p1, q1, p2, q2):
        R = self.task["R"]
        if self.reward_mode == "const":
            A = np.ones_like(R)
        elif self.reward_mode == "group_centered":
            mean_R = (q1[:, :, None] * q2 * R).sum((1, 2), keepdims=True)
            A = R - mean_R
        else:
            raise ValueError(self.reward_mode)
        Rvec2 = A                                          # [C, V, V]
        Rvec1 = (q2 * A).sum(-1)                           # [C, V]
        return Rvec1, Rvec2

    def step(self):
        p1, p2, q1, q2 = self.dists()
        Rvec1, Rvec2 = self.advantages(p1, q1, p2, q2)
        # The official objective (Eq. 13) divides by the total token count, so a
        # prefix's expected contribution is weighted by the probability of
        # visiting it; both factors are applied here.
        norm = 2.0                                      # tokens per rollout
        g1 = rule_grad(p1, q1, Rvec1, self.wname, self.sc) / norm
        g2 = rule_grad(p2, q2, Rvec2, self.wname, self.sc) / norm
        if self.cfg.get("visit_weight", True):
            g2 = g2 * q1[:, :, None]
        d1, d2 = None, None
        if self.cfg.get("track_split"):
            dr1, sg1 = decompose(p1, q1, Rvec1, self.wname)
            dr2, sg2 = decompose(p2, q2, Rvec2, self.wname)
            dr1, sg1 = dr1 / norm, sg1 / norm
            dr2, sg2 = dr2 / norm, sg2 / norm
            if self.cfg.get("visit_weight", True):
                dr2 = dr2 * q1[:, :, None]
                sg2 = sg2 * q1[:, :, None]
            d1 = (dr1, sg1)
            d2 = (dr2, sg2)
        self.apply_grads(g1, g2)
        t1, t2 = self.params.logits()
        self.tim.push(t1, t2)
        return dict(p1=p1, p2=p2, q1=q1, q2=q2, Rvec1=Rvec1, Rvec2=Rvec2,
                    g1=g1, g2=g2, split=(d1, d2))

    def apply_grads(self, g1, g2):
        self.params.grad_apply(g1, g2, self.cfg["lr"])

    def metrics(self):
        """Diagnostics that do not depend on the method."""
        p1, p2, q1, q2 = self.dists()
        t1, t2 = self.params.logits()
        R = self.task["R"]
        p1_0 = softmax(self.logits_0[0])
        p2_0 = softmax(self.logits_0[1])
        # exact expected reward of the current trainer policy
        rew_p = (p1[:, :, None] * p2 * R).sum((1, 2))
        rew_q = (q1[:, :, None] * q2 * R).sum((1, 2))
        return {
            "tv_pq": float(np.mean(np.concatenate([tv(p1, q1).ravel(),
                                                  tv(p2, q2).ravel()]))),
            "kl_qp": float(np.mean(np.concatenate([kl(q1, p1).ravel(),
                                                   kl(q2, p2).ravel()]))),
            "tv_p0": float(np.mean(np.concatenate([tv(p1, p1_0).ravel(),
                                                   tv(p2, p2_0).ravel()]))),
            "reward_trainer": float(np.mean(rew_p)),
            "reward_sampler": float(np.mean(rew_q)),
            "entropy": float(np.mean(np.concatenate([
                (-(p1 * np.log(np.maximum(p1, 1e-300))).sum(-1)).ravel(),
                (-(p2 * np.log(np.maximum(p2, 1e-300))).sum(-1)).ravel()]))),
            "logit_norm": float(np.sqrt((t1 ** 2).sum() + (t2 ** 2).sum())),
        }


def run(cfg, steps=200):
    toy = Toy(cfg)
    hist = []
    for it in range(steps):
        info = toy.step()
        m = toy.metrics()
        m["step"] = it + 1
        if cfg.get("track_split"):
            (dr1, sg1), (dr2, sg2) = info["split"]
            m["update_norm"] = float(np.sqrt((info["g1"] ** 2).sum() + (info["g2"] ** 2).sum()))
            m["drift_norm"] = float(np.sqrt((dr1 ** 2).sum() + (dr2 ** 2).sum()))
            m["signal_norm"] = float(np.sqrt((sg1 ** 2).sum() + (sg2 ** 2).sum()))
            allg = np.concatenate([info["g1"].ravel(), info["g2"].ravel()])
            alld = np.concatenate([dr1.ravel(), dr2.ravel()])
            na, nd = np.linalg.norm(allg), np.linalg.norm(alld)
            m["cos_update_drift"] = float(allg @ alld / (na * nd)) if na * nd > 0 else 0.0
        hist.append(m)
    return toy, hist


# ------------------------------------------------------------ sampled control

def _one_hot(idx, V):
    out = np.zeros(idx.shape + (V,))
    np.put_along_axis(out, idx[..., None], 1.0, axis=-1)
    return out


def sample_step(toy, rng, group=8, reward_noise=0.0):
    """One SGD step on a *sampled* batch: `group` rollouts per context,
    group-centered advantages, and the official per-token weights / centering
    term (weights are detached functions of p_v/q_v, the centering term is
    evaluated exactly over the vocabulary of every visited prefix).  The
    expected update equals Toy.step()'s, so a sampled run is a direct control
    for the exact-expectation dynamics -- and the only place where IS variance
    and MIS dropping actually bite."""
    p1, p2, q1, q2 = toy.dists()
    C, V = q1.shape

    a1 = np.stack([rng.choice(V, size=group, p=q1[c]) for c in range(C)])
    q2_rows = np.take_along_axis(q2, a1[:, :, None], axis=1)        # [C,G,V]
    a2 = np.stack([[rng.choice(V, p=q2_rows[c, g]) for g in range(group)]
                   for c in range(C)])
    R = np.take_along_axis(
        np.take_along_axis(toy.task["R"], a1[:, :, None], axis=1),
        a2[:, :, None], axis=2)[..., 0]
    if reward_noise:
        R = R + reward_noise * rng.standard_normal(R.shape)
    A = R - R.mean(1, keepdims=True)

    t1, t2 = toy.params.logits()
    logp1 = log_softmax(t1)
    logp2 = log_softmax(t2)
    p2_logp_rows = np.take_along_axis(logp2, a1[:, :, None], axis=1)  # [C,G,V]
    p2_rows = np.exp(p2_logp_rows)                                    # probabilities

    f = wfun(toy.wname)
    detach = (lambda r: np.ones_like(r)) if f is None else f
    w1_all = detach(p1 / np.maximum(q1, 1e-300))                    # [C,V]
    w2_all = detach(np.exp(p2_logp_rows - np.log(np.maximum(q2_rows, 1e-300))))
    w1 = np.take_along_axis(w1_all, a1, axis=1)                     # [C,G]
    w2 = np.take_along_axis(w2_all, a2[:, :, None], axis=2)[..., 0]

    g1 = (A * w1)[:, :, None] * (_one_hot(a1, V) - p1[:, None, :])
    g2 = (A * w2)[:, :, None] * (_one_hot(a2, V) - p2_rows)
    if toy.sc:
        wq1 = q1 * w1_all
        cbar1 = wq1 - wq1.sum(-1, keepdims=True) * p1
        g1 = g1 - A.sum(1)[:, None, None] * cbar1[:, None, :]
        wq2 = q2_rows * w2_all
        cbar2 = wq2 - wq2.sum(-1, keepdims=True) * p2_rows
        g2 = g2 - A[:, :, None] * cbar2

    d1 = g1.sum(1) / (2.0 * group)
    d2 = np.zeros_like(q2)
    for c in range(C):
        np.add.at(d2[c], a1[c], g2[c])
    d2 /= 2.0 * group

    toy.params.grad_apply(d1, d2, toy.cfg["lr"])
    t1n, t2n = toy.params.logits()
    toy.tim.push(t1n, t2n)


def run_sampled(cfg, steps=200, group=8, reward_noise=0.05, seed=0):
    toy = Toy(cfg)
    rng = np.random.default_rng(seed)
    hist = []
    for it in range(steps):
        sample_step(toy, rng, group=group, reward_noise=reward_noise)
        m = toy.metrics()
        m["step"] = it + 1
        hist.append(m)
    return toy, hist


def vec_batch_update(rng, toy, group, nbatch):
    """Vectorized sample_step gradient for `nbatch` independent batches of
    `group` rollouts per context; returns the summed (un-normalised) updates."""
    p1, p2, q1, q2 = toy.dists()
    C, V = p1.shape
    n = nbatch * group
    a1 = np.stack([rng.choice(V, size=n, p=q1[c]) for c in range(C)])
    q2_rows = np.take_along_axis(q2, a1[:, :, None], axis=1)          # [C,nb,V]
    a2 = np.stack([[rng.choice(V, p=q2_rows[c, g]) for g in range(n)]
                   for c in range(C)])
    R = np.take_along_axis(
        np.take_along_axis(toy.task["R"], a1[:, :, None], axis=1),
        a2[:, :, None], axis=2)[..., 0]
    A = (R.reshape(C, nbatch, group)
         - R.reshape(C, nbatch, group).mean(2, keepdims=True)).reshape(C, n)
    logp2 = log_softmax(toy.params.logits()[1])
    p2_rows = np.exp(np.take_along_axis(logp2, a1[:, :, None], axis=1))
    f = wfun(toy.wname)
    detach = (lambda r: np.ones_like(r)) if f is None else f
    w1_all = detach(p1 / np.maximum(q1, 1e-300))
    w2_all = detach(np.exp(np.take_along_axis(logp2, a1[:, :, None], axis=1)
                           - np.log(np.maximum(q2_rows, 1e-300))))
    w1 = np.take_along_axis(w1_all, a1, axis=1)
    w2 = np.take_along_axis(w2_all, a2[:, :, None], axis=2)[..., 0]
    g1 = (A * w1)[:, :, None] * (_one_hot(a1, V) - p1[:, None, :])
    g2 = (A * w2)[:, :, None] * (_one_hot(a2, V) - p2_rows)
    if toy.sc:
        wq1 = q1 * w1_all
        cbar1 = wq1 - wq1.sum(-1, keepdims=True) * p1
        Ab = A.reshape(C, nbatch, group)
        g1 = (g1.reshape(C, nbatch, group, V)
              - Ab.sum(2)[:, :, None, None] * cbar1[:, None, None, :])
        wq2 = q2_rows * w2_all
        cbar2 = wq2 - wq2.sum(-1, keepdims=True) * p2_rows
        g2 = g2 - A[:, :, None] * cbar2
    d1 = g1.reshape(C, nbatch, group, V).sum((1, 2))
    d2 = np.zeros_like(q2)
    for c in range(C):
        np.add.at(d2[c], a1[c], g2[c])
    return {"d1": d1, "d2": d2}


def check_sampled_vs_exact(group=8, blocks=30, per_block=200, seed=1,
                           methods=None, lanes=None, V=16, C=2):
    """The sampled rule's expected update equals (1 - 1/group) times the exact
    one (group centering shrinks the advantage by that factor).  Checked against
    block standard errors so the agreement is testable, not eyeballed."""
    methods = methods or ["vanilla", "SC", "TIS", "MIS", "SC+TIS", "SC+MIS"]
    lanes = lanes or [("none", {}), ("quant3b", dict(tim="quant", bits=3)),
                      ("noise0.8", dict(tim="noise", sigma=0.8)),
                      ("stale4", dict(tim="stale", stale_k=4))]
    rows = []
    for lane_name, lane in lanes:
        for m in methods:
            rng = np.random.default_rng(seed)
            cfg = dict(V=V, C=C, sigma_r=0.35, seed=3, init_scale=0.7, tim="none",
                       method=m, reward_mode="group_centered", lr=1.0)
            cfg.update(lane)
            toy = Toy(cfg)
            exact = _exact_grads(toy)
            samps = [vec_batch_update(rng, toy, group, per_block)
                     for _ in range(blocks)]
            est = {k: np.array([s[k] for s in samps]) for k in ("d1", "d2")}
            rec = {"lane": lane_name, "method": m, "group": group}
            norm = 2.0 * group * per_block
            for k in ("d1", "d2"):
                emp = est[k].mean(0) / norm
                sem = est[k].std(0, ddof=1) / np.sqrt(blocks) / norm
                ref = (1.0 - 1.0 / group) * exact[k]
                z = np.linalg.norm(emp - ref) / max(np.sqrt((sem ** 2).sum()), 1e-300)
                cos = float((emp.ravel() @ ref.ravel())
                            / max(np.linalg.norm(emp) * np.linalg.norm(ref), 1e-300))
                rec[f"{k}_z"] = float(z)
                rec[f"{k}_cos"] = cos
            rows.append(rec)
    return rows


def _exact_grads(toy):
    """The exact expected update of Toy.step(), without applying it."""
    p1, p2, q1, q2 = toy.dists()
    Rvec1, Rvec2 = toy.advantages(p1, q1, p2, q2)
    g1 = rule_grad(p1, q1, Rvec1, toy.wname, toy.sc) / 2.0
    g2 = rule_grad(p2, q2, Rvec2, toy.wname, toy.sc) / 2.0
    if toy.cfg.get("visit_weight", True):
        g2 = g2 * q1[:, :, None]
    return {"d1": g1, "d2": g2}
