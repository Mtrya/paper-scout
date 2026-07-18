"""Toy World-Action Model (WAM) for probing BadWAM's world-action drift attacks.

A minimal, fully-numpy reconstruction of the WAM design pattern:

    observation o (image) --encoder--> latent h
    h --imagination head--> predicted future video z (K decoded frames)
    (h [, z]) --action head--> action chunk a (H steps x 3 channels)

Three variants mirror the three FastWAM variants attacked in the paper:

    direct : action head reads h only; no imagination loss (FastWAM
             action-only inference: video expert used as static context).
    joint  : action head reads the same h that imagination is decoded from
             (FastWAMJoint: action tokens attend to jointly-denoised video
             tokens).
    idm    : action head reads ONLY the decoded imagination z
             (FastWAMIDM: imagine video first, then decode actions from the
             frozen imagination K/V cache).  This is the strict
             "imagine-then-act" architecture: the input image can only
             affect actions *through* the imagination.

The synthetic task is 2D point-mass goal reaching.  The observation is a
32x32 grayscale render (agent blob + goal blob, distinct shapes), mapped to
[-1, 1] to match FastWAM's input normalization, so the paper's L-inf budget
eps=0.06 means the same thing here.

Actions: 3 channels = (dx, dy) velocity (continuous) + "gripper" (bang-bang,
closes when near the goal) -- mirroring LIBERO's xyz/rot/gripper split so we
can check whether attacks concentrate on continuous channels (paper Fig. 2).

Everything (training, white-box attack) uses exact analytic gradients --
no autodiff needed.
"""

from __future__ import annotations

import numpy as np

IMG = 32          # observation resolution (IMG x IMG, grayscale)
FUT_IMG = 16      # decoded future frame resolution
K_FUT = 4         # number of predicted future frames
H_ACT = 4         # action chunk horizon
A_DIM = 3         # (dx, dy, gripper)
HID1, HID2 = 256, 128

DT = 0.15         # env step size
A_MAX = 0.5       # action clip (env)
KP = 2.0          # expert proportional gain
GRIP_RADIUS = 0.15  # expert gripper-close radius
SUCCESS_RADIUS = 0.08


# ---------------------------------------------------------------- rendering
def _blob(grid_x, grid_y, cx, cy, sigma, amp):
    return amp * np.exp(-((grid_x - cx) ** 2 + (grid_y - cy) ** 2) / (2 * sigma ** 2))


def render_scene(agent, goal, size=IMG):
    """agent: small sharp gaussian blob; goal: ring (annulus) -- visually
    distinct even under overlap, like color-coded objects. Returns [0,1]."""
    lin = np.linspace(0, 1, size)
    gx, gy = np.meshgrid(lin, lin)
    s = 1.0 / size
    img = _blob(gx, gy, agent[0], agent[1], 1.6 * s, 1.0)
    r = np.sqrt((gx - goal[0]) ** 2 + (gy - goal[1]) ** 2)
    img = img + 0.75 * np.exp(-((r - 4.0 * s) ** 2) / (2 * (1.4 * s) ** 2))
    return np.clip(img, 0.0, 1.0)


def render_future(agent_positions, goal, size=FUT_IMG):
    """K frames of the agent at successive positions against the static goal."""
    frames = [render_scene(p, goal, size=size) for p in agent_positions]
    return np.stack(frames, 0)  # [K, size, size] in [0,1]


def to_model_range(img):
    return 2.0 * img - 1.0  # [0,1] -> [-1,1], FastWAM-style


# ---------------------------------------------------------------- expert/env
def expert_rollout(agent, goal, steps=H_ACT):
    """Expert action chunk + resulting positions (used as training targets)."""
    p = np.array(agent, dtype=np.float64)
    acts, positions = [], []
    for _ in range(steps):
        a = np.clip(KP * (goal - p), -A_MAX, A_MAX)
        grip = 1.0 if np.linalg.norm(goal - p) < GRIP_RADIUS else -1.0
        acts.append(np.array([a[0], a[1], grip]))
        p = p + a * DT
        positions.append(p.copy())
    return np.array(acts), np.array(positions)  # [H,3], [H,2]


def env_step(agent, action, goal):
    a = np.clip(np.asarray(action[:2], dtype=np.float64), -A_MAX, A_MAX)
    return agent + a * DT


# ---------------------------------------------------------------- the model
def _xavier(rng, fan_in, fan_out):
    return rng.normal(0, np.sqrt(2.0 / (fan_in + fan_out)), (fan_in, fan_out))


class ToyWAM:
    """MLP world-action model with exact analytic gradients."""

    def __init__(self, variant, seed=0, img_weight=1.0):
        assert variant in {"direct", "joint", "idm"}
        self.variant = variant
        self.img_weight = img_weight if variant != "direct" else 0.0
        rng = np.random.default_rng(seed)
        n_in = IMG * IMG
        n_z = K_FUT * FUT_IMG * FUT_IMG
        n_a = H_ACT * A_DIM
        self.params = {
            "W1": _xavier(rng, n_in, HID1), "b1": np.zeros(HID1),
            "W2": _xavier(rng, HID1, HID2), "b2": np.zeros(HID2),
            "Wz": _xavier(rng, HID2, n_z) * 0.1, "bz": np.zeros(n_z),
        }
        if variant in {"direct", "joint"}:
            self.params.update({"Wa": _xavier(rng, HID2, n_a) * 0.1, "ba": np.zeros(n_a)})
        else:  # idm: action decoded from imagination only
            self.params.update({
                "Wi1": _xavier(rng, n_z, HID2), "bi1": np.zeros(HID2),
                "Wi2": _xavier(rng, HID2, n_a) * 0.1, "bi2": np.zeros(n_a),
            })

    # ---------------------------------------------------------- forward
    @staticmethod
    def _center(o):
        """Per-image mean subtraction (LayerNorm-like DC removal).

        Without this the first tanh layer saturates from the large DC offset
        of [-1,1] images (real WAMs avoid this via VAE latents + LayerNorm).
        Centering is linear, so input Jacobians just gain a factor C.
        """
        return o - o.mean(axis=-1, keepdims=True)

    def forward(self, o):
        """o: [..., IMG*IMG] in [-1,1]. Returns dict with h, z, a (last two flat)."""
        p = self.params
        o = self._center(o)
        h1 = np.tanh(o @ p["W1"] + p["b1"])
        h = np.tanh(h1 @ p["W2"] + p["b2"])
        z = h @ p["Wz"] + p["bz"]                      # future video, [-1,1]-ish
        if self.variant in {"direct", "joint"}:
            a = h @ p["Wa"] + p["ba"]
        else:
            q = np.tanh(z @ p["Wi1"] + p["bi1"])
            a = q @ p["Wi2"] + p["bi2"]
        cache = (o, h1, h)
        if self.variant == "idm":
            cache = cache + (q,)
        return {"h": h, "z": z, "a": a}, cache

    def predict(self, o):
        """Single-observation query interface (mirrors the attacker's QueryOutput)."""
        out, _ = self.forward(o.reshape(1, -1))
        a = out["a"][0].reshape(H_ACT, A_DIM)
        z = None if self.variant == "direct" else out["z"][0].reshape(K_FUT, FUT_IMG, FUT_IMG)
        return a, z

    # ---------------------------------------------------------- gradients
    def loss_and_grads(self, O, A_star, Z_star):
        """MSE on action chunk (+ MSE on future video unless direct). Full batch."""
        p = self.params
        B = O.shape[0]
        out, cache = self.forward(O)
        o, h1, h = cache[:3]
        n_a = H_ACT * A_DIM

        da = 2.0 * (out["a"] - A_star) / (B * n_a)
        dz = (2.0 * self.img_weight * (out["z"] - Z_star) / (B * Z_star.shape[1])
              if self.img_weight > 0 else np.zeros_like(out["z"]))

        loss = float(np.mean((out["a"] - A_star) ** 2))
        if self.img_weight > 0:
            loss += float(self.img_weight * np.mean((out["z"] - Z_star) ** 2))

        g = {}
        if self.variant in {"direct", "joint"}:
            g["Wa"] = h.T @ da
            g["ba"] = da.sum(0)
            dh = da @ p["Wa"].T
        else:
            q = cache[3]
            g["Wi2"] = q.T @ da
            g["bi2"] = da.sum(0)
            dq = da @ p["Wi2"].T
            dz = dz + (dq * (1 - q ** 2)) @ p["Wi1"].T
            g["Wi1"] = out["z"].T @ (dq * (1 - q ** 2))
            g["bi1"] = (dq * (1 - q ** 2)).sum(0)
            dh = np.zeros_like(h)

        g["Wz"] = h.T @ dz
        g["bz"] = dz.sum(0)
        dh = dh + dz @ p["Wz"].T
        dh1 = (dh * (1 - h ** 2)) @ p["W2"].T
        g["W2"] = h1.T @ (dh * (1 - h ** 2))
        g["b2"] = (dh * (1 - h ** 2)).sum(0)
        # loss_and_grads uses the centered input cached in `o`, so the
        # returned W1 gradient is w.r.t. centered input; the parameter
        # gradient is identical because sum_k dC_ik/do_j ... DC cancels.
        g["W1"] = o.T @ (dh1 * (1 - h1 ** 2))
        g["b1"] = (dh1 * (1 - h1 ** 2)).sum(0)
        return loss, g

    def train(self, O, A_star, Z_star, steps=3000, lr=1e-2, lr_min=1e-3,
              log_every=500, batch_size=None, seed=0):
        """Adam with cosine lr decay lr -> lr_min; full-batch or minibatch."""
        m = {k: np.zeros_like(v) for k, v in self.params.items()}
        v = {k: np.zeros_like(v) for k, v in self.params.items()}
        b1, b2, eps = 0.9, 0.999, 1e-8
        hist = []
        rng = np.random.default_rng(seed)
        n = O.shape[0]
        for t in range(1, steps + 1):
            frac = 0.5 * (1 + np.cos(np.pi * t / steps))
            lr_t = lr_min + (lr - lr_min) * frac
            if batch_size and batch_size < n:
                idx = rng.choice(n, batch_size, replace=False)
                loss, g = self.loss_and_grads(O[idx], A_star[idx], Z_star[idx])
            else:
                loss, g = self.loss_and_grads(O, A_star, Z_star)
            for k in self.params:
                m[k] = b1 * m[k] + (1 - b1) * g[k]
                v[k] = b2 * v[k] + (1 - b2) * g[k] ** 2
                mh = m[k] / (1 - b1 ** t)
                vh = v[k] / (1 - b2 ** t)
                self.params[k] -= lr_t * mh / (np.sqrt(vh) + eps)
            if t % log_every == 0 or t == 1:
                hist.append((t, loss))
        return hist

    # ---------------------------------------------------------- input jacobians
    def input_jacobians(self, o):
        """Exact d(a)/d(o) and d(z)/d(o) at a single observation o [IMG*IMG]."""
        p = self.params
        out, cache = self.forward(o.reshape(1, -1))
        _, h1, h = cache[:3]
        # d h / d o_c  (HID2 x n_in); o_c = centered input (cache stores it)
        dh1 = (1 - h1 ** 2)[:, :, None] * p["W1"].T[None]          # [1,HID1,n_in]
        Jh = ((1 - h ** 2)[:, :, None] * (p["W2"].T @ dh1[0])[None])[0]  # [HID2,n_in]
        # chain through input centering: d o_c / d o = I - 1/n (symmetric)
        Jh = Jh - Jh.mean(axis=1, keepdims=True)
        Jz = p["Wz"].T @ Jh                                        # [n_z,n_in]
        if self.variant in {"direct", "joint"}:
            Ja = p["Wa"].T @ Jh
        else:
            q = cache[3]
            Jq = (1 - q ** 2)[:, :, None] * (p["Wi1"].T @ Jz)[None]  # [1,HID2,n_in]
            Ja = p["Wi2"].T @ Jq[0]
        return Ja, Jz

    def input_grad_objective(self, o, a_clean, z_clean, lam):
        """d/do of [ ||a(o)-a_clean|| - lam*||z(o)-z_clean|| ] at o [IMG*IMG]."""
        out, _ = self.forward(o.reshape(1, -1))
        a = out["a"][0]
        da = a - a_clean.reshape(-1)
        na = np.linalg.norm(da) + 1e-12
        Ja, Jz = self.input_jacobians(o)
        g = Ja.T @ (da / na)
        if lam > 0 and self.variant != "direct":
            dz = out["z"][0] - z_clean.reshape(-1)
            nz = np.linalg.norm(dz) + 1e-12
            g = g - lam * (Jz.T @ (dz / nz))
        return g
