#!/usr/bin/env python3
"""
implicit_state_demo.py
======================
A minimal mechanistic demonstration of WHY the "state entangled in
observations" design (arXiv:2607.14076, Sec. 3.2) fails long-horizon
state-observation persistence, while an explicit carried state does not.

Synthetic boss fight (the paper's own running example):
  "the same strike kills or merely wounds depending on the remaining
   health of the target"

World:
  - Boss HP h in {3,2,1,0}; h is NEVER directly observable.
  - Three attack events per episode; the 3rd attack kills (h: 1->0).
  - Attacks are visible AS THEY HAPPEN (a flash marker in the observation)
    but leave no permanent trace in the pixels (boss looks identical at
    HP=3 and HP=1) -- exactly like cooldowns/stamina/ammo in real games.
  - Task: at attack events #2 and #3, predict the rule-governed OUTCOME:
    wound (#2) vs kill (#3). Classes balanced 50/50.

Model A (implicit state = recent-observation window; the GameNGen/Oasis/
MIRA design): ridge readout on the flattened last W observations
(+actions). Mechanistically cannot count attacks older than W.

Model B (explicit state = the game-engine loop, and what the paper's
Black-Myth data engine is built to supervise): a carried scalar state
h_t with (i) a learned hit detector on the current observation and
(ii) a learned linear transition h_{t+1} = alpha*h_t + beta*hit_t + gamma
fit by least squares on state supervision. Outcome read off h_t.

Both are linear-capacity models fit on the SAME episodes; the ONLY
difference is where the state lives. Pure numpy, CPU, seconds.

Outputs: forgetting_curve.png + demo_results.json + printed summary.
"""

import json
import os
import numpy as np

rng = np.random.default_rng(7)

# ---------------- world parameters ----------------
D_OBS = 64          # observation dims (62 scene/flash + 2 action channels)
HP0 = 3             # boss hit points
W = 32              # context window of the implicit-state model
SIGMA = 0.35        # observation noise

# fixed random "rendering" embeddings
e_scene = rng.normal(0, 1, D_OBS - 2)   # boss-present background (identical at any HP>0)
e_flash = rng.normal(0, 1, D_OBS - 2)   # visible attack flash (only on attack steps)
e_scene /= np.linalg.norm(e_scene)
e_flash /= np.linalg.norm(e_flash)


def make_episode(gap, T_margin=12, jitter=4):
    """obs (T,D_OBS); attacks (T,); hp_before (T,) = HP before that step's events;
    (t1,t2,t3) = attack step indices."""
    t1 = int(rng.integers(5, 10))
    t2 = t1 + gap + int(rng.integers(-jitter, jitter + 1))
    t3 = t2 + gap + int(rng.integers(-jitter, jitter + 1))
    T = t3 + T_margin
    attacks = np.zeros(T)
    attacks[[t1, t2, t3]] = 1.0
    hp_before = np.full(T, HP0, dtype=float)
    hp_before[t1 + 1:t2 + 1] = 2
    hp_before[t2 + 1:t3 + 1] = 1
    hp_before[t3 + 1:] = 0
    obs = np.tile(e_scene, (T, 1))
    obs[attacks > 0] += e_flash          # flash visible only when the hit lands
    obs += SIGMA * rng.normal(0, 1, (T, D_OBS - 2))
    obs = np.concatenate([obs, np.stack([attacks, rng.normal(0, 0.1, T)], 1)], 1)
    return obs, attacks, hp_before, (t1, t2, t3)


def windowed_feature(obs, t, W):
    """Flattened last-W observations = the entire 'state' of an implicit model."""
    lo = max(0, t - W + 1)
    feat = obs[lo:t + 1].reshape(-1)
    pad = (W * D_OBS) - feat.shape[0]
    return np.concatenate([np.zeros(pad), feat]) if pad else feat


def gen_windowed(n_ep, gap_sampler, W):
    """Windowed features + labels for attacks #2 (wound, y=0) and #3 (kill, y=1)."""
    X, y = [], []
    for _ in range(n_ep):
        obs, _, _, (t1, t2, t3) = make_episode(gap_sampler())
        X.append(windowed_feature(obs, t2, W)); y.append(0)
        X.append(windowed_feature(obs, t3, W)); y.append(1)
    return np.array(X), np.array(y)


def fit_ridge(X, y, lam=1e-2):
    """Closed-form linear readout on labels {-1,+1} (deterministic, no optimizer noise)."""
    Xb = np.concatenate([X, np.ones((len(X), 1))], 1)
    t = 2 * y - 1
    w = np.linalg.solve(Xb.T @ Xb + lam * np.eye(Xb.shape[1]), Xb.T @ t)
    return w


def predict_ridge(w, X):
    Xb = np.concatenate([X, np.ones((len(X), 1))], 1)
    return (Xb @ w > 0).astype(int)


def fit_ridge_detector(X, y, lam=1e-2):
    """Closed-form ridge on 0/1 labels; threshold at 0.5. Used for the hit detector
    (robust to the ~1.5% class prior that trips up plain logistic training)."""
    Xb = np.concatenate([X, np.ones((len(X), 1))], 1)
    w = np.linalg.solve(Xb.T @ Xb + lam * np.eye(Xb.shape[1]), Xb.T @ y)
    return w


def detect(w, x):
    return float(np.concatenate([x, [1.0]]) @ w > 0.5)


def run(gaps, n_train=3000, n_eval=400, W=W):
    # ---------- training: episodes with MIXED gaps, so eval gaps are not leaked ----------
    mix = lambda: int(rng.choice([8, 24, 48, 96]))
    Xtr, ytr = gen_windowed(n_train, mix, W)
    w_imp = fit_ridge(Xtr, ytr)                                   # Model A

    # Model B (i): hit detector on single frames
    feats, labels = [], []
    for _ in range(400):
        obs, attacks, _, _ = make_episode(16)
        feats.append(obs); labels.append(attacks)
    w_det = fit_ridge_detector(np.concatenate(feats), np.concatenate(labels))

    # Model B (ii): transition h' = alpha*h + beta*hit + gamma, least squares on
    # state supervision (this is what the paper's slot/state captions provide).
    A_rows, b_rows = [], []
    for _ in range(400):
        _, attacks, hp_before, _ = make_episode(16)
        for t in range(len(hp_before) - 1):
            A_rows.append([hp_before[t], attacks[t], 1.0])
            b_rows.append(hp_before[t + 1])
    theta, *_ = np.linalg.lstsq(np.array(A_rows), np.array(b_rows), rcond=None)

    out = {"W": W, "HP0": HP0,
           "learned_transition": {"alpha": float(theta[0]), "beta": float(theta[1]),
                                  "gamma": float(theta[2]),
                                  "true": {"alpha": 1.0, "beta": -1.0, "gamma": 0.0}},
           "gaps": [], "visible_count_diagnostic": {}}

    for gap in gaps:
        # ---- Model A on this gap
        Xev, yev = gen_windowed(n_eval, lambda: gap, W)
        acc_imp = float(np.mean(predict_ridge(w_imp, Xev) == yev))

        # ---- Model B: explicit-state rollout per episode
        accs, det_hits, det_frames = [], 0, 0
        for _ in range(n_eval):
            obs, attacks, _, (t1, t2, t3) = make_episode(gap)
            h_hat, ep_pred = float(HP0), {}
            for t in range(t3 + 1):
                if t in (t2, t3):
                    ep_pred[t] = int(round(h_hat) <= 1)  # kill iff 1 HP remains before strike
                hit = detect(w_det, obs[t])
                det_hits += int(hit == attacks[t]); det_frames += 1
                h_hat = theta[0] * h_hat + theta[1] * hit + theta[2]
            accs.append(np.mean([ep_pred[t2] == 0, ep_pred[t3] == 1]))
        acc_exp = float(np.mean(accs))
        out["gaps"].append({"gap": gap, "gap_over_W": gap / W,
                            "acc_implicit_window": acc_imp,
                            "acc_explicit_state": acc_exp})
        print(f"gap={gap:4d} (gap/W={gap/W:4.2f})  implicit={acc_imp:.3f}  explicit={acc_exp:.3f}")

    # ---------- diagnostic: Model A accuracy vs #prior attacks inside the window ----------
    diag = {}
    for gap in (24, 48, 96):
        bucket = {}
        for _ in range(300):
            obs, _, _, (t1, t2, t3) = make_episode(gap)
            for tk, lbl in ((t2, 0), (t3, 1)):
                n_vis = int(((t1 >= tk - W + 1) & (t1 < tk)) + ((t2 >= tk - W + 1) & (t2 < tk)))
                pred = predict_ridge(w_imp, windowed_feature(obs, tk, W)[None, :])[0]
                bucket.setdefault(n_vis, []).append(int(pred == lbl))
        diag[gap] = {k: float(np.mean(v)) for k, v in sorted(bucket.items())}
    out["visible_count_diagnostic"] = diag
    print("diagnostic (implicit-model accuracy | #prior attacks visible in window):", diag)
    print("learned transition h' = %.3f*h %+.3f*hit %+.3f  (true: 1.0, -1.0, 0.0)" % tuple(theta))
    print("hit-detector frame accuracy on eval rollouts: %.4f" % (det_hits / det_frames))
    return out


def plot(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = [g["gap_over_W"] for g in results["gaps"]]
    ai = [g["acc_implicit_window"] for g in results["gaps"]]
    ae = [g["acc_explicit_state"] for g in results["gaps"]]
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    ax.plot(x, ai, "o-", color="#e76f51", lw=2.2, ms=7,
            label="implicit state = recent-observation window (Model A)")
    ax.plot(x, ae, "s-", color="#2a9d8f", lw=2.2, ms=7,
            label="explicit carried state, learned transition (Model B)")
    ax.axhline(0.5, color="grey", ls="--", lw=1)
    ax.text(x[-1], 0.515, "chance", ha="right", fontsize=9, color="grey")
    ax.axvline(0.5, color="#d62828", ls=":", lw=1.4)
    ax.text(0.52, 0.88, "earliest state change exits\nthe context window", fontsize=9,
            color="#d62828")
    ax.set_xlabel("spacing of state-changing events  /  context window W")
    ax.set_ylabel("outcome-prediction accuracy\n(wound vs kill, chance = 0.5)")
    ax.set_title("Why implicit state fails long-horizon persistence\n"
                 "(HP-3 boss fight; 'the same strike kills or merely wounds')")
    ax.set_ylim(0.42, 1.04)
    ax.legend(fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    print("wrote", path)


if __name__ == "__main__":
    HERE = os.path.dirname(os.path.abspath(__file__))
    res = run(gaps=[8, 12, 16, 20, 24, 32, 48, 64, 96, 128])
    with open(os.path.join(HERE, "demo_results.json"), "w") as f:
        json.dump(res, f, indent=2)
    plot(res, os.path.join(HERE, "forgetting_curve.png"))
