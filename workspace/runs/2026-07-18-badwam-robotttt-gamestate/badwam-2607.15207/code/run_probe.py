"""BadWAM toy probe: does world-action decoupling emerge in a minimal WAM?

Stages (run in order; models.npz must exist -- see train_models.py):
  python3 run_probe.py attacks <variant>   # closed-loop attack sweeps
  python3 run_probe.py mechanism           # input-Jacobian analysis
  python3 run_probe.py figures             # plots + (results.json written at each stage)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from toy_wam import (IMG, FUT_IMG, K_FUT, H_ACT, A_DIM, DT, KP, A_MAX, SUCCESS_RADIUS,
                     ToyWAM, render_scene, render_future, to_model_range,
                     expert_rollout, env_step)
from attacks import (QueryFn, query_search_attack, random_attack, whitebox_pgd,
                     d_act, d_img)
from train_models import load_models

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parent.parent / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)
RESULTS = HERE / "results.json"

EPS, SIGMA, STEP, ITERS = 0.06, 0.02, 0.02, 8   # paper appendix A.3 defaults
T_REPLAN = 20


# ------------------------------------------------------------------ data
def sample_scene(rng):
    while True:
        agent = rng.uniform(0.15, 0.85, 2)
        goal = rng.uniform(0.15, 0.85, 2)
        if np.linalg.norm(agent - goal) > 0.3:
            return agent, goal


def make_dataset(n, seed):
    """States sampled ALONG expert rollouts (like a real demonstration
    dataset), so all goal separations -- including the near-goal regime the
    closed loop must traverse -- are covered."""
    rng = np.random.default_rng(seed)
    O, A, Z = [], [], []
    while len(O) < n:
        agent = rng.uniform(0.1, 0.9, 2)
        goal = rng.uniform(0.1, 0.9, 2)
        if np.linalg.norm(agent - goal) < 0.15:
            continue
        for _ in range(int(rng.integers(0, 8))):
            a = np.clip(KP * (goal - agent), -A_MAX, A_MAX)
            agent = agent + a * DT
        if np.linalg.norm(agent - goal) < 0.03 or not np.all((agent > 0.02) & (agent < 0.98)):
            continue
        acts, positions = expert_rollout(agent, goal)
        O.append(to_model_range(render_scene(agent, goal)).reshape(-1))
        A.append(acts.reshape(-1))
        Z.append(to_model_range(render_future(positions, goal)).reshape(-1))
    return np.array(O), np.array(A), np.array(Z)


# ------------------------------------------------------------------ eval
def run_episode(model, agent0, goal, attack=None, lam=0.0, seed=0, log_chunks=False,
                task="grasp"):
    """Closed loop, receding horizon.
    attack: None | 'random' | 'spsa' (paper recipe, m=1) | 'spsa_hi' (m=16) | 'whitebox'.
    task='grasp': the agent must close the gripper inside GRIP_RADIUS of the
    goal (a close outside is an irreversible mis-grasp -> failure), then hold
    position.  This mirrors real manipulation fragility: some failures cannot
    be undone by the next replan, unlike pure reaching."""
    from toy_wam import GRIP_RADIUS
    p = np.array(agent0, dtype=np.float64)
    logs = {"d_act": [], "d_img": [], "chan": [], "horizon": [], "traj": [p.copy()],
            "queries": 0}
    qf = QueryFn(model)
    grasped = False
    failed = False
    for t in range(T_REPLAN):
        o = to_model_range(render_scene(p, goal))
        o_flat = o.reshape(-1)
        clean_a, clean_z = model.predict(o_flat)
        if attack in {"spsa", "spsa_hi"}:
            delta, info = query_search_attack(model, o, eps=EPS, sigma=SIGMA, step=STEP,
                                              iters=ITERS, lam=lam, seed=seed * 1000 + t,
                                              query_fn=qf, n_dir=1 if attack == "spsa" else 16)
        elif attack == "whitebox":
            delta, info = whitebox_pgd(model, o, eps=EPS, lam=lam, seed=seed * 1000 + t)
        elif attack == "random":
            delta = random_attack(model, o, eps=EPS, seed=seed * 1000 + t)
        else:
            delta = np.zeros_like(o_flat)
        adv_a, adv_z = model.predict(np.clip(o_flat + delta, -1, 1))
        logs["d_act"].append(d_act(adv_a, clean_a))
        logs["d_img"].append(d_img(adv_z, clean_z))
        if log_chunks:
            logs["chan"].append(np.abs(adv_a - clean_a).mean(0))               # [A_DIM]
            logs["horizon"].append(np.linalg.norm(adv_a - clean_a, axis=-1))   # [H_ACT]
        a0 = clean_a[0] if attack is None else adv_a[0]
        p = env_step(p, a0, goal)
        logs["traj"].append(p.copy())
        if task == "grasp":
            dist = np.linalg.norm(p - goal)
            grip_close = a0[2] > 0.0
            if not grasped and grip_close:
                if dist < GRIP_RADIUS:
                    grasped = True
                else:
                    failed = True
                    break
            if grasped and dist > 2 * GRIP_RADIUS:
                failed = True
                break
    if task == "grasp":
        success = grasped and not failed
    else:
        success = bool(np.linalg.norm(p - goal) < SUCCESS_RADIUS)
    logs["success"] = bool(success)
    logs["final_dist"] = float(np.linalg.norm(p - goal))
    logs["queries"] = qf.n_queries
    return logs


def sweep(model, variant, configs, log_chunks=False):
    out = {}
    for name, attack, lam, n_episodes in configs:
        t0 = time.time()
        eps_logs = []
        for ep in range(n_episodes):
            rng = np.random.default_rng(10_000 + ep)
            agent, goal = sample_scene(rng)
            eps_logs.append(run_episode(model, agent, goal, attack=attack, lam=lam,
                                        seed=ep, log_chunks=log_chunks))
        succ = np.mean([l["success"] for l in eps_logs])
        da = np.mean([np.mean(l["d_act"]) for l in eps_logs])
        dz = np.mean([np.mean(l["d_img"]) for l in eps_logs])
        rec = {"attack": name, "success": float(succ), "d_act": float(da),
               "d_img": float(dz), "decoupling": float(da / (dz + 1e-6)),
               "queries_per_replan": float(np.mean([l["queries"] / T_REPLAN for l in eps_logs])),
               "wall_s": float(time.time() - t0)}
        if log_chunks:
            rec["chan_shift"] = np.mean([np.mean(l["chan"], 0) for l in eps_logs], 0).tolist()
            rec["horizon_shift"] = np.mean([np.mean(l["horizon"], 0) for l in eps_logs], 0).tolist()
        if name == "spsahi_lam0.1":
            rec["per_episode"] = [{"d_act": float(np.mean(l["d_act"])),
                                   "d_img": float(np.mean(l["d_img"])),
                                   "success": l["success"]} for l in eps_logs]
        out[name] = rec
        print(f"  [{variant:6s}] {name:22s} succ={succ:5.1%}  D_act={da:7.4f}  "
              f"D_img={dz:7.3f}  decpl={rec['decoupling']:8.5f}  ({rec['wall_s']:.0f}s)",
              flush=True)
    return out


def load_results():
    if RESULTS.exists():
        return json.load(open(RESULTS))
    return {"config": {"eps": EPS, "sigma": SIGMA, "step": STEP, "iters": ITERS,
                       "t_replan": T_REPLAN}}


# ------------------------------------------------------------------ stages
def stage_attacks(variant, group):
    results = load_results()
    models = load_models()
    m = models[variant]
    N_LO, N_HI = 80, 40
    if group == "basic":
        configs = [("clean", None, 0.0, N_LO), ("random", "random", 0.0, N_LO),
                   ("spsa_lam0", "spsa", 0.0, N_LO)]
        if variant != "direct":
            configs += [("spsa_lam0.1", "spsa", 0.1, N_LO), ("spsa_lam1", "spsa", 1.0, N_LO)]
    elif group == "hi":  # high-budget black-box (m=16 directions/update)
        configs = [("spsahi_lam0", "spsa_hi", 0.0, N_HI)]
        if variant != "direct":
            configs += [(f"spsahi_lam{lam:g}", "spsa_hi", lam, N_HI)
                        for lam in (0.03, 0.1, 0.3, 1.0)]
    elif group == "wb":  # white-box gradient reference
        configs = [("wb_lam0", "whitebox", 0.0, N_HI)]
        if variant != "direct":
            configs += [(f"wb_lam{lam:g}", "whitebox", lam, N_HI)
                        for lam in (0.01, 0.03, 0.1, 0.3, 1.0)]
    else:
        raise SystemExit(f"unknown group {group}")
    results.setdefault(variant, {}).update(
        sweep(m, variant, configs,
              log_chunks=(variant != "direct" and group in {"basic", "hi"})))
    json.dump(results, open(RESULTS, "w"), indent=1)
    print("wrote", RESULTS)


def stage_mechanism():
    results = load_results()
    models = load_models()
    mech = {}
    for variant in ["joint", "idm"]:
        m = models[variant]
        pr_a, pr_z, cross, invis = [], [], [], []
        amp = {"spsa_a": [], "spsa_z": [], "rand_a": [], "rand_z": [],
               "spsa_ip_a": [], "spsa_ip_z": []}
        spec_a, spec_z, k = None, None, None
        for i in range(24):
            rng = np.random.default_rng(20_000 + i)
            agent, goal = sample_scene(rng)
            o = to_model_range(render_scene(agent, goal)).reshape(-1)
            Ja, Jz = m.input_jacobians(o)
            sa = np.linalg.svd(Ja, compute_uv=False)
            sz = np.linalg.svd(Jz, compute_uv=False)
            spec_a, spec_z = sa, sz
            pr_a.append((sa ** 2).sum() ** 2 / (sa ** 4).sum())
            pr_z.append((sz ** 2).sum() ** 2 / (sz ** 4).sum())
            Vz = np.linalg.svd(Jz, full_matrices=False)[2]
            k = int(np.argmax(np.cumsum(sz ** 2) / (sz ** 2).sum() > 0.99)) + 1
            Proj = Vz[:k].T @ Vz[:k]
            invis.append(float(1 - np.linalg.norm(Ja @ Proj, "fro") ** 2
                               / np.linalg.norm(Ja, "fro") ** 2))
            cross.append(float(np.linalg.norm(Ja @ Vz[:k].T, "fro") / np.linalg.norm(Ja, "fro")))
            d_spsa, _ = query_search_attack(m, o.reshape(IMG, IMG), eps=EPS, sigma=SIGMA,
                                            step=STEP, iters=ITERS, lam=0.0, seed=i, n_dir=16)
            d_ip, _ = query_search_attack(m, o.reshape(IMG, IMG), eps=EPS, sigma=SIGMA,
                                          step=STEP, iters=ITERS, lam=0.1, seed=10_000 + i,
                                          n_dir=16)
            d_rand = random_attack(m, o.reshape(IMG, IMG), eps=EPS, seed=i)
            for key, d in (("spsa", d_spsa), ("spsa_ip", d_ip), ("rand", d_rand)):
                n = np.linalg.norm(d) + 1e-12
                amp[f"{key}_a"].append(float(np.linalg.norm(Ja @ d) / n))
                amp[f"{key}_z"].append(float(np.linalg.norm(Jz @ d) / n))
        mech[variant] = {
            "pr_action": float(np.mean(pr_a)), "pr_imagination": float(np.mean(pr_z)),
            "k99_img": k,
            "frac_action_sensitivity_outside_img_topk": float(np.mean(invis)),
            "cross_sensitivity": float(np.mean(cross)),
            "amp_action_per_unit_norm": {k2: float(np.mean(v)) for k2, v in amp.items()
                                         if k2.endswith("_a")},
            "amp_img_per_unit_norm": {k2: float(np.mean(v)) for k2, v in amp.items()
                                      if k2.endswith("_z")},
            "spec_action": spec_a.tolist(), "spec_img_first64": spec_z[:64].tolist(),
        }
        print(f"  {variant:6s} PR(Ja)={mech[variant]['pr_action']:.1f} "
              f"PR(Jz)={mech[variant]['pr_imagination']:.1f} "
              f"k99={k} "
              f"invisible-action-sens={mech[variant]['frac_action_sensitivity_outside_img_topk']:.1%}",
              flush=True)
    results["mechanism"] = mech
    json.dump(results, open(RESULTS, "w"), indent=1)
    print("wrote", RESULTS)


def stage_scatter():
    """Per-episode (d_img, d_act, success) under the imagination-preserving
    WHITE-BOX attack (lam=0.1) -- the only toy attack strong enough to produce
    both outcomes; used for the paper-Fig.-1 analog scatter."""
    results = load_results()
    models = load_models()
    for variant in ["joint", "idm"]:
        m = models[variant]
        per = []
        for ep in range(60):
            rng = np.random.default_rng(30_000 + ep)
            agent, goal = sample_scene(rng)
            l = run_episode(m, agent, goal, attack="whitebox", lam=0.1, seed=ep)
            per.append({"d_act": float(np.mean(l["d_act"])),
                        "d_img": float(np.mean(l["d_img"])),
                        "success": l["success"]})
        results.setdefault(variant, {})["wb_scatter"] = {"per_episode": per}
        n_fail = sum(1 for p in per if not p["success"])
        print(f"  {variant}: {n_fail}/60 failures logged", flush=True)
    json.dump(results, open(RESULTS, "w"), indent=1)
    print("wrote", RESULTS)


def stage_figures():
    results = load_results()
    models = load_models()
    make_figures(results, models)


def make_figures(results, models):
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 150})
    colors = {"joint": "#C44E52", "idm": "#4C72B0"}

    def lam_sweep(r, prefix):
        pts = []
        for key in r:
            if key.startswith(prefix + "_lam"):
                lam = float(key[len(prefix) + 4:])
                pts.append((lam, r[key]["success"], r[key]["d_act"], r[key]["d_img"]))
        pts.sort()
        return [np.array([p[i] for p in pts]) for i in range(4)]

    # --- Fig P1: lambda sweep (paper Fig 13 analog) + Pareto frontier -----
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.3))
    ax0b = axes[0].twinx()
    ax0b.spines["right"].set_visible(True)
    for variant in ["joint", "idm"]:
        r = results[variant]
        for prefix, style, lab in (("spsahi", "o-", "black-box m=16"), ("wb", "^--", "white-box")):
            lams, succ, da, dz = lam_sweep(r, prefix)
            if len(lams) == 0:
                continue
            axes[0].semilogx(np.maximum(lams, 1e-4), 100 * succ, style, color=colors[variant],
                             alpha=0.9 if prefix == "spsahi" else 0.55,
                             label=f"{variant} {lab}")
            if prefix == "spsahi":
                ax0b.semilogx(np.maximum(lams, 1e-4), da / (dz + 1e-6), "o:", ms=3,
                              color=colors[variant], alpha=0.6)
            axes[1].plot(dz, da, style, color=colors[variant],
                         alpha=0.9 if prefix == "spsahi" else 0.55, label=f"{variant} {lab}")
        axes[1].plot([r["random"]["d_img"]], [r["random"]["d_act"]], "s", color="gray")
        axes[1].plot([r["spsa_lam0"]["d_img"]], [r["spsa_lam0"]["d_act"]], "P", ms=9,
                     color=colors[variant], mec="k")
    ax0b.set_ylabel(r"decoupling $D_{act}/D_{img}$ (dotted, black-box)")
    axes[1].plot([], [], "s", color="gray", label="random noise")
    axes[1].plot([], [], "P", ms=9, color="k", label="paper recipe (m=1)")
    axes[0].set_xlabel(r"future-preserving weight $\lambda$")
    axes[0].set_ylabel("closed-loop success (%)")
    axes[0].set_title("Attack strength vs. preservation weight")
    axes[0].legend(frameon=False, fontsize=7)
    axes[1].set_xlabel(r"imagination drift $D_{\mathrm{img}}$ (L2, future video)")
    axes[1].set_ylabel(r"action drift $D_{\mathrm{act}}$ (L2, action chunk)")
    axes[1].set_title("World-action decoupling frontier")
    axes[1].legend(frameon=False, fontsize=7)
    fig.suptitle("BadWAM toy probe: action-only ($\\lambda$=0) vs imagination-preserving attacks", y=1.02)
    fig.tight_layout()
    fig.savefig(ASSETS / "probe_lambda_frontier.png", bbox_inches="tight")
    plt.close(fig)

    # --- Fig P2: per-episode scatter, paper Fig 1 analog -------------------
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), sharey=True)
    for ax, variant in zip(axes, ["joint", "idm"]):
        per = results[variant]["wb_scatter"]["per_episode"]
        for ok, col, lab in ((False, "#DD8452", "failure"), (True, "#8172B3", "success")):
            xs = [p["d_img"] for p in per if p["success"] == ok]
            ys = [p["d_act"] for p in per if p["success"] == ok]
            ax.scatter(xs, ys, s=14, alpha=0.75, color=col,
                       label=f"{lab} (n={len(xs)})")
        ax.set_xlabel(r"episode-mean predicted-future distance $D_{\mathrm{img}}$")
        ax.set_title(f"{variant} WAM, img-preserving attack ($\\lambda$=0.1)")
        ax.legend(frameon=False)
    axes[0].set_ylabel(r"episode-mean action distance $D_{\mathrm{act}}$")
    fig.suptitle("Toy analog of paper Fig. 1: action shifts predict failure; future shifts overlap", y=1.02)
    fig.tight_layout()
    fig.savefig(ASSETS / "probe_desync_scatter.png", bbox_inches="tight")
    plt.close(fig)

    # --- Fig P3: mechanism --------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.3))
    for variant, col in colors.items():
        spec = np.array(results["mechanism"][variant]["spec_img_first64"])
        axes[0].semilogy(spec / spec[0], color=col, label=f"{variant} $J_z$")
        spec_a = np.array(results["mechanism"][variant]["spec_action"])
        axes[0].semilogy(np.arange(len(spec_a)) * 5, spec_a / spec_a[0], "o--", ms=3,
                         color=col, alpha=0.6, label=f"{variant} $J_a$")
    axes[0].set_xlabel("singular index (action spectrum x5 for visibility)")
    axes[0].set_ylabel("normalized singular value")
    axes[0].set_title("Input sensitivity spectra are low-rank")
    axes[0].legend(frameon=False, fontsize=7)
    x = np.arange(3)
    w = 0.35
    for i, variant in enumerate(["joint", "idm"]):
        mech = results["mechanism"][variant]
        amps = [mech["amp_action_per_unit_norm"]["rand_a"] / (mech["amp_img_per_unit_norm"]["rand_z"] + 1e-12),
                mech["amp_action_per_unit_norm"]["spsa_a"] / (mech["amp_img_per_unit_norm"]["spsa_z"] + 1e-12),
                mech["amp_action_per_unit_norm"]["spsa_ip_a"] / (mech["amp_img_per_unit_norm"]["spsa_ip_z"] + 1e-12)]
        axes[1].bar(x + i * w - w / 2, amps, w, color=colors[variant], label=f"{variant} WAM")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["random $\\delta$", "action-only", "img-preserving"])
    axes[1].set_ylabel(r"$(\|J_a\delta\|/\|\delta\|)\,/\,(\|J_z\delta\|/\|\delta\|)$")
    axes[1].set_title("Attack steers into action-sensitive,\nimagination-insensitive directions")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(ASSETS / "probe_mechanism.png", bbox_inches="tight")
    plt.close(fig)

    # --- Fig P4: channel / horizon profiles (paper Fig 2 analog) -----------
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2))
    chan_labels = ["dx", "dy", "gripper"]
    for i, variant in enumerate(["joint", "idm"]):
        prof = results[variant]["spsahi_lam0"]["chan_shift"]
        axes[0].bar(np.arange(3) + i * 0.35 - 0.175, prof, 0.35,
                    color=colors[variant], label=f"{variant} WAM")
        hor = results[variant]["spsahi_lam0"]["horizon_shift"]
        axes[1].plot(np.arange(len(hor)), hor, "o-", color=colors[variant], label=f"{variant} WAM")
    axes[0].set_xticks(np.arange(3)); axes[0].set_xticklabels(chan_labels)
    axes[0].set_ylabel("mean |Δaction| per channel")
    axes[0].set_title("Channel-level shift (action-only attack)")
    axes[0].legend(frameon=False)
    axes[1].set_xlabel("chunk step"); axes[1].set_ylabel("L2 shift per chunk step")
    axes[1].set_title("Horizon-level shift")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(ASSETS / "probe_channels.png", bbox_inches="tight")
    plt.close(fig)

    # --- Fig P5: qualitative episode (paper Fig 4 analog) -------------------
    m = models["joint"]
    rng = np.random.default_rng(777)  # seed 0: clean succeeds, attack fails
    agent, goal = sample_scene(rng)
    clean_log = run_episode(m, agent, goal, attack=None, seed=0)
    adv_log = run_episode(m, agent, goal, attack="whitebox", lam=0.1, seed=0)
    fig = plt.figure(figsize=(8.6, 3.6))
    ax1 = fig.add_subplot(1, 3, 1)
    scene = render_scene(agent, goal)
    ax1.imshow(scene, cmap="viridis", origin="lower", extent=[0, 1, 0, 1])
    ctraj = np.array(clean_log["traj"]); atraj = np.array(adv_log["traj"])
    ax1.plot(ctraj[:, 0], ctraj[:, 1], "o-", color="white", ms=3, lw=1.2,
             label=f"clean ({'success' if clean_log['success'] else 'fail'})")
    ax1.plot(atraj[:, 0], atraj[:, 1], "s--", color="#FF6E6E", ms=3, lw=1.2,
             label=f"img-pres. attack ({'success' if adv_log['success'] else 'fail'})")
    ax1.plot(*goal, "*", color="gold", ms=14)
    ax1.legend(frameon=False, fontsize=7, loc="upper right")
    ax1.set_title("Closed-loop trajectories (grasp task)")
    o0 = to_model_range(render_scene(agent, goal))
    clean_a, clean_z = m.predict(o0.reshape(-1))
    delta, info = whitebox_pgd(m, o0, eps=EPS, lam=0.1, seed=5)
    adv_a, adv_z = m.predict(np.clip(o0.reshape(-1) + delta, -1, 1))
    ax2 = fig.add_subplot(1, 3, 2)
    strip_c = np.concatenate([(clean_z[k] + 1) / 2 for k in range(K_FUT)], axis=1)
    strip_a = np.concatenate([(adv_z[k] + 1) / 2 for k in range(K_FUT)], axis=1)
    ax2.imshow(np.concatenate([strip_c, strip_a], axis=0), cmap="viridis", origin="lower")
    ax2.set_title(f"Imagined futures (clean top, adv bottom)\n$D_{{act}}$={d_act(adv_a, clean_a):.3f}  $D_{{img}}$={d_img(adv_z, clean_z):.3f}")
    ax2.set_xticks([]); ax2.set_yticks([])
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(np.abs(strip_a - strip_c) * 8, cmap="magma", origin="lower")
    ax3.set_title("abs diff ×8 (paper Fig. 4 style)")
    ax3.set_xticks([]); ax3.set_yticks([])
    fig.tight_layout()
    fig.savefig(ASSETS / "probe_qualitative.png", bbox_inches="tight")
    plt.close(fig)
    print("figures saved to", ASSETS)


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage == "attacks":
        stage_attacks(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "basic")
    elif stage == "mechanism":
        stage_mechanism()
    elif stage == "scatter":
        stage_scatter()
    elif stage == "figures":
        stage_figures()
    else:
        raise SystemExit("usage: run_probe.py [attacks VARIANT GROUP|mechanism|scatter|figures]")
