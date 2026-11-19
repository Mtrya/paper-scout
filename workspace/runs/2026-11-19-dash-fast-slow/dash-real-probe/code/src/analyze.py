"""Aggregate arm/seed results, paired contrasts over held-out episodes, feature-drift diagnostics, figures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

H = 8


def load_runs(out: Path, arms, seeds):
    rows, per_ep = [], []
    for arm in arms:
        for s in seeds:
            sp = out / "pred" / f"summary_{arm}_s{s}.json"
            if not sp.exists():
                continue
            summ = json.loads(sp.read_text())
            m = json.loads((out / "pred" / f"meta_{arm}_s{s}.json").read_text())
            task_arr = np.array(m["tasks"])
            anchors = np.array(m["anchors"])
            epstart = np.full(anchors.size, -1, np.int64)
            by_task = {}
            for e in m["episodes"]:
                by_task.setdefault(e["task"], []).append(e)
            for t, lst in by_task.items():
                lst.sort(key=lambda e: e["start"])
                starts = np.array([e["start"] for e in lst])
                sel = task_arr == t
                k = np.clip(np.searchsorted(starts, anchors[sel], side="right") - 1, 0, len(starts) - 1)
                epstart[sel] = starts[k]
            for key, val in summ.items():
                cond, d = key.rsplit("_d", 1)
                rows.append(dict(arm=arm, seed=s, cond=cond, delta=int(d), nmse=val["nmse"]))
                f = out / "pred" / f"{arm}_s{s}_{cond}_d{d}.npz"
                if f.exists():
                    z = np.load(f)
                    if z["err"].ndim == 1:                       # constant predictor: already averaged over dims
                        err_dim = None
                        err_scalar = z["err"]
                        e0_dim = None
                    else:
                        err_dim = z["err"]                       # (N, 13) squared error, mean over horizon
                        err_scalar = err_dim.mean(1)
                        e0_dim = z["err0"] if "err0" in z else None
                    for e in np.unique(epstart[epstart >= 0]):
                        m = epstart == e
                        if err_dim is None:
                            groups = dict(osc=np.nan, grip=np.nan, kp=np.nan, osc0=np.nan)
                        else:
                            groups = dict(osc=float(err_dim[m][:, :6].mean()), grip=float(err_dim[m][:, 6].mean()),
                                          kp=float(err_dim[m][:, 7:].mean()),
                                          osc0=float(e0_dim[m][:, :6].mean()) if e0_dim is not None else float("nan"))
                        per_ep.append(dict(arm=arm, seed=s, cond=cond, delta=int(d),
                                           task=str(task_arr[m][0]), ep_seed=int(e),
                                           nmse=float(err_scalar[m].mean()), **groups))
    return rows, per_ep


def agg(rows):
    out = {}
    for r in rows:
        out.setdefault((r["arm"], r["cond"], r["delta"]), []).append(r["nmse"])
    return {k: (float(np.mean(v)), float(np.std(v, ddof=1) if len(v) > 1 else 0.0), len(v)) for k, v in out.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--packed", required=True)
    ap.add_argument("--arms", nargs="+", default=["sync_nofresh", "sync_fresh", "desync8_nofresh", "desync8_fresh"])
    ap.add_argument("--extra-arms", nargs="*", default=["desync4_nofresh", "desync4_fresh", "leak_nofresh", "leak_fresh"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--figdir", default="")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "fig").mkdir(exist_ok=True)
    figdir = Path(args.figdir) if args.figdir else out / "fig"
    figdir.mkdir(parents=True, exist_ok=True)

    arms = [a for a in args.arms + list(args.extra_arms)]
    rows, per_ep = load_runs(out, arms, args.seeds)
    a = agg(rows)
    with open(out / "results_long.csv", "w") as f:
        f.write("arm,seed,cond,delta,nmse\n")
        for r in sorted(rows, key=lambda r: (r["arm"], r["cond"], r["delta"], r["seed"])):
            f.write(f"{r['arm']},{r['seed']},{r['cond']},{r['delta']},{r['nmse']:.6f}\n")
    grp = {}
    for r in per_ep:
        k = (r["arm"], r["cond"], r["delta"])
        grp.setdefault(k, {"osc": [], "grip": [], "kp": [], "osc0": []})
        for g in ("osc", "grip", "kp", "osc0"):
            grp[k][g].append(r.get(g, float("nan")))
    with open(out / "results_groups.csv", "w") as f:
        f.write("arm,cond,delta,osc,grip,kp,osc_step0,n_episode_seed_means\n")
        for k in sorted(grp):
            v = grp[k]
            f.write(f"{k[0]},{k[1]},{k[2]},{np.mean(v['osc']):.6f},{np.mean(v['grip']):.6f},"
                    f"{np.mean(v['kp']):.6f},{np.mean(v['osc0']):.6f},{len(v['osc'])}\n")
    with open(out / "results_agg.csv", "w") as f:
        f.write("arm,cond,delta,nmse_mean,nmse_sd,n_seeds\n")
        for k in sorted(a, key=lambda k: (k[0], k[1], k[2])):
            m, sd, n = a[k]
            f.write(f"{k[0]},{k[1]},{k[2]},{m:.6f},{sd:.6f},{n}\n")
    with open(out / "per_episode.csv", "w") as f:
        f.write("arm,seed,cond,delta,task,ep_seed,nmse,osc,grip,kp,osc0\n")
        for r in per_ep:
            f.write(f"{r['arm']},{r['seed']},{r['cond']},{r['delta']},{r['task']},{r['ep_seed']},"
                    f"{r['nmse']:.6f},{r['osc']:.6f},{r['grip']:.6f},{r['kp']:.6f},{r['osc0']:.6f}\n")

    # paired contrasts over episodes (mean over seeds), with bootstrap CI
    idx = {}
    for r in per_ep:
        idx.setdefault((r["arm"], r["cond"], r["delta"]), {})[(r["task"], r["ep_seed"])] = (
            idx.get((r["arm"], r["cond"], r["delta"]), {}).get((r["task"], r["ep_seed"]), []) + [r["nmse"]])
    contrasts = []
    rng = np.random.default_rng(0)

    def ep_map(arm, cond, d):
        m = {}
        for r in per_ep:
            if r["arm"] == arm and r["cond"] == cond and r["delta"] == d:
                m.setdefault((r["task"], r["ep_seed"]), []).append(r["nmse"])
        return {k: float(np.mean(v)) for k, v in m.items()}

    def contrast(name, x, y):
        keys = sorted(set(x) & set(y))
        if not keys:
            return
        diff = np.array([x[k] - y[k] for k in keys])
        bs = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)])
        contrasts.append(dict(name=name, n_episodes=len(keys), delta_mean=float(diff.mean()),
                              ci_lo=float(np.percentile(bs, 2.5)), ci_hi=float(np.percentile(bs, 97.5)),
                              rel=float(diff.mean() / np.mean([y[k] for k in keys]))))

    for d in (0, 1, 2, 4, 8):
        contrast(f"desync8-fresh_minus_sync-fresh_d{d}", ep_map("desync8_fresh", "main", d), ep_map("sync_fresh", "main", d))
        contrast(f"desync8-nofresh_minus_sync-nofresh_d{d}", ep_map("desync8_nofresh", "main", d),
                 ep_map("sync_nofresh", "main", d))
        contrast(f"fresh_minus_nofresh_sync_d{d}", ep_map("sync_fresh", "main", d), ep_map("sync_nofresh", "main", d))
        contrast(f"fresh_minus_nofresh_desync_d{d}", ep_map("desync8_fresh", "main", d),
                 ep_map("desync8_nofresh", "main", d))
    with open(out / "contrasts.csv", "w") as f:
        f.write("name,n_episodes,diff_mean,ci_lo,ci_hi,rel\n")
        for c in contrasts:
            f.write(f"{c['name']},{c['n_episodes']},{c['delta_mean']:.6f},{c['ci_lo']:.6f},{c['ci_hi']:.6f},{c['rel']:.4f}\n")

    # per-dimension NMSE (13 action dims) and physical RMSE for the OSC dims
    per_dim = {}
    for arm in arms:
        for s_ in args.seeds:
            for key in ("main_d0", "main_d8"):
                f = out / "pred" / f"{arm}_s{s_}_{key.split('_d')[0]}_d{key.split('_d')[1]}.npz"
                if not f.exists():
                    continue
                e = np.load(f)["err"].mean(0)
                per_dim.setdefault((arm, key), []).append(e)
    norm = json.loads((Path(args.packed) / "norm.json").read_text())
    act_std = np.array(norm["act_std"], np.float32)
    with open(out / "per_dim.csv", "w") as f:
        f.write("arm,cond_delta," + ",".join(f"d{j}" for j in range(13)) + "\n")
        for k in sorted(per_dim):
            f.write(f"{k[0]},{k[1]}," + ",".join(f"{v:.5f}" for v in np.mean(per_dim[k], 0)) + "\n")
    with open(out / "physical_rmse.csv", "w") as f:
        f.write("arm,cond_delta,pos_mm,rot_mrad,grip_raw\n")
        for k in sorted(per_dim):
            m = np.mean(per_dim[k], 0)                      # per-dim NMSE = per-dim MSE in normalised units
            rmse = np.sqrt(np.maximum(m, 0)) * act_std       # back to raw action units
            pos = float((rmse[:3] * 50.0).mean())           # 1.0 OSC position unit = 0.05 m
            rot = float((rmse[3:6] * 500.0).mean())         # 1.0 OSC rotation unit = 0.5 rad
            f.write(f"{k[0]},{k[1]},{pos:.3f},{rot:.2f},{rmse[6]:.4f}\n")

    # feature / image drift as a function of delta
    from packed import PackedSplit
    val = PackedSplit(Path(args.packed), "val")
    anchors = val.anchors(stride=8)
    rng2 = np.random.default_rng(3)
    sub = rng2.choice(anchors, size=min(600, anchors.size), replace=False)
    drift = []
    for d in (0, 1, 2, 4, 8):
        v0 = np.asarray(val.vis[sub]).astype(np.float32)
        v1 = np.asarray(val.vis[sub - d]).astype(np.float32)
        cos = (v0 * v1).sum(-1) / (np.linalg.norm(v0, axis=-1) * np.linalg.norm(v1, axis=-1) + 1e-8)
        i0 = np.asarray(val.img[sub]).astype(np.float32)
        i1 = np.asarray(val.img[sub - d]).astype(np.float32)
        pix = np.abs(i0 - i1).mean() / 255.0
        drift.append(dict(delta=d, vis_cos=float(cos.mean()), vis_cos_per_layer=cos.mean((0, 2)).tolist(),
                          img_mae=pix))
    with open(out / "drift.csv", "w") as f:
        f.write("delta,vis_cos,img_mae\n")
        for r in drift:
            f.write(f"{r['delta']},{r['vis_cos']:.5f},{r['img_mae']:.5f}\n")
    (out / "drift.json").write_text(json.dumps(drift, indent=1))

    # figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6))
    styles = {"sync_nofresh": ("#7a7a7a", "--o"), "sync_fresh": ("#1f77b4", "-o"),
              "desync8_nofresh": ("#d17b00", "--s"), "desync8_fresh": ("#2ca02c", "-s"),
              "desync4_nofresh": ("#d17b00", ":^"), "desync4_fresh": ("#2ca02c", ":^")}
    ax = axes[0]
    for arm, (c, st) in styles.items():
        xs, ys, es = [], [], []
        for d in (0, 1, 2, 4, 8):
            k = (arm, "main", d)
            if k in a:
                xs.append(d); ys.append(a[k][0]); es.append(a[k][1])
        if xs:
            ax.errorbar(xs, ys, yerr=es, color=c, fmt=st, capsize=2, label=arm)
    ax.set_xlabel("slow-context staleness Δ (frames @20 Hz)")
    ax.set_ylabel("open-loop chunk NMSE")
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.set_title("(a) dose-response to staleness")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)

    ax = axes[1]
    conds = ["main", "zero_vis", "shuf_vis", "zero_txt", "other_txt"]
    w = 0.8 / len(conds)
    for j, (arm, (c, st)) in enumerate(styles.items()):
        vals = [a.get((arm, cond, 8), (np.nan,))[0] for cond in conds]
        ax.bar(np.arange(len(conds)) + j * w - 0.4, vals, width=w, color=c, label=arm)
    ax.set_xticks(np.arange(len(conds)), conds, fontsize=7, rotation=15)
    ax.set_ylabel("NMSE at Δ=8")
    ax.set_title("(b) which channel carries the correction")
    ax.grid(alpha=0.25, axis="y")

    ax = axes[2]
    ds = [r["delta"] for r in drift]
    ax.plot(ds, [r["vis_cos"] for r in drift], "-o", color="#444", label="VLM visual-token cosine (mean layer)")
    ax.set_xlabel("Δ (frames)")
    ax.set_ylabel("cosine similarity")
    ax.set_title("(c) how stale is stale")
    for l in range(len(drift[0]["vis_cos_per_layer"])):
        ax.plot(ds, [r["vis_cos_per_layer"][l] for r in drift], lw=0.7, alpha=0.5)
    ax2 = ax.twinx()
    ax2.plot(ds, [r["img_mae"] for r in drift], "--s", color="#c44", label="per-pixel |Δimage|")
    ax2.set_ylabel("mean |Δpixel| / 255")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figdir / "fig_dash_probe.png", dpi=170)
    print("wrote", out / "results_agg.csv", "and", figdir / "fig_dash_probe.png")
    for k in sorted(a, key=lambda k: (k[0], k[1], k[2])):
        if k[1] == "main":
            print(f"  {k[0]:16s} Δ={k[2]}  NMSE={a[k][0]:.4f} ± {a[k][1]:.4f}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
