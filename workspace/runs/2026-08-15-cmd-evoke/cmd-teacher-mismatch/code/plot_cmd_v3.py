import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, json, sys
sys.path.insert(0, "code/cmd-probe")
from cmd_teacher_mismatch import make_world, controls, distill, deploy_err, acausal_ratio
from cmd_prefix_ablation import distill_prefix, gen_true

c_tr, _ = controls(32, 4000, seed=1)
c_te, _ = controls(32, 2000, seed=2)
ars = [0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95]
ratios = [acausal_ratio(*make_world(d=8, ar=a, sigma=0.3, seed=10)[:2], 0.3) for a in ars]

# panel b: 相同训练预算(n_iter)下 bidir-causal 部署误差差(瞬态代价)
iters = [1, 2, 4, 8, 16, 32]
diff_by_ar = {}
for ar in [0.7, 0.8, 0.9]:
    A, B = make_world(d=8, ar=ar, sigma=0.3, seed=10)
    ds = []
    for k in iters:
        Wc, Uc = distill(A, B, 0.3, c_tr, "causal", n_iter=k, seed=3)
        Wb, Ub = distill(A, B, 0.3, c_tr, "bidir", n_iter=k, seed=3)
        ec = deploy_err(A, B, 0.3, c_te, Wc, Uc, seed=4)
        eb = deploy_err(A, B, 0.3, c_te, Wb, Ub, seed=4)
        ds.append(eb - ec)
    diff_by_ar[ar] = ds

# panel c: prefix 对照(重跑)
ars2 = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
errs = {"base": [], "prefix": []}
for a in ars2:
    A, B = make_world(d=8, ar=a, sigma=0.3, seed=10)
    x_tr = gen_true(A, B, 0.3, c_tr, seed=6)
    Wba, Uba = distill_prefix(A, B, 0.3, c_tr, x_tr, "base", seed=3)
    Wp, Up = distill_prefix(A, B, 0.3, c_tr, x_tr, "prefix", seed=3)
    errs["base"].append(deploy_err(A, B, 0.3, c_te, Wba, Uba, seed=4))
    errs["prefix"].append(deploy_err(A, B, 0.3, c_te, Wp, Up, seed=4))

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
ax = axes[0]
ax.plot(ars, ratios, "o-", color="#1f77b4", lw=2)
ax.set_xlabel("memory strength ar"); ax.set_ylabel("acausal gradient ratio")
ax.set_title("(a) Bidirectional teacher supervision\nweight on future control c_{t+1} (analytic)")
ax.grid(alpha=0.3)

ax = axes[1]
for ar, ds in diff_by_ar.items():
    ax.plot(iters, ds, "o-", label=f"ar={ar}")
ax.axhline(0, color="gray", ls=":")
ax.set_xscale("log", base=2); ax.set_xticks(iters); ax.set_xticklabels([str(i) for i in iters])
ax.set_xlabel("training budget (rollout-fit iterations)"); ax.set_ylabel("deployment MSE diff (bidir - causal)")
ax.set_title("(b) Mismatch cost is a training transient:\nsame budget, bidirectional student lags")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

ax = axes[2]
ax.plot(ars2, errs["base"], "o-", color="#9467bd", label="teacher sees true prefix (base CMD)")
ax.plot(ars2, errs["prefix"], "s-", color="#ff7f0e", label="teacher sees student rollout (Prefix Scoring)")
ax.axhline(0.09, color="gray", ls=":", label="noise floor")
ax.set_xlabel("memory strength ar"); ax.set_ylabel("online deployment MSE")
ax.set_title("(c) Prefix-context mismatch")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

fig.suptitle("CMD probe: distillation supervision must match the deployed information set "
             "(linear-Gaussian world, d=8, T=32)", y=1.02, fontsize=12)
fig.tight_layout()
fig.savefig("code/cmd-probe/cmd_probe_fig.png", dpi=160, bbox_inches="tight")
print("saved fig")

data = dict(ars_ratio=ars, ratios=ratios, iters=iters, diff_by_ar=diff_by_ar,
            ars_prefix=ars2, err_base=errs["base"], err_prefix=errs["prefix"],
            noise_floor=0.09,
            note="(b): both students converge to the true dynamics W=A (benign fixed point); "
                 "the cost of acausal supervision is a slower transient at equal budget.")
with open("code/cmd-probe/cmd_probe_data.json", "w") as f:
    json.dump(data, f, indent=2)
print("saved json")
