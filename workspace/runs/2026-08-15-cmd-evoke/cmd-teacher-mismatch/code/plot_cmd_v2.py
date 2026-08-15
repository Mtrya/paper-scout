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
ars2 = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
errs = {"causal": [], "bidir": [], "base": [], "prefix": []}
for a in ars2:
    A, B = make_world(d=8, ar=a, sigma=0.3, seed=10)
    x_tr = gen_true(A, B, 0.3, c_tr, seed=6)
    Wc, Uc = distill(A, B, 0.3, c_tr, "causal", seed=3)
    Wb, Ub = distill(A, B, 0.3, c_tr, "bidir", seed=3)
    Wba, Uba = distill_prefix(A, B, 0.3, c_tr, x_tr, "base", seed=3)
    Wp, Up = distill_prefix(A, B, 0.3, c_tr, x_tr, "prefix", seed=3)
    errs["causal"].append(deploy_err(A, B, 0.3, c_te, Wc, Uc, seed=4))
    errs["bidir"].append(deploy_err(A, B, 0.3, c_te, Wb, Ub, seed=4))
    errs["base"].append(deploy_err(A, B, 0.3, c_te, Wba, Uba, seed=4))
    errs["prefix"].append(deploy_err(A, B, 0.3, c_te, Wp, Up, seed=4))

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
ax = axes[0]
ax.plot(ars, ratios, "o-", color="#1f77b4", lw=2)
ax.set_xlabel("memory strength ar"); ax.set_ylabel("acausal gradient ratio")
ax.set_title("(a) Bidirectional teacher supervision\nweight on future control c_{t+1}")
ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(ars2, errs["causal"], "s--", color="#2ca02c", label="causal teacher")
ax.plot(ars2, errs["bidir"], "o-", color="#d62728", label="bidirectional teacher")
ax.axhline(0.09, color="gray", ls=":", label="noise floor")
ax.set_xlabel("memory strength ar"); ax.set_ylabel("online deployment MSE")
ax.set_title("(b) Information-boundary mismatch")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

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
print("saved")

data = dict(ars_ratio=ars, ratios=ratios, ars_err=ars2,
            err_causal=errs["causal"], err_bidir=errs["bidir"],
            err_base=errs["base"], err_prefix=errs["prefix"],
            noise_floor=0.09)
with open("code/cmd-probe/cmd_probe_data.json", "w") as f:
    json.dump(data, f, indent=2)
print("json saved")
