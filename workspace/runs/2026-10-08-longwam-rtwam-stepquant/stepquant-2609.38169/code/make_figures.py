"""Figures for the STEPQuant thread: half-life vs error scatter, error growth curves,
oracle comparison, NLL comparison."""
import json
import os

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np

_CJK = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
font_manager.fontManager.addfont(_CJK)
plt.rcParams["font.family"] = [font_manager.FontProperties(fname=_CJK).get_name(), "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

import csv
layers, heads, tau, Du, frel = [], [], [], [], []
with open(os.path.join(HERE, "probe_a_perhead.csv")) as f:
    for row in csv.DictReader(f):
        layers.append(int(row["layer"]))
        heads.append(int(row["head"]))
        tau.append(float(row["half_life"]))
        Du.append(float(row["cum_sq_err"]))
        frel.append(float(row["final_rel_err"]))
layers = np.array(layers); tau = np.array(tau); Du = np.array(Du); frel = np.array(frel)
sa = json.load(open(os.path.join(HERE, "probe_a_summary.json")))
sb = json.load(open(os.path.join(HERE, "probe_b_summary.json")))
sc = json.load(open(os.path.join(HERE, "probe_c_summary.json")))

# Fig 1: half-life vs cumulative INT6 squared error
fig, ax = plt.subplots(figsize=(6.4, 4.6))
sc_ = ax.scatter(tau, Du, c=layers, cmap="viridis", s=22, alpha=0.85, edgecolors="none")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("门控半衰期 τ = ln2 / (−E[log α])(token)")
ax.set_ylabel("INT6 累积平方状态误差 $D_u = \\sum_t \\|E_t\\|_F^2$")
ax.set_title(f"340M 纯 GDN(96 头):Spearman ρ = {sa['spearman_tau_vs_cumerr']:.3f}(论文 Qwen ρ ≈ 0.80)\n"
             f"最长寿 1/4 头占误差 {sa['longest_quarter_error_share']*100:.1f}%(论文 52.5%)")
plt.colorbar(sc_, label="层号")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_halflife_error.png"), dpi=150); plt.close(fig)

# Fig 2: error growth curves by half-life quartile
err = np.load(os.path.join(HERE, "probe_a_errcurve.npy"))  # [L, H, T]
tau_flat_order = np.argsort(tau)
q = len(tau) // 4
groups = {"最短寿 1/4": tau_flat_order[:q], "中间 1/2": tau_flat_order[q:-q], "最长寿 1/4": tau_flat_order[-q:]}
fig, ax = plt.subplots(figsize=(6.4, 4.4))
T = err.shape[-1]
for name, idx in groups.items():
    curve = err.reshape(-1, T)[idx].mean(0)
    ax.plot(np.arange(1, T + 1), curve, label=name)
ax.set_yscale("log")
ax.set_xlabel("解码步 t")
ax.set_ylabel("相对 Frobenius 误差 $\\|S_q - S_{ref}\\|_F/\\|S_{ref}\\|_F$")
ax.set_title("均匀 INT6 状态量化:误差随步数增长(按门控半衰期分组,96 头均值)")
ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_error_growth.png"), dpi=150); plt.close(fig)

# Fig 3: oracle vs native cumulative squared readout error
fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3), sharey=False)
for ax, bits in zip(axes, [6, 8]):
    cu = np.load(os.path.join(HERE, f"probe_b_curves_int{bits}.npy"))
    inj = sb["inject_step"]
    tt = np.arange(1, cu.shape[1] + 1)
    ax.plot(tt[inj + 1:], np.cumsum(cu[0])[inj + 1:], label="native delta 反馈")
    ax.plot(tt[inj + 1:], np.cumsum(cu[1])[inj + 1:], label="exact-read oracle(无自纠)")
    ax.set_yscale("log")
    ax.set_xlabel("解码步 t")
    ratio = sb[str(bits)]["ratio_total"]
    paper = sb["paper_ratio_int6"] if bits == 6 else sb["paper_ratio_int8"]
    ax.set_title(f"INT{bits}:oracle/native = {ratio:.1f}×(论文 {paper}×)")
    ax.legend()
axes[0].set_ylabel("逐头平均的累积平方读出误差")
fig.suptitle("单点注入(step 256)后,delta 自纠对量化误差的清除作用(340M GDN,96 头 × 3 段)")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_oracle.png"), dpi=150); plt.close(fig)

# Fig 4: NLL comparison
nc = {n: np.load(os.path.join(HERE, f"probe_c_nll_{n}.npy")) for n in ("fp32_ref", "uniform6", "lifetime_8_6_4")}
fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), gridspec_kw={"width_ratios": [1.2, 1]})
tt = np.arange(1, nc["fp32_ref"].shape[1] + 1)
smooth = lambda x, w=64: np.convolve(x, np.ones(w) / w, "valid")
axes[0].plot(tt[len(tt) - len(smooth(nc["fp32_ref"].mean(0))):], smooth(nc["fp32_ref"].mean(0)), label="FP32", color="black")
for name, lab, col in [("uniform6", "均匀 INT6", "tab:red"), ("lifetime_8_6_4", "寿命感知 8/6/4(简化)", "tab:blue")]:
    axes[0].plot(tt[len(tt) - len(smooth(nc[name].mean(0))):], smooth(nc[name].mean(0)), label=lab, color=col, alpha=0.85)
axes[0].set_xlabel("位置 t"); axes[0].set_ylabel("NLL(64 位滑动均值)")
axes[0].set_title("教师强制 NLL 曲线(WikiText-2 test,3 段均值)")
axes[0].legend()
names = ["uniform6", "lifetime_8_6_4"]
excess = [sc[n]["excess_nll"] for n in names]
bars = axes[1].bar(["均匀 INT6", "寿命感知\n8/6/4"], excess, color=["tab:red", "tab:blue"])
for b, v in zip(bars, excess):
    axes[1].text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom")
axes[1].set_ylabel("excess NLL(相对 FP32)")
axes[1].set_title(f"平均 excess NLL(FP32 基线 NLL {sc['fp32_ref']['nll_mean']:.2f})")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_nll.png"), dpi=150); plt.close(fig)
print("figures done")
