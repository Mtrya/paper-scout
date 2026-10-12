"""Render all probe figures (PNG) from results/."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
TR = HERE / "traces"
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

plt.rcParams.update({"figure.dpi": 130, "font.size": 9})
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Droid Sans Fallback", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def fig_a1_delta_dist():
    z = np.load(TR / "probe_a_traces.npz")
    stats = {r[0]: r for r in csv.reader((RES / "probe_a_head_stats.csv").read_text().splitlines())}
    rows = list(csv.DictReader((RES / "probe_a_head_stats.csv").read_text().splitlines()))
    dep = np.zeros((24, 4), bool)
    near_mask = np.zeros((24, 4), bool)
    for r in rows:
        dep[int(r["layer"]), int(r["head"])] = r["input_dep"] == "1"
        near_mask[int(r["layer"]), int(r["head"])] = r["near_one"] == "1"
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
    for ax, cls, title in zip(axes, ["natural", "code", "repetitive"],
                              ["自然文本 (WikiText-2)", "代码 (CyFA 源码)", "重复/周期文本"]):
        d = z[f"{cls}_delta"]  # [L,B,T,H]
        flat = d.transpose(0, 3, 1, 2).reshape(24, 4, -1)
        near = flat[near_mask].ravel() if near_mask.any() else np.array([])
        idep = flat[dep].ravel() if dep.any() else np.array([])
        ax.hist(near, bins=60, range=(0, 1), density=True, alpha=0.6,
                label=f"近一头 ({int(near_mask.sum())})", color="#4878cf")
        ax.hist(idep, bins=60, range=(0, 1), density=True, alpha=0.6,
                label=f"输入依赖头 ({int(dep.sum())})", color="#d65f5f")
        ax.set_title(title)
        ax.set_xlabel("时钟步长 $\\delta_t$")
        ax.legend(frameon=False)
    axes[0].set_ylabel("密度")
    fig.suptitle("三类输入上的时钟步长分布(按头类分层)")
    fig.tight_layout()
    fig.savefig(FIG / "fig_a1_delta_dist.png")
    plt.close(fig)


def fig_a2_head_map():
    cross = json.loads((RES / "probe_a_cross_class.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(11, 5.5), sharex=True, sharey=True)
    for ax, cls, title in zip(axes, ["natural", "code", "repetitive"],
                              ["自然文本", "代码", "重复/周期文本"]):
        dm = np.array(cross[cls]["delta_mean_per_head"])  # [L,H]
        im = ax.imshow(dm, aspect="auto", vmin=0, vmax=1, cmap="viridis")
        ax.set_title(title)
        ax.set_xticks(range(4), [f"H{h}" for h in range(4)])
        if ax is axes[0]:
            ax.set_ylabel("层")
    fig.colorbar(im, ax=axes, label="平均时钟步长 $\\bar\\delta$")
    fig.suptitle("逐头平均时钟步长(24 层 × 4 头):同一头在不同输入上的节奏")
    fig.tight_layout()
    fig.savefig(FIG / "fig_a2_head_map.png")
    plt.close(fig)


def fig_a3_intervention():
    s = json.loads((RES / "probe_a_summary.json").read_text())
    iv = s["interventions"]
    paper = iv["paper_table8"]
    labels = ["原始时钟", "头均值替换", "循环平移 257", "$\\delta\\equiv 1$"]
    mine = [iv["baseline_nll"], iv["mean"], iv["shift257"], iv["one"]]
    ref = [paper["baseline"], paper["mean"], paper["shift257"], paper["one"]]
    x = np.arange(4)
    w = 0.38
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.bar(x - w / 2, mine, w, label="本实验(8 段 held-out)", color="#4878cf")
    ax.bar(x + w / 2, ref, w, label="论文 Table 8(16 段 held-out)", color="#999999")
    for xi, v in zip(x - w / 2, mine):
        ax.text(xi, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    for xi, v in zip(x + w / 2, ref):
        ax.text(xi, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("held-out NLL")
    ax.set_title("时钟干预实验:本实验复测 vs 论文 Table 8")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_a3_intervention.png")
    plt.close(fig)


def fig_b1_halflife_vs_cycle():
    rows = list(csv.DictReader((RES / "probe_b_perhead.csv").read_text().splitlines()))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharex=True, sharey=True)
    colors = {"natural": "#4878cf", "code": "#6acc65", "repetitive": "#d65f5f"}
    titles = {"natural": "自然文本", "code": "代码", "repetitive": "重复/周期文本"}
    for ax, cls in zip(axes, ["natural", "code", "repetitive"]):
        sub = [r for r in rows if r["class"] == cls]
        tpc = np.array([float(r["tokens_per_cycle"]) for r in sub])
        hl = np.array([float(r["half_life_tokens"]) for r in sub])
        ax.scatter(tpc, hl, c=colors[cls], s=14, alpha=0.8)
        xs = np.linspace(30, max(tpc.max() * 1.1, 200), 100)
        for dpc, ls, lab in [(0.1, "--", "整周期衰减=0.1"), (0.01, ":", "0.01"), (1e-3, "-.", "0.001")]:
            # decay per cycle = exp(ln(0.5)/hl * tpc) = dpc -> hl = tpc * ln2 / -ln(dpc)
            ax.plot(xs, xs * np.log(2) / (-np.log(dpc)), ls, c="k", lw=0.8, label=lab if cls == "natural" else None)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(titles[cls])
        ax.set_xlabel("整周期所需 token 数(127/$\\bar\\delta$)")
    axes[0].set_ylabel("遗忘门半衰期(token)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("回绕安全性:周期长度 vs 半衰期(点越低于等值线,回绕残迹越小)")
    fig.tight_layout()
    fig.savefig(FIG / "fig_b1_halflife_vs_cycle.png")
    plt.close(fig)


def fig_b2_residual_at_wrap():
    rows = list(csv.DictReader((RES / "probe_b_perhead.csv").read_text().splitlines()))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    titles = {"natural": "自然文本", "code": "代码", "repetitive": "重复/周期文本"}
    for ax, cls in zip(axes, ["natural", "code", "repetitive"]):
        sub = [r for r in rows if r["class"] == cls]
        sub.sort(key=lambda r: -float(r["resid_wrap_p99"]))
        p99 = np.array([float(r["resid_wrap_p99"]) for r in sub])
        mx = np.array([float(r["resid_wrap_max"]) for r in sub])
        bm = np.array([float(r["beta_median"]) for r in sub])
        x = np.arange(len(sub))
        ax.bar(x, mx, color="#d65f5f", alpha=0.55, label="最大残迹")
        ax.bar(x, p99, color="#4878cf", label="p99 残迹")
        ax.step([0, len(sub) - 1], [np.median(bm)] * 2, color="k", ls="--",
                label="新写入强度中位 $\\beta$")
        ax.set_yscale("log")
        ax.set_title(titles[cls])
        ax.set_xlabel("头(按 p99 降序)")
    axes[0].set_ylabel("回绕时残迹强度")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("写入回绕到 age-0 槽时的剩余强度(远低于新写入即安全)")
    fig.tight_layout()
    fig.savefig(FIG / "fig_b2_residual_at_wrap.png")
    plt.close(fig)


def fig_c1_spectra():
    z = np.load(RES / "probe_c_spectra.npz")
    s = json.loads((RES / "probe_c_summary.json").read_text())
    layers = [0, 5, 11, 17, 23]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(layers)))
    for l, c in zip(layers, cmap):
        axes[0].plot(z[f"hk_spec_l{l}"], color=c, label=f"L{l}")
        axes[1].plot(z[f"hv_spec_l{l}"], color=c, label=f"L{l}")
    for ax, t in zip(axes, ["键状态 $\\bar K$ (256×128)", "值状态 $\\bar V$ (128×256)"]):
        ax.set_yscale("log")
        ax.set_xlabel("奇异值序号")
        ax.set_ylabel("归一化奇异值 $\\sigma_i/\\sigma_1$")
        ax.set_title(t)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("自然文本 t=2048 处循环状态谱(头平均)")
    fig.tight_layout()
    fig.savefig(FIG / "fig_c1_spectra.png")
    plt.close(fig)


def fig_c2_clock_effect():
    s = json.loads((RES / "probe_c_summary.json").read_text())
    base = {r["layer"]: r for r in s["t2048"]["layers"]}
    mean = {r["layer"]: r for r in s["t2048_mean_clock"]["layers"]}
    layers = sorted(base)
    x = np.arange(len(layers))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar(x - w / 2, [base[l]["hk_pr_mean"] for l in layers], w,
           label="真实时钟", color="#4878cf")
    ax.bar(x + w / 2, [mean[l]["hk_pr_mean"] for l in layers], w,
           label="均值时钟(消融)", color="#d65f5f")
    ax.set_xticks(x[::3], [str(l) for l in layers[::3]])
    ax.set_xlabel("层")
    ax.set_ylabel("有效秩(参与率 PR)")
    ax.set_title("键状态有效秩:真实时钟 vs 均值时钟(t=2048,头平均)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_c2_clock_effect.png")
    plt.close(fig)


def fig_b3_nll_vs_aliasing():
    s = json.loads((RES / "probe_b2_summary.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, cls, title in zip(axes, ["natural", "code"], ["自然文本", "代码"]):
        d = s[cls]
        x = d["binned_aliased_mass"]
        y = d["binned_nll"]
        ax.bar(range(len(y)), y, color="#4878cf", alpha=0.8)
        ax.axhline(d["mean_nll"], color="k", ls="--", lw=0.8, label=f"均值 {d['mean_nll']:.3f}")
        ax.set_xticks(range(len(y)), ["0-50%", "50-90%", "90-99%", "99-100%"])
        ax.set_xlabel("活混叠质量分位桶(按 token)")
        ax.set_title(f"{title}: NLL vs 混叠(位置控制后 Spearman={d['position_controlled_spearman_mean']:+.3f})")
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("平均 token NLL")
    fig.suptitle("回绕混叠最重的 1% token 并未显著更难预测")
    fig.tight_layout()
    fig.savefig(FIG / "fig_b3_nll_vs_aliasing.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_a1_delta_dist()
    fig_a2_head_map()
    fig_a3_intervention()
    fig_b1_halflife_vs_cycle()
    fig_b2_residual_at_wrap()
    fig_b3_nll_vs_aliasing()
    fig_c1_spectra()
    fig_c2_clock_effect()
    print("figures written to", FIG)
