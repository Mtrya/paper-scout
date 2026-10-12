"""Figures for the DiffusionART toy thread. Reads results/*.json, writes figs/*.png."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

OUT = Path(__file__).resolve().parent
FIG = OUT / "figs"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 150})


def load(name):
    return json.loads((OUT / "results" / name).read_text())


def vector(pair):
    if isinstance(pair, dict):
        pair = [pair["x"], pair["y"]]
    return torch.tensor(pair, dtype=torch.float64)


def fig_force_identity():
    data = load("prop1.json")
    rows = data["states"]
    meas = torch.tensor([vector(r["F"]).norm() for r in rows])
    pred = torch.tensor([vector(r["F_pred"]).norm() for r in rows])
    resid = torch.tensor([vector(r["residual"]).norm() for r in rows])
    guid = torch.tensor([vector(r["guidance"]).norm() for r in rows])

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
    ax = axes[0]
    ax.scatter(pred, meas, s=22, c="tab:blue")
    lim = [max(1e-4, pred.min()) * 0.7, meas.max() * 1.4]
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("closed form |2g + (2q-1)e|")
    ax.set_ylabel("measured |F_NFT| (autodiff)")
    ax.set_title("prop. 1 force identity, %d states\nmax err %.1e" % (len(rows), max(r["F_err"] for r in rows)))

    ax = axes[1]
    ax.scatter(guid, resid, s=22, c="tab:red")
    ax.set_xscale("log")
    ax.set_yscale("log")
    for r in rows:
        if r["residual_over_guidance"] > 2:
            ax.annotate(
                f"t={r['t']:.1f} {r['path']}",
                (float(vector(r["guidance"]).norm()), float(vector(r["residual"]).norm())),
                fontsize=6,
                xytext=(2, 2),
                textcoords="offset points",
            )
    ax.set_xlabel("guidance |2g|")
    ax.set_ylabel("consistency residual |(2q-1)e|")
    ax.set_title("residual is largest where guidance is smallest")

    ax = axes[2]
    for c, marker in ((0, "o"), (1, "s")):
        for path, ls in (("hi", "-"), ("lo", "--"), ("mid", ":")):
            sel = [r for r in rows if r["c"] == c and r["path"] == path]
            ax.plot([r["t"] for r in sel], [2 * r["q"] - 1 for r in sel], ls, marker=marker, lw=1, ms=3,
                    label=f"c{c} {path}")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("noise level t")
    ax.set_ylabel("statewise 2q - 1")
    ax.set_title("q is not 1/2 statewise")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_force_identity.png")
    plt.close(fig)


def fig_aggregation():
    data = load("prop1.json")
    rows = data["states"]
    agg = data["aggregation"]
    scale = data["reward_scale"]
    gaps = data["solver_gap"]

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
    ax = axes[0]
    names = ["guidance", "residual"]
    per_state = [agg["mean_norm_guidance"], agg["mean_norm_residual"]]
    aggregated = [agg["norm_mean_guidance"], agg["norm_mean_residual"]]
    xs = torch.arange(2)
    ax.bar(xs - 0.18, per_state, width=0.34, label="mean of norms")
    ax.bar(xs + 0.18, aggregated, width=0.34, label="norm of mean (aggregate)")
    ax.set_xticks(xs, ["2g (guidance)", "(2q-1)e (residual)"])
    ax.set_ylabel("output-space force")
    ax.legend(fontsize=7)
    ax.set_title(
        "aggregation over %d states: coherence %.2f vs %.2f"
        % (agg["n_states"], agg["coherence_guidance"], agg["coherence_residual"])
    )
    for i, row in enumerate(rows):
        ax.plot([], [])
    ax = axes[1]
    gains = [r["gain"] for r in scale]
    ax.plot(gains, [r["mean_abs2q1"] for r in scale], "o-", label="mean |2q-1|")
    ax.plot(gains, [r["mean_ratio"] for r in scale], "s--", label="mean residual/guidance")
    ax.set_xscale("log")
    ax.set_xlabel("advantage gain (reward spread / Z_c)")
    ax.set_ylim(0, max([r["mean_ratio"] for r in scale]) * 1.25)
    ax.legend(fontsize=7)
    ax.set_title("sharpening the reward sharpens both terms")
    ax = axes[2]
    steps = [r["steps"] for r in gaps]
    ax.plot(steps, [r["mean_rel_e"] for r in gaps], "o-", label="|e| / |v_fwd|")
    ax2 = ax.twinx()
    ax2.plot(steps, [r["mean_norm_e"] for r in gaps], "s--", color="tab:red", label="|e|")
    ax2.set_ylabel("|e|", color="tab:red")
    ax.set_xscale("log")
    ax.set_xlabel("rollout solver steps")
    ax.set_ylabel("relative forward-reverse gap")
    ax.set_title("gap vs solver accuracy")
    fig.tight_layout()
    fig.savefig(FIG / "fig2_aggregation.png")
    plt.close(fig)


def fig_kernels():
    km = load("kernel_map.json")
    C = km["C_art_force"]
    B = km["B_restoration"]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))
    ax = axes[0]
    xs = torch.arange(len(C))
    ax.bar(xs - 0.2, [r["art_force_norm"] for r in C], width=0.4, label="ART force |F|")
    ax.bar(xs + 0.2, [r["nft_force_norm"] for r in C], width=0.4, label="NFT force |F|")
    ax.set_xticks(xs, [r["ensemble"] for r in C])
    ax.set_xlabel("conditional ensemble at one state (same state, same G)")
    ax.set_ylabel("force norm from the shipped kernel")
    ax.legend(fontsize=7)
    ax.set_title("ART force is ensemble-independent; NFT force is not")
    ax = axes[1]
    betas = [r["beta"] for r in B[: len(B) // 2]]
    kernel_h = [r["kernel_hessian"] for r in B[: len(B) // 2]]
    paper_h = [r["paper_hessian"] for r in B[: len(B) // 2]]
    ax.loglog(betas, kernel_h, "o-", label="shipped nft_loss (∝ beta)")
    ax.loglog(betas, paper_h, "s--", label="paper Eq.1 (∝ beta^2)")
    for b, kh in zip(betas, kernel_h):
        ax.annotate(f"{kh:.2f}", (b, kh), fontsize=6, xytext=(4, -2), textcoords="offset points")
    ax.set_xlabel("beta")
    ax.set_ylabel("restoration Hessian (batch sum)")
    ax.legend(fontsize=7)
    ax.set_title("restoration strength: code vs paper")
    fig.tight_layout()
    fig.savefig(FIG / "fig3_kernels.png")
    plt.close(fig)


def fig_train():
    data = load("train.json")
    runs = data["runs"]
    order = [
        ("nft_onpolicy_1u", "NFT on-policy, 1 update", "tab:red"),
        ("nft_onpolicy_2u", "NFT on-policy, 2 updates", "tab:orange"),
        ("nft_ema_1u", "NFT EMA (eta=0.5), 1 update", "tab:green"),
        ("nft_ema_noanchor", "NFT EMA rollouts, no anchor", "tab:brown"),
        ("art_1u", "ART, 1 update", "tab:blue"),
        ("art_2u", "ART, 2 updates + mask", "tab:purple"),
        ("art_2u_nomask", "ART, 2 updates, no mask", "tab:cyan"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
    for name, label, color in order:
        if name not in runs:
            continue
        hist = runs[name]["history"]
        xs = [h["iter"] for h in hist]
        axes[0].plot(xs, [h["reward"] for h in hist], color=color, lw=1.4, label=label)
        axes[1].plot(xs, [h["energy_to_target"] for h in hist], color=color, lw=1.4)
        axes[2].plot(xs, [h["spread"] for h in hist], color=color, lw=1.4)
    axes[0].axhline(-0.627, color="k", ls=":", lw=0.8)
    axes[0].text(2, -0.60, "reward floor of a perfect mode fit (-0.63)", fontsize=6)
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("mean reward (higher better)")
    axes[0].set_title("training reward")
    axes[0].legend(fontsize=6)
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("energy distance to target mode")
    axes[1].set_title("distribution match (lower better)")
    axes[2].axhline(0.707, color="k", ls=":", lw=0.8)
    axes[2].set_xlabel("iteration")
    axes[2].set_ylabel("endpoint spread ||std||")
    axes[2].set_title("diversity: target spread 0.71")
    fig.tight_layout()
    fig.savefig(FIG / "fig4_train.png")
    plt.close(fig)


def fig_restoration():
    data = load("train.json")
    runs = data["runs"]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))
    ax = axes[0]
    names, out_ratio, par_ratio = [], [], []
    for name in ("nft_onpolicy_2u", "art_2u", "art_2u_nomask"):
        if name in runs and runs[name]["instrument"]:
            m = runs[name]["instrument"][0]
            names.append(name)
            out_ratio.append(m["ratio_output"])
            par_ratio.append(m["ratio_param"])
    xs = torch.arange(len(names))
    ax.bar(xs - 0.2, out_ratio, width=0.4, label="output-force ratio")
    ax.bar(xs + 0.2, par_ratio, width=0.4, label="parameter-gradient ratio")
    ax.axhline(1.0, color="k", ls=":", lw=0.8)
    ax.set_xticks(xs, names, rotation=12, fontsize=7)
    ax.set_ylabel("restoration / guidance")
    ax.set_title("second update: what survives aggregation")
    ax.legend(fontsize=7)
    ax = axes[1]
    p2 = load("prop2.json")
    for name, rewards in p2["beta_two_update"].items():
        ax.plot(rewards, lw=1.2, label=f"2 updates, {name}")
    for name, rewards in p2["beta_single_update"]["adam"].items():
        ax.plot(rewards, ls="--", lw=1.0, label=f"1 update, {name.split('_')[0]}")
    ax.set_xlabel("iteration")
    ax.set_ylabel("mean reward")
    ax.set_yscale("symlog")
    ax.legend(fontsize=6, ncol=2)
    ax.set_title("beta sweep")
    fig.tight_layout()
    fig.savefig(FIG / "fig5_restoration.png")
    plt.close(fig)


def main():
    if (OUT / "results" / "prop1.json").exists():
        fig_force_identity()
        fig_aggregation()
    if (OUT / "results" / "kernel_map.json").exists():
        fig_kernels()
    if (OUT / "results" / "train.json").exists():
        fig_train()
    if (OUT / "results" / "prop2.json").exists() and (OUT / "results" / "train.json").exists():
        fig_restoration()
    for path in sorted(FIG.glob("*.png")):
        print(f"{path.name}: {path.stat().st_size/1024:.0f} KB")


if __name__ == "__main__":
    main()
