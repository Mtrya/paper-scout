"""Figures for the HLAR replication thread."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
if not os.path.isdir(RES):
    RES = os.path.join(HERE, "..", "results")
OUT = os.path.join(HERE, "figures")
if not os.path.isdir(OUT):
    OUT = os.path.join(HERE, "..", "figures")
ASSETS = os.path.join(HERE, "..", "..", "assets")
if not os.path.isdir(ASSETS):
    ASSETS = os.path.join(HERE, "..", "..", "runs", "2026-10-19-whyfar-linearmem-natura-astra", "assets")
os.makedirs(OUT, exist_ok=True)
os.makedirs(ASSETS, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 160})


def save(fig, name):
    for d in (OUT, ASSETS):
        p = os.path.join(d, name)
        fig.savefig(p, bbox_inches="tight", facecolor="white")
        im = Image.open(p).convert("RGB")
        arr = np.asarray(im)
        mask = (arr < 250).any(2)
        ys, xs = np.where(mask)
        if len(xs):
            im = im.crop((max(xs.min() - 8, 0), max(ys.min() - 8, 0),
                          min(xs.max() + 8, im.width), min(ys.max() + 8, im.height)))
        if max(im.size) > 1600:
            s = 1600 / max(im.size)
            im = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
        im.save(p)
    plt.close(fig)


def j_heatmap():
    Jv = np.load(os.path.join(RES, "screen_J_validation.npy"))
    Jt = np.load(os.path.join(RES, "screen_J_test.npy"))
    vmax = max(np.abs(Jv).max(), np.abs(Jt).max())
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), constrained_layout=True)
    for ax, J, title in zip(axes, [Jv, Jt], ["validation (dev)", "test (confirmation)"]):
        im = ax.imshow(J, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xlabel("head")
        ax.set_ylabel("layer")
        ax.set_title(title)
        ax.set_xticks(range(4))
        ax.set_yticks(range(0, 24, 4))
        ax.add_patch(plt.Rectangle((1.5, 19.5), 1, 1, fill=False, ec="black", lw=1.4))
        ax.text(2.6, 20.5, "L20H2", fontsize=7, va="center")
        ax.plot(2, 11, marker="*", ms=9, mfc="yellow", mec="black", mew=0.5)
        ax.text(2.6, 11.5, "L11H2", fontsize=7, va="center")
    cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cb.set_label("J = mean donor-margin shift (logits)")
    fig.suptitle("Pre-query state swap, native LM head: max |J| = 0.004 logits", y=1.04)
    save(fig, "hlar_j_heatmap.png")


def retention_distance():
    rv = json.load(open(os.path.join(RES, "state_retention_validation.json")))
    rt = json.load(open(os.path.join(RES, "state_retention_test.json")))
    ds = sorted(int(k) for k in rv)
    heads = {(11, 2): "L11H2 (tau rank 1)", (7, 2): "L7H2 (rank 2)",
             (14, 2): "L14H2 (rank 4)", (17, 3): "L17H3 (rank 6)",
             (20, 2): "L20H2 (paper hub, rank 22)"}
    fig, ax = plt.subplots(figsize=(4.6, 3.2), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(heads)))
    xs = np.array(ds, dtype=float)
    for (l, h), lab in heads.items():
        yv = np.array([rv[str(d)]["rel"][l][h] for d in ds])
        yt = np.array([rt[str(d)]["rel"][l][h] for d in ds])
        ax.plot(xs, (yv + yt) / 2, "-o", ms=3, lw=1.4, color=colors[list(heads).index((l, h))],
                label=lab)
        ax.fill_between(xs, np.minimum(yv, yt), np.maximum(yv, yt), color=colors[list(heads).index((l, h))],
                        alpha=0.15, lw=0)
    ax.set_xscale("symlog", linthresh=32)
    ax.set_xticks([0, 64, 128, 256, 512, 1024])
    ax.set_xticklabels(["0", "64", "128", "256", "512", "1024"])
    ax.set_xlabel("filler tokens between the last presentation and the query")
    ax.set_ylabel(r"$\Vert S_{\rm donor}-S_{\rm rec}\Vert / \Vert S_{\rm rec}\Vert$")
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("Only the slowest heads retain the fact at long delay")
    save(fig, "hlar_retention_distance.png")


def tau_cross():
    d = json.load(open(os.path.join(RES, "tau_cross.json")))
    tau = np.array(d["tau"])
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), constrained_layout=True, sharey=True)
    for ax, dist in zip(axes, ["0", "512"]):
        rv = json.load(open(os.path.join(RES, "state_retention_validation.json")))
        for split, mk, alpha in [("validation", "o", 0.85), ("test", "x", 0.55)]:
            rs = json.load(open(os.path.join(RES, f"state_retention_{split}.json")))
            R = np.array(rs[dist]["rel"])
            ax.scatter(tau.ravel(), R.ravel(), s=14, marker=mk, alpha=alpha,
                       label=split, edgecolors="none" if mk == "o" else None)
        for (l, h), lab in {(20, 2): "L20H2", (11, 2): "L11H2", (17, 3): "L17H3"}.items():
            ax.annotate(lab, (tau[l, h], np.array(rv[dist]["rel"])[l, h]),
                        textcoords="offset points", xytext=(3, 3), fontsize=7)
        rho = d["splits"]["validation"][dist]["spearman_tau_rel"]
        ax.set_xscale("log")
        ax.set_xlabel("gate half-life tau (tokens, STEPQuant)")
        ax.set_title(f"d = {dist}   rho = {rho:+.2f}")
        ax.axhline(0, color="grey", lw=0.6)
    axes[0].set_ylabel("relative donor-recipient state difference")
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle("Slow-forgetting heads carry the fact (all 96 heads)", y=1.03)
    save(fig, "hlar_tau_cross.png")


def load_fig():
    lv = json.load(open(os.path.join(RES, "load_validation.json")))
    lt = json.load(open(os.path.join(RES, "load_test.json")))
    ls = json.load(open(os.path.join(RES, "load_state_validation_L11H2.json")))
    ks = [1, 2, 4, 8, 16]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), constrained_layout=True)
    ax = axes[0]
    ax.plot(ks, [lv[str(k)]["acc"] for k in ks], "-o", label="validation")
    ax.plot(ks, [lt[str(k)]["acc"] for k in ks], "--s", label="test")
    ax.axhline(0.25, color="grey", lw=0.8, ls=":")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.set_xticklabels(ks)
    ax.set_ylim(0, 0.6)
    ax.set_xlabel("number of active bindings K (fixed length 626)")
    ax.set_ylabel("native four-choice accuracy")
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("Native readout: flat at chance")
    ax = axes[1]
    ax.plot(ks, [ls[str(k)]["rel_state"] for k in ks], "-o", color="C2")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.set_xticklabels(ks)
    ax.set_xlabel("K")
    ax.set_ylabel("L11H2 relative state difference")
    ax.set_title("State carries a weaker but real load signal")
    save(fig, "hlar_load.png")


def write_fig():
    w = json.load(open(os.path.join(RES, "write_validation_L11H2.json")))
    conds = ["clean", "block", "donor", "random"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), constrained_layout=True)
    ax = axes[0]
    x = np.arange(len(conds))
    ax.bar(x - 0.18, [w[c]["d_recipient"] for c in conds], 0.36, label="to recipient state")
    ax.bar(x + 0.18, [w[c]["d_donor"] for c in conds], 0.36, label="to donor state")
    ax.set_xticks(x)
    ax.set_xticklabels(conds)
    ax.set_ylabel("Frobenius distance (L11H2)")
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("Write replacement lands on the donor")
    ax = axes[1]
    ax.bar(x - 0.18, [w[c]["acc"] for c in conds], 0.36, label="native acc")
    ax.bar(x + 0.18, [w[c]["donor_rate"] for c in conds], 0.36, label="donor rate")
    ax.axhline(0.25, color="grey", lw=0.8, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels(conds)
    ax.set_ylim(0, 0.6)
    ax.set_ylabel("rate")
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("Native readout: no transfer")
    save(fig, "hlar_write.png")


def baseline_fig():
    b = json.load(open(os.path.join(RES, "baseline_validation.json")))
    fig, ax = plt.subplots(figsize=(3.6, 2.8), constrained_layout=True)
    vals = [0.25, b["acc"], 0.469]
    labs = ["chance", "GDN-340M\n(this run)", "paper\n(GDN-340M)"]
    bars = ax.bar(labs, vals, color=["grey", "C3", "C0"])
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v*100:.1f}%",
                ha="center", fontsize=8)
    ax.set_ylim(0, 0.6)
    ax.set_ylabel("four-choice accuracy")
    ax.set_title("Native task competence")
    save(fig, "hlar_baseline.png")


if __name__ == "__main__":
    j_heatmap()
    retention_distance()
    tau_cross()
    load_fig()
    write_fig()
    baseline_fig()
    print("figures written")
