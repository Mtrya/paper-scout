"""Plot U-OPSD probe results: per-class fate before/after training."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CLASSES = ["maj_correct", "split_wrong", "unanimous_wrong", "low_signal"]
LABELS = {"maj_correct": "多数正确",
          "split_wrong": "多数错(不一致)",
          "unanimous_wrong": "全体一致错",
          "low_signal": "低信号(<2可解析)"}
COLORS = {"maj_correct": "#1f77b4", "split_wrong": "#d62728",
          "unanimous_wrong": "#9467bd", "low_signal": "#7f7f7f"}


def main():
    with open("eval_base.json") as f:
        base = json.load(f)["train"]
    finals = {}
    for fname in sorted(os.listdir(".")):
        if fname.endswith("_metrics.json") and "merged" in fname:
            step = fname.split("_")[1]
            finals[step] = json.load(open(fname))

    os.makedirs("plots", exist_ok=True)
    if not finals:
        print("no merged metrics found")
        return

    last_step = sorted(finals, key=int)[-1]
    fin = finals[last_step]
    print(f"comparing base vs merged_{last_step}")

    # ---- per-class bar chart
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    x = np.arange(len(CLASSES))
    w = 0.38
    ax = axes[0]
    base_maj = [base.get(c, {}).get("maj8_acc", 0) for c in CLASSES]
    fin_maj = [fin.get(c, {}).get("maj8_acc", 0) for c in CLASSES]
    base_n = [base.get(c, {}).get("n", 0) for c in CLASSES]
    ax.bar(x - w/2, base_maj, w, label="base", color="#9ecae1")
    ax.bar(x + w/2, fin_maj, w, label=f"U-OPSD {last_step}步", color="#3182bd")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{LABELS[c]}\n(n={base_n[i]})" for i, c in enumerate(CLASSES)], fontsize=8)
    ax.set_ylabel("maj@8 正确率")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    aw = ["wrong_agree", "mean_unique"]
    for i, key in enumerate(aw):
        b = [base.get(c, {}).get(key, 0) for c in CLASSES[:3]]
        f = [fin.get(c, {}).get(key, 0) for c in CLASSES[:3]]
        xx = x[:3] + (i - 0.5) * w
        ax.bar(xx, b, w/2, label=["base", "trained"][0] if i == 0 else None, color="#9ecae1")
        ax.bar(xx + w/4, f, w/4, label="trained" if i == 0 else None, color="#3182bd")
    ax.set_xticks(x[:3])
    ax.set_xticklabels([LABELS[c] for c in CLASSES[:3]], fontsize=8)
    ax.set_title("wrong_agree(错误共识一致度) & mean_unique(多样性)", fontsize=8)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig("plots/per_class.png", dpi=160)
    plt.close(fig)

    # ---- overall summary print
    print(f"{'':18s} {'maj8':>8s} {'pass1':>8s}")
    print(f"{'base':18s} {base['overall_maj8']:8.3f} {base['overall_pass1']:8.3f}")
    for s in sorted(finals, key=int):
        print(f"step{s:13s} {finals[s]['overall_maj8']:8.3f} {finals[s]['overall_pass1']:8.3f}")
    print()
    for c in CLASSES:
        b, f = base.get(c, {}), fin.get(c, {})
        print(f"{LABELS[c]:12s} n={b.get('n',0):3d}  maj8 {b.get('maj8_acc',0):.3f} -> {f.get('maj8_acc',0):.3f}"
              f" | wrong_agree {b.get('wrong_agree',0):.3f} -> {f.get('wrong_agree',0):.3f}"
              f" | unique {b.get('mean_unique',0):.2f} -> {f.get('mean_unique',0):.2f}")


if __name__ == "__main__":
    main()
