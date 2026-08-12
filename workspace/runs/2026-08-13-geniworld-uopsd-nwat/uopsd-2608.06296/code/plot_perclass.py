"""Plot U-OPSD per-class results (base vs merged150, train + held)."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CLASSES = ["maj_correct", "split_wrong", "unanimous_wrong", "low_signal"]
LABELS = {"maj_correct": "majority correct",
          "split_wrong": "wrong majority\n(split)",
          "unanimous_wrong": "wrong majority\n(unanimous)",
          "low_signal": "low signal\n(<2 parsable)"}


def main():
    res = json.load(open("uopsd_results/final_metrics.json"))
    os_ = __import__("os")
    os_.makedirs("plots", exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4))
    for row, split in enumerate(["train", "held"]):
        b = res["base"].get(split, {})
        f = res["merged150"].get(split, {})
        x = np.arange(len(CLASSES))
        w = 0.36
        ax = axes[row][0]
        base_maj = [b.get(c, {}).get("maj8_acc", 0) for c in CLASSES]
        fin_maj = [f.get(c, {}).get("maj8_acc", 0) for c in CLASSES]
        ns = [b.get(c, {}).get("n", 0) for c in CLASSES]
        ax.bar(x - w/2, base_maj, w, label="base", color="#9ecae1")
        ax.bar(x + w/2, fin_maj, w, label="U-OPSD (150 steps)", color="#3182bd")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{LABELS[c]}\n(n={ns[i]})" for i, c in enumerate(CLASSES)], fontsize=8)
        ax.set_ylabel("maj@8 accuracy")
        ax.set_title(f"{split} prompts", fontsize=10)
        ax.set_ylim(0, 1.08)
        ax.legend(frameon=False, fontsize=7)
        ax.grid(alpha=0.3, axis="y")

        ax = axes[row][1]
        # wrong_agree for the two wrong classes
        xx = np.arange(2)
        for i, c in enumerate(["split_wrong", "unanimous_wrong"]):
            bv = b.get(c, {}).get("wrong_agree", 0)
            fv = f.get(c, {}).get("wrong_agree", 0)
            ax.bar(xx[i] - w/3, bv, w/1.5, color="#9ecae1", label="base" if i == 0 else None)
            ax.bar(xx[i] + w/3, fv, w/1.5, color="#e67e22", label="U-OPSD" if i == 0 else None)
        ax.set_xticks(xx)
        ax.set_xticklabels(["wrong majority\n(split)", "wrong majority\n(unanimous)"], fontsize=8)
        ax.set_ylabel("wrong-consensus agreement")
        ax.set_ylim(0, 1.15)
        ax.set_title("amplification signature", fontsize=10)
        ax.legend(frameon=False, fontsize=7)
        ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig("plots/per_class.png", dpi=160)
    plt.close(fig)
    print("saved plots/per_class.png")


if __name__ == "__main__":
    main()
