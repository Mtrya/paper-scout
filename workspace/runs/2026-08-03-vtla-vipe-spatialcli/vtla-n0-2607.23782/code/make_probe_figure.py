"""Grouped bar chart for the synthetic z-sensitivity probe (two seeds).

Usage (local scout-exp env has matplotlib + Noto Sans CJK):
  code/scout-exp/bin/python make_probe_figure.py probe_synth_s1000.json probe_synth_s5000.json ../assets/probe-bars.png
"""

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = ["Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

GROUPS = [
    ("null", "z_null\n触觉差分清零"),
    ("shuffle", "z_shuffle\n触觉换样本"),
    ("vlswap", "z_vlswap\n场景+指令换样本"),
    ("padpert", "z_padpert\n截断指令尾部"),
]


def main():
    datas = [json.load(open(p)) for p in sys.argv[1:-1]]
    out_path = sys.argv[-1]
    seed_names = [f"合成批次 seed{1000 if i == 0 else 5000}" for i in range(len(datas))]

    labels = [lbl for _, lbl in GROUPS]
    x = np.arange(len(GROUPS))
    width = 0.36
    colors = ["#2980b9", "#e67e22"]

    fig, ax = plt.subplots(figsize=(8.0, 4.2), dpi=150)
    for i, (d, name) in enumerate(zip(datas, seed_names)):
        vals = [d["groups"][k]["one_minus_cos_cent"] for k, _ in GROUPS]
        bars = ax.bar(x + (i - 0.5) * width, vals, width, label=f"{name},R={d['R_tactile_over_vl']:.2f}", color=colors[i])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x, labels, fontsize=9)
    ax.set_ylabel("1 − 居中余弦相似度(相对 z_real,越大=z 移动越多)")
    ax.set_title(
        "N0-VTLA 放出权重(arch C)上的 z 敏感性探针:VL 扰动让 z 移动得比触觉扰动更多\n"
        "R = 触觉敏感度 / VL敏感度,两个合成批次 R = 0.41 / 0.28,均远小于 1(健康值应 ≫1)"
    )
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
