#!/usr/bin/env python3
"""DreamX prior-attribution probe: CALVIN normalized MSE bar chart with paper anchors."""
import json
import glob
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = sys.argv[1] if len(sys.argv) > 1 else "results_full"

def get(tag, seed=42):
    f = f"{OUT}/{tag}_seed{seed}.json"
    return json.load(open(f))["best_val_mse"]

rows = []  # (label, mse, color, hatch)
# ours
ours = [
    ("Cosmos DiT L6 (attn)", "dit_l6"),
    ("Cosmos DiT L13 (attn)", "dit_l13"),
    ("Cosmos DiT L20 (attn)", "dit_l20"),
    ("Cosmos DiT L27 (attn)", "dit_l27"),
    ("Cosmos DiT L27 pool4 (attn)", "dit_l27_pool4"),
    ("Cosmos VAE tokens (attn)", "vae"),
    ("Cosmos VAE flat (MLP)", "vae_mlp"),
    ("Cosmos DiT L27 pool4 flat (MLP)", "dit_l27_pool4_mlp"),
    ("Cosmos DiT L27 mean (MLP)", "dit_l27_mean_mlp"),
]
for label, tag in ours:
    try:
        vals = [get(tag, s) for s in (42, 0, 1) if glob.glob(f"{OUT}/{tag}_seed{s}.json")]
        color = "#2f9e44" if tag.endswith("mlp") else "#8ac926"
        rows.append((label, np.mean(vals), np.std(vals) if len(vals) > 1 else 0.0, color))
    except FileNotFoundError:
        print("missing:", tag)

anchors = [
    ("DINOv3 frozen (LARY T2)", 0.22, "#868e96"),
    ("V-JEPA2 frozen (LARY T2)", 0.27, "#868e96"),
    ("Wan2.2-VAE (LARY T2)", 0.39, "#868e96"),
    ("DiLA (video, DreamX T1)", 0.43, "#868e96"),
    ("LAPA (2-frame VQ)", 0.96, "#adb5bd"),
    ("mean floor A0", 0.987, "#ced4da"),
    ("DreamX (trained, full video)", 0.15, "#e8590c"),
]
for label, mse, color in anchors:
    rows.append((label, mse, 0.0, color))

rows.sort(key=lambda r: -r[1])
fig, ax = plt.subplots(figsize=(9.5, 5.2))
y = np.arange(len(rows))[::-1]
for yi, (label, mse, std, color) in zip(y, rows):
    ax.barh(yi, mse, xerr=std, color=color, alpha=0.88, height=0.62)
    ax.text(mse + 0.012, yi, f"{mse:.3f}" + (f"±{std:.3f}" if std > 0 else ""),
            va="center", fontsize=9)
ax.set_yticks(y)
ax.set_yticklabels([r[0] for r in rows], fontsize=9.5)
ax.set_xlabel("LARYBench CALVIN action regression: normalized MSE ↓ (val, n=11,969)")
ax.set_xlim(0, 1.30)
ax.axvline(0.987, color="#ced4da", lw=0.8, ls=":")
ax.set_title("Frozen Cosmos-Predict2.5 prior vs LARYBench anchors (green: our attn probe; dark green: LARY-style MLP)", fontsize=10.5)
plt.tight_layout()
plt.savefig("dreamx_probe_calvin.png", dpi=170)
print("saved dreamx_probe_calvin.png")
