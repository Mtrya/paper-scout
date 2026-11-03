"""Reanalysis of the released Probing-Spatial-State tables.

Outputs: CSV summaries and figures for the report. No model runs; all numbers
come from the released validation tables (seed 42).
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REL = os.path.join(ROOT, "Probing_Spatial_State_Release", "results")
OUT = os.path.join(ROOT, "wamreadout", "out")
os.makedirs(OUT, exist_ok=True)

DESIGNATED = {
    "dino": 0, "f3r": 24, "videomae": 23, "vjepa2": 23, "qwen2_5_vl": -1,
    "wan": 20, "cogvideox": 20, "aether": 1, "minimax_h3": 30, "lingbot": 30,
}
LABEL = {
    "dino": "DINOv2-L", "f3r": "Fast3R", "videomae": "VideoMAE-L",
    "vjepa2": "V-JEPA2-L", "qwen2_5_vl": "Qwen2.5-VL", "wan": "Wan",
    "cogvideox": "CogVideoX", "aether": "Aether", "minimax_h3": "MiniMax-H3",
    "lingbot": "LingBot-Video",
}


def jget(s, k):
    try:
        return json.loads(s).get(k)
    except Exception:
        return None


def load():
    canon = pd.read_csv(os.path.join(REL, "canonical_v2_val_seed42.csv"))
    c3 = pd.read_csv(os.path.join(REL, "c3_from_c1_val_seed42.csv"))
    return canon, c3


def main_table(df):
    rows = []
    depth = df[df.probe == "streaming_depth"].copy()
    depth["auc30"] = depth.secondary_json.map(lambda s: jget(s, "mean_Auc_30"))
    for m, layer in DESIGNATED.items():
        r = {"model": m, "label": LABEL[m], "layer": layer}
        for probe, name in [
            ("streaming_depth", "depth"), ("streaming_mask", "inst"),
            ("view_consistency", "view"), ("ego_belief_v2", "persist"),
            ("action_dynamics", "direct"), ("path_integration", "path"),
        ]:
            sub = df[(df.model == m) & (df.probe == probe) & (df.layer == layer)]
            r[name] = sub.primary_value.mean() if len(sub) else np.nan
        sub = depth[(depth.model == m) & (depth.layer == layer)]
        r["pose"] = sub.auc30.mean() if len(sub) else np.nan
        rows.append(r)
    return pd.DataFrame(rows).set_index("model")


def action_share(df):
    c1 = df[df.probe == "action_dynamics"].copy()
    for k in ["no_action_global_R@1", "shuffled_action_global_R@1"]:
        c1[k] = c1.secondary_json.map(lambda s: jget(s, k))
    c1["static"] = c1[["no_action_global_R@1", "shuffled_action_global_R@1"]].max(axis=1)
    c1["gain"] = c1.primary_value - c1.static
    rows = []
    for m, layer in DESIGNATED.items():
        sub = c1[(c1.model == m) & (c1.layer == layer)]
        if len(sub) == 0:
            continue
        r1, st, ga = sub.primary_value.mean(), sub.static.mean(), sub.gain.mean()
        rows.append({"model": m, "label": LABEL[m], "R1": r1, "static": st,
                     "gain": ga, "share": ga / r1 if r1 > 0 else np.nan})
    return pd.DataFrame(rows).set_index("model").sort_values("R1", ascending=False)


def c3_summary(c3):
    p = c3[c3.variant == "c3_from_c1"].copy()
    for k in ["intervention_margin", "action_sensitivity_cosine",
              "target_sensitivity_cosine", "counterfactual_consistency"]:
        p[k] = p.secondary_json.map(lambda s: jget(s, k))
    g = p.groupby("model").agg(
        validity=("primary_value", "mean"),
        margin=("intervention_margin", "mean"),
        action_sens=("action_sensitivity_cosine", "mean"),
        target_sens=("target_sensitivity_cosine", "mean"),
        consistency=("counterfactual_consistency", "mean"),
    )
    g["label"] = [LABEL[m] for m in g.index]
    return g


def loo_table(T):
    H = pd.DataFrame({
        "depth": -T.depth, "inst": -T.inst, "view": T.view,
        "persist": -T.persist, "direct": T.direct, "path": T.path, "pose": T.pose,
    })
    rows = []
    for a, b in [("persist", "pose"), ("persist", "depth"), ("direct", "depth"),
                 ("direct", "inst"), ("direct", "pose"), ("path", "depth")]:
        sub = H[[a, b]].dropna()
        r, p = spearmanr(sub[a], sub[b])
        loos = []
        for drop in sub.index:
            q = sub.drop(drop)
            rr, _ = spearmanr(q[a], q[b])
            loos.append(rr)
        rows.append({"pair": f"{a}~{b}", "n": len(sub), "rho": r, "p": p,
                     "loo_min": min(loos), "loo_max": max(loos)})
    return pd.DataFrame(rows)


def fig_action_share(A, path):
    A = A.sort_values("R1")
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    y = np.arange(len(A))
    ax.barh(y, A.static, color="#d9d9d9", label="no-action / shuffled baseline")
    ax.barh(y, A.gain, left=A.static, color="#2f6f9f", label="action-attributable")
    ax.set_yticks(y, A.label)
    ax.set_xlabel("Direct-prediction Global R@1")
    ax.set_title("How much of future-feature retrieval needs the action?")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    for i, (_, r) in enumerate(A.iterrows()):
        ax.text(r.R1 + 0.004, i, f"{r.share*100:.0f}%", va="center", fontsize=8)
    ax.set_xlim(0, 0.29)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_pose_persist(T, path):
    T = T.dropna(subset=["pose", "persist"])
    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    ax.scatter(T.pose, T.persist, s=42, color="#2f6f9f")
    for m, r in T.iterrows():
        ax.annotate(LABEL[m], (r.pose, r.persist), xytext=(4, 4),
                    textcoords="offset points", fontsize=8)
    ax.set_xlabel("Camera-pose AUC@30 (higher is better)")
    ax.set_ylabel("Object-persistence angular error (deg, lower is better)")
    ax.invert_yaxis()
    ax.set_title("Persistence follows camera pose\nSpearman rho=+0.80, n=9", fontsize=10)
    ax.set_xlim(0, 0.42)
    fig.tight_layout(pad=1.2)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_c3(C, path):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    x = C.action_sens.clip(lower=1e-6)
    colors = ["#2f6f9f" if m == "dino" else "#a6a6a6" for m in C.index]
    ax.scatter(x, C.margin, s=46, c=colors)
    offsets = {"dino": (-58, 4), "vjepa2": (-20, 8), "videomae": (4, -10),
               "qwen2_5_vl": (4, 4), "fast3r": (4, 6), "wan": (4, -10),
               "lingbot": (4, 4), "aether": (4, 6), "cogvideox": (4, 6)}
    for m, r in C.iterrows():
        ax.annotate(r.label, (max(r.action_sens, 1e-6), r.margin),
                    xytext=offsets.get(m, (4, 4)), textcoords="offset points", fontsize=8)
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(1e-3, color="#bbbbbb", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("action sensitivity (cosine, log scale)")
    ax.set_ylabel("intervention margin")
    ax.set_title("Only DINOv2 moves its prediction enough to matter", fontsize=10)
    fig.tight_layout(pad=1.2)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    canon, c3 = load()
    T = main_table(canon)
    A = action_share(canon)
    C = c3_summary(c3)
    L = loo_table(T)
    T.to_csv(os.path.join(OUT, "probe_main_table.csv"))
    A.to_csv(os.path.join(OUT, "probe_action_share.csv"))
    C.to_csv(os.path.join(OUT, "probe_c3_summary.csv"))
    L.to_csv(os.path.join(OUT, "probe_loo.csv"), index=False)
    fig_action_share(A, os.path.join(OUT, "fig_action_share.png"))
    fig_pose_persist(T, os.path.join(OUT, "fig_pose_persist.png"))
    fig_c3(C, os.path.join(OUT, "fig_c3.png"))
    print(T.round(4).to_string())
    print("\nAction share:\n", A.round(4).to_string())
    print("\nC3:\n", C.round(4).to_string())
    print("\nLOO:\n", L.round(3).to_string())


if __name__ == "__main__":
    main()
