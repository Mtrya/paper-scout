"""Produce report figures from roundtrip + rope probe results."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figs")
os.makedirs(FIG, exist_ok=True)

DEPTHS = [5, 10, 20, 40, 80]
REGIMES = ["pendulum", "r0.5", "r3.0", "r10.0", "r20.0", "r28.0"]
LABELS = {"pendulum": "pendulum (volume-preserving)",
          "r0.5": "r=0.5 (globally stable origin)",
          "r3.0": "r=3", "r10.0": "r=10 (stable fixed pts)",
          "r20.0": "r=20", "r28.0": "r=28 (strange attractor)"}
COLORS = {"pendulum": "#111111", "r0.5": "#d62728", "r3.0": "#ff7f0e", "r10.0": "#2ca02c",
          "r20.0": "#1f77b4", "r28.0": "#7f3fbf"}

S = json.load(open(os.path.join(RES, "roundtrip_summary.json")))

# --- Fig 1: growth of E / C / delta with depth, three regimes ---
fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.6), sharey=False)
for ax, reg, title in [(axes[0], "pendulum", "pendulum  (volume-preserving control)"),
                       (axes[1], "r0.5", "Lorenz r = 0.5  (origin globally stable)"),
                       (axes[2], "r28.0", "Lorenz r = 28  (strange attractor)")]:
    e = S[reg]
    ax.plot(DEPTHS, e["E_med"], "o-", color="#1f77b4", lw=2, label="forward rollout error $E_i$ (median)")
    ax.plot(DEPTHS, e["C_med"], "s-", color="#d62728", lw=2, label="round-trip $C_i$ (median)")
    ax.plot(DEPTHS, e["delta_med"], "^--", color="#888888", lw=1.5, label="reverse-leg error $\\delta_i$ (median)")
    ax.set_yscale("log")
    ax.set_xlabel("rollout depth $i$ (steps)")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3, which="both")
axes[0].set_ylabel("error (log scale)")
axes[0].legend(fontsize=8, loc="upper left")
fig.suptitle("Round-trip $C_i$ is calibrated when the inverse is stable — and explodes with the reverse leg when it is not",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(os.path.join(FIG, "fig_roundtrip_growth.png"), dpi=160)
plt.close(fig)

# --- Fig 2: Spearman(C, E) vs depth, all regimes ---
fig, ax = plt.subplots(figsize=(6.4, 3.6))
for reg in REGIMES:
    rho = [x if x is not None else np.nan for x in S[reg]["rho_CE"]]
    ax.plot(DEPTHS, rho, "o-", color=COLORS[reg], lw=1.8, ms=4, label=LABELS[reg])
ax.axhspan(0.91, 0.98, color="green", alpha=0.15, label="paper's reported range (0.91-0.98)")
ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel("rollout depth $i$ (steps)")
ax.set_ylabel("Spearman $\\rho(C_i, E_i)$ across 128 ICs")
ax.set_ylim(-0.3, 1.05)
ax.grid(alpha=0.3)
ax.legend(fontsize=7.5, loc="lower left")
ax.set_title("Round-trip $C_i$ never ranks forward error $E_i$ in our surrogate — in any regime", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_roundtrip_rho.png"), dpi=160)
plt.close(fig)

# --- Fig 3: RoPE survival vs theta*Dt ---
R = json.load(open(os.path.join(RES, "rope_probe.json")))
fig, ax = plt.subplots(figsize=(6.4, 3.6))
for cfg, marker, alpha in [(3, "o", 0.85), (2, "s", 0.45), (1, "^", 0.3)]:
    b = R["B"][cfg]
    ax.scatter(b["theta_Dt"], b["survival"], marker=marker, s=22,
               color="#d62728", alpha=alpha,
               label=f"naive mean (M={b['M']}, Δt={b['Dt']})" if cfg == 3 else None)
    if cfg == 3:
        ax.scatter(b["theta_Dt"], b["canon_survival"], marker="o", s=22,
                   color="#1f77b4", alpha=0.85,
                   label=f"canonical (un-rotate→mean→re-rotate)")
ax.axvline(2.0, color="k", ls="--", lw=1)
ax.text(2.15, 0.05, "θ·Δt = 2", fontsize=8)
ax.set_xlabel("θ·Δt  (phase drift per frequency pair, radians)")
ax.set_ylabel("pair survival (norm after averaging / before)")
ax.set_ylim(-0.05, 1.05)
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
ax.set_title("Naive averaging of RoPE-rotated keys cancels high-frequency pairs", fontsize=9.5)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_rope_survival.png"), dpi=160)
plt.close(fig)

# --- Fig 4: softmax read-mass ratio, naive vs canonical ---
J = json.load(open(os.path.join(RES, "rope_softmax.json")))
fig, ax = plt.subplots(figsize=(5.6, 3.4))
x = np.arange(len(J))
w = 0.35
ax.bar(x - w/2, [j["naive_ratio_median"] for j in J], w, color="#d62728", label="naive mean")
ax.bar(x + w/2, [j["canon_ratio_median"] for j in J], w, color="#1f77b4", label="canonical")
for i, j in enumerate(J):
    ax.text(i - w/2, j["naive_ratio_median"] + 0.004, f"{j['naive_ratio_median']:.3f}", ha="center", fontsize=8)
    ax.text(i + w/2, j["canon_ratio_median"] + 0.004, f"{j['canon_ratio_median']:.3f}", ha="center", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels([f"M={j['M']}, stride={j['stride']}\n(Δt={j['Dt']}, n={j['n']})" for j in J], fontsize=9)
ax.set_ylabel("softmax mass on summary slot\n/ mass full cache puts on sources (median)")
ax.set_ylim(0, max(j["canon_ratio_median"] for j in J) * 1.25)
ax.grid(alpha=0.3, axis="y")
ax.legend(fontsize=8)
ax.set_title("Compressed slot readability: 8 KV heads × 16 queries × 3 layers", fontsize=9.5)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_rope_softmax.png"), dpi=160)
plt.close(fig)

print("figs written:", os.listdir(FIG))
