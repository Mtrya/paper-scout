"""Build the probe figures from results/*.json."""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = os.path.join(os.path.dirname(__file__), "results")
OUT = sys.argv[1] if len(sys.argv) > 1 else RES

BUCKET_X = {"8-16": 12, "16-32": 24, "32-64": 48, "64-128": 96,
            "128-256": 192, "256-384": 320, "384-512": 448}

STYLES = {
    "ttt":   dict(color="#1a9850", marker="o", lw=2.2, label="TTT-MLP (RoboTTT mechanism)"),
    "delta": dict(color="#4575b4", marker="s", lw=2.2, label="Gated delta rule (linear attn / GDN-lite)"),
    "attn":  dict(color="#d73027", marker="^", lw=2.2, label="Causal attention (KV-cached)"),
    "noctx": dict(color="#888888", marker="x", lw=1.5, ls=":", label="No context (per-step MLP)"),
}


def load(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


def fig_scaling():
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    ax = axes[0]
    for key, fname in [("ttt", "ttt_T512.json"), ("delta", "delta_T512.json"),
                       ("attn", "attn_T512.json"), ("noctx", "noctx.json")]:
        r = load(fname)
        xs = [BUCKET_X[b] for b in r["r2_by_context_bucket"]]
        ys = list(r["r2_by_context_bucket"].values())
        ax.plot(xs, ys, **STYLES[key])
    r64 = load("ttt_T64.json")
    xs = [BUCKET_X[b] for b in r64["r2_by_context_bucket"]]
    ys = list(r64["r2_by_context_bucket"].values())
    ax.plot(xs, ys, color="#1a9850", marker="o", mfc="none", lw=1.5,
            ls="--", label="TTT-MLP, trained at 64-step context")
    ax.set_ylim(-1.7, 1.02)
    ax.set_xscale("log", base=2)
    ax.set_xticks([16, 32, 64, 128, 256, 512])
    ax.set_xticklabels(["16", "32", "64", "128", "256", "512"])
    ax.set_xlabel("context length (elapsed timesteps of the rollout)")
    ax.set_ylabel("$R^2$ of action prediction\n(unsaturated steps, held-out tasks)")
    ax.set_title("In-context task identification vs context length\n"
                 "(dip at 16-32 = first trial, no cross-phase context yet)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")

    # panel (b): 2x-horizon extrapolation (T=1024, trained at 512)
    ax = axes[1]
    r = load("eval_long_T1024.json")
    x1024 = {"256-384": 320, "384-512": 448, "512-640": 576, "640-768": 704,
             "768-896": 832, "896-1024": 960}
    for key, name in [("ttt", "ttt_T512"), ("delta", "delta_T512"),
                      ("attn", "attn_T512")]:
        xs = [x1024[b] for b in r[name]]
        ys = list(r[name].values())
        ax.plot(xs, ys, **{k: v for k, v in STYLES[key].items() if k != "ls"})
    xs = [x1024[b] for b in r["ttt_T64"]]
    ys = list(r["ttt_T64"].values())
    ax.plot(xs, ys, color="#1a9850", marker="o", mfc="none", lw=1.5, ls="--",
            label="TTT-MLP, trained at 64-step context")
    ax.axvline(512, color="#bbbbbb", lw=1, ls="--")
    ax.text(520, -2.6, "training window (512)", fontsize=8, color="#888888")
    ax.set_ylim(-3.2, 1.02)
    ax.set_xscale("log", base=2)
    ax.set_xticks([256, 512, 1024])
    ax.set_xticklabels(["256", "512", "1024"])
    ax.set_xlabel("context length (2x beyond the 512-step training window)")
    ax.set_ylabel("$R^2$ of action prediction")
    ax.set_title("Horizon extrapolation: fast-weight drift vs gated-state "
                 "stability\n(models trained at 512, evaluated to 1024)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    p = os.path.join(OUT, "robotttt-probe-scaling.png")
    fig.savefig(p, dpi=160)
    print("wrote", p)


def fig_latency():
    r = load("latency_d256.json")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    xs_all = list(range(1, r["T"] + 1))
    for m in ["ttt", "delta", "attn"]:
        step = r["mixers"][m]["step_ms"]
        # light smoothing: mean over 32-step blocks
        bl = 32
        ys = [sum(step[i:i + bl]) / len(step[i:i + bl])
              for i in range(0, len(step), bl)]
        xs = [i + bl / 2 for i in range(0, len(step), bl)]
        ax.plot(xs, ys, **{k: v for k, v in STYLES[m].items()
                           if k not in ("ls",)})
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("context position t (batch=1, single CPU thread)")
    ax.set_ylabel("per-step latency (ms)")
    ax.set_title(f"Per-step cost vs context (d={r['d']}, {r['layers']} layers)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8.5)

    ax = axes[1]
    for m in ["ttt", "delta", "attn"]:
        step = r["mixers"][m]["step_ms"]
        cum, c = [], 0.0
        for v in step:
            c += v
            cum.append(c)
        ax.plot(xs_all[::16], cum[::16], **{k: v for k, v in STYLES[m].items()
                                            if k not in ("ls",)})
    ax.set_xlabel("timesteps decoded")
    ax.set_ylabel("cumulative latency (ms)")
    ax.set_title("Total rollout cost: O(T^2) vs O(T)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(OUT, "robotttt-probe-latency.png")
    fig.savefig(p, dpi=160)
    print("wrote", p)


def fig_diag():
    r = load("ttt_diag.json")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    ax = axes[0]
    ax.plot(range(1, len(r["inner_loss_l0"]) + 1), r["inner_loss_l0"],
            color="#1a9850", lw=1.8)
    for p in range(48, 512, 48):
        ax.axvline(p, color="#dddddd", lw=0.7, zorder=0)
    ax.set_xlabel("context position t (phase boundaries in grey)")
    ax.set_ylabel("inner TTT loss  " + r"$\|f_W(k_t)-v_t\|^2$")
    ax.set_title("Fast model fits the stream better as context grows\n"
                 "(layer-0 self-supervised loss, before update)")
    ax.grid(alpha=0.3)
    ax = axes[1]
    ax.plot(range(1, len(r["drift_l0"]) + 1), r["drift_l0"],
            color="#984ea3", lw=1.8)
    ax.set_xlabel("context position t")
    ax.set_ylabel(r"$\|W_t - W_0\|_F / \|W_0\|_F$")
    ax.set_title("Fast-weight drift over a 512-step rollout")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(OUT, "robotttt-probe-ttt-inner.png")
    fig.savefig(p, dpi=160)
    print("wrote", p)


if __name__ == "__main__":
    fig_scaling()
    fig_latency()
    fig_diag()
