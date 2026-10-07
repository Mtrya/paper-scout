"""Analyse the online-TTT arms: paired curves, stability metrics, B1-B5 verdicts, figures.

    python analyze.py <results-dir> <analysis-out-dir> [<assets-dir>]

Writes arms_results.jsonl / arms_eval.jsonl / arms_updates.jsonl / curves.json /
summary.json into the analysis dir, and ttt-*.png into the assets dir.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import numpy as np

ARMS = ["frozen", "loop-imitate", "rft", "rft-settlement", "ascent", "loop-imitate-frozen-gen"]
NICE = {"frozen": "frozen (base)", "loop-imitate": "loop-imitate (ungated)",
        "rft": "RFT (verified-only)", "rft-settlement": "RFT + settlement gate",
        "ascent": "ASCENT (hindsight self-distill)",
        "loop-imitate-frozen-gen": "loop-imit, frozen generation"}
COLORS = {"frozen": "#444444", "loop-imitate": "#d62728", "rft": "#ff7f0e",
          "rft-settlement": "#2ca02c", "ascent": "#1f77b4",
          "loop-imitate-frozen-gen": "#9467bd"}
WINDOW = 40


def load(path):
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path)]


def trailing(xs, w=WINDOW):
    if len(xs) < w:
        return []
    c = np.cumsum([0.0] + list(xs))
    return [(c[i + w] - c[i]) / w for i in range(len(xs) - w + 1)]


def boot_ci(vals, n=10000, seed=0):
    vals = np.asarray(vals, dtype=float)
    if len(vals) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vals), size=(n, len(vals)))
    m = vals[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    res = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else res
    assets = sys.argv[3] if len(sys.argv) > 3 else ""
    os.makedirs(out, exist_ok=True)
    stream = {a: load(f"{res}/{a}_stream.jsonl") for a in ARMS}
    evals = {a: load(f"{res}/{a}_eval.jsonl") for a in ARMS}
    updates = {a: load(f"{res}/{a}_updates.jsonl") for a in ARMS}
    runs = [a for a in ARMS if stream[a]]
    base = stream["frozen"]
    print("loaded: " + ", ".join(f"{a}={len(stream[a])}" for a in runs), flush=True)

    def seg(which, n):
        return (0, n) if which == "all" else ((0, n // 3) if which == "first"
                                              else (2 * n // 3, n))

    def agg(a, which="all"):
        r = stream[a]
        n = min(len(r), len(base))
        i0, i1 = seg(which, n)
        s = r[i0:i1]
        if not s:
            return {}
        m = lambda f: float(np.mean([f(x) for x in s]))              # noqa: E731
        return {"n": len(s), "succ": m(lambda x: x["success"]),
                "fmt": m(lambda x: x["format_valid_rate"]),
                "inv": m(lambda x: x["invalid_turns"]),
                "inv_rate": m(lambda x: x["invalid_turns"] / max(1, x["n_turns"])),
                "H": m(lambda x: x["mean_tok_entropy"] or 0.0),
                "H1": m(lambda x: x["first_tok_entropy"] or 0.0),
                "kl": m(lambda x: x["seq_kl_cur_base"] or 0.0),
                "tok": m(lambda x: x["n_resp_tokens"]),
                "turns": m(lambda x: x["n_turns"])}

    def paired(a, which="all"):
        r = stream[a]
        n = min(len(r), len(base))
        i0, i1 = seg(which, n)
        return [r[i]["success"] - base[i]["success"] for i in range(i0, i1)]

    summary = {}
    print(f"\n{'arm':30s} {'n':>4s} {'succ':>6s} {'Δbase':>7s} {'95% CI':>17s} "
          f"{'lateΔ':>7s} {'fmt':>6s} {'inv/t':>6s} {'H':>7s} {'KL↑':>6s} {'tok/t':>6s}",
          flush=True)
    for a in runs:
        g, gl = agg(a, "all"), agg(a, "last")
        d, dl = paired(a), paired(a, "last")
        lo, hi = boot_ci(d)
        llo, lhi = boot_ci(dl)
        summary[a] = {"all": g, "first": agg(a, "first"), "last": gl,
                      "paired_delta": float(np.mean(d)) if d else None,
                      "paired_delta_ci": [lo, hi],
                      "paired_delta_last": float(np.mean(dl)) if dl else None,
                      "paired_delta_last_ci": [llo, lhi]}
        print(f"{NICE.get(a,a):30s} {g['n']:4d} {g['succ']:6.3f} {np.mean(d):+7.3f} "
              f"[{lo:+.3f},{hi:+.3f}] {np.mean(dl):+7.3f} {g['fmt']:6.3f} "
              f"{g['inv_rate']:6.3f} {g['H']:7.3f} {g['kl']:6.3f} "
              f"{g['tok']/max(1e-9,g['turns']):6.1f}", flush=True)

    print("\nper-family paired Δ success vs base (all / late third)")
    print(f"{'arm':30s} " + " ".join(f"{f:>15s}" for f in
                                     ("strxform", "calc", "world", "listops")))
    fam = defaultdict(dict)
    for a in runs:
        if a == "frozen":
            continue
        row = f"{NICE.get(a,a):30s} "
        for f in ("strxform", "calc", "world", "listops"):
            n = min(len(stream[a]), len(base))
            idx = [i for i in range(n) if stream[a][i]["family"] == f]
            d = [stream[a][i]["success"] - base[i]["success"] for i in idx]
            dl = [stream[a][i]["success"] - base[i]["success"] for i in idx
                  if i >= 2 * n // 3]
            fam[a][f] = {"delta_all": float(np.mean(d)) if d else None,
                         "delta_late": float(np.mean(dl)) if dl else None, "n": len(idx)}
            row += f"{np.mean(d):+.2f}/{np.mean(dl) if dl else 0:+.2f}".rjust(16)
        summary[a]["family_delta"] = fam[a]
        print(row, flush=True)

    print("\nreal-text probe NLL and validation-set curves (first -> last eval point)")
    for a in runs:
        e = evals[a]
        if not e:
            continue
        s0, s1 = e[0], e[-1]
        summary[a].update({
            "probe_nll_first": s0.get("probe_nll"), "probe_nll_last": s1.get("probe_nll"),
            "probe_nll_delta": (s1.get("probe_nll") or 0) - (s0.get("probe_nll") or 0),
            "val_success_first": s0.get("val_success"), "val_success_last": s1.get("val_success"),
            "val_fmt_first": s0.get("val_format_valid"), "val_fmt_last": s1.get("val_format_valid"),
            "val_curve": [[x["task_idx"], x.get("val_success")] for x in e],
            "nll_curve": [[x["task_idx"], x.get("probe_nll")] for x in e],
            "val_fmt_curve": [[x["task_idx"], x.get("val_format_valid")] for x in e],
            "val_kl_curve": [[x["task_idx"], x.get("val_seq_kl_cur_base")] for x in e],
        })
        print(f"  {NICE.get(a,a):30s} nll {s0.get('probe_nll'):.4f}->{s1.get('probe_nll'):.4f} "
              f"({s1.get('probe_nll')-s0.get('probe_nll'):+.4f})  "
              f"val_succ {s0.get('val_success'):.3f}->{s1.get('val_success'):.3f}  "
              f"val_fmt {s0.get('val_format_valid'):.3f}->{s1.get('val_format_valid'):.3f}",
              flush=True)
    for a in runs:
        u = updates[a]
        if not u:
            continue
        used = [x for x in u if not x.get("skipped")]
        rej = [x for x in used if x.get("rejected")]
        summary[a].update({"update_buffers": len(u), "update_used": len(used),
                           "update_rejected": len(rej),
                           "mean_gated": float(np.mean([x.get("gated_n", 0) for x in u])),
                           "mean_update_wall": float(np.mean([x["wall_s"] for x in u])),
                           "update_wall_total": float(np.sum([x["wall_s"] for x in u]))})
        print(f"  {NICE.get(a,a):30s} buffers={len(u)} applied={len(used)} "
              f"rejected={len(rej)} mean gated/buffer={summary[a]['mean_gated']:.1f} "
              f"total update time={summary[a]['update_wall_total']:.0f}s", flush=True)

    # ---- data dumps ----------------------------------------------------------------
    with open(f"{out}/arms_results.jsonl", "w") as f:
        for a in runs:
            for x in stream[a]:
                f.write(json.dumps(x) + "\n")
    with open(f"{out}/arms_eval.jsonl", "w") as f:
        for a in runs:
            for x in evals[a]:
                f.write(json.dumps(x) + "\n")
    with open(f"{out}/arms_updates.jsonl", "w") as f:
        for a in runs:
            for x in updates[a]:
                f.write(json.dumps(x) + "\n")
    curves = {}
    for a in runs:
        n = min(len(stream[a]), len(base))
        curves[a] = {
            "succ_window": trailing([x["success"] for x in stream[a]]),
            "paired_delta_window": trailing(
                [stream[a][i]["success"] - base[i]["success"] for i in range(n)]),
            "fmt_window": trailing([x["format_valid_rate"] for x in stream[a]]),
            "H_window": trailing([x["mean_tok_entropy"] or 0.0 for x in stream[a]]),
            "KL_window": trailing([x["seq_kl_cur_base"] or 0.0 for x in stream[a]]),
            "inv_rate_window": trailing(
                [x["invalid_turns"] / max(1, x["n_turns"]) for x in stream[a]]),
            "window": WINDOW,
        }
    with open(f"{out}/curves.json", "w") as f:
        json.dump(curves, f)
    with open(f"{out}/summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nwrote {out}/arms_results.jsonl, arms_eval.jsonl, arms_updates.jsonl, "
          f"curves.json, summary.json", flush=True)

    if not assets:
        return
    os.makedirs(assets, exist_ok=True)

    # ---- report figures -------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 150, "font.size": 12, "axes.titlesize": 13,
                         "axes.labelsize": 12, "legend.fontsize": 10,
                         "xtick.labelsize": 11, "ytick.labelsize": 11,
                         "axes.grid": True, "grid.alpha": 0.3,
                         "axes.spines.top": False, "axes.spines.right": False})

    def W(a, key, transform=None):
        if key in curves.get(a, {}):
            return curves[a][key]
        v = [(transform(x[key]) if transform else x[key]) for x in stream[a]]
        v = [0.0 if x is None else x for x in v]
        return trailing(v, WINDOW)

    xs = lambda c: range(WINDOW, WINDOW + len(c))          # noqa: E731

    # Figure 1: success
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for a in runs:
        c = W(a, "success")
        axes[0].plot(xs(c), c, label=NICE.get(a, a), color=COLORS[a], lw=2)
    axes[0].set_title(f"Stream success rate (trailing {WINDOW}-task window)")
    axes[0].set_xlabel("task index in the arriving stream")
    axes[0].set_ylabel("success rate")
    axes[0].set_ylim(0, 1.0)
    axes[0].axvspan(160, 240, color="grey", alpha=0.10)
    axes[0].text(168, 0.03, "hardest phase (L3)", fontsize=9, color="#555555")
    axes[0].legend(loc="upper right", framealpha=0.9)
    for a in runs:
        if a == "frozen":
            continue
        c = curves[a]["paired_delta_window"]
        if c:
            axes[1].plot(xs(c), c, label=NICE.get(a, a), color=COLORS[a], lw=2)
    axes[1].axhline(0, color="k", lw=1)
    axes[1].set_title("Paired Δ success vs frozen base, same task order")
    axes[1].set_xlabel("task index in the arriving stream")
    axes[1].set_ylabel("Δ success rate")
    axes[1].legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(f"{assets}/ttt-success-curves.png")
    plt.close(fig)
    print(f"wrote {assets}/ttt-success-curves.png", flush=True)

    # Figure 2: stability
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.4))
    for a in runs:
        c = W(a, "fmt_window")
        axes[0].plot(xs(c), c, label=NICE.get(a, a), color=COLORS[a], lw=2)
    axes[0].set_title("Format-valid turn rate")
    axes[0].set_xlabel("task index"); axes[0].set_ylim(0, 1.0)
    axes[0].legend(loc="lower left", framealpha=0.9)
    for a in runs:
        c = W(a, "H_window")
        axes[1].plot(xs(c), c, label=NICE.get(a, a), color=COLORS[a], lw=2)
    axes[1].set_title("Mean per-token entropy of generated responses")
    axes[1].set_xlabel("task index"); axes[1].set_ylabel("nats")
    for a in runs:
        c = W(a, "KL_window")
        axes[2].plot(xs(c), c, label=NICE.get(a, a), color=COLORS[a], lw=2)
    axes[2].set_title("KL(current ‖ base) on generated responses")
    axes[2].set_xlabel("task index"); axes[2].set_ylabel("nats/token")
    fig.tight_layout()
    fig.savefig(f"{assets}/ttt-stability.png")
    plt.close(fig)
    print(f"wrote {assets}/ttt-stability.png", flush=True)

    # Figure 3: probes
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.4))
    for a in runs:
        e = evals[a]
        if not e:
            continue
        axes[0].plot([x["task_idx"] for x in e], [x.get("val_success") for x in e],
                     marker="o", ms=5, lw=2, label=NICE.get(a, a), color=COLORS[a])
        axes[1].plot([x["task_idx"] for x in e], [x.get("probe_nll") for x in e],
                     marker="o", ms=5, lw=2, label=NICE.get(a, a), color=COLORS[a])
        axes[2].plot([x["task_idx"] for x in e], [x.get("val_format_valid") for x in e],
                     marker="o", ms=5, lw=2, label=NICE.get(a, a), color=COLORS[a])
    axes[0].set_title("Fixed 20-task validation success")
    axes[0].set_xlabel("task index"); axes[0].set_ylim(0, 1); axes[0].legend(framealpha=0.9)
    axes[1].set_title("NLL on 50 held-out real English passages")
    axes[1].set_xlabel("task index"); axes[1].set_ylabel("nats/token")
    axes[2].set_title("Validation format-valid rate")
    axes[2].set_xlabel("task index"); axes[2].set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(f"{assets}/ttt-probes.png")
    plt.close(fig)
    print(f"wrote {assets}/ttt-probes.png", flush=True)

    # Figure 4: per-family deltas
    fams = ("strxform", "calc", "world", "listops")
    shown = [a for a in runs if a != "frozen"]
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    w = 0.8 / max(1, len(shown))
    for i, a in enumerate(shown):
        vals = [fam[a][f]["delta_late"] or 0.0 for f in fams]
        ax.bar(np.arange(len(fams)) + (i - (len(shown) - 1) / 2) * w, vals, width=w,
               label=NICE.get(a, a), color=COLORS[a])
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(range(len(fams)))
    ax.set_xticklabels(["string\nxform", "arith +\ncalc tool", "text-world\nagent",
                        "list ops +\ntool"])
    ax.set_ylabel("Δ success (late third) vs base")
    ax.set_title("Per-family paired success gain, final third of the stream")
    ax.legend(framealpha=0.9, fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{assets}/ttt-family-deltas.png")
    plt.close(fig)
    print(f"wrote {assets}/ttt-family-deltas.png", flush=True)


if __name__ == "__main__":
    main()
