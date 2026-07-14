#!/usr/bin/env python3
"""
Parse the RoboDojo paper leaderboard tables and compute diagnostic gaps.

Inputs:
    papers/robotics/robodojo-2607.04434.md (the paper markdown)

Outputs (written to RUN_ASSETS/robodojo/):
    sim_ranking.csv
    sim_gaps.csv
    real_ranking.csv
    sim_real_overlap.csv
    summary.md
    sim_top10_success.png
    sim_dimension_gaps.png
    sim_real_scatter.png
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

PAPER = Path(__file__).resolve().parents[4] / "papers" / "robotics" / "robodojo-2607.04434.md"
RUN_ASSETS = Path(__file__).resolve().parents[2] / "assets" / "robodojo"

SIM_DIMS = ["Generalization", "Precision", "Long-Horizon", "Memory", "Open", "Average"]


def read_lines(path: Path) -> List[str]:
    return path.read_text(encoding="utf-8").splitlines()


def parse_sim_table(lines: List[str]) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    """Return (policy_rows, human_row) where each row maps dimension -> success_rate."""
    policies: Dict[str, Dict[str, float]] = {}
    human: Dict[str, float] | None = None

    in_table = False
    for line in lines:
        if "Table 1: RoboDojo Simulation Benchmark Leaderboard" in line:
            in_table = True
            continue
        if not in_table:
            continue
        if line.startswith("### 5.2"):
            break
        if not line.strip():
            continue
        if line.startswith("Alg "):
            continue

        cells = re.findall(r"(\d+\.\d+)\s*/\s*(\d+\.\d+)%", line)
        if len(cells) != 6:
            continue

        # Name ends at first ')' (the citation). Human Expert has no citation.
        if line.startswith("Human Expert"):
            name = "Human Expert (Teleop)"
            human = {dim: float(sr) for dim, (_, sr) in zip(SIM_DIMS, cells)}
            continue

        close = line.find(")")
        name = line[: close + 1].strip() if close != -1 else line.strip()
        policies[name] = {dim: float(sr) for dim, (_, sr) in zip(SIM_DIMS, cells)}

    if human is None:
        raise RuntimeError("Could not parse human expert row in simulation table")
    return policies, human


def parse_real_table(lines: List[str]) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    """Return (policy_rows, human_row) for real-world overall average."""
    policies: Dict[str, Dict[str, float]] = {}
    human: Dict[str, float] = {"score": 100.0, "success": 100.0}

    in_table = False
    current_policy: str | None = None
    for line in lines:
        if "Table 2: RoboDojo Real-World Benchmark Leaderboard" in line:
            in_table = True
            continue
        if not in_table:
            continue
        if line.startswith("## 6."):
            break
        if not line.strip():
            continue
        if line.startswith("Policy "):
            continue
        if line.startswith("Human Expert"):
            continue

        # Policy rows contain a citation link; embodiment rows do not.
        is_policy_row = "arxiv.org/html" in line or "et al.," in line
        if is_policy_row:
            close = line.find(")")
            current_policy = line[: close + 1].strip() if close != -1 else line.strip()
            # Overall Avg is the last score/success pair on the first (policy) row.
            m = re.search(r"(\d+\.\d+)\s*/\s*(\d+\.\d+)$", line.strip())
            if m and current_policy:
                policies[current_policy] = {
                    "score": float(m.group(1)),
                    "success": float(m.group(2)),
                }

    return policies, human


def normalize_name(name: str) -> str:
    """Very light normalization so sim/real names can be matched."""
    return name.strip().replace("\\pi_", "pi_").replace("\\", "").replace(" ", " ")


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_sim_top10(policies: Dict[str, Dict[str, float]], human: Dict[str, float], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ranked = sorted(
        ((n, r["Average"]) for n, r in policies.items()), key=lambda x: x[1], reverse=True
    )[:10]
    names = [r[0] for r in ranked]
    vals = [r[1] for r in ranked]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(names)), vals, color="steelblue")
    ax.axvline(human["Average"], color="crimson", linestyle="--", label="Human teleop (76.03%)")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Average success rate (%)")
    ax.set_title("Top 10 policies on RoboDojo simulation (average success)")
    ax.legend()
    for bar, v in zip(bars, vals):
        ax.text(v + 0.5, bar.get_y() + bar.get_height() / 2, f"{v:.2f}", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_dimension_gaps(
    policies: Dict[str, Dict[str, float]], human: Dict[str, float], out: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dims = SIM_DIMS[:-1]  # exclude Average
    best = []
    best_names = []
    human_vals = []
    for dim in dims:
        ranked = sorted(policies.items(), key=lambda x: x[1][dim], reverse=True)
        top_name, top_row = ranked[0]
        best.append(top_row[dim])
        best_names.append(top_name)
        human_vals.append(human[dim])

    x = list(range(len(dims)))
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.35
    ax.bar([i - width / 2 for i in x], list(human_vals), width, label="Human teleop", color="crimson")
    ax.bar([i + width / 2 for i in x], list(best), width, label="Best policy", color="steelblue")
    ax.set_xticks(x)
    ax.set_xticklabels(dims, rotation=15, ha="right")
    ax.set_ylabel("Success rate (%)")
    ax.set_title("Per-dimension best policy vs human teleop")
    ax.legend()
    for i, (hv, bv, bn) in enumerate(zip(human_vals, best, best_names)):
        ax.text(i - width / 2, hv + 1, f"{hv:.1f}", ha="center", fontsize=7)
        ax.text(i + width / 2, bv + 1, f"{bv:.1f}\n({bn[:20]}", ha="center", fontsize=6)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_sim_real_scatter(
    sim: Dict[str, Dict[str, float]], real: Dict[str, Dict[str, float]], out: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pairs = []
    labels = []
    for name, real_row in real.items():
        sim_name = next((k for k in sim if normalize_name(k) == normalize_name(name)), None)
        if sim_name is None:
            continue
        pairs.append((sim[sim_name]["Average"], real_row["success"]))
        labels.append(name)

    xs, ys = zip(*pairs) if pairs else ([], [])
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(xs, ys, s=80, color="steelblue", zorder=3)
    for x, y, lab in zip(xs, ys, labels):
        ax.annotate(lab[:25], (x, y), textcoords="offset points", xytext=(5, 5), fontsize=7)
    ax.plot([0, max(xs or [1])], [0, max(xs or [1])], "k--", alpha=0.3, label="y=x")
    ax.set_xlabel("Simulation average success rate (%)")
    ax.set_ylabel("Real-world overall success rate (%)")
    ax.set_title("Simulation vs real-world performance (overlapping policies)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main() -> int:
    if not PAPER.exists():
        print(f"Paper markdown not found: {PAPER}", file=sys.stderr)
        return 1
    RUN_ASSETS.mkdir(parents=True, exist_ok=True)

    lines = read_lines(PAPER)
    sim_policies, sim_human = parse_sim_table(lines)
    real_policies, real_human = parse_real_table(lines)

    # Simulation ranking by average success.
    sim_ranked = sorted(sim_policies.items(), key=lambda x: x[1]["Average"], reverse=True)
    sim_ranking_rows = []
    for name, row in sim_ranked:
        sim_ranking_rows.append(
            {
                "rank": len(sim_ranking_rows) + 1,
                "policy": name,
                **{d: f"{row[d]:.2f}" for d in SIM_DIMS},
            }
        )
    write_csv(
        RUN_ASSETS / "sim_ranking.csv",
        sim_ranking_rows,
        ["rank", "policy"] + SIM_DIMS,
    )

    # Per-dimension gaps to human.
    gap_rows = []
    for dim in SIM_DIMS[:-1]:
        best_name, best_row = max(sim_policies.items(), key=lambda x: x[1][dim])
        gap_rows.append(
            {
                "dimension": dim,
                "human_score": f"{sim_human[dim]:.2f}",
                "best_policy": best_name,
                "best_success": f"{best_row[dim]:.2f}",
                "absolute_gap_pp": f"{sim_human[dim] - best_row[dim]:.2f}",
                "relative_gap_pct": f"{(sim_human[dim] - best_row[dim]) / sim_human[dim] * 100:.1f}",
            }
        )
    best_overall_name, best_overall_row = max(sim_policies.items(), key=lambda x: x[1]["Average"])
    gap_rows.append(
        {
            "dimension": "Average",
            "human_score": f"{sim_human['Average']:.2f}",
            "best_policy": best_overall_name,
            "best_success": f"{best_overall_row['Average']:.2f}",
            "absolute_gap_pp": f"{sim_human['Average'] - best_overall_row['Average']:.2f}",
            "relative_gap_pct": f"{(sim_human['Average'] - best_overall_row['Average']) / sim_human['Average'] * 100:.1f}",
        }
    )
    write_csv(
        RUN_ASSETS / "sim_gaps.csv",
        gap_rows,
        ["dimension", "human_score", "best_policy", "best_success", "absolute_gap_pp", "relative_gap_pct"],
    )

    # Real-world ranking.
    real_ranked = sorted(real_policies.items(), key=lambda x: x[1]["success"], reverse=True)
    real_ranking_rows = [
        {"rank": i + 1, "policy": name, "score": f"{row['score']:.2f}", "success": f"{row['success']:.2f}"}
        for i, (name, row) in enumerate(real_ranked)
    ]
    write_csv(RUN_ASSETS / "real_ranking.csv", real_ranking_rows, ["rank", "policy", "score", "success"])

    # Overlap sim vs real.
    overlap_rows = []
    for name, real_row in real_policies.items():
        sim_name = next((k for k in sim_policies if normalize_name(k) == normalize_name(name)), None)
        if sim_name is None:
            continue
        overlap_rows.append(
            {
                "policy": name,
                "sim_success": f"{sim_policies[sim_name]['Average']:.2f}",
                "sim_rank": next(i for i, (n, _) in enumerate(sim_ranked, 1) if n == sim_name),
                "real_success": f"{real_row['success']:.2f}",
                "real_rank": next(i for i, (n, _) in enumerate(real_ranked, 1) if n == name),
                "sim_to_real_drop_pp": f"{sim_policies[sim_name]['Average'] - real_row['success']:.2f}",
            }
        )
    overlap_rows.sort(key=lambda x: float(x["sim_success"]), reverse=True)
    for i, row in enumerate(overlap_rows, 1):
        row["rank"] = i
    write_csv(
        RUN_ASSETS / "sim_real_overlap.csv",
        overlap_rows,
        ["rank", "policy", "sim_success", "sim_rank", "real_success", "real_rank", "sim_to_real_drop_pp"],
    )

    # Plots.
    plot_sim_top10(sim_policies, sim_human, RUN_ASSETS / "sim_top10_success.png")
    plot_dimension_gaps(sim_policies, sim_human, RUN_ASSETS / "sim_dimension_gaps.png")
    plot_sim_real_scatter(sim_policies, real_policies, RUN_ASSETS / "sim_real_scatter.png")

    # Summary markdown.
    summary = []
    summary.append("# RoboDojo leaderboard probe summary\n")
    summary.append(f"- Simulation policies parsed: {len(sim_policies)}")
    summary.append(f"- Real-world policies parsed: {len(real_policies)}")
    summary.append(f"- Best simulation policy (avg success): {best_overall_name} @ {best_overall_row['Average']:.2f}%")
    summary.append(f"- Human teleop simulation avg success: {sim_human['Average']:.2f}%")
    summary.append(f"- Overall sim gap to human: {sim_human['Average'] - best_overall_row['Average']:.2f} pp")
    best_real_name, best_real_row = max(real_policies.items(), key=lambda x: x[1]["success"])
    summary.append(f"- Best real-world policy (overall success): {best_real_name} @ {best_real_row['success']:.2f}%")
    summary.append(f"- Overall real gap to human: {real_human['success'] - best_real_row['success']:.2f} pp\n")

    summary.append("## Largest per-dimension sim gaps to human teleop\n")
    for row in gap_rows:
        summary.append(
            f"- **{row['dimension']}**: human {row['human_score']}%, best ({row['best_policy']}) "
            f"{row['best_success']}% → gap {row['absolute_gap_pp']} pp ({row['relative_gap_pct']}% relative)"
        )
    summary.append("")

    summary.append("## Sim-vs-real overlap (policies in both leaderboards)\n")
    for row in overlap_rows:
        summary.append(
            f"- {row['policy']}: sim {row['sim_success']}% (rank {row['sim_rank']}) → "
            f"real {row['real_success']}% (rank {row['real_rank']}); drop {row['sim_to_real_drop_pp']} pp"
        )
    summary.append("")

    summary.append("## Artifacts produced\n")
    for f in sorted(RUN_ASSETS.glob("*")):
        if f.suffix in {".csv", ".png", ".md"}:
            summary.append(f"- `{f.name}`")

    (RUN_ASSETS / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    print(f"Artifacts written to {RUN_ASSETS}")
    print((RUN_ASSETS / "summary.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
