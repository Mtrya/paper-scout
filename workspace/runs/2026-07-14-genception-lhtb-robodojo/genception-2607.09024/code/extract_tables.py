#!/usr/bin/env python3
"""
Probe: extract quantitative evidence from the GenCeption arXiv HTML.

Reads the local paper.html, pulls Table 1 (specialist/generalist comparison)
and Table 2 (pre-training & scaling ablation), writes tidy JSON/CSV artifacts,
and produces two report-facing plots under
runs/2026-07-14-genception-lhtb-robodojo/assets/genception/.
"""

import json
import csv
from pathlib import Path
from bs4 import BeautifulSoup
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

THREAD = Path(__file__).resolve().parent.parent  # thread dir
RUN = THREAD.parent
ASSETS = RUN / "assets" / "genception"
ASSETS.mkdir(parents=True, exist_ok=True)
HTML = THREAD / "paper.html"


def parse_tables(html_path):
    soup = BeautifulSoup(open(html_path, encoding="utf-8").read(), "html.parser")
    return [
        [[cell.get_text(strip=True) for cell in row.find_all(["th", "td"])]
         for row in table.find_all("tr")]
        for table in soup.find_all("table")
    ]


def clean_num(x):
    x = x.replace("\\sim", "").replace("∼", "").replace("\\downarrow", "").replace("\\uparrow", "")
    x = x.replace("↓", "").replace("↑", "").replace("$", "").strip()
    if x in ("-", "", "∼"):
        return None
    mult = 1.0
    if x.endswith("K"):
        mult = 1e3
        x = x[:-1]
    elif x.endswith("M"):
        mult = 1e6
        x = x[:-1]
    try:
        return float(x) * mult
    except ValueError:
        return x.strip()


def table1_to_records(rows):
    # Table 1 uses colspans in the header rows, so we use a hand-verified
    # column mapping derived from the task/metric layout in the paper.
    # Data rows have 14 value columns after Method and Backbone.
    col_map = [
        ("Normals", "Sintel", "mAE"),
        ("Normals", "Hi4D", "mAE"),
        ("Depth", "Sintel", "AbsRel"),
        ("Depth", "KITTI", "AbsRel"),
        ("Depth", "ETH3D", "AbsRel"),
        ("Depth", "Goliath", "AbsRel"),
        ("Cam. Pose", "Sintel", "ATE"),
        ("Cam. Pose", "Sintel", "RPE-T"),
        ("Cam. Pose", "Sintel", "RPE-R"),
        ("Foreground Seg.", "V.Mat.", "MSE"),
        ("Foreground Seg.", "P.Mat.", "MSE"),
        ("Expression-Referring Seg.", "Ref-DAVIS", "J&F"),
        ("Expression-Referring Seg.", "MeViS", "J&F"),
        ("3D Human", "EMDB", "MPJPE"),
    ]
    records = []
    for r in rows[3:]:
        method, backbone = r[0], r[1]
        vals = r[2:]
        for i, (task, bench, metric) in enumerate(col_map):
            if i >= len(vals):
                continue
            v = clean_num(vals[i])
            if v is None:
                continue
            records.append({
                "method": method,
                "backbone": backbone,
                "task": task,
                "benchmark": bench,
                "metric": metric,
                "value": v,
            })
    return records


def table2_to_records(rows):
    # rows[0]=header, rows[1]=sub-header, then data rows
    records = []
    for r in rows[2:]:
        method = r[0]
        model_size = clean_num(r[1])
        datasets = clean_num(r[2])
        videos = clean_num(r[3])
        frames = r[4]
        sintel_abs = clean_num(r[5])
        sintel_d1 = clean_num(r[6])
        kitti_abs = clean_num(r[7])
        kitti_d1 = clean_num(r[8])
        eth_abs = clean_num(r[9])
        eth_d1 = clean_num(r[10])
        avg_abs = clean_num(r[11])
        avg_d1 = clean_num(r[12])
        records.append({
            "method": method,
            "model_size_b": model_size,
            "datasets": datasets,
            "videos": videos,
            "frames_text": frames,
            "sintel_absrel": sintel_abs,
            "sintel_delta1": sintel_d1,
            "kitti_absrel": kitti_abs,
            "kitti_delta1": kitti_d1,
            "eth3d_absrel": eth_abs,
            "eth3d_delta1": eth_d1,
            "avg_absrel": avg_abs,
            "avg_delta1": avg_d1,
        })
    return records


def save_json_csv(records, name):
    json_path = ASSETS / f"{name}.json"
    csv_path = ASSETS / f"{name}.csv"
    json_path.write_text(json.dumps(records, indent=2))
    with open(csv_path, "w", newline="") as f:
        if records:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
    return json_path, csv_path


def plot_scaling(table2):
    genception = [r for r in table2 if "WAN 2.1" in r["method"]]
    ssl_baselines = [r for r in table2 if r["method"].startswith("Ours") and "WAN 2.1" not in r["method"]]
    specialist = [r for r in table2 if not r["method"].startswith("Ours")]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [1.2, 1]})

    # Left: pretraining comparison at 7.5K videos
    ax = axes[0]
    pre = genception + ssl_baselines
    pre = [r for r in pre if r["videos"] == 7500]
    names = []
    vals = []
    colors = []
    for r in pre:
        if "WAN 2.1" in r["method"]:
            name = r["method"].replace("Ours - WAN 2.1 - ", "WAN ")
            colors.append("#4e79a7")
        elif "VideoMae" in r["method"]:
            name = r["method"].replace("Ours - VideoMae V2 - ", "VMae ")
            colors.append("#f28e2c")
        else:
            name = r["method"].replace("Ours - ", "")
            colors.append("#e15759")
        names.append(name)
        vals.append(r["avg_absrel"])
    bars = ax.barh(names, vals, color=colors)
    ax.set_xlabel("Average AbsRel ↓")
    ax.set_title("Pretraining objective at 7.5K videos")
    ax.grid(axis="x", ls="--", alpha=0.4)
    for bar, v in zip(bars, vals):
        ax.text(v + 0.005, bar.get_y() + bar.get_height()/2, f"{v:.3f}", va="center", fontsize=8)

    # Right: data/model scaling + specialists
    ax = axes[1]
    colors_size = {"1.3B": "#e15759", "14B": "#4e79a7"}
    for r in genception:
        size = f"{r['model_size_b']:.1f}B" if isinstance(r["model_size_b"], float) else str(r["model_size_b"])
        videos = r["videos"]
        avg_abs = r["avg_absrel"]
        ax.scatter(videos, avg_abs, c=colors_size.get(size, "#333"), s=100, zorder=3)
    for r in specialist:
        name = r["method"].split("[")[0].replace(" - G", "").replace(" - Ω", "")
        videos = r["videos"]
        avg_abs = r["avg_absrel"]
        ax.scatter(videos, avg_abs, marker="s", c="#59a14f", s=80, zorder=2)
        ax.annotate(name, (videos, avg_abs), textcoords="offset points", xytext=(5, 0), fontsize=7, va="center")

    ax.set_xlabel("Training videos (reported, log scale)")
    ax.set_ylabel("Average AbsRel ↓")
    ax.set_title("Scaling & specialists")
    ax.set_xscale("log")
    ax.grid(True, which="both", ls="--", alpha=0.4)
    from matplotlib.lines import Line2D
    legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#e15759", markersize=8, label="GenCeption 1.3B"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#4e79a7", markersize=8, label="GenCeption 14B"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#59a14f", markersize=8, label="Specialists"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=8)

    fig.suptitle("GenCeption depth scaling and pretraining ablation", fontsize=12, y=1.02)
    fig.tight_layout()
    path = ASSETS / "scaling_depth.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print("Saved", path)
    return path


def plot_sota_depth(table1):
    # Pull KITTI AbsRel for depth methods + GenCeption variants
    kitti = [r for r in table1 if r["benchmark"].startswith("KITTI") and isinstance(r["value"], float)]
    kitti.sort(key=lambda r: r["value"])

    fig, ax = plt.subplots(figsize=(8, 4))
    names = [r["method"].replace(" [", "\n[").replace(" - ", "\n") for r in kitti]
    vals = [r["value"] for r in kitti]
    colors = ["#4e79a7" if "Ours" in r["method"] else "#bab0ab" for r in kitti]
    bars = ax.barh(names, vals, color=colors)
    ax.set_xlabel("KITTI AbsRel ↓", fontsize=10)
    ax.set_title("Depth estimation on KITTI (lower is better)", fontsize=11)
    ax.grid(axis="x", ls="--", alpha=0.4)
    for bar, v in zip(bars, vals):
        ax.text(v + 0.003, bar.get_y() + bar.get_height()/2, f"{v:.3f}",
                va="center", fontsize=8)
    fig.tight_layout()
    path = ASSETS / "sota_kitti_depth.png"
    fig.savefig(path, dpi=200)
    print("Saved", path)
    return path


def main():
    tables = parse_tables(HTML)
    # The HTML contains a tiny empty table first; real tables are 1 and 2.
    table1_rows = tables[1]
    table2_rows = tables[2]

    t1 = table1_to_records(table1_rows)
    t2 = table2_to_records(table2_rows)

    save_json_csv(t1, "table1_sota_comparison")
    save_json_csv(t2, "table2_scaling_ablation")

    plot_scaling(t2)
    plot_sota_depth(t1)

    print(f"Extracted {len(t1)} measurement cells from Table 1, {len(t2)} rows from Table 2.")
    print("Artifacts:", ASSETS)


if __name__ == "__main__":
    main()
