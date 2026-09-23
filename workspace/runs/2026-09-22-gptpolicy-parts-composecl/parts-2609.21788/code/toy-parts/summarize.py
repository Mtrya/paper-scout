"""汇总 results/*.json → results/summary.md(均值±标准差,3 种子)。
样本效率:评估快照曲线上 σ3 成功率首次达到 50% 的环境步数(线性插值;未达到记 >60k)。
用法: python summarize.py [--results results]
"""

import argparse
import datetime
import json
import os

import numpy as np

from plot import load, GRID, curves

BUDGET = 60000


def steps_to_thr(run, key="s3", thr=0.5):
    xs = [s["step"] for s in run["snapshots"]]
    vs = [s[key] for s in run["snapshots"]]
    if len(xs) == 1:
        return None if vs[0] >= thr else float("inf")
    for i in range(1, len(xs)):
        if vs[i] >= thr:
            x0, x1, v0, v1 = xs[i - 1], xs[i], vs[i - 1], vs[i]
            frac = 0.0 if v1 == v0 else (thr - v0) / (v1 - v0)
            return min(x0 + frac * (x1 - x0), BUDGET)
    return float("inf")


def ms(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "—"
    return f"{np.mean(vals):.2f}±{np.std(vals):.2f}"


def ms_steps(vals):
    vals = np.array(vals, dtype=float)
    n_cens = int(np.sum(np.isinf(vals)))
    fin = vals[~np.isinf(vals)]
    if len(fin) == 0:
        return f">{BUDGET//1000}k(全部截尾)"
    s = f"{np.mean(fin)/1000:.1f}k±{np.std(fin)/1000:.1f}k"
    if n_cens:
        s += f"(另 {n_cens} 种子 >{BUDGET//1000}k)"
    return s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    args = p.parse_args()
    R = args.results
    lines = []
    lines.append("# Toy-PARTS 结果汇总")
    lines.append("")
    lines.append(f"生成时间:{datetime.datetime.now().isoformat(timespec='seconds')};"
                 f"预算 {BUDGET//1000}k 环境步/run,3 种子(0/1/2),"
                 "评估快照 100 episodes(base 评估 500 eps×3 种子)。")
    lines.append("")
    lines.append("共享 base(冻结):σ1=抓取、σ2=绕障搬运、σ3=精密插入。")

    # base 行
    a_runs = load(R, "A", 0.0)
    if a_runs:
        for k in ("s1", "s2", "s3", "full"):
            vals = [r["snapshots"][0][k] for r in a_runs]
            lines.append(f"- base {k}: {np.mean(vals):.2f}±{np.std(vals):.2f}")
    lines.append("")

    lines.append("## 表 1:ε=0,四方法对比(最终 = 60k 步处评估)")
    lines.append("")
    lines.append("| 方法 | 最终全任务成功率 | 最终瓶颈 σ3 成功率 | σ3 达 50% 所需步数 | 训练尝试数 |")
    lines.append("|---|---|---|---|---|")
    label = {"A": "A base-only", "B": "B 全任务 RL(稀疏 R)",
             "C": "C PARTS(重训)", "D": "D PARTS 无重训"}
    for m in ("A", "B", "C", "D"):
        runs = load(R, m, 0.0)
        if not runs:
            continue
        full = [r["snapshots"][-1]["full"] for r in runs]
        s3 = [r["snapshots"][-1]["s3"] for r in runs]
        eff = [steps_to_thr(r) for r in runs]
        nat = [len(r.get("attempts", [])) for r in runs]
        nat_s = f"{np.mean(nat):.0f}±{np.std(nat):.0f}" if nat and nat[0] else "—"
        lines.append(f"| {label[m]} | {ms(full)} | {ms(s3)} | {ms_steps(eff)} | {nat_s} |")
    lines.append("")

    lines.append("## 表 2:方法 C 的 verifier 噪声扫描(σ3 局部奖励以概率 ε 翻转)")
    lines.append("")
    lines.append("ε=0.5 为规格外扩展点(标签零信息量,锚定曲线终点)。")
    lines.append("")
    lines.append("| ε | 最终 σ3 成功率 | 最终全任务成功率 | σ3 达 50% 步数 | 训练尝试数 |")
    lines.append("|---|---|---|---|---|")
    for e in (0.0, 0.05, 0.15, 0.3, 0.5):
        runs = load(R, "C", e)
        full = [r["snapshots"][-1]["full"] for r in runs]
        s3 = [r["snapshots"][-1]["s3"] for r in runs]
        eff = [steps_to_thr(r) for r in runs]
        nat = [len(r.get("attempts", [])) for r in runs]
        nat_s = f"{np.mean(nat):.0f}±{np.std(nat):.0f}" if nat and nat[0] else "—"
        lines.append(f"| {e} | {ms(s3)} | {ms(full)} | {ms_steps(eff)} | {nat_s} |")
    lines.append("")
    lines.append("图:`fig_training.png`(训练曲线×4 方法),`fig_noise.png`(噪声衰减)。")

    out = os.path.join(R, "summary.md")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
