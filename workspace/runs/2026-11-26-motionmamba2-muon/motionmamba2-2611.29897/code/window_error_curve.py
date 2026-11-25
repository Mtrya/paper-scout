"""HiPHI 逐帧到目标距离曲线 + 子集选择性检查。

问题:长视界 Root CE(末端根位置到命令目标点的距离)随视界增长,
是随机游走型(误差 ~ t^0.5)、线性漂移型(误差 ~ t^1),还是目标导向型
(先收敛再稳定/发散)?这决定"有界记忆长程控制"的失效模式。

方法:npz 按 case_id 命名,benchmark 资产的 cases.jsonl 给出每个 case 的
root_goal_xz_m,两者连接即可对每一帧计算 |root(t) - goal|。对齐帧取均值
得到群体距离曲线;对曲线后半段做 log-log 斜率拟合得增长指数。

子集选择性:--limit 48 取文件序前 48 个 locked_test case;用
hiphi_inputs.npz 的 initial_physical 末端根位置计算初始根-目标距离,
对比前 48 族与全部 locked_test 的分布,检查子集难度是否有偏。

用法: python3 window_error_curve.py <mm2-out>/hiphi/<task> [out.json]
       (cases/inputs 路径按仓库默认布局解析,可用环境变量覆盖)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(os.environ.get("MM2_REPO", str(Path(__file__).resolve().parents[4] / "code" / "motionmamba2")))
ASSETS = REPO / "benchmarks" / "benchmark_assets"


def load_cases():
    cases = [json.loads(l) for l in (ASSETS / "hiphi_cases.jsonl").read_text().splitlines() if l.strip()]
    return cases, {c["case_id"]: c for c in cases}


def distance_curves(folder, by_id):
    curves, missed = [], 0
    for p in sorted(Path(folder).glob("*.npz")):
        case_id = p.stem.rsplit("_seed_", 1)[0]
        case = by_id.get(case_id)
        if case is None:
            missed += 1
            continue
        d = np.load(p, allow_pickle=False)
        root = d["root"][:, [0, 2]]
        curves.append(np.linalg.norm(root - np.asarray(case["root_goal_xz_m"]), axis=1) * 100)
    return curves, missed


def growth_curve(curves):
    stack = np.stack(curves)  # [R, T]
    mean = stack.mean(0)
    T = mean.shape[0]
    half = T // 2
    t = np.arange(1, T + 1)[half:]
    slope = float(np.polyfit(np.log(t), np.log(np.maximum(mean[half:], 1e-9)), 1)[0])
    nwin = T // 52
    per_win = [float(mean[i * 52:(i + 1) * 52].mean()) for i in range(nwin)]
    return dict(n_rollouts=int(stack.shape[0]), frames=T, mean_curve=[float(x) for x in mean],
               per_win_mean_cm=per_win, tail_loglog_slope=slope,
               final_cm=float(mean[-1]))


def displacement_stats(folder):
    rows = []
    for p in sorted(Path(folder).glob("*.npz")):
        root = np.load(p, allow_pickle=False)["root"]
        pos = root[:, [0, 2]] - root[0, [0, 2]]
        T = pos.shape[0]
        nwin = T // 52
        if nwin < 2:
            continue
        per_win = [float(np.linalg.norm(pos[(i + 1) * 52 - 1] - pos[i * 52])) for i in range(nwin)]
        step = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        rows.append(dict(per_win=per_win, final=float(np.linalg.norm(pos[-1])),
                         path=float(np.cumsum(step)[-1])))
    if not rows:
        return None
    nwin = len(rows[0]["per_win"])
    rows = [r for r in rows if len(r["per_win"]) == nwin]
    disps = np.array([r["per_win"] for r in rows])
    ac = [float(np.corrcoef(d[:-1], d[1:])[0, 1]) for d in disps if d.std() > 0]
    return dict(n_rollouts=len(rows), nwin=nwin,
                per_win_disp_mean_m=float(disps.mean()), per_win_disp_cv=float(disps.std() / disps.mean()),
                straightness=float(np.mean([r["final"] / r["path"] for r in rows])),
                lag1_autocorr=float(np.mean(ac)) if ac else None)


def subset_selectivity(limit):
    cases, _ = load_cases()
    locked = [(i, c) for i, c in enumerate(cases) if c["split"] == "locked_test"]
    data = np.load(ASSETS / "hiphi_inputs.npz", allow_pickle=False)["initial_physical"]
    d0 = np.array([np.linalg.norm(data[i, -1, [0, 2]] - np.asarray(c["root_goal_xz_m"]))
                   for i, c in locked])
    head = d0[:limit] if limit else d0
    return dict(n_locked=len(locked), limit=limit,
                init_dist_head_mean_m=float(head.mean()), init_dist_all_mean_m=float(d0.mean()),
                init_dist_head_median_m=float(np.median(head)), init_dist_all_median_m=float(np.median(d0)))


def main(folder, out_path, limit=None):
    _, by_id = load_cases()
    curves, missed = distance_curves(folder, by_id)
    if missed:
        print(f"warning: {missed} npz without matching case")
    out = {"folder": str(folder)}
    if curves:
        out["distance_to_goal"] = growth_curve(curves)
        g = out["distance_to_goal"]
        print(f"rollouts={g['n_rollouts']} frames={g['frames']} final={g['final_cm']:.2f}cm "
              f"tail_loglog_slope={g['tail_loglog_slope']:.3f}")
        print("per-window mean distance (cm):", [round(x, 2) for x in g["per_win_mean_cm"]])
    else:
        print("no npz found")
    disp = displacement_stats(folder)
    if disp:
        out["displacement"] = disp
        print(f"displacement: mean={disp['per_win_disp_mean_m']:.3f}m cv={disp['per_win_disp_cv']:.3f} "
              f"straightness={disp['straightness']:.3f} lag1_ac={disp['lag1_autocorr']:.3f}")
    if limit is not None:
        out["subset"] = subset_selectivity(limit)
        s = out["subset"]
        print(f"subset: head init-dist {s['init_dist_head_mean_m']:.3f}m vs all {s['init_dist_all_mean_m']:.3f}m")
    Path(out_path).write_text(json.dumps(out, indent=1))
    print("written", out_path)


if __name__ == "__main__":
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else None
    main(sys.argv[1], sys.argv[2], limit)
