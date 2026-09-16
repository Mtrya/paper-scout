#!/usr/bin/env python
# SyncWorld 可观性探针分析:逐 episode 转动占比 × 各条件指标,输出 JSON。
# 在实例上跑: /root/sw-env/bin/python analyze_probe.py [out.json]
import csv, json, os, pickle, sys
import numpy as np

WR = "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/ky26021/embodied-research/syncworld"
EVALSET = f"{WR}/SyncWorld-Evaluation/evaluation_libero"
OUT = f"{WR}/outputs"
TAGS = [t for t in ["pilot_full", "full", "null", "droprot", "droptrans"]
        if os.path.isdir(os.path.join(OUT, t))]
# 输出目录里的 episode_{k} 是过滤后列表的位置;EPIDX 把它映射回 discover_leaves 的全局序号
EPIDX = [int(x) for x in os.environ.get("EPIDX", "").split(",") if x.strip()]

def episode_leaves(root):
    # 与 eval 脚本 discover_leaves 同口径:跳过 calibration 目录
    leaves = []
    for dirpath, dirnames, filenames in os.walk(root):
        if os.path.basename(dirpath) == "calibration":
            dirnames[:] = []
            continue
        if "pose.pkl" in filenames or "actions.pkl" in filenames:
            leaves.append(dirpath)
    return sorted(leaves)

def motion(leaf):
    """与 eval 脚本 _load_pose 同口径:优先 pose.pkl(gripper_matrix T,4,4)。"""
    pp = os.path.join(leaf, "pose.pkl")
    if not os.path.isfile(pp):
        return None, None
    d = pickle.load(open(pp, "rb"))
    mats = np.asarray(d["gripper_matrix"], dtype=np.float64)
    if len(mats) < 2:
        return None, None
    dR = mats[1:, :3, :3] @ np.transpose(mats[:-1, :3, :3], (0, 2, 1))
    dp = mats[1:, :3, 3] - mats[:-1, :3, 3]
    tr = np.trace(dR, axis1=1, axis2=2)
    ang = np.arccos(np.clip((tr - 1) / 2, -1, 1))
    return float(np.linalg.norm(dp, axis=1).sum()), float(ang.sum())

def load_metrics(tag):
    """<out>/<tag>/<view>/metrics_full_episode.csv -> {(ep,view): {psnr,ssim,lpips}} 帧数加权"""
    res = {}
    tdir = os.path.join(OUT, tag)
    for view in sorted(os.listdir(tdir)):
        p = os.path.join(tdir, view, "metrics_full_episode.csv")
        if not os.path.isfile(p):
            continue
        acc = {}
        for r in csv.DictReader(open(p)):
            if r["episode"] == "average":
                continue
            k = int(str(r["episode"]).split("_")[-1])   # "episode_0003" -> 3(位置)
            ep = EPIDX[k] if EPIDX else k               # 映射回全局序号
            key = (ep, view)
            n = float(r["num_frames_eval"])
            a = acc.setdefault(key, [0.0, 0.0, 0.0, 0.0])
            for i, k in enumerate(["psnr", "ssim", "lpips"]):
                v = float(r[k])
                if v == v:  # skip nan
                    a[i] += v * n
            a[3] += n
        for key, a in acc.items():
            if a[3] > 0:
                res[key] = {k: a[i] / a[3] for i, k in enumerate(["psnr", "ssim", "lpips"])}
    return res

leaves = episode_leaves(EVALSET)
tm, rm = [], []
for leaf in leaves:
    t, r = motion(leaf)
    tm.append(t); rm.append(r)
tv = np.array([x for x in tm if x is not None])
rv = np.array([x for x in rm if x is not None])
t_med, r_med = float(np.median(tv)), float(np.median(rv))

metrics = {tag: load_metrics(tag) for tag in TAGS}
rows = []
for i, leaf in enumerate(leaves):
    if tm[i] is None:
        continue
    tn, rn = tm[i] / t_med, rm[i] / r_med
    row = {"episode": i, "leaf": os.path.relpath(leaf, EVALSET),
           "trans_motion": tm[i], "rot_motion": rm[i],
           "rot_share": rn / (tn + rn)}
    for view in set(k[1] for m in metrics.values() for k in m if k[0] == i):
        for tag in TAGS:
            v = metrics[tag].get((i, view))
            if v:
                row[f"{tag}:{view}"] = v
    rows.append(row)

out = {"tags": TAGS, "t_med": t_med, "r_med": r_med, "episodes": rows}
dest = sys.argv[1] if len(sys.argv) > 1 else f"{WR}/outputs/probe_analysis.json"
json.dump(out, open(dest, "w"), ensure_ascii=False, indent=1)
print("wrote", dest, "episodes:", len(rows), "tags:", TAGS)
