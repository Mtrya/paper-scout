#!/usr/bin/env python
# 在实例上给 eval_gripperhead_fdm_rollout.py 幂等加 --episode-indices 支持
import re
P = "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/ky26021/embodied-research/syncworld/SyncWorld/examples/eval_gripperhead_fdm_rollout.py"
src = open(P).read()
if "episode-indices" in src:
    print("already patched"); raise SystemExit(0)

old1 = 'g.add_argument("--max-episodes", type=int, default=0, help="cap #episodes (0 = all)")'
new1 = old1 + '\n    g.add_argument("--episode-indices", default="", help="comma-separated 0-based indices into the sorted discovered leaves (evaluated in given order)")'
assert old1 in src
src = src.replace(old1, new1)

old2 = """    leaves = discover_leaves(args.eval_set, args.calib_dirname)
    if args.max_episodes > 0:"""
new2 = """    leaves = discover_leaves(args.eval_set, args.calib_dirname)
    if getattr(args, "episode_indices", ""):
        _keep = [int(x) for x in args.episode_indices.split(",") if x.strip()]
        leaves = [leaves[i] for i in _keep]
    if args.max_episodes > 0:"""
assert old2 in src
src = src.replace(old2, new2)
open(P, "w").write(src)
print("patched OK")
