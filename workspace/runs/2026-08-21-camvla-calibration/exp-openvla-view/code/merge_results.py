"""合并远端三个结果 JSON(main baseline + worker A/B),按 (task,theta,mode,sign,episode) 去重。"""
import json
import sys

files = sys.argv[1:]
rows = []
seen = set()
for f in files:
    with open(f) as fh:
        data = json.load(fh)
    for r in data:
        key = (r["task"], r["theta_deg"], r["mode"], r["rescue_sign"], r["episode"])
        if key in seen:
            continue
        seen.add(key)
        rows.append(r)
print("merged records:", len(rows))
json.dump(rows, open("merged_results.json", "w"), indent=1)

# 快速汇总
import collections
c = collections.defaultdict(list)
for r in rows:
    c[(r["mode"], r["theta_deg"], r["rescue_sign"])].append(bool(r["success"]))
for k in sorted(c, key=str):
    v = c[k]
    print(f"{str(k):>45}  {sum(v)}/{len(v)}  {sum(v)/len(v):.2f}")
