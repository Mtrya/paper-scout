#!/bin/bash
# dl5: Qwen3-14B (base) from ModelScope, parallel shards
set -u
W=/inspire/qb-ilm/project/cq-scientific-cooperation-zone/ky26021
D=$W/opd-pawb/models/Qwen3-14B-Instruct
mkdir -p $D
python3 - "$D" <<'PYEOF'
import json, os, sys, subprocess, concurrent.futures
dst = sys.argv[1]
repo = "Qwen/Qwen3-14B"
u = f"https://modelscope.cn/api/v1/models/{repo}/repo/files?Recursive=true"
d = json.load(__import__('urllib.request', fromlist=['urlopen']).urlopen(u))
files = d['Data']['Files']
big = [f for f in files if f['Size'] > 50_000_000]
small = [f for f in files if f['Size'] <= 50_000_000]
def grab(f):
    name, size = f['Name'], f['Size']
    out = os.path.join(dst, name)
    if os.path.exists(out) and abs(os.path.getsize(out) - size) < 10:
        return f"skip {name}"
    url = f"https://modelscope.cn/models/{repo}/resolve/master/{name}"
    for attempt in range(5):
        r = subprocess.run(["curl", "-sL", "-C", "-", "--max-time", "7200", "-o", out, url])
        if r.returncode == 0 and abs(os.path.getsize(out) - size) < 10:
            return f"ok {name}"
        print(f"retry {name} attempt {attempt}", flush=True)
    return f"FAIL {name}"
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
    for msg in ex.map(grab, big + small):
        print(msg, flush=True)
print("ALL DONE", flush=True)
PYEOF
du -sh $D
