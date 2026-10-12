"""Dense parallel scan for target episodes inside AgiBotWorld-Beta parameters tars.

Layout (probed): tars are grouped into GB-scale contiguous task blocks; episodes
are in ARBITRARY order within a block, ~3.3MB per episode; small param JSONs are
interleaved with ~0.5-1.1MB *_aligned.json siblings, so a 1MB window per 2MB
stride touches every episode (L_episode >= S + W).

Transport: curl subprocesses (urllib wedged after ~200 requests; curl -L handles
the ModelScope 302 + Range and sustains ~12 parallel 1MB windows per second).

Saves every observed (task, episode) header to a map JSON, extracts the 6 small
camera param JSONs for the target episode, writes recalc/agibot-orig/{task}_{episode}.json.
"""
import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tarhunt import scan_headers

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'agibot-orig')
os.makedirs(OUT, exist_ok=True)
MB = 1024 * 1024
BASE = 'https://modelscope.cn/datasets/agibot_world/agibot_world_beta/resolve/master/parameters'

WANT_FILES = ('head_intrinsic_params.json', 'head_extrinsic_params.json',
              'hand_left_intrinsic_params.json', 'hand_left_extrinsic_params.json',
              'hand_right_intrinsic_params.json', 'hand_right_extrinsic_params.json')

# (tar, size, task, episode, scan_lo_GB, scan_hi_GB)
TARGETS = [
    ('648533-663145', 48447580160, 410, 649000, 12.5, 16.5),
    ('663146-674519', 48441067520, 380, 665152, 25.0, 29.5),
    ('834407-841619', 48437524480, 599, 835856, 34.0, 40.0),
]

HDR_RE = re.compile(r'(\d+)/(\d+)/parameters/camera/([\w.]+\.json)$')

def fetch_range(url, off, span, retries=4):
    for i in range(retries):
        try:
            p = subprocess.run(
                ['curl', '-sL', '--max-time', '25', '-r', f'{off}-{off+span-1}', url],
                capture_output=True, timeout=40)
            if p.returncode == 0 and p.stdout:
                return p.stdout
        except subprocess.TimeoutExpired:
            pass
        time.sleep(0.4 * (i + 1))
    raise RuntimeError(f'fetch failed at {off}')

def scan_one(url, off):
    buf = fetch_range(url, off, MB)
    rows = []
    for hoff, name, size in scan_headers(buf, off):
        m = HDR_RE.match(name)
        if m:
            rows.append({'task': int(m.group(1)), 'ep': int(m.group(2)),
                         'file': m.group(3), 'off': hoff, 'size': size})
    return off, rows

def main():
    for tar, size, task, episode, lo_gb, hi_gb in TARGETS:
        outpath = os.path.join(OUT, f'{task}_{episode}.json')
        mappath = os.path.join(OUT, f'scanmap_{tar}.json')
        if os.path.exists(outpath):
            print('have', outpath, flush=True)
            continue
        url = f'{BASE}/{tar}.tar'
        lo = int(lo_gb * 1e9) // (2 * MB) * (2 * MB)
        hi = min(int(hi_gb * 1e9), size - MB)
        offsets = list(range(lo, hi, 2 * MB))
        print(f'== {tar} task {task} ep {episode}: scanning [{lo/MB:.0f},{hi/MB:.0f}]MB, {len(offsets)} windows', flush=True)
        found_ep = {}
        target_hits = []
        errs = 0
        t0 = time.time()
        done = 0
        with ThreadPoolExecutor(max_workers=12) as ex:
            futs = {ex.submit(scan_one, url, off): off for off in offsets}
            for fut in as_completed(futs):
                done += 1
                try:
                    off, rows = fut.result()
                except Exception:
                    errs += 1
                    continue
                for r in rows:
                    if r['task'] == task:
                        found_ep.setdefault(r['ep'], off)
                        if r['ep'] == episode:
                            target_hits.append(r)
                if done % 200 == 0:
                    print(f'  {done}/{len(offsets)} {time.time()-t0:.0f}s eps={len(found_ep)} hits={len(target_hits)} errs={errs}', flush=True)
                if target_hits:
                    for f in futs:
                        f.cancel()
                    break
        print(f'  scan done {time.time()-t0:.0f}s: {len(found_ep)} task-{task} eps, hits={len(target_hits)}, errs={errs}', flush=True)
        with open(mappath, 'w') as fp:
            json.dump({'tar': tar, 'task': task, 'episode': episode,
                       'scan_range_mb': [lo / MB, hi / MB],
                       'episodes_seen': {str(k): v for k, v in sorted(found_ep.items())},
                       'target_hits': target_hits}, fp, indent=1)
        if not target_hits:
            print('  EPISODE NOT FOUND in range', flush=True)
            continue
        anchor = target_hits[0]['off']
        start = max(0, anchor - 7 * MB) // 512 * 512
        collected = {}
        pos = start
        while pos < anchor + 7 * MB:
            buf = fetch_range(url, pos, MB)
            for hoff, name, hsize in scan_headers(buf, pos):
                m = HDR_RE.match(name)
                if m and int(m.group(1)) == task and int(m.group(2)) == episode:
                    fn = m.group(3)
                    if fn in WANT_FILES and fn not in collected and 0 < hsize < 65536:
                        data_off = (hoff // 512 + 1) * 512
                        collected[fn] = json.loads(fetch_range(url, data_off, hsize).decode())
            pos += MB
        print('  collected:', sorted(collected), flush=True)
        with open(outpath, 'w') as fp:
            json.dump({'tar': tar, 'task': task, 'episode': episode, 'files': collected}, fp, indent=1)
    print('done', flush=True)

if __name__ == '__main__':
    main()
