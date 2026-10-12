"""Map episode ordering around known task-block hits via ladder probes."""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tarhunt import fetch_range, scan_headers

BASE = 'https://modelscope.cn/datasets/agibot_world/agibot_world_beta/resolve/master/parameters'
MB = 1024 * 1024

CASES = [
    ('648533-663145', 48447580160, 410, 649000, 14495000576),
    ('663146-674519', 48441067520, 380, 665152, 28454000640),
    ('834407-841619', 48437524480, 599, 835856, 38654000640),
]

def probe_eps(url, off):
    buf = fetch_range(url, off, MB)
    out = []
    for hoff, name, size in scan_headers(buf, off):
        m = re.match(r'(\d+)/(\d+)/', name)
        if m:
            out.append((int(m.group(1)), int(m.group(2))))
    return out

for tar, size, task, episode, hit in CASES:
    url = f'{BASE}/{tar}.tar'
    print(f'===== {tar} task {task} target ep {episode} (known hit @{hit/MB:.0f}MB)')
    for delta_mb in [0, 8, -8, 32, -32, 128, -128, 512, -512, 1024, -1024, 2048, -2048]:
        off = hit + delta_mb * MB
        if off < 0 or off > size - MB:
            continue
        off = off // 512 * 512
        try:
            eps = probe_eps(url, off)
        except Exception as e:
            print(f'  {delta_mb:+5d}MB: ERR {e}')
            continue
        teps = sorted({e for t, e in eps})
        teps_t = sorted({e for t, e in eps if t == task})
        print(f'  {delta_mb:+5d}MB off={off/MB:8.0f}MB: all={teps[:6]}{"..." if len(teps) > 6 else ""} task{task}={teps_t[:8]}', flush=True)
