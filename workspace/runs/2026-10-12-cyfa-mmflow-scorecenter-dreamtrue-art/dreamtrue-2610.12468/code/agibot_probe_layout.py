"""Empirically probe the layout of AgiBotWorld-Beta parameters tars: dump all
{task}/{episode}/ headers found in windows at several offsets."""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tarhunt import fetch_range, scan_headers

BASE = 'https://modelscope.cn/datasets/agibot_world/agibot_world_beta/resolve/master/parameters'

PROBES = {
    # tar: (size, [offsets in GB])
    '648533-663145': (48447580160, [0, 1.55, 5, 10, 14.495, 14.6, 15, 16, 20, 30, 40, 47]),
    '663146-674519': (48441067520, [28.454]),
    '834407-841619': (48437524480, [38.654]),
}

for tar, (size, offs_gb) in PROBES.items():
    url = f'{BASE}/{tar}.tar'
    print(f'===== {tar} ({size/1e9:.1f} GB)')
    for g in offs_gb:
        off = int(g * 1e9) // 512 * 512
        buf = fetch_range(url, off, 1024 * 1024)
        rows = []
        for hoff, name, hsize in scan_headers(buf, off):
            m = re.match(r'(\d+)/(\d+)/', name)
            tag = f'{m.group(1)}/{m.group(2)}' if m else '??'
            rows.append((tag, name.split('/')[-1], hsize))
        eps = {}
        for tag, fn, hsize in rows:
            eps.setdefault(tag, []).append((fn, hsize))
        summary = {t: (len(v), sum(s for _, s in v)) for t, v in eps.items()}
        print(f'  @{g}GB off={off}: {len(rows)} headers, episodes={summary}')
        for tag, v in list(eps.items())[:3]:
            print(f'     {tag}: {v[:8]}')
