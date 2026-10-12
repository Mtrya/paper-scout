"""Continuation scan: extend ranges for targets not found in the first pass.
Usage: agibot_dense_scan_ext.py  (edits EXT list inline)"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import agibot_dense_scan as core

def main():
    core.TARGETS = []
    for tar, size, task, episode, lo_mb, hi_mb in EXT:
        outpath = os.path.join(core.OUT, f'{task}_{episode}.json')
        if os.path.exists(outpath):
            print('have', outpath, flush=True)
            continue
        core.TARGETS.append((tar, size, task, episode, lo_mb * core.MB / 1e9, hi_mb * core.MB / 1e9))
    core.main()

EXT = [
    ('663146-674519', 48441067520, 380, 665152, 28133, 29600),
    ('834407-841619', 48437524480, 599, 835856, 38147, 40200),
]

if __name__ == '__main__':
    main()
