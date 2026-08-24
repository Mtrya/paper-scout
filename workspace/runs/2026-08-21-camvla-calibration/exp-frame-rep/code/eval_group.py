"""Evaluate one group of (ckpt, head, mount, seed, label) jobs, write results JSON.

Run as independent processes (one per group) to avoid fork+torch-thread-pool
deadlocks. Output: checkpoints/eval_cache/<group>.json
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

from b2 import eval_yaw_sweep, load_model
import run_b2 as R

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, 'checkpoints')

torch.set_num_threads(1)  # eval is render-dominated; 1 thread avoids oversubscription

# (ckpt_name, head, mount, seed, emb_label) — ckpt_name w/o .pt suffix
GROUPS = {
    'g1': [  # T1 + scan 1k
        ('t1_base', 'base', 'E1', 100, None),
        ('t1_base_pose', 'base_pose', 'E1', 100, None),
        ('t1_cam', 'cam', 'E1', 100, None),
        ('scan1000_base_pose', 'base_pose', 'E1', 101, None),
        ('scan1000_cam', 'cam', 'E1', 101, None),
    ],
    'g2': [  # T2 + scan 4k
        ('t2_base', 'base', 'E1', 101, None),
        ('t2_base_pose', 'base_pose', 'E1', 101, None),
        ('t2_cam', 'cam', 'E1', 101, None),
        ('scan4000_base_pose', 'base_pose', 'E1', 101, None),
        ('scan4000_cam', 'cam', 'E1', 101, None),
    ],
    'g3': [  # DENSE
        ('dense_base', 'base', 'E1', 102, None),
        ('dense_base_pose', 'base_pose', 'E1', 102, None),
        ('dense_cam', 'cam', 'E1', 102, None),
    ],
    'g4': [  # B2b cam (no label)
        ('b2b_cam', 'cam', 'E1', 110, None),
        ('b2b_cam', 'cam', 'E2', 110, None),
        ('b2b_cam', 'cam', 'E3', 110, None),
    ],
    'g5': [  # B2b base_pose
        ('b2b_base_pose', 'base_pose', 'E1', 110, None),
        ('b2b_base_pose', 'base_pose', 'E2', 110, None),
        ('b2b_base_pose', 'base_pose', 'E3', 110, None),
    ],
    'g6': [  # B2b base_pose_label (E3 label OOD = zero vector)
        ('b2b_base_pose_label', 'base_pose_label', 'E1', 110, np.array([1.0, 0.0])),
        ('b2b_base_pose_label', 'base_pose_label', 'E2', 110, np.array([0.0, 1.0])),
        ('b2b_base_pose_label', 'base_pose_label', 'E3', 110, np.array([0.0, 0.0])),
    ],
    'g7': [  # B2b base_ref (reference floor)
        ('b2b_base', 'base', 'E1', 110, None),
        ('b2b_base', 'base', 'E2', 110, None),
        ('b2b_base', 'base', 'E3', 110, None),
    ],
}


def run_group(gid: str) -> dict:
    jobs = GROUPS[gid]
    out = {}
    for ckpt_name, head, mount, seed, label in jobs:
        path = os.path.join(CKPT, f'{ckpt_name}.pt')
        if not os.path.exists(path):
            out[f'{ckpt_name}|{mount}'] = {'error': f'missing {path}'}
            continue
        m = load_model(path, head)
        t0 = time.time()
        d = eval_yaw_sweep(m, head, R.YAW_GRID, R.N_EPS, seed=seed,
                           mount_name=mount, emb_label=label)
        s, se = R.mean_success(d)
        print(f'  [{gid}] {ckpt_name} @ {mount}: mean_success={s:.3f} '
              f'(se {se:.3f}) in {time.time()-t0:.0f}s', flush=True)
        out[f'{ckpt_name}|{mount}'] = d
    return out


def main():
    gid = sys.argv[1]
    cache = os.path.join(CKPT, 'eval_cache')
    os.makedirs(cache, exist_ok=True)
    out = run_group(gid)
    with open(os.path.join(cache, f'{gid}.json'), 'w') as f:
        json.dump(out, f, indent=1, default=float)
    print(f'[{gid}] saved {len(out)} jobs', flush=True)


if __name__ == '__main__':
    main()
