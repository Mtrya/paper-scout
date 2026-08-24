"""Parallel driver for B2a/B2b (3 workers x 4 torch threads on a 12-core box).

Same protocols as run_b2.py, but training and evaluation are spread over a
multiprocessing pool so the machine's cores are actually used (the sequential
driver measured ~160 min; this one fits the time budget). Results structure is
identical to run_b2.py's, and the final figure step reuses run_b2.make_figures.

Usage:  python par_run.py
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import time

# limit BLAS/OMP threads BEFORE numpy/torch are imported: with N workers each
# spawning its own torch + OpenBLAS thread pools, thread oversubscription made
# training ~5x slower (measured via /proc/<pid>/task count and ctxt switches).
# numpy splat-rendering is elementwise, so OpenBLAS threads buy nothing here.
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import numpy as np
import torch

import run_b2 as R  # constants, spec fns, summarize, make_figures
from b2 import Dataset, eval_yaw_sweep, load_model, train

N_WORKERS = 3
THREADS_PER_WORKER = 4


def worker_init():
    torch.set_num_threads(THREADS_PER_WORKER)


def load_ds(name: str) -> Dataset:
    z = np.load(os.path.join(R.CKPT, f'dataset_{name}.npz'))
    return Dataset(z['imgs'], z['deltas'], z['yaws'], z['emb'])


# ---- stage jobs --------------------------------------------------------------

def build_job(name: str, n_eps: int, seed: int, spec_fn):
    R.get_dataset(name, n_eps, seed, spec_fn, resume=True)


def train_job(args) -> str:
    ds_name, head, out_name = args
    path = os.path.join(R.CKPT, f'{out_name}_{head}.pt')
    if os.path.exists(path):
        print(f'[par] {out_name}_{head}: exists, skipped', flush=True)
        return path
    torch.set_num_threads(THREADS_PER_WORKER)  # sequential stage: this IS the main process
    ds = load_ds(ds_name)
    m = train(ds, head, out_path=path)
    return path


def eval_job(args):
    ckpt_name, head, mount_name, seed, emb_label = args
    ckpt_path = os.path.join(R.CKPT, f'{ckpt_name}.pt')
    m = load_model(ckpt_path, head)
    d = eval_yaw_sweep(m, head, R.YAW_GRID, R.N_EPS, seed=seed,
                       mount_name=mount_name, emb_label=emb_label)
    return d


def main():
    mp.set_start_method('fork')
    t0 = time.time()

    # ---------- stage A: build any missing datasets ----------
    to_build = [
        ('t2_1000', 1000, 11, R.spec_t2),
        ('t2_4000', 4000, 11, R.spec_t2),
        ('b2b_5000', 5000, 13, R.spec_b2b),
    ]
    missing = []
    for name, n, seed, spec in to_build:
        if not os.path.exists(os.path.join(R.CKPT, f'dataset_{name}.npz')):
            missing.append((name, n, seed, spec))
    if missing:
        print(f'[par] building {len(missing)} datasets: {[m[0] for m in missing]}', flush=True)
        with mp.Pool(N_WORKERS, initializer=worker_init) as pool:
            pool.starmap(build_job, missing)
    else:
        print('[par] all datasets cached', flush=True)

    # ---------- stage B: train all heads ----------
    jobs = [
        # T1 (base already trained)
        ('t1_5000', 'base_pose', 't1'), ('t1_5000', 'cam', 't1'),
        # T2
        ('t2_4500', 'base', 't2'), ('t2_4500', 'base_pose', 't2'), ('t2_4500', 'cam', 't2'),
        # DENSE
        ('dense_4500', 'base', 'dense'), ('dense_4500', 'base_pose', 'dense'),
        ('dense_4500', 'cam', 'dense'),
        # SCAN (T2 protocol; 16k point dropped to fit the 90-min time budget)
        ('t2_1000', 'base_pose', 'scan1000'), ('t2_1000', 'cam', 'scan1000'),
        ('t2_4000', 'base_pose', 'scan4000'), ('t2_4000', 'cam', 'scan4000'),
        # B2b (base_ref uses head 'base')
        ('b2b_5000', 'cam', 'b2b'), ('b2b_5000', 'base_pose', 'b2b'),
        ('b2b_5000', 'base_pose_label', 'b2b'), ('b2b_5000', 'base', 'b2b'),
    ]
    # drop already-trained ones (e.g. t1_base)
    pending = []
    for ds_name, head, out_name in jobs:
        if not os.path.exists(os.path.join(R.CKPT, f'{out_name}_{head}.pt')):
            pending.append((ds_name, head, out_name))
    print(f'[par] training {len(pending)} models sequentially (measured: parallel training has ~no throughput gain on this box)...', flush=True)
    for ds_name, head, out_name in pending:
        print(f'  - {out_name}_{head}  (ds={ds_name})', flush=True)
    for args in pending:
        train_job(args)
    print(f'[par] training done in {time.time()-t0:.0f}s', flush=True)

    # ---------- stage C: evaluate all heads ----------
    eval_jobs = []
    for proto, head, ckpt_name, seed in [
            ('t1', 'base', 't1_base', 100), ('t1', 'base_pose', 't1_base_pose', 100),
            ('t1', 'cam', 't1_cam', 100),
            ('t2', 'base', 't2_base', 101), ('t2', 'base_pose', 't2_base_pose', 101),
            ('t2', 'cam', 't2_cam', 101),
            ('dense', 'base', 'dense_base', 102), ('dense', 'base_pose', 'dense_base_pose', 102),
            ('dense', 'cam', 'dense_cam', 102),
            ('scan', 'base_pose', 'scan1000_base_pose', 101), ('scan', 'cam', 'scan1000_cam', 101),
            ('scan', 'base_pose', 'scan4000_base_pose', 101), ('scan', 'cam', 'scan4000_cam', 101)]:
        eval_jobs.append((ckpt_name, head, 'E1', seed, None))

    # B2b: 4 conditions x 3 mounts, seed 110
    b2b_heads = {'cam': 'cam', 'base_pose': 'base_pose', 'base_pose_label': 'base_pose_label',
                 'base_ref': 'base'}
    for cond, head in b2b_heads.items():
        ckpt_name = f'b2b_{head}'
        for mount in ['E1', 'E2', 'E3']:
            label = R.EMB_LABEL[mount] if cond == 'base_pose_label' else None
            eval_jobs.append((ckpt_name, head, mount, 110, label))

    print(f'[par] evaluating {len(eval_jobs)} groups x {len(R.YAW_GRID)} yaws x {R.N_EPS} eps...', flush=True)
    results = {}
    # per-group results indexed by job id
    def fmt_job(job):
        ckpt_name, head, mount, seed, label = job
        return ckpt_name, mount
    out = {}
    with mp.Pool(N_WORKERS, initializer=worker_init) as pool:
        for job, d in zip(eval_jobs, pool.imap(eval_job, eval_jobs)):
            out[job] = d
    # ---- assemble results in run_b2.py's structure ----
    for proto, head, ckpt_name, seed in [
            ('t1', 'base', 't1_base', 100), ('t1', 'base_pose', 't1_base_pose', 100),
            ('t1', 'cam', 't1_cam', 100),
            ('t2', 'base', 't2_base', 101), ('t2', 'base_pose', 't2_base_pose', 101),
            ('t2', 'cam', 't2_cam', 101),
            ('dense', 'base', 'dense_base', 102), ('dense', 'base_pose', 'dense_base_pose', 102),
            ('dense', 'cam', 'dense_cam', 102)]:
        key = {'t1': 'b2a_t1', 't2': 'b2a_t2', 'dense': 'b2a_dense'}[proto]
        results.setdefault(key, {})[head] = out[(ckpt_name, 'E1')]
    results['b2a_scan'] = {}
    for nk, ckpt_base, head_list in [('1000', 'scan1000', ['base_pose', 'cam']),
                                     ('4000', 'scan4000', ['base_pose', 'cam'])]:
        results['b2a_scan'][nk] = {}
        for head in head_list:
            results['b2a_scan'][nk][head] = out[(f'{ckpt_base}_{head}', 'E1')]
    results['b2b_conditions'] = {}
    for cond, head in b2b_heads.items():
        results['b2b_conditions'][cond] = {}
        for mount in ['E1', 'E2', 'E3']:
            results['b2b_conditions'][cond][mount] = out[(f'b2b_{head}', mount)]
    results['b2b_summary'] = R.summarize(results['b2b_conditions'])
    results['config'] = {
        'yaw_grid': list(R.YAW_GRID), 'n_eps_eval': R.N_EPS, 'epochs': 15, 'lr': 1e-3,
        'batch_size': 256, 'train_seed': 1,
        'mounts': {k: v for k, v in R.MOUNTS.items()},
        'pose_input': 'R_bc(9) + t_b/2.5(3)', 'emb_label': 'one-hot(2)',
        'driver': 'par_run.py (3 workers x 4 threads)',
    }
    with open(os.path.join(R.HERE, 'results.json'), 'w') as f:
        json.dump(results, f, indent=1, default=float)

    # ---- summary lines ----
    for key in ['b2a_t1', 'b2a_t2', 'b2a_dense']:
        for h, d in results[key].items():
            s, se = R.mean_success(d)
            print(f'  {key}-{h}: mean_success={s:.3f} (se {se:.3f})', flush=True)
    for nk, d in results['b2a_scan'].items():
        for h in ['base_pose', 'cam']:
            s, se = R.mean_success(d[h])
            print(f'  SCAN n={nk} {h}: mean_success={s:.3f} (se {se:.3f})', flush=True)
    for h, by_mount in results['b2b_summary'].items():
        line = ' | '.join(f"{m}: sr={v['mean_success']:.3f} fd={v['mean_final_dist']:.3f}"
                          for m, v in by_mount.items())
        print(f'  B2b-{h}: {line}', flush=True)

    R.make_figures(results)
    print(f'\nAll done in {time.time()-t0:.0f}s. results.json + figures saved.', flush=True)


if __name__ == '__main__':
    main()
