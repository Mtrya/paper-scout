"""Merge eval_cache/*.json into results.json (run_b2.py structure) + figures."""
from __future__ import annotations

import json
import os

import run_b2 as R

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, 'checkpoints', 'eval_cache')


def load(gid: str) -> dict:
    with open(os.path.join(CACHE, f'{gid}.json')) as f:
        return json.load(f)


def main():
    g = {gid: load(gid) for gid in ['g1', 'g2', 'g3', 'g4', 'g5', 'g6', 'g7']}
    results = {}

    def get(gid, ckpt, mount='E1'):
        return g[gid][f'{ckpt}|{mount}']

    results['b2a_t1'] = {h: get('g1', f't1_{h}') for h in ['base', 'base_pose', 'cam']}
    results['b2a_t2'] = {h: get('g2', f't2_{h}') for h in ['base', 'base_pose', 'cam']}
    results['b2a_dense'] = {h: get('g3', f'dense_{h}') for h in ['base', 'base_pose', 'cam']}
    results['b2a_scan'] = {
        '1000': {h: get('g1', f'scan1000_{h}') for h in ['base_pose', 'cam']},
        '4000': {h: get('g2', f'scan4000_{h}') for h in ['base_pose', 'cam']},
    }
    results['b2b_conditions'] = {
        'cam': {m: get('g4', 'b2b_cam', m) for m in ['E1', 'E2', 'E3']},
        'base_pose': {m: get('g5', 'b2b_base_pose', m) for m in ['E1', 'E2', 'E3']},
        'base_pose_label': {m: get('g6', 'b2b_base_pose_label', m) for m in ['E1', 'E2', 'E3']},
        'base_ref': {m: get('g7', 'b2b_base', m) for m in ['E1', 'E2', 'E3']},
    }
    results['b2b_summary'] = R.summarize(results['b2b_conditions'])
    results['config'] = {
        'yaw_grid': list(R.YAW_GRID), 'n_eps_eval': R.N_EPS, 'epochs': 15, 'lr': 1e-3,
        'batch_size': 256, 'train_seed': 1,
        'mounts': {k: v for k, v in R.MOUNTS.items()},
        'pose_input': 'R_bc(9) + t_b/2.5(3)', 'emb_label': 'one-hot(2)',
        'driver': 'par_run.py train (sequential) + eval_group.py eval (independent processes)',
        'scan_16k_dropped': '16k scan point dropped to fit time budget; 1k/4k/4.5k plotted',
    }
    with open(os.path.join(HERE, 'results.json'), 'w') as f:
        json.dump(results, f, indent=1, default=float)

    # ---- print key numbers ----
    for key, label in [('b2a_t1', 'T1'), ('b2a_t2', 'T2'), ('b2a_dense', 'DENSE')]:
        for h in ['base', 'base_pose', 'cam']:
            s, se = R.mean_success(results[key][h])
            print(f'{label}-{h}: mean={s:.3f} (se {se:.3f})')
    for nk in ['1000', '4000']:
        for h in ['base_pose', 'cam']:
            s, se = R.mean_success(results['b2a_scan'][nk][h])
            print(f'SCAN n={nk} {h}: mean={s:.3f} (se {se:.3f})')
    for h, by_mount in results['b2b_summary'].items():
        line = ' | '.join(f"{m}: sr={v['mean_success']:.3f} fd={v['mean_final_dist']:.3f}"
                          for m, v in by_mount.items())
        print(f'B2b-{h}: {line}')

    R.make_figures(results)
    print('results.json + figures saved.')


if __name__ == '__main__':
    main()
