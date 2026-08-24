"""Orchestrate B2a (pose conditioning) and B2b (cross-embodiment pooling).

Usage:
    python run_b2.py            # full run
    python run_b2.py --resume   # reuse existing datasets/checkpoints, only re-evaluate

Protocols (all hyperparameters = experiment B unless noted):
  B2a T1    : yaw=0 train, 5000 eps (exp-B T1 protocol; pose constant -> degenerate control)
  B2a T2    : yaw in {-30,0,30} train, 4500 eps (exp-B T2 protocol; pose varies over 3 views)
  B2a DENSE : yaw~U(-45,45) train, 4500 eps (pose input dense)
  B2a SCAN  : T2 protocol at {1k, 4k, 16k} eps, base_pose vs cam (nested prefixes of the T2
              stream; 4.5k main T2 adds another point)
  B2b       : yaw=0, 5000 eps alternating E1/E2 mounts (50/50), conditions
              cam / base_pose / base_pose_label / base_ref; eval E1,E2,E3 x 19-yaw grid
Outputs: results.json, figures/, checkpoints/.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

import blob_world as bw
from b2 import MOUNTS, build_dataset, eval_yaw_sweep, load_model, train

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, 'checkpoints')
FIG = os.path.join(HERE, 'figures')
os.makedirs(CKPT, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

torch.set_num_threads(4)  # small tensors: >4 threads causes barrier thrash (measured ~20x slower)

YAW_GRID = np.arange(-45.0, 45.01, 5.0)
N_EPS = 50
EMB_LABEL = {'E1': np.array([1.0, 0.0]), 'E2': np.array([0.0, 1.0]),
             'E3': np.array([0.0, 0.0])}  # E3 label is out-of-distribution (zero vector)

# ---- spec functions: (yaw, mount) per episode, sharing the dataset rng stream ------
def spec_t1(rng, i):
    return 0.0, 'E1'

def spec_t2(rng, i):
    return float(rng.choice([-30.0, 0.0, 30.0])), 'E1'

def spec_dense(rng, i):
    return float(rng.uniform(-45.0, 45.0)), 'E1'

def spec_b2b(rng, i):
    return 0.0, ('E1' if i % 2 == 0 else 'E2')


def get_dataset(name: str, n_eps: int, seed: int, spec_fn, resume: bool) -> object:
    path = os.path.join(CKPT, f'dataset_{name}.npz')
    if os.path.exists(path) and resume:
        z = np.load(path)
        from b2 import Dataset
        ds = Dataset(z['imgs'], z['deltas'], z['yaws'], z['emb'])
        print(f'loaded cached dataset {name} ({ds.n} samples)', flush=True)
        return ds
    t0 = time.time()
    print(f'building dataset {name}: {n_eps} eps, seed {seed}...', flush=True)
    ds = build_dataset(n_eps, seed, spec_fn)
    np.savez_compressed(path, imgs=ds.imgs.numpy(), deltas=ds.deltas_b.numpy(),
                        yaws=ds.yaws, emb=ds.emb)
    print(f'  {name}: {ds.n} samples in {time.time()-t0:.0f}s', flush=True)
    return ds


def get_model(ds, name: str, head: str, resume: bool):
    path = os.path.join(CKPT, f'{name}_{head}.pt')
    if os.path.exists(path) and resume:
        m = load_model(path, head)
        print(f'loaded {path}', flush=True)
        return m
    m = train(ds, head, out_path=path)
    return m


def mean_success(d: dict) -> tuple[float, float]:
    sr = np.array([d[k]['success_rate'] for k in d])
    return float(sr.mean()), float(sr.std(ddof=1) / np.sqrt(len(sr)))


def summarize(cond_evals: dict) -> dict:
    """cond_evals: {head: {mount: {yaw: stats}}} -> per-head per-mount grid means."""
    out = {}
    for head, by_mount in cond_evals.items():
        out[head] = {}
        for mount, d in by_mount.items():
            sr, sse = mean_success(d)
            fd = np.mean([d[k]['mean_final_dist'] for k in d])
            out[head][mount] = {'mean_success': sr, 'success_se_of_mean': sse,
                                'mean_final_dist': float(fd)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()
    resume = args.resume
    t0 = time.time()
    results = {}

    # ================= B2a =================
    print('\n========== B2a: pose conditioning ==========', flush=True)

    # --- datasets ---
    d_t1 = get_dataset('t1_5000', 5000, 10, spec_t1, resume)
    d_t2 = get_dataset('t2_4500', 4500, 11, spec_t2, resume)

    # --- T1: single-view (pose constant -> degenerate control) ---
    b2a = {}
    print('\n--- B2a T1: train yaw=0, 5000 eps ---', flush=True)
    b2a['t1'] = {}
    for h in ['base', 'base_pose', 'cam']:
        m = get_model(d_t1, 't1', h, resume)
        b2a['t1'][h] = eval_yaw_sweep(m, h, YAW_GRID, N_EPS, seed=100, mount_name='E1')
        s, se = mean_success(b2a['t1'][h])
        print(f'  T1-{h}: mean_success={s:.3f} (se {se:.3f})', flush=True)
    results['b2a_t1'] = b2a['t1']
    with open(os.path.join(HERE, 'results.json'), 'w') as f:
        json.dump(results, f, indent=1, default=float)

    # --- T2: multi-view (pose varies over 3 views) ---
    print('\n--- B2a T2: train yaw in {-30,0,30}, 4500 eps ---', flush=True)
    b2a['t2'] = {}
    for h in ['base', 'base_pose', 'cam']:
        m = get_model(d_t2, 't2', h, resume)
        b2a['t2'][h] = eval_yaw_sweep(m, h, YAW_GRID, N_EPS, seed=101, mount_name='E1')
        s, se = mean_success(b2a['t2'][h])
        print(f'  T2-{h}: mean_success={s:.3f} (se {se:.3f})', flush=True)
    results['b2a_t2'] = b2a['t2']
    with open(os.path.join(HERE, 'results.json'), 'w') as f:
        json.dump(results, f, indent=1, default=float)

    # --- DENSE: continuous yaw training (pose input dense) ---
    print('\n--- B2a DENSE: train yaw~U(-45,45), 4500 eps ---', flush=True)
    d_dense = get_dataset('dense_4500', 4500, 12, spec_dense, resume)
    b2a['dense'] = {}
    for h in ['base', 'base_pose', 'cam']:
        m = get_model(d_dense, 'dense', h, resume)
        b2a['dense'][h] = eval_yaw_sweep(m, h, YAW_GRID, N_EPS, seed=102, mount_name='E1')
        s, se = mean_success(b2a['dense'][h])
        print(f'  DENSE-{h}: mean_success={s:.3f} (se {se:.3f})', flush=True)
    results['b2a_dense'] = b2a['dense']
    with open(os.path.join(HERE, 'results.json'), 'w') as f:
        json.dump(results, f, indent=1, default=float)

    # --- sample-size scan at the T2 protocol ---
    print('\n--- B2a SCAN: T2 protocol, {1k,4k,16k} eps (base_pose vs cam) ---', flush=True)
    b2a['scan'] = {}
    for n_eps in [1000, 4000, 16000]:
        d = get_dataset(f't2_{n_eps}', n_eps, 11, spec_t2, resume)
        b2a['scan'][str(n_eps)] = {}
        for h in ['base_pose', 'cam']:
            m = get_model(d, f'scan{n_eps}', h, resume)
            b2a['scan'][str(n_eps)][h] = eval_yaw_sweep(m, h, YAW_GRID, N_EPS, seed=101, mount_name='E1')
            s, se = mean_success(b2a['scan'][str(n_eps)][h])
            print(f'  SCAN n={n_eps} {h}: mean_success={s:.3f} (se {se:.3f})', flush=True)
    results['b2a_scan'] = b2a['scan']
    with open(os.path.join(HERE, 'results.json'), 'w') as f:
        json.dump(results, f, indent=1, default=float)
    del d_t1, d_t2

    # ================= B2b =================
    print('\n========== B2b: cross-embodiment pooling ==========', flush=True)
    d_b2b = get_dataset('b2b_5000', 5000, 13, spec_b2b, resume)
    b2b_cond_evals = {}
    for h in ['cam', 'base_pose', 'base_pose_label', 'base_ref']:
        head = 'base' if h == 'base_ref' else h
        m = get_model(d_b2b, 'b2b', h, resume)
        b2b_cond_evals[h] = {}
        for mount in ['E1', 'E2', 'E3']:
            label = EMB_LABEL[mount] if h == 'base_pose_label' else None
            b2b_cond_evals[h][mount] = eval_yaw_sweep(m, head, YAW_GRID, N_EPS, seed=110,
                                                      mount_name=mount, emb_label=label)
    summ = summarize(b2b_cond_evals)
    for h, by_mount in summ.items():
        line = ' | '.join(f"{m}: sr={v['mean_success']:.3f} fd={v['mean_final_dist']:.3f}"
                          for m, v in by_mount.items())
        print(f'  B2b-{h}: {line}', flush=True)
    results['b2b_conditions'] = b2b_cond_evals
    results['b2b_summary'] = summ
    with open(os.path.join(HERE, 'results.json'), 'w') as f:
        json.dump(results, f, indent=1, default=float)

    results['config'] = {
        'yaw_grid': list(YAW_GRID), 'n_eps_eval': N_EPS, 'epochs': 15, 'lr': 1e-3,
        'batch_size': 256, 'train_seed': 1,
        'mounts': {k: v for k, v in MOUNTS.items()},
        'pose_input': 'R_bc(9) + t_b/2.5(3)', 'emb_label': 'one-hot(2)',
    }
    with open(os.path.join(HERE, 'results.json'), 'w') as f:
        json.dump(results, f, indent=1, default=float)

    make_figures(results)
    print(f'\nAll done in {time.time()-t0:.0f}s. results.json + figures saved.', flush=True)


def make_figures(results: dict):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for f in ['/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc',
              '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc']:
        if os.path.exists(f):
            font_manager.fontManager.addfont(f)
    plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': 0.3,
                         'font.family': 'Noto Serif CJK SC',
                         'axes.unicode_minus': False})

    def xs_of(d):
        return [float(k) for k in d]

    # ---- fig 1: 19-yaw generalization (T1 / T2 / DENSE) ----------------------
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.0), sharey=True)
    styles = [('base', 'tab:red', 'Base(基座系,无位姿)', '-'),
              ('base_pose', 'tab:green', 'Base+位姿输入', '--'),
              ('cam', 'tab:blue', 'Cam(相机系+GT旋转)', '-')]
    panels = [('T1', results['b2a_t1'], 'T1: 单视角训练 (yaw=0)\n位姿输入恒定 → 无学习信号'),
              ('T2', results['b2a_t2'], 'T2: 多视角训练 (yaw∈{-30,0,30}°)\n位姿输入 3 个值'),
              ('DENSE', results['b2a_dense'], 'DENSE: 连续视角训练 (yaw~U(-45,45)°)\n位姿输入稠密')]
    for ax, (key, data, title) in zip(axes, panels):
        for h, color, lab, ls in styles:
            if h not in data:
                continue
            xs = xs_of(data[h])
            sr = [data[h][k]['success_rate'] for k in data[h]]
            se = [data[h][k]['success_se'] for k in data[h]]
            ax.errorbar(xs, sr, yerr=se, fmt='o', ms=3, capsize=2, lw=1.6,
                        color=color, label=lab, ls=ls)
        ax.axvline(0, color='k', ls=':', lw=0.8)
        ax.set_xlabel('相机 yaw (度)')
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(title, fontsize=9)
    axes[0].set_ylabel('闭环成功率 (n=50/点)')
    axes[0].legend(fontsize=7.5, loc='lower left')
    fig.suptitle('B2a: 给 Base 头相机位姿输入后,19-yaw 泛化 (E1 本体, 5000/4500/4500 eps)', y=1.03)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig_b2a1_yaw_generalization.png'), dpi=170, bbox_inches='tight')
    plt.close(fig)

    # ---- fig 2: sample complexity (T2 protocol) ------------------------------
    scan = results['b2a_scan']
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    x_all = {'1000': 1e3, '4000': 4e3, '16000': 16e3}
    x_main = {'4500': 4.5e3}
    for h, color, lab, ls in [('base_pose', 'tab:green', 'Base+位姿', '--'),
                              ('cam', 'tab:blue', 'Cam', '-')]:
        xs, ys, yse = [], [], []
        for nk, d in scan.items():
            if h not in d:
                continue
            xs.append(x_all[nk])
            ys.append(mean_success(d[h])[0])
            yse.append(mean_success(d[h])[1])
        # add the main T2 point (4500 eps)
        d4500 = results['b2a_t2'][h]
        xs.append(x_main['4500']); ys.append(mean_success(d4500)[0]); yse.append(mean_success(d4500)[1])
        ax.errorbar(xs, ys, yerr=yse, fmt='o', ms=5, capsize=3, lw=1.6, color=color, label=lab, ls=ls)
    ax.set_xscale('log')
    ax.set_xticks([1e3, 4e3, 4.5e3, 16e3])
    ax.set_xticklabels(['1k', '4k', '4.5k(主)', '16k'])
    ax.set_xlabel('演示集数 (T2 协议: yaw∈{-30,0,30}°)')
    ax.set_ylabel('19-yaw 网格平均闭环成功率')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    ax.set_title('B2a: 样本复杂度 (平均成功率, n=50/点)')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig_b2a2_sample_complexity.png'), dpi=170, bbox_inches='tight')
    plt.close(fig)

    # ---- fig 3: B2b cross-embodiment (2x3: success/final-dist x E1/E2/E3) ----
    summ = results['b2b_summary']
    conds = ['cam', 'base_pose', 'base_pose_label', 'base_ref']
    colors = {'cam': 'tab:blue', 'base_pose': 'tab:green',
              'base_pose_label': 'tab:orange', 'base_ref': 'gray'}
    labels = {'cam': 'Cam(无标签)', 'base_pose': 'Base+位姿',
              'base_pose_label': 'Base+位姿+本体标签', 'base_ref': 'Base(参考地板)'}
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.2), sharey='row')
    mount_titles = {'E1': 'E1 (训练本体 A)\nh=0.6 d=2.5', 'E2': 'E2 (训练本体 B)\nh=1.0 d=2.2',
                    'E3': 'E3 (未见本体)\nh=0.8 d=2.35'}
    xpos = np.arange(len(conds))
    w = 0.7
    for j, mount in enumerate(['E1', 'E2', 'E3']):
        sr = [summ[c][mount]['mean_success'] for c in conds]
        sse = [summ[c][mount]['success_se_of_mean'] for c in conds]
        fd = [summ[c][mount]['mean_final_dist'] for c in conds]
        fse = [0.01] * len(conds)  # se of the mean over yaw grid (final dist), display-only
        for i, c in enumerate(conds):
            axes[0, j].bar(xpos[i], sr[i], w, color=colors[c],
                           yerr=sse[i], capsize=3, label=labels[c] if j == 0 else None)
            axes[1, j].bar(xpos[i], fd[i], w, color=colors[c], alpha=0.75)
        axes[0, j].set_title(mount_titles[mount], fontsize=9.5)
        axes[0, j].axhline(1.0, color='gray', ls='--', lw=0.8)
        axes[1, j].axhline(bw.SUCCESS_THRESH, color='gray', ls='--', lw=0.8)
        axes[0, j].set_xticks(xpos)
        axes[0, j].set_xticklabels(['Cam', 'B+P', 'B+P+L', 'Base'], fontsize=7.5)
        axes[1, j].set_xticklabels(['Cam', 'B+P', 'B+P+L', 'Base'], fontsize=7.5)
    axes[0, 0].set_ylabel('19-yaw 平均成功率 (n=50/点)')
    axes[1, 0].set_ylabel('19-yaw 平均终距')
    axes[0, 0].legend(fontsize=7.5, loc='lower left')
    axes[0, 0].set_ylim(0, 1.05)
    fig.suptitle('B2b: 跨本体 50/50 混合训练 (yaw=0 单视角), 条件对比 (B+P=Base+位姿, L=本体标签)', y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig_b2b_cross_embodiment.png'), dpi=170, bbox_inches='tight')
    plt.close(fig)

    # ---- fig 4 (supplementary): cam flatness across mounts -------------------
    conds4 = results['b2b_conditions']
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for mount, color, lab in [('E1', 'tab:blue', 'E1 (训练)'), ('E2', 'tab:orange', 'E2 (训练)'),
                              ('E3', 'tab:green', 'E3 (未见)')]:
        d = conds4['cam'][mount]
        xs = xs_of(d)
        sr = [d[k]['success_rate'] for k in d]
        se = [d[k]['success_se'] for k in d]
        ax.errorbar(xs, sr, yerr=se, fmt='o', ms=3, capsize=2, lw=1.6, color=color, label=lab)
    ax.set_xlabel('相机 yaw (度)')
    ax.set_ylabel('闭环成功率')
    ax.set_ylim(0.5, 1.05)
    ax.legend(fontsize=8)
    ax.set_title('B2b 补充: Cam 头跨本体 19-yaw 成功率 (混合训练, 无本体标签)')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig_b2b_cam_flatness.png'), dpi=170, bbox_inches='tight')
    plt.close(fig)
    print('figures saved to', FIG, flush=True)


if __name__ == '__main__':
    main()
