"""Run the full experiment: three tests -> figures + results.json.

Usage:
    python run_all.py            # full run
    python run_all.py --resume   # skip trainings whose checkpoints exist

Tests
-----
T1 entanglement:   train at yaw=0 only; closed-loop eval across yaw in [-45,45].
T2 multi-view:     train at yaws {-30,0,30}; eval across the same grid.
T3 error structure:Cam head at yaw=0 with extrinsics error: iid / static / AR(1)
                   (rho=0.5, 0.9), amplitude sigma in {0..20} deg, at query rates
                   K=1 (re-point every step) and K=5 (open-loop segments).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

import blob_world as bw
from evaluate import eval_error_structures, eval_yaw_sweep
from models import Policy
from train import build_dataset, train

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, 'checkpoints')
FIG = os.path.join(HERE, 'figures')
os.makedirs(CKPT, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

torch.set_num_threads(6)  # fewer threads: less thermal throttling on this machine

YAW_GRID = np.arange(-45.0, 45.01, 5.0)
AMPS_DEG = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0]
N_EPS_EVAL = 50


def load_model(path: str) -> Policy:
    m = Policy(3)
    m.load_state_dict(torch.load(path, weights_only=True))
    m.eval()
    return m


def make_figures(res, seed_images=True):
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

    def errbar(ax, xs, y, se, *a, **kw):
        ax.errorbar(xs, y, yerr=se, capsize=2.5, lw=1.6, *a, **kw)

    # ---- fig1: single-view training, yaw sweep ------------------------------
    t1 = res['test1']
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    for key, color, lab in [('base', 'tab:red', 'Base 头(基座系)'),
                            ('cam', 'tab:blue', 'Cam 头(相机系+GT旋转)')]:
        d = t1[key]
        xs = [float(k) for k in d]
        sr = [d[k]['success_rate'] for k in d]
        se = [d[k]['success_se'] for k in d]
        fd = [d[k]['mean_final_dist'] for k in d]
        fse = [d[k]['final_dist_se'] for k in d]
        errbar(axes[0], xs, sr, se, color=color, label=lab, marker='o', ms=3)
        errbar(axes[1], xs, fd, fse, color=color, label=lab, marker='o', ms=3)
    axes[0].axvline(0, color='k', ls=':', lw=0.8)
    axes[1].axvline(0, color='k', ls=':', lw=0.8)
    axes[0].set_xlabel('相机 yaw (度, 训练于 0°)')
    axes[0].set_ylabel('闭环成功率')
    axes[0].set_ylim(-0.05, 1.05)
    axes[1].set_xlabel('相机 yaw (度)')
    axes[1].set_ylabel('平均最终距离')
    axes[1].axhline(bw.SUCCESS_THRESH, color='gray', ls='--', lw=0.8)
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.suptitle('测试1: 单视角训练 (yaw=0°), 视角偏移闭环评测 (n=50/点)', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig1_test1_singleview_yawsweep.png'), dpi=160, bbox_inches='tight')
    plt.close(fig)

    # ---- fig2: multi-view training ------------------------------------------
    t2 = res['test2']
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    for tname, marker in [('test1', 'o'), ('test2', 's')]:
        for key, color, lab in [('base', 'tab:red', 'Base 头'),
                                ('cam', 'tab:blue', 'Cam 头')]:
            d = res[tname][key]
            xs = [float(k) for k in d]
            sr = [d[k]['success_rate'] for k in d]
            se = [d[k]['success_se'] for k in d]
            fd = [d[k]['mean_final_dist'] for k in d]
            fse = [d[k]['final_dist_se'] for k in d]
            lbl = lab if tname == 'test1' else None
            ls = '-' if tname == 'test1' else '--'
            axes[0].errorbar(xs, sr, yerr=se, fmt=marker, ls=ls, ms=3, capsize=2,
                             color=color, label=('单视角 ' + lbl if tname == 'test1' and key == 'base'
                                                 else ('单视角 ' + lab if tname == 'test1' and key == 'cam'
                                                       else ('多视角 ' + lab))))
            axes[1].errorbar(xs, fd, yerr=fse, fmt=marker, ls=ls, ms=3, capsize=2, color=color)
    for ax in axes:
        for v in [-30.0, 0.0, 30.0]:
            ax.axvline(v, color='gray', ls=':', lw=0.8)
    axes[0].set_xlabel('相机 yaw (度)')
    axes[0].set_ylabel('闭环成功率')
    axes[0].set_ylim(-0.05, 1.05)
    axes[1].set_xlabel('相机 yaw (度)')
    axes[1].set_ylabel('平均最终距离')
    axes[1].axhline(bw.SUCCESS_THRESH, color='gray', ls='--', lw=0.8)
    axes[0].legend(fontsize=7.5)
    fig.suptitle('测试2: 多视角训练 (yaw∈{-30,0,30}°) vs 单视角 (n=50/点)', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig2_test2_multiview_yawsweep.png'), dpi=160, bbox_inches='tight')
    plt.close(fig)

    # ---- fig3: error temporal structure -------------------------------------
    t3 = res['test3']
    t3t2 = res.get('test3_t2', t3)
    for kk, K in enumerate([1, 5]):
        for label, dK in [('t1cam', t3[f'K{K}']), ('t2cam', t3t2[f'K{K}'])]:
            fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
            colors = {'iid': 'tab:green', 'static': 'tab:red', 'ar0.5': 'tab:orange', 'ar0.9': 'tab:purple'}
            labels = {'iid': 'iid(每步独立)', 'static': '静态偏差(整集固定)',
                      'ar0.5': 'AR(1) ρ=0.5', 'ar0.9': 'AR(1) ρ=0.9'}
            for key in ['iid', 'static', 'ar0.5', 'ar0.9']:
                d = dK[key]
                xs = [float(k) for k in d]
                sr = [d[k]['success_rate'] for k in d]
                se = [d[k]['success_se'] for k in d]
                fd = [d[k]['mean_final_dist'] for k in d]
                fse = [d[k]['final_dist_se'] for k in d]
                errbar(axes[0], xs, sr, se, color=colors[key], label=labels[key], marker='o', ms=3)
                errbar(axes[1], xs, fd, fse, color=colors[key], label=labels[key], marker='o', ms=3)
            axes[0].set_xlabel('外参误差 σ (度, 高斯, 三种形态同边缘分布)')
            axes[0].set_ylabel('闭环成功率')
            axes[0].set_ylim(-0.05, 1.05)
            axes[1].set_xlabel('外参误差 σ (度)')
            axes[1].set_ylabel('平均最终距离')
            axes[1].axhline(bw.SUCCESS_THRESH, color='gray', ls='--', lw=0.8)
            axes[0].legend(fontsize=8)
            axes[1].legend(fontsize=8)
            who = 'T1-Cam(单视角训练)' if label == 't1cam' else 'T2-Cam(多视角训练)'
            fig.suptitle(f'测试3: {who} 外参误差时间结构 (重规划间隔 K={K} 步, n=50/点)', y=1.02)
            fig.tight_layout()
            fig.savefig(os.path.join(FIG, f'fig3_test3_errorstruct_{label}_K{K}.png'), dpi=160, bbox_inches='tight')
            plt.close(fig)

    # ---- fig4: sample views -------------------------------------------------
    if seed_images:
        fig, axes = plt.subplots(2, 4, figsize=(11, 5.5))
        rng = np.random.default_rng(7)
        ee, tgt = bw.sample_state(rng)
        for i, yaw in enumerate([-45.0, -22.5, 0.0, 22.5]):
            img = bw.render(ee, tgt, yaw)
            axes[0, i].imshow(img)
            axes[0, i].set_title(f'yaw = {yaw:+.0f}°')
            axes[0, i].axis('off')
        rng = np.random.default_rng(8)
        ee, tgt = bw.sample_state(rng)
        for i, yaw in enumerate([30.0, 45.0, 0.0, -30.0]):
            img = bw.render(ee, tgt, yaw)
            axes[1, i].imshow(img)
            axes[1, i].set_title(f'yaw = {yaw:+.0f}°')
            axes[1, i].axis('off')
        fig.suptitle('示例渲染: 绿=EE, 红=目标, 灰=固定干扰斑 (同一场景不同视角)', y=1.01)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, 'fig4_sample_views.png'), dpi=160, bbox_inches='tight')
        plt.close(fig)

    print('figures saved to', FIG, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--resume', action='store_true', help='skip existing trainings')
    ap.add_argument('--n-single', type=int, default=5000)
    ap.add_argument('--n-multi', type=int, default=4500)
    ap.add_argument('--epochs', type=int, default=15)
    args = ap.parse_args()

    t0 = time.time()
    results = {}

    # ---- datasets & training ------------------------------------------------
    ds1_path = os.path.join(CKPT, f'dataset_single_{args.n_single}.npz')
    if os.path.exists(ds1_path) and args.resume:
        z = np.load(ds1_path)
        from train import DemoDataset
        ds1 = DemoDataset(z['imgs'], z['deltas'], z['yaws'])
        print('loaded cached single-view dataset', flush=True)
    else:
        print('building single-view dataset (yaw=0, %d eps)...' % args.n_single, flush=True)
        ds1 = build_dataset([0.0], args.n_single, seed=10)
        np.savez_compressed(ds1_path, imgs=ds1.imgs.numpy(), deltas=ds1.deltas.numpy(),
                            yaws=ds1.yaws.numpy())

    ck = {h: os.path.join(CKPT, f't1_{h}_n{args.n_single}.pt') for h in ['base', 'cam']}
    models_t1 = {}
    for h in ['base', 'cam']:
        if os.path.exists(ck[h]) and args.resume:
            models_t1[h] = load_model(ck[h])
            print(f'loaded {ck[h]}', flush=True)
        else:
            models_t1[h] = train(ds1, h, args.epochs, lr=1e-3, batch_size=256, seed=1, out_path=ck[h])
    del ds1

    ds2_path = os.path.join(CKPT, f'dataset_multi_{args.n_multi}.npz')
    if os.path.exists(ds2_path) and args.resume:
        z = np.load(ds2_path)
        from train import DemoDataset
        ds2 = DemoDataset(z['imgs'], z['deltas'], z['yaws'])
        print('loaded cached multi-view dataset', flush=True)
    else:
        print('building multi-view dataset (yaws=-30/0/30, %d eps)...' % args.n_multi, flush=True)
        ds2 = build_dataset([-30.0, 0.0, 30.0], args.n_multi, seed=11)
        np.savez_compressed(ds2_path, imgs=ds2.imgs.numpy(), deltas=ds2.deltas.numpy(),
                            yaws=ds2.yaws.numpy())

    ck2 = {h: os.path.join(CKPT, f't2_{h}_n{args.n_multi}.pt') for h in ['base', 'cam']}
    models_t2 = {}
    for h in ['base', 'cam']:
        if os.path.exists(ck2[h]) and args.resume:
            models_t2[h] = load_model(ck2[h])
            print(f'loaded {ck2[h]}', flush=True)
        else:
            models_t2[h] = train(ds2, h, args.epochs, lr=1e-3, batch_size=256, seed=1, out_path=ck2[h])
    del ds2

    # ---- Test 1 & 2: yaw sweep ----------------------------------------------
    print('\n=== Test 1: single-view yaw sweep ===', flush=True)
    results['test1'] = {h: eval_yaw_sweep(models_t1[h], h, YAW_GRID, N_EPS_EVAL, seed=100)
                        for h in ['base', 'cam']}
    for h in ['base', 'cam']:
        d = results['test1'][h]
        print(f'  T1-{h}: yaw=0 sr={d[0.0]["success_rate"]:.3f} | '
              f'yaw=20 sr={d[20.0]["success_rate"]:.3f} | yaw=45 sr={d[45.0]["success_rate"]:.3f}', flush=True)

    print('\n=== Test 2: multi-view yaw sweep ===', flush=True)
    results['test2'] = {h: eval_yaw_sweep(models_t2[h], h, YAW_GRID, N_EPS_EVAL, seed=101)
                        for h in ['base', 'cam']}
    for h in ['base', 'cam']:
        d = results['test2'][h]
        print(f'  T2-{h}: yaw=-30 sr={d[-30.0]["success_rate"]:.3f} | yaw=0 sr={d[0.0]["success_rate"]:.3f} | '
              f'yaw=30 sr={d[30.0]["success_rate"]:.3f} | yaw=15 sr={d[15.0]["success_rate"]:.3f}', flush=True)

    # ---- Test 3: error temporal structure (Cam heads, eval at trained view) --
    print('\n=== Test 3: error temporal structure ===', flush=True)
    results['test3'] = {}
    results['test3_t2'] = {}
    for label, cam_model in [('t1', models_t1['cam']), ('t2', models_t2['cam'])]:
        results['test3' if label == 't1' else 'test3_t2'] = {}
        for K in [1, 5]:
            dst = results['test3' if label == 't1' else 'test3_t2']
            dst[f'K{K}'] = eval_error_structures(cam_model, 'cam',
                                                 AMPS_DEG, N_EPS_EVAL, seed=102,
                                                 yaw=0.0, query_k=K)
            for key in ['iid', 'static', 'ar0.5', 'ar0.9']:
                d = dst[f'K{K}'][key]
                print(f'  {label} K={K} {key}: sigma=0 sr={d[0.0]["success_rate"]:.3f} | '
                      f'sigma=10 sr={d[10.0]["success_rate"]:.3f} | sigma=20 sr={d[20.0]["success_rate"]:.3f}', flush=True)

    results['config'] = {
        'yaw_grid': list(YAW_GRID), 'amps_deg': AMPS_DEG, 'n_eps_eval': N_EPS_EVAL,
        'n_single': args.n_single, 'n_multi': args.n_multi, 'epochs': args.epochs,
        'v_max': bw.V_MAX, 'success_thresh': bw.SUCCESS_THRESH, 'max_steps': bw.MAX_STEPS,
    }

    with open(os.path.join(HERE, 'results.json'), 'w') as f:
        json.dump(results, f, indent=1, default=float)

    make_figures(results)
    print(f'\nAll done in {time.time()-t0:.0f}s. Results: {os.path.join(HERE, "results.json")}', flush=True)


if __name__ == '__main__':
    main()
