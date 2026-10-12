"""Plot recalculation results: calibration deltas (init->refined and dataset-provided->refined).

Input: recalc/recalc_results.json (produced by recalc_calibration.py)
       recalc/agibot_orig_vs_refined.json (produced by agibot_orig_compare.py;
       AgiBot orig deltas are after absorbing the fitted fixed parent-frame change)
Output: recalc/dt-recalc-calib-deltas.png
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
res = json.load(open(os.path.join(HERE, 'recalc_results.json')))
agi_path = os.path.join(HERE, 'agibot_orig_vs_refined.json')
agi = json.load(open(agi_path)) if os.path.exists(agi_path) else None

DS_LABEL = {'AgibotWorld': 'AgiBot', 'droid_raw': 'DROID', 'robomind_franka': 'RoboMIND'}
DS_COLOR = {'AgibotWorld': '#1f77b4', 'droid_raw': '#d62728', 'robomind_franka': '#2ca02c'}


def short_ep(key):
    parts = key.split('/')
    if parts[0] == 'AgibotWorld':
        return f'A:{parts[2]}/{parts[3]}'
    if parts[0] == 'droid_raw':
        # droid_raw/param/1.0.1/<lab>/<status>/<ts>/optimize
        toks = parts[5].split('_')
        return f'D:{parts[3]} {" ".join(toks[1:3])}'
    # robomind_franka/param/<task>/<status>/<ts>/optimize
    return f'R:{parts[2][:12]}…/{parts[4]}'


def collect():
    """rows keyed by (dataset, episode, camera): {'init': (t,r,fx), 'orig': (t,r,fx)}"""
    rows = {}
    for r in res['init_vs_refined']:
        rows.setdefault((r['dataset'], r['calib_key'], r['camera']), {})['init'] = (
            r['trans_err_m'] * 100, r['rot_err_deg'],
            None if r['fx_rel_err'] is None else r['fx_rel_err'] * 100)
    for r in res['droid_original_vs_refined']:
        rows.setdefault(('droid_raw', r['calib_key'], r['camera']), {})['orig'] = (
            r['trans_err_m'] * 100, r['rot_err_deg'], r['fx_rel_err'] * 100)
    if agi:
        for cam, v in agi['per_camera'].items():
            fx_by_ep = {r['episode']: r['fx_rel_refined_vs_orig'] for r in v['intrinsics']}
            for d in v['consistency']['per_episode_delta_after_frame_fix']:
                key = f"AgibotWorld/param/{d['episode']}/optimize"
                rows.setdefault(('AgibotWorld', key, cam), {})['orig'] = (
                    d['trans_delta_m'] * 100, d['rot_delta_deg'],
                    fx_by_ep[d['episode']] * 100)
    return rows


def main():
    rows = collect()
    keys = sorted(rows, key=lambda k: (list(DS_LABEL).index(k[0]), k[1], k[2]))
    groups = []  # (label, color, [(kind, t, r, fx)])
    for ds, ep, cam in keys:
        entry = rows[(ds, ep, cam)]
        g = []
        if 'init' in entry:
            g.append(('init→refined',) + entry['init'])
        if 'orig' in entry:
            g.append(('orig→refined',) + entry['orig'])
        groups.append((f'{short_ep(ep)}\n{cam}', DS_COLOR[ds], g))

    # x layout: up to 2 bars per group
    xt, labels = [], []
    bars = {0: [], 1: [], 2: []}  # per panel lists of (x, val, color, hatch)
    pos = 0.0
    for label, color, g in groups:
        w = len(g)
        for i, (kind, t, r, fx) in enumerate(g):
            hatch = '//' if kind == 'orig→refined' else ''
            bars[0].append((pos, t, color, hatch))
            bars[1].append((pos, r, color, hatch))
            if fx is not None:
                bars[2].append((pos, fx, color, hatch))
            pos += 1.0
        xt.append(pos - (w + 1) / 2)
        labels.append(label)
        pos += 0.6

    fig, axes = plt.subplots(3, 1, figsize=(14, 11))
    titles = ['Extrinsic translation delta to reference', 'Extrinsic rotation delta to reference',
              'Focal length: refined vs reference (relative change in fx)']
    ylabels = ['|Δt| (cm, log)', 'ΔR (deg, log)', 'Δfx / fx_ref (%)']
    for pi, ax in enumerate(axes):
        for x, v, c, h in bars[pi]:
            ax.bar(x, v, color=c, hatch=h, edgecolor='black' if h else None, linewidth=0.4, width=0.85)
        if pi < 2:
            ax.set_yscale('log')
            ax.axhline(1.0, color='gray', lw=0.5, ls='--')
        else:
            ax.axhline(0.0, color='black', lw=0.8)
        ax.set_ylabel(ylabels[pi])
        ax.set_title(titles[pi])
        ax.grid(axis='y', alpha=0.3)
        ax.set_xlim(-0.8, pos)
    axes[2].set_xticks(xt)
    axes[2].set_xticklabels(labels, fontsize=6.5, rotation=90)
    axes[2].set_xlabel('episode × camera  (A=AgiBot, D=DROID, R=RoboMIND)')
    handles = [Patch(color=DS_COLOR[k], label=DS_LABEL[k]) for k in DS_LABEL]
    handles += [Patch(facecolor='white', edgecolor='black', label='init (robot YAML) → refined'),
                Patch(facecolor='white', edgecolor='black', hatch='//',
                      label='dataset-provided → refined (AgiBot: after fixed frame fit)')]
    axes[0].legend(handles=handles, loc='upper left', fontsize=8)
    fig.tight_layout()
    out = os.path.join(HERE, 'dt-recalc-calib-deltas.png')
    fig.savefig(out, dpi=150, facecolor='white')
    print('wrote', out)


if __name__ == '__main__':
    main()
