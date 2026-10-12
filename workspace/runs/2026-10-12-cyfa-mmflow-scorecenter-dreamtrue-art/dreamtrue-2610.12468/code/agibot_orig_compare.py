"""Compare AgiBotWorld-Beta dataset-provided camera params (hunted from the 48GB
parameters tars) against DreamTrue's refined calibration for the same 3 episodes
(410/649000, 380/665152, 599/835856).

The original extrinsic {rotation_matrix, translation_vector} frame is undocumented
in the dataset. Strategy:
  1. cross-episode consistency: is R_orig ~constant per camera across episodes?
     (a camera rigid w.r.t. its parent frame => original is cam->some fixed frame)
  2. solve the fixed parent-frame change A minimizing sum_i ||R_refined_i - A R_orig_i||^2
     (24 signed permutations, plus a general-rotation Kabsch solution). A general A
     fits to the noise floor for all 3 cameras; adding a right factor does not help,
     so the camera-frame conventions agree and only the parent frame differs.
  3. translation: with A fixed, per-episode offset b_i = t_refined - A t_orig must be
     constant across episodes; its spread is the residual.

Intrinsics are reported verbatim: original params lack an image size; refined are
640x360. Original fx~322 / fy~429 (fy/fx ~ 1.33) is anomalous on its own.
"""
import glob
import json
import itertools
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
MINI = os.path.join(BASE, 'condition-mini/condition')

def quat_wxyz_to_R(q):
    w, x, y, z = np.asarray(q, dtype=float)
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])

C_SP2CV = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=float)

def rot_angle_deg(R):
    c = (np.trace(R) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))

def signed_permutations():
    out = []
    for perm in itertools.permutations(range(3)):
        P = np.zeros((3, 3))
        for i, j in enumerate(perm):
            P[i, j] = 1
        for signs in itertools.product([1, -1], repeat=3):
            M = P * np.array(signs)[:, None]
            if abs(round(np.linalg.det(M)) - 1) == 0 and round(np.linalg.det(M)) == 1:
                out.append(M)
    return out

PERMS = signed_permutations()
assert len(PERMS) == 24

def kabsch_left(Xs, Ys):
    """R minimizing sum ||R X_i - Y_i||_F^2 (X_i, Y_i are 3x3)."""
    M = sum(Xs[i] @ Ys[i].T for i in range(len(Xs)))
    U, _, Vt = np.linalg.svd(M)
    D = np.diag([1, 1, np.linalg.det(Vt.T @ U.T)])
    return Vt.T @ D @ U.T

def main():
    orig = {}
    for fp in sorted(glob.glob(os.path.join(HERE, 'agibot-orig', '*_*.json'))):
        if 'scanmap' in fp:
            continue
        d = json.load(open(fp))
        orig[f"{d['task']}/{d['episode']}"] = d['files']
    refined = {}
    for fp in sorted(glob.glob(os.path.join(MINI, 'calibration/AgibotWorld/param/*/*/optimize/*.json'))):
        key = fp.split('/param/')[1].split('/optimize/')[0]
        refined.setdefault(key, {})[os.path.basename(fp)[:-5]] = json.load(open(fp))

    cams = ['head', 'hand_left', 'hand_right']
    result = {'episodes': sorted(orig), 'per_camera': {}}
    for cam in cams:
        Ro, to, Rr, tr = [], [], [], []
        eps_ids = []
        intr_rows = []
        for ep in sorted(orig):
            if ep not in refined:
                continue
            o = orig[ep].get(f'{cam}_extrinsic_params.json', {}).get('extrinsic')
            i = orig[ep].get(f'{cam}_intrinsic_params.json', {}).get('intrinsic')
            r_pose = refined[ep]['local_pose_best'][cam]
            r_intr = refined[ep]['intrinsic_best'][cam]
            if o is None:
                continue
            eps_ids.append(ep)
            Ro.append(np.array(o['rotation_matrix']))
            to.append(np.array(o['translation_vector']))
            Rr.append(quat_wxyz_to_R(r_pose['q']))   # SAPIEN cam frame rel mount
            tr.append(np.array(r_pose['p']))
            oi = {k: i[k] for k in ('fx', 'fy', 'ppx', 'ppy', 'k1', 'k2', 'k3', 'p1', 'p2')}
            ri = {k: r_intr[k] for k in ('fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'k3', 'p1', 'p2', 'w', 'h')}
            intr_rows.append({
                'episode': ep,
                'orig': oi,
                'refined': ri,
                'orig_fy_over_fx': oi['fy'] / oi['fx'],
                # hypothesis: original intrinsics refer to a 640x480 frame, the
                # DreamTrue pipeline works at 640x360 -> fy, ppy scale by 0.75
                'fx_rel_refined_vs_orig': ri['fx'] / oi['fx'] - 1,
                'fy_rel_refined_vs_orig_x0.75': ri['fy'] / (oi['fy'] * 0.75) - 1,
                'dcx_px': ri['cx'] - oi['ppx'],
                'dcy_px_vs_ppy_x0.75': ri['cy'] - oi['ppy'] * 0.75,
            })
        Ro, to, Rr, tr = map(np.array, (Ro, to, Rr, tr))
        n = len(Ro)
        # 1) cross-episode consistency
        cons = {
            'n_episodes': n,
            'orig_rot_max_pairwise_deg': max(rot_angle_deg(Ro[i].T @ Ro[j]) for i in range(n) for j in range(i + 1, n)) if n > 1 else 0,
            'orig_trans_max_pairwise_m': max(float(np.linalg.norm(to[i] - to[j])) for i in range(n) for j in range(i + 1, n)) if n > 1 else 0,
            'refined_rot_max_pairwise_deg': max(rot_angle_deg(Rr[i].T @ Rr[j]) for i in range(n) for j in range(i + 1, n)) if n > 1 else 0,
            'refined_trans_max_pairwise_m': max(float(np.linalg.norm(tr[i] - tr[j])) for i in range(n) for j in range(i + 1, n)) if n > 1 else 0,
        }
        # 2) best fixed left/right signed-permutation basis change
        best = None
        for A in PERMS:
            for B in PERMS:
                err = max(rot_angle_deg((A @ Ro[i] @ B).T @ Rr[i]) for i in range(n))
                if best is None or err < best[0]:
                    best = (err, A, B)
        err, A, B = best
        # general fixed left basis change (parent-frame change), camera frame kept
        A1 = kabsch_left(Ro, Rr)
        errA1 = max(rot_angle_deg((A1 @ Ro[i]).T @ Rr[i]) for i in range(n))
        # translation with A FIXED: per-episode offset b_i = t_r - A t_o; spread is the residual
        b_i = np.array([tr[i] - A1 @ to[i] for i in range(n)])
        b_mean = b_i.mean(axis=0)
        trans_res = [float(np.linalg.norm(b_i[i] - b_mean)) for i in range(n)]
        # per-episode refinement deltas after absorbing the fixed frame change
        per_ep_delta = [{
            'episode': eps_ids[i],
            'rot_delta_deg': rot_angle_deg((A1 @ Ro[i]).T @ Rr[i]),
            'trans_delta_m': float(np.linalg.norm(tr[i] - (A1 @ to[i] + b_mean))),
        } for i in range(n)]
        cons.update({
            'best_left_A': A.tolist(), 'best_right_B': B.tolist(),
            'rot_residual_max_deg_after_basis': err,
            'general_A': A1.tolist(),
            'general_A_axis_angle_deg': rot_angle_deg(A1),
            'rot_residual_max_deg_general_A_only': errA1,
            'trans_offset_b_spread_m': trans_res,
            'trans_offset_b_mean': b_mean.tolist(),
            'per_episode_delta_after_frame_fix': per_ep_delta,
        })
        # also raw delta without basis change for scale intuition
        cons['raw_rot_delta_max_deg'] = max(rot_angle_deg(Ro[i].T @ Rr[i]) for i in range(n)) if n else None
        result['per_camera'][cam] = {'consistency': cons, 'intrinsics': intr_rows}

    outp = os.path.join(HERE, 'agibot_orig_vs_refined.json')
    with open(outp, 'w') as f:
        json.dump(result, f, indent=1)
    print('wrote', outp)
    for cam in cams:
        c = result['per_camera'][cam]['consistency']
        print(cam, json.dumps({k: v for k, v in c.items() if not isinstance(v, list)}, indent=1))

if __name__ == '__main__':
    main()
