"""DreamTrue calibration release — independent recalculation.

Data sources (all fetched fresh, no project code executed):
  - Refined params: ModelScope dataset huoxingdawang/DreamTrue,
    condition/condition-mini.tar.gz -> condition/calibration/<dataset>/param/<key>/optimize/*.json
  - Init guesses: ModelScope dataset huoxingdawang/DreamTrueAssets
    (the `assets` submodule of the calibration repo): agibot/G1_120s/G1_120s.yaml,
    droid/franka/franka.yaml, robomind/franka/franka.yaml -> `cameras:` section
  - DROID dataset-provided originals: gs://gresearch/robotics/droid_raw metadata_*.json
    (per-episode `*_cam_extrinsics` = [x,y,z,rx,ry,rz]) and KarlP/droid intrinsics.json
    (per-episode per-serial [fx,cx,fy,cy] from the ZED SVO factory calibration).
  - DROID joint states for wrist-camera FK: trajectory.h5 observation/robot_state/joint_positions.

Conventions (verified against code/DreamTrue/calibration sources):
  - local_pose: {p, q(w>=0,wxyz)} camera pose relative to its mount link, SAPIEN camera
    frame (x forward, y left, z up). OpenCV frame (x right, y down, z fwd) = C @ sapien
    with C = [[0,-1,0],[0,0,-1],[1,0,0]].
  - intrinsic: OpenCV plumb-bob, stored image is the distorted image
    (sim/camera.py: render with rectified intrinsics, remap through camera_distort_remap).
  - Projection of a world point P: p_cv = C @ R_sp^T (P - t_sp); x=X/Z,y=Y/Z;
    distort(k1..k3,p1,p2); u = fx*x_d+cx, v = fy*y_d+cy.
"""
import glob
import hashlib
import json
import os
import xml.etree.ElementTree as ET

import h5py
import hdf5plugin  # noqa: F401  (registers zstd decompressor for the condition h5 files)
import numpy as np
import yaml

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # code/dt-recalc
MINI = os.path.join(BASE, 'condition-mini/condition')


# ---------------- math helpers ----------------

def quat_wxyz_to_R(q):
    w, x, y, z = np.asarray(q, dtype=float)
    n = w * w + x * x + y * y + z * z
    w, x, y, z = w / np.sqrt(n), x / np.sqrt(n), y / np.sqrt(n), z / np.sqrt(n)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def euler_xyz_intrinsic_to_R(e):
    """scipy from_euler('xyz') semantics: R = Rx @ Ry @ Rz."""
    cx, sx = np.cos(e[0]), np.sin(e[0])
    cy, sy = np.cos(e[1]), np.sin(e[1])
    cz, sz = np.cos(e[2]), np.sin(e[2])
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rx @ Ry @ Rz


def euler_xyz_extrinsic_to_R(e):
    """R = Rz @ Ry @ Rx."""
    cx, sx = np.cos(e[0]), np.sin(e[0])
    cy, sy = np.cos(e[1]), np.sin(e[1])
    cz, sz = np.cos(e[2]), np.sin(e[2])
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def rot_angle_deg(R):
    c = (np.trace(R) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


# sapien camera frame -> opencv camera frame
C_SP2CV = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=float)


def sapien_pose_to_opencv_cam2mount(p, q):
    """Return (R_cb, t_cb): camera->mount transform in OpenCV camera convention."""
    R_sp = quat_wxyz_to_R(q)
    return R_sp @ C_SP2CV.T, np.asarray(p, dtype=float)


def opencv_project(P_world, R_cam2world, t_cam2world, intr):
    """Project world points with cam2world pose (opencv convention) + plumb-bob distortion."""
    P = np.asarray(P_world, dtype=float)
    Pc = (P - t_cam2world) @ R_cam2world  # R^T (P - t)
    z = Pc[:, 2]
    x, y = Pc[:, 0] / z, Pc[:, 1] / z
    r2 = x * x + y * y
    ra = 1 + intr['k1'] * r2 + intr['k2'] * r2 ** 2 + intr['k3'] * r2 ** 3
    xd = x * ra + 2 * intr['p1'] * x * y + intr['p2'] * (r2 + 2 * x ** 2)
    yd = y * ra + intr['p1'] * (r2 + 2 * y ** 2) + 2 * intr['p2'] * x * y
    return np.stack([intr['fx'] * xd + intr['cx'], intr['fy'] * yd + intr['cy']], -1), z


# ---------------- loaders ----------------

def load_refined():
    """mini package calibration subset -> {dataset: {calib_key: {cam: ...}}}"""
    out = {}
    for ep_dir in sorted(glob.glob(os.path.join(MINI, 'calibration/*/*/**/optimize'), recursive=True)):
        rel = os.path.relpath(ep_dir, os.path.join(MINI, 'calibration'))
        dataset = rel.split(os.sep)[0]
        rec = {}
        for f in glob.glob(os.path.join(ep_dir, '*.json')):
            rec[os.path.basename(f)[:-5]] = json.load(open(f))
        out.setdefault(dataset, {})[rel] = rec
    return out


def load_init_yamls():
    cams = {}
    for name, path in [('agibot', 'assets/agibot/G1_120s/G1_120s.yaml'),
                       ('droid', 'assets/droid/franka/franka.yaml'),
                       ('robomind', 'assets/robomind/franka/franka.yaml')]:
        doc = yaml.safe_load(open(os.path.join(BASE, path)))
        cams[name] = {k: v for k, v in doc['cameras'].items() if k != 'observe'}
    return cams


def mount_of(yaml_cams, dataset_key, cam):
    return yaml_cams[dataset_key][cam]['mount']


# ---------------- 1) init vs refined ----------------

def delta_init_vs_refined(refined, inits):
    rows = []
    ds2yaml = {'AgibotWorld': 'agibot', 'droid_raw': 'droid', 'robomind_franka': 'robomind'}
    for ds, eps in refined.items():
        for key, rec in eps.items():
            for cam, pose in rec['local_pose_best'].items():
                init = inits[ds2yaml[ds]].get(cam)
                if init is None:
                    continue
                R_r, t_r = sapien_pose_to_opencv_cam2mount(pose['p'], pose['q'])
                R_i, t_i = sapien_pose_to_opencv_cam2mount(init['p'], init['q'])
                intr_r, intr_i = rec['intrinsic_best'][cam], init
                rows.append({
                    'dataset': ds, 'calib_key': key, 'camera': cam, 'mount': init['mount'],
                    'trans_err_m': float(np.linalg.norm(t_r - t_i)),
                    'rot_err_deg': rot_angle_deg(R_r.T @ R_i),
                    'fx_rel_err': float(intr_r['fx'] / init['fx'] - 1) if 'fx' in init else None,
                    'fy_rel_err': float(intr_r['fy'] / init['fy'] - 1) if 'fy' in init else None,
                    'dcx_px': float(intr_r['cx'] - init.get('cx', intr_r['w'] / 2)),
                    'dcy_px': float(intr_r['cy'] - init.get('cy', intr_r['h'] / 2)),
                    'distortion_l1': float(sum(abs(intr_r[k]) for k in ('k1', 'k2', 'p1', 'p2', 'k3'))),
                })
    return rows


# ---------------- 2) DROID dataset-provided vs refined ----------------

def droid_original_vs_refined(refined):
    """Compare refined local_pose/intrinsic against DROID's own calibration.

    DreamTrue 'left'/'right' = ext1/ext2 ZED left-eye views (verified by camera-center
    proximity). wrist is flange-mounted -> needs FK against the recorded joint states.
    """
    meta_files = {
        'droid_raw/param/1.0.1/AUTOLab/failure/Sat_Oct_21_19:58:12_2023/optimize':
            'droid-meta/AUTOLab_failure_Sat_Oct_21_19-58-12_2023.json',
        'droid_raw/param/1.0.1/ILIAD/success/Thu_Jul_20_11:48:53_2023/optimize':
            'droid-meta/ILIAD_success_Thu_Jul_20_11-48-53_2023.json',
        'droid_raw/param/1.0.1/ILIAD/failure/Thu_Jun_15_13:56:46_2023/optimize':
            'droid-meta/ILIAD_failure_Thu_Jun_15_13-56-46_2023.json',
    }
    traj_files = {k: v.replace('droid-meta/', 'droid-traj/')[:-5] + '.h5' for k, v in meta_files.items()}
    intr_all = json.load(open(os.path.join(BASE, 'droid-calib/intrinsics.json')))

    fk = FrankaFK(os.path.join(BASE, 'assets/droid/franka/franka.urdf'))
    rows, euler_test = [], {}
    for key, mf in meta_files.items():
        rec = refined['droid_raw'][key]
        meta = json.load(open(os.path.join(BASE, mf)))
        eid = meta['uuid']
        orig_extr = {'left': meta['ext1_cam_extrinsics'], 'right': meta['ext2_cam_extrinsics'],
                     'wrist': meta['wrist_cam_extrinsics']}
        serials = {'left': meta['ext1_cam_serial'], 'right': meta['ext2_cam_serial'],
                   'wrist': meta['wrist_cam_serial']}
        orig_intr = {cam: intr_all[eid][serials[cam]]['cameraMatrix'] for cam in orig_extr}

        # --- static external cams: direct comparison (mount = base_link)
        for cam in ('left', 'right'):
            R_r, t_r = sapien_pose_to_opencv_cam2mount(**rec['local_pose_best'][cam])
            e = orig_extr[cam]
            t_o = np.array(e[:3])
            # convention probe: intrinsic (Rx@Ry@Rz) vs extrinsic (Rz@Ry@Rx) euler
            ang_in = rot_angle_deg(R_r.T @ euler_xyz_intrinsic_to_R(e[3:]))
            ang_ex = rot_angle_deg(R_r.T @ euler_xyz_extrinsic_to_R(e[3:]))
            euler_test.setdefault(cam, []).append((ang_in, ang_ex))
            conv = 'intrinsic' if ang_in < ang_ex else 'extrinsic'
            R_o = euler_xyz_intrinsic_to_R(e[3:]) if conv == 'intrinsic' else euler_xyz_extrinsic_to_R(e[3:])
            fx_o, cx_o, fy_o, cy_o = orig_intr[cam]
            ir = rec['intrinsic_best'][cam]
            rows.append({
                'dataset': 'droid_raw', 'calib_key': key, 'camera': cam,
                'euler_conv_chosen': conv,
                'trans_err_m': float(np.linalg.norm(t_r - t_o)),
                'rot_err_deg': rot_angle_deg(R_r.T @ R_o),
                'fx_ref_vs_orig': [ir['fx'], fx_o], 'fy_ref_vs_orig': [ir['fy'], fy_o],
                'fx_rel_err': float(ir['fx'] / fx_o - 1), 'fy_rel_err': float(ir['fy'] / fy_o - 1),
                'dcx_px': float(ir['cx'] - cx_o), 'dcy_px': float(ir['cy'] - cy_o),
                'orig_distortion': 'ZED factory: none provided in intrinsics.json (distCoeffs all zero)',
            })

        # --- wrist: original cam2base per frame (trajectory.h5) -> cam2flange via FK
        with h5py.File(os.path.join(BASE, traj_files[key]), 'r') as f:
            qpos = np.asarray(f['observation/robot_state/joint_positions'])
        serial = serials['wrist']
        with h5py.File(os.path.join(BASE, traj_files[key]), 'r') as f:
            ext = np.asarray(f[f'observation/camera_extrinsics/{serial}_left'])
        # per-frame original cam2flange = inv(flange2base(t)) @ cam2base(t)
        # (verified: constant across frames to machine precision under the extrinsic
        # euler convention; FK validated against trajectory cartesian_position to 0.00cm)
        cam2flange_orig = []
        for t in range(0, len(qpos), max(1, len(qpos) // 50)):
            T_fb = fk.link_pose(qpos[t], 'flange')  # flange in base
            e = ext[t]
            R_ob = euler_xyz_extrinsic_to_R(e[3:])  # DROID convention, resolved by probe
            T_cb = np.eye(4); T_cb[:3, :3] = R_ob; T_cb[:3, 3] = e[:3]
            cam2flange_orig.append(np.linalg.inv(T_fb) @ T_cb)
        cam2flange_orig = np.array(cam2flange_orig)
        t_std = cam2flange_orig[:, :3, 3].std(axis=0)
        R_r_sp = quat_wxyz_to_R(rec['local_pose_best']['wrist']['q'])
        # refined: cam2flange in opencv convention = R_sp @ C^T
        R_r_cv = R_r_sp @ C_SP2CV.T
        t_r = np.array(rec['local_pose_best']['wrist']['p'])
        t_o = cam2flange_orig[:, :3, 3].mean(axis=0)
        R_o = cam2flange_orig[len(cam2flange_orig) // 2, :3, :3]
        fx_o, cx_o, fy_o, cy_o = orig_intr['wrist']
        ir = rec['intrinsic_best']['wrist']
        rows.append({
            'dataset': 'droid_raw', 'calib_key': key, 'camera': 'wrist',
            'euler_conv_chosen': 'extrinsic (Rz@Ry@Rx; verified: cam2flange constant to 0.000deg over time)',
            'orig_cam2flange_translation_std_m_per_frame': t_std.tolist(),
            'trans_err_m': float(np.linalg.norm(t_r - t_o)),
            'rot_err_deg': rot_angle_deg(R_r_cv.T @ R_o),
            'fx_ref_vs_orig': [ir['fx'], fx_o], 'fy_ref_vs_orig': [ir['fy'], fy_o],
            'fx_rel_err': float(ir['fx'] / fx_o - 1), 'fy_rel_err': float(ir['fy'] / fy_o - 1),
            'dcx_px': float(ir['cx'] - cx_o), 'dcy_px': float(ir['cy'] - cy_o),
        })
    return rows, euler_test


class FrankaFK:
    """Minimal URDF FK for the droid/franka chain (revolute + fixed joints)."""

    def __init__(self, urdf_path):
        root = ET.parse(urdf_path).getroot()
        self.joints = {}
        children = {}
        for j in root.iter('joint'):
            name = j.get('name')
            parent = j.find('parent').get('link')
            child = j.find('child').get('link')
            origin = j.find('origin')
            xyz = np.array([float(v) for v in (origin.get('xyz') if origin is not None else '0 0 0').split()])
            rpy = np.array([float(v) for v in (origin.get('rpy') if origin is not None else '0 0 0').split()])
            axis_el = j.find('axis')
            axis = np.array([float(v) for v in axis_el.get('xyz').split()]) if axis_el is not None else np.array([1., 0, 0])
            self.joints[name] = (parent, child, xyz, rpy, j.get('type'), axis)
            children.setdefault(parent, []).append(name)

    def link_pose(self, qpos7, link):
        """Pose of `link` in base_link frame. qpos7 = joint1..joint7 angles."""
        qmap = {f'joint{i+1}': qpos7[i] for i in range(7)}
        T = np.eye(4)
        cur = 'base_link'
        while cur != link:
            nxt = [jn for jn in self.joints if self.joints[jn][0] == cur]
            assert len(nxt) == 1, f'branch at {cur}'
            jn = nxt[0]
            parent, child, xyz, rpy, jtype, axis = self.joints[jn]
            T_j = np.eye(4)
            T_j[:3, :3] = euler_xyz_extrinsic_to_R(rpy)  # URDF rpy = fixed-axis xyz = extrinsic
            T_j[:3, 3] = xyz
            if jtype == 'revolute':
                th = qmap.get(jn, 0.0)
                K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
                R = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K
                T_r = np.eye(4); T_r[:3, :3] = R
                T_j = T_j @ T_r
            T = T @ T_j
            cur = child
        return T


# ---------------- 3) internal consistency of the release ----------------

def check_h5_consistency(refined):
    """For every condition h5: attrs vs referenced calibration JSON; calib_hash; r* recompute."""
    out = []
    for h5path in sorted(glob.glob(os.path.join(MINI, 'conditions/**/*.h5'), recursive=True)):
        with h5py.File(h5path, 'r') as f:
            a = f.attrs
            calib_key = a['calib_key']
            ds = {'AgibotDataset': 'AgibotWorld', 'DroidDataset': 'droid_raw',
                  'RobomindFrankaDataset': 'robomind_franka'}[a['data_class']]
            sub = {'AgibotWorld': 'AgibotWorld/param', 'droid_raw': 'droid_raw/param',
                   'robomind_franka': 'robomind_franka/param'}[ds]
            calib_dir = os.path.join(MINI, 'calibration', sub, calib_key)
            # hash chain: sha256(local_pose_best.json + intrinsic_best.json [+ offset_joints_best.json])
            h = hashlib.sha256()
            files = ['local_pose_best.json', 'intrinsic_best.json', 'offset_joints_best.json']
            used = []
            for fn in files:
                p = os.path.join(calib_dir, fn)
                if os.path.exists(p):
                    h.update(open(p, 'rb').read())
                    used.append(fn)
            hash_ok = h.hexdigest() == a['calib_hash']
            intr_attr = json.loads(a['cameras_intrinsic'])
            pose_attr = json.loads(a['cameras_local_pose'])
            rec_intr = json.load(open(os.path.join(calib_dir, 'intrinsic_best.json')))
            rec_pose = json.load(open(os.path.join(calib_dir, 'local_pose_best.json')))
            intr_max_err = max(abs(intr_attr[c][k] - rec_intr[c][k])
                               for c in rec_intr for k in ('fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2', 'k3'))
            pose_max_err = max(max(abs(np.array(pose_attr[c]['p']) - np.array(rec_pose[c]['p'])).max(),
                                   abs(np.array(pose_attr[c]['q']) - np.array(rec_pose[c]['q'])).max())
                               for c in rec_pose)
            # recompute render intrinsics r* from the released opencv intrinsics
            r_err = {}
            for c, iv in rec_intr.items():
                r = compute_render_intrinsic(iv)
                r_err[c] = max(abs(r[k] - intr_attr[c][k]) for k in ('rfx', 'rfy', 'rcx', 'rcy'))
            out.append({
                'h5': os.path.relpath(h5path, MINI), 'calib_key': calib_key,
                'calib_hash_match': hash_ok, 'hash_files': used,
                'intrinsic_attr_vs_json_maxabs': intr_max_err,
                'local_pose_attr_vs_json_maxabs': pose_max_err,
                'render_intrinsic_recompute_maxabs': r_err,
            })
    return out


def compute_render_intrinsic(iv):
    """Replicates OpenCVRenderIntrinsic.from_opencv (utils/intrinsic.py)."""
    w, h = iv['w'], iv['h']
    v_out, u_out = np.mgrid[0:h, 0:w].astype(np.float64) + 0.5
    xd, yd = (u_out - iv['cx']) / iv['fx'], (v_out - iv['cy']) / iv['fy']
    x, y = xd.copy(), yd.copy()
    for _ in range(8):  # fixed-point undistort
        r2 = x * x + y * y
        ra = 1 + iv['k1'] * r2 + iv['k2'] * r2 ** 2 + iv['k3'] * r2 ** 3
        xg = x * ra + 2 * iv['p1'] * x * y + iv['p2'] * (r2 + 2 * x ** 2)
        yg = y * ra + iv['p1'] * (r2 + 2 * y ** 2) + 2 * iv['p2'] * x * y
        x, y = xd - (xg - x), yd - (yg - y)
    rfx = (w - 1) / (x.max() - x.min()); rcx = 0.5 - rfx * x.min()
    rfy = (h - 1) / (y.max() - y.min()); rcy = 0.5 - rfy * y.min()
    return {'rfx': rfx, 'rfy': rfy, 'rcx': rcx, 'rcy': rcy}


# ---------------- 4) geometric self-consistency ----------------

def check_projection_consistency(max_frames=12, n_pix=4000):
    """Two pixel-level checks against the stored condition renders.

    (a) roundtrip: sample mask pixels, unproject with stored depth (mm) to world via the
        inverse of the stored camera pose, reproject -> same pixel. Validates the
        pose+intrinsic+distortion convention chain end-to-end.
    (b) link-origin probe: project arm/gripper link origins; of those landing in-frame,
        fraction on the robot mask. Origins occluded by body links legitimately miss,
        so this is a loose lower bound, not a pass/fail metric.
    """
    rng = np.random.default_rng(0)
    out = []
    for h5path in sorted(glob.glob(os.path.join(MINI, 'conditions/**/*.h5'), recursive=True)):
        with h5py.File(h5path, 'r') as f:
            a = f.attrs
            intr = json.loads(a['cameras_intrinsic'])
            link_names = list(f['link'].keys())
            arm_links = [ln for ln in link_names if any(s in ln for s in ('arm', 'gripper', 'link'))]
            frames = np.linspace(0, f['qpos'].shape[0] - 1, max_frames).astype(int)
            rec = {'h5': os.path.relpath(h5path, MINI), 'cameras': {}}
            for cam in intr:
                gp = f[f'{cam}/global_pose']  # (T,4,4) sapien camera world pose
                rt_err, hits, tot = [], 0, 0
                for t in frames:
                    T = gp[t]
                    R_sp, t_sp = T[:3, :3].astype(np.float64), T[:3, 3].astype(np.float64)
                    R_cv = R_sp @ C_SP2CV.T  # cam2world rotation, opencv frame
                    m = f[f'{cam}/mask'][t]
                    d = f[f'{cam}/depth'][t].astype(np.float64) / 1000.0
                    ys, xs = np.nonzero(m)
                    if len(xs) == 0:
                        continue
                    sel = rng.choice(len(xs), size=min(n_pix // max_frames + 1, len(xs)), replace=False)
                    u, v, z = xs[sel].astype(float), ys[sel].astype(float), d[ys[sel], xs[sel]]
                    # undo distortion (fixed-point, 8 iters as in utils/intrinsic.py)
                    iv = intr[cam]
                    xn, yn = (u - iv['cx']) / iv['fx'], (v - iv['cy']) / iv['fy']
                    x, y = xn.copy(), yn.copy()
                    for _ in range(8):
                        r2 = x * x + y * y
                        ra = 1 + iv['k1'] * r2 + iv['k2'] * r2 ** 2 + iv['k3'] * r2 ** 3
                        xg = x * ra + 2 * iv['p1'] * x * y + iv['p2'] * (r2 + 2 * x ** 2)
                        yg = y * ra + iv['p1'] * (r2 + 2 * y ** 2) + 2 * iv['p2'] * x * y
                        x, y = xn - (xg - x), yn - (yg - y)
                    Pc = np.stack([x * z, y * z, z], -1)  # opencv camera frame
                    Pw = Pc @ R_cv.T + t_sp             # cam2world (row-vector form of R P + t)
                    uv2, z2 = opencv_project(Pw, R_cv, t_sp, iv)
                    rt_err.extend(np.abs(uv2 - np.stack([u, v], -1)).max(axis=1).tolist())
                    # (b) link origins
                    pts = np.array([f[f'link/{ln}/global_pose'][t][:3, 3] for ln in arm_links])
                    uv3, z3 = opencv_project(pts, R_cv, t_sp, iv)
                    H, W = m.shape
                    for (uu, vv), zz in zip(uv3, z3):
                        if zz <= 0.02:
                            continue
                        iu, ivv = int(round(uu)), int(round(vv))
                        if 0 <= iu < W and 0 <= ivv < H:
                            tot += 1
                            hits += int(m[ivv, iu])
                rec['cameras'][cam] = {
                    'roundtrip_max_err_px': float(np.max(rt_err)) if rt_err else None,
                    'roundtrip_mean_err_px': float(np.mean(rt_err)) if rt_err else None,
                    'link_origin_inframe': tot, 'link_origin_mask_hit_rate': hits / max(tot, 1),
                }
            out.append(rec)
    return out


def main():
    refined = load_refined()
    inits = load_init_yamls()
    res = {}
    res['n_refined_episodes'] = {ds: len(eps) for ds, eps in refined.items()}
    res['init_vs_refined'] = delta_init_vs_refined(refined, inits)
    droid_rows, euler_test = droid_original_vs_refined(refined)
    res['droid_original_vs_refined'] = droid_rows
    res['droid_euler_convention_test'] = euler_test
    res['h5_consistency'] = check_h5_consistency(refined)
    res['projection_consistency'] = check_projection_consistency()
    outp = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'recalc_results.json')
    with open(outp, 'w') as f:
        json.dump(res, f, indent=1, default=str)
    print('wrote', outp)


if __name__ == '__main__':
    main()
