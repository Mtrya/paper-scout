"""B2a/B2b follow-up experiments on blob world (frame-representation control).

Question: experiment B showed the base-frame head fails under view shift while the
camera-frame head is flat-robust. Is that a difference of *representation geometry* or
just of *information availability* -- i.e. does giving the base head the camera pose as
input close the gap?

B2a (pose conditioning, single embodiment E1 = the original camera mount):
  - T1    : train at yaw=0 only (exp-B protocol). NOTE: pose is CONSTANT in training ->
            it carries no learning signal (degenerate control, reported as such).
  - T2    : train at yaws {-30,0,30} (exp-B multi-view protocol) -> pose varies.
  - Scan  : T2 protocol at {1k, 4k} episodes, base_pose vs cam (4.5k main T2 adds a point).
B2b (cross-embodiment pooling, hand-eye mount as the embodiment):
  - E1 (h=0.6, d=2.5) = original mount; E2 (h=1.0, d=2.2); E3 (h=0.8, d=2.35, unseen).
  - 50/50 mixed E1+E2 single-view (yaw=0) training, conditions:
      cam (no label), base_pose (pose in), base_pose_label (pose + one-hot emb),
      base_ref (neither -- reference floor).
  - Eval on the 19-yaw grid x {E1, E2, E3} with ground-truth extrinsics at execution.

Camera-frame targets are R_cb(yaw, emb) @ delta_base, an isometry per sample, so MSE
magnitudes are identical across heads; only the output frame (and conditioning) differs.
"""
from __future__ import annotations

import numpy as np
import torch

import blob_world as bw
from models_pose import PolicyX

# hand-eye mounts: (height, dist) of the camera rig above / away from the base origin
MOUNTS = {
    'E1': {'height': 0.6, 'dist': 2.5},   # = original blob_world camera
    'E2': {'height': 1.0, 'dist': 2.2},   # higher, closer
    'E3': {'height': 0.8, 'dist': 2.35},  # unseen midpoint (interpolation target)
}

POS_T_SCALE = 2.5  # translation scaled by 1/this to O(1) in the pose vector


def cam_extrinsics(yaw: float, mount: dict) -> tuple[np.ndarray, np.ndarray]:
    """(R_bc, t_b) for a mount with canonical pose rotated around base z by yaw."""
    t0 = np.array([0.0, -mount['dist'], mount['height']])
    zc0 = -t0 / np.linalg.norm(t0)
    xc0 = np.cross(zc0, np.array([0.0, 0.0, 1.0]))
    xc0 /= np.linalg.norm(xc0)
    yc0 = np.cross(zc0, xc0)
    R0 = np.stack([xc0, yc0, zc0], axis=1)
    return bw.rz(yaw) @ R0, bw.rz(yaw) @ t0


def pose_vec(yaw: float, mount: dict) -> np.ndarray:
    """12-dim conditioning vector: R_bc flattened (9) + t_b scaled (3)."""
    R, t = cam_extrinsics(yaw, mount)
    return np.concatenate([R.ravel(), t / POS_T_SCALE])


def render(ee: np.ndarray, tgt: np.ndarray, yaw: float, mount: dict) -> np.ndarray:
    """Same splat renderer as blob_world.render, but with mount-parameterized extrinsics."""
    R_bc, t_b = cam_extrinsics(yaw, mount)
    img = np.zeros((bw.IMG, bw.IMG, 3), dtype=np.float32)
    uu, vv = np.meshgrid(np.arange(bw.IMG), np.arange(bw.IMG))
    blobs = [(ee, bw.EE_COLOR, bw.BLOB_RADIUS), (tgt, bw.TGT_COLOR, bw.BLOB_RADIUS)]
    for d in bw.DISTRACTORS:
        blobs.append((d, bw.DIST_COLOR, bw.DIST_RADIUS))
    for pos, color, radius in blobs:
        p_c = bw.project(R_bc, t_b, pos)
        if p_c is None:
            continue
        u0 = bw.FOCAL * p_c[0] / p_c[2] + bw.CX
        v0 = bw.FOCAL * p_c[1] / p_c[2] + bw.CY
        if not (-6 <= u0 <= bw.IMG + 6 and -6 <= v0 <= bw.IMG + 6):
            continue
        sig = bw.FOCAL * radius / p_c[2]
        d2 = (uu - u0) ** 2 + (vv - v0) ** 2
        mask = bw.BLOB_PEAK * np.exp(-d2 / (2.0 * sig * sig))
        for c in range(3):
            img[..., c] += mask * color[c]
    return np.clip(img, 0.0, 1.0)


# ---- dataset ----------------------------------------------------------------

class Dataset:
    """Rendered uint8 images + base-frame deltas + per-sample (yaw, emb)."""

    def __init__(self, imgs_u8: np.ndarray, deltas_b: np.ndarray,
                 yaws: np.ndarray, emb: np.ndarray):
        self.imgs = torch.from_numpy(np.ascontiguousarray(imgs_u8))  # N,H,W,C uint8
        self.deltas_b = torch.from_numpy(deltas_b.astype(np.float32))  # N,3
        self.yaws = yaws.astype(np.float64)                            # N,
        self.emb = emb.astype(np.int64)                                # N, in {0,1}
        self.n = len(imgs_u8)
        # per-sample conditioning: pose (N,12) + one-hot emb (N,2)
        poses = np.zeros((self.n, 12), dtype=np.float32)
        for i in range(self.n):
            poses[i] = pose_vec(float(self.yaws[i]), MOUNTS[EMB_NAMES[self.emb[i]]])
        self.poses = torch.from_numpy(poses)
        oh = np.zeros((self.n, 2), dtype=np.float32)
        oh[np.arange(self.n), self.emb] = 1.0
        self.onehot = torch.from_numpy(oh)


EMB_NAMES = ['E1', 'E2']


def build_dataset(n_episodes: int, seed: int, spec_fn) -> Dataset:
    """Generate oracle demos, render every step at the episode's (yaw, mount).

    spec_fn(rng, i) -> (yaw, mount_name); the rng stream is shared in the same order as
    experiment B's build_dataset (spec draw, then gen_episode), so identical spec_fns and
    seeds produce nested datasets (a 1k-episode dataset is a prefix of the 4k one).
    """
    rng = np.random.default_rng(seed)
    states_all, deltas_all, yaws_all, emb_all = [], [], [], []
    for i in range(n_episodes):
        yaw, mname = spec_fn(rng, i)
        states, deltas, tgt = bw.gen_episode(rng)
        for st in states:
            states_all.append(np.concatenate([st, tgt]))
        deltas_all.append(deltas)
        yaws_all.append([yaw] * len(states))
        emb_all.append([0 if mname == 'E1' else 1] * len(states))
    states_all = np.asarray(states_all)
    deltas_all = np.concatenate(deltas_all)
    yaws_all = np.concatenate(yaws_all)
    emb_all = np.concatenate(emb_all)
    n = len(states_all)
    imgs = np.empty((n, bw.IMG, bw.IMG, 3), dtype=np.uint8)
    for k in range(n):
        ee = states_all[k, :3]
        tgt = states_all[k, 3:]
        imgs[k] = (render(ee, tgt, float(yaws_all[k]), MOUNTS[EMB_NAMES[emb_all[k]]]) * 255).astype(np.uint8)
    return Dataset(imgs, deltas_all, yaws_all, emb_all)


# ---- training ---------------------------------------------------------------

EXTRA_DIM = {'base': 0, 'cam': 0, 'base_pose': 12, 'base_pose_label': 14}
TARGET_FRAME = {'base': 'base', 'cam': 'cam',
                'base_pose': 'base', 'base_pose_label': 'base'}


def train(dataset: Dataset, head: str, epochs: int = 15, lr: float = 1e-3,
          batch_size: int = 256, seed: int = 1, out_path: str = '') -> PolicyX:
    """BC-train one head. Same hyperparameters/seed scheme as experiment B."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = PolicyX(3, EXTRA_DIM[head])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    lossf = torch.nn.MSELoss()
    n = dataset.n
    # per-(yaw, emb) camera-to-base rotations for the cam targets
    yaw_vals = np.unique(dataset.yaws)
    yaw_ids = np.searchsorted(yaw_vals, dataset.yaws).astype(np.int64)
    R_cb = np.zeros((len(yaw_vals), 2, 3, 3), dtype=np.float32)
    for yi, yv in enumerate(yaw_vals):
        for e, mname in enumerate(EMB_NAMES):
            R, _ = cam_extrinsics(float(yv), MOUNTS[mname])
            R_cb[yi, e] = R.T
    R_cb = torch.from_numpy(R_cb)
    if TARGET_FRAME[head] == 'cam':
        target_all = torch.einsum('nij,nj->ni', R_cb[yaw_ids, dataset.emb], dataset.deltas_b)
    else:
        target_all = dataset.deltas_b

    print(f'[{head}] train {n} samples x {epochs} epochs', flush=True)
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(n)
        total, cnt = 0.0, 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            x = dataset.imgs[idx].permute(0, 3, 1, 2).float() / 255.0
            tgt = target_all[idx]
            extra = None
            if head == 'base_pose':
                extra = dataset.poses[idx]
            elif head == 'base_pose_label':
                extra = torch.cat([dataset.poses[idx], dataset.onehot[idx]], dim=1)
            pred = model(x, extra) if EXTRA_DIM[head] else model(x)
            loss = lossf(pred, tgt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
            cnt += len(idx)
        if (ep + 1) % 5 == 0 or ep == epochs - 1:
            print(f'  [{head}] ep {ep+1}/{epochs} mse={total/cnt:.6f}', flush=True)
    if out_path:
        torch.save(model.state_dict(), out_path)
        print(f'[{head}] saved {out_path}', flush=True)
    return model


def load_model(path: str, head: str) -> PolicyX:
    m = PolicyX(3, EXTRA_DIM[head])
    m.load_state_dict(torch.load(path, weights_only=True))
    m.eval()
    return m


# ---- closed-loop evaluation --------------------------------------------------

@torch.no_grad()
def rollout(model: PolicyX, head: str, yaw: float, mount_name: str,
            rng: np.random.Generator, emb_label: np.ndarray | None = None) -> tuple[float, bool, int]:
    """One closed-loop episode at (yaw, mount), GT extrinsics at execution.

    Cam head: predicts camera-frame delta, synthesized with the episode's own R_bc.
    Base* heads: predict base-frame delta directly; pose/label are fed as conditioning.
    """
    model.eval()
    ee, tgt = bw.sample_state(rng)
    p = ee.copy()
    mount = MOUNTS[mount_name]
    R_bc, _ = cam_extrinsics(yaw, mount)
    for t in range(bw.MAX_STEPS):
        img = render(p, tgt, yaw, mount)
        x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
        if head == 'cam':
            cmd = model(x).squeeze(0).numpy()
            d_b = R_bc @ cmd
        else:
            if head in ('base_pose', 'base_pose_label'):
                extra = torch.from_numpy(pose_vec(yaw, mount)).float().unsqueeze(0)
                if head == 'base_pose_label':
                    extra = torch.cat([extra,
                                       torch.from_numpy(emb_label.astype(np.float32)).unsqueeze(0)], dim=1)
                cmd = model(x, extra).squeeze(0).numpy()
            else:
                cmd = model(x).squeeze(0).numpy()
            d_b = cmd
        p = bw.step_world(p, d_b, rng)
        if np.linalg.norm(p - tgt) < bw.SUCCESS_THRESH:
            return np.linalg.norm(p - tgt), True, t + 1
    return np.linalg.norm(p - tgt), False, bw.MAX_STEPS


def _stats(bools: np.ndarray, floats: np.ndarray) -> dict:
    n = len(bools)
    p = float(np.mean(bools))
    return {
        'n': n,
        'success_rate': p,
        'success_se': float(np.sqrt(max(p * (1 - p), 1e-12) / n)),
        'mean_final_dist': float(np.mean(floats)),
        'final_dist_se': float(np.std(floats, ddof=1) / np.sqrt(n)),
    }


def eval_yaw_sweep(model: PolicyX, head: str, yaws: np.ndarray, n_eps: int, seed: int,
                   mount_name: str, emb_label: np.ndarray | None = None) -> dict:
    """Closed-loop success/final-dist vs yaw at one mount (same seed scheme as exp B)."""
    out = {}
    for yaw in yaws:
        rng = np.random.default_rng(seed + 10000 + int(round(yaw * 100)))
        su, fd = [], []
        for _ in range(n_eps):
            d, ok, _ = rollout(model, head, float(yaw), mount_name, rng, emb_label)
            su.append(ok)
            fd.append(d)
        out[float(yaw)] = _stats(np.array(su), np.array(fd))
    return out
