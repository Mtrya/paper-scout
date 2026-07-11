"""
PhysisForcing mechanism probe (pure NumPy).

Builds a tiny synthetic manipulation clip, extracts a physics-informative region
mask from dense trajectories + depth, and implements the two PhysisForcing
auxiliary losses exactly as described in the paper:

* pixel-level trajectory alignment  L^phy_pix
* semantic-level relational alignment L^phy_sem

We then run a small feature-learning experiment: a student feature map is
initialized randomly and optimized with different loss combinations.  The
diagnostic shows that adding the physics losses pulls the student toward
physically consistent trajectories and relational structure even when the
plain reconstruction target is noisy / incomplete.

No giant diffusion model is required; the point is to make the forcing
mechanism concrete and inspectable.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(0)

# -----------------------------------------------------------------------------
# Synthetic world: a foreground square moves right; background is static noise.
# -----------------------------------------------------------------------------
FRAMES, HEI, WID, FCH = 8, 16, 16, 8
SQ = 4                         # square size

def make_world():
    video = np.zeros((FRAMES, 3, HEI, WID), dtype=np.float32)
    depth = np.ones((HEI, WID), dtype=np.float32)
    traj = {}                    # query_id -> (FRAMES, 2) float coordinates [x, y]

    # background gradient + static checkerboard
    yg, xg = np.mgrid[0:HEI, 0:WID]
    bg = (np.sin(xg * 0.8) * np.cos(yg * 0.8))[None, ...].astype(np.float32)
    for t in range(FRAMES):
        video[t] = np.tile(bg, (3, 1, 1)) * 0.3

    # foreground square moving right
    cx0, cy0 = 3, 7
    for t in range(FRAMES):
        cx = cx0 + t
        cy = cy0
        video[t, 0, cy:cy+SQ, cx:cx+SQ] = 0.9
        video[t, 1, cy:cy+SQ, cx:cx+SQ] = 0.2
        video[t, 2, cy:cy+SQ, cx:cx+SQ] = 0.2
        depth[cy:cy+SQ, cx:cx+SQ] = 0.5

    # query grid of points
    qy, qx = np.mgrid[0:HEI:2, 0:WID:2]
    qpts = np.stack([qx.ravel(), qy.ravel()], axis=1).astype(np.float32)  # (N,2) x,y
    for idx, (x0, y0) in enumerate(qpts):
        # if inside square, move with it; otherwise stay
        if SQ - 1 >= x0 - cx0 >= 0 and SQ - 1 >= y0 - cy0 >= 0:
            traj[idx] = np.array([[cx0 + t + (x0 - cx0), cy0 + (y0 - cy0)] for t in range(FRAMES)], dtype=np.float32)
        else:
            traj[idx] = np.tile([x0, y0], (FRAMES, 1))
    return video, depth, qpts, traj

def physics_mask(qpts, traj, depth, eps=1e-6):
    """Eqs. (1)-(4) in the paper, simplified to integer pixel coordinates."""
    N = qpts.shape[0]
    a = np.zeros(N, dtype=np.float32)
    r = np.zeros(N, dtype=np.float32)
    for i in range(N):
        p = traj[i]  # (FRAMES,2)
        a[i] = np.linalg.norm(p[1:] - p[:-1], axis=1).sum()
        x0, y0 = int(np.round(p[0, 0])), int(np.round(p[0, 1]))
        y0 = np.clip(y0, 0, HEI-1); x0 = np.clip(x0, 0, WID-1)
        r[i] = 1.0 / (depth[y0, x0] + eps)
    q = a * r
    thresh = q.mean()
    keep = q >= thresh
    M = np.zeros((FRAMES, HEI, WID), dtype=np.float32)
    for i in range(N):
        if not keep[i]:
            continue
        for t in range(FRAMES):
            x, y = int(np.round(traj[i][t, 0])), int(np.round(traj[i][t, 1]))
            x = np.clip(x, 0, WID-1); y = np.clip(y, 0, HEI-1)
            M[t, y, x] = 1.0
    return M, keep

# -----------------------------------------------------------------------------
# Teacher feature map: encodes object identity + absolute coordinates.
# -----------------------------------------------------------------------------
def make_teacher_features(video):
    """Teacher F^u used for the semantic relation target."""
    yg, xg = np.mgrid[0:HEI, 0:WID]
    yg = yg.astype(np.float32) / HEI
    xg = xg.astype(np.float32) / WID
    F = np.empty((FRAMES, HEI, WID, FCH), dtype=np.float32)
    obj = (video[:, 0] > 0.5).astype(np.float32)  # (FRAMES,HEI,WID)
    for t in range(FRAMES):
        F[t, :, :, 0] = obj[t] * 1.0
        F[t, :, :, 1] = (1.0 - obj[t]) * 1.0
        F[t, :, :, 2] = np.sin(np.pi * yg)
        F[t, :, :, 3] = np.cos(np.pi * yg)
        F[t, :, :, 4] = np.sin(np.pi * xg)
        F[t, :, :, 5] = np.cos(np.pi * xg)
        F[t, :, :, 6] = yg
        F[t, :, :, 7] = xg
    return F

def make_input_features(video):
    """Input features for the tiny linear student (RGB + coords + time + noise)."""
    yg, xg = np.mgrid[0:HEI, 0:WID]
    yg = yg.astype(np.float32) / HEI
    xg = xg.astype(np.float32) / WID
    x = np.empty((FRAMES, HEI, WID, 7), dtype=np.float32)
    for t in range(FRAMES):
        x[t, :, :, :3] = video[t].transpose(1, 2, 0)
        x[t, :, :, 3] = yg
        x[t, :, :, 4] = xg
        x[t, :, :, 5] = t / FRAMES
        x[t, :, :, 6] = 1.0  # bias
    return x

# -----------------------------------------------------------------------------
# Losses and gradients
# -----------------------------------------------------------------------------
def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

def pixel_trajectory_loss(F, qpts, traj, M_idx):
    """
    L^phy_pix = (1/|M|) sum_i M_i ||P_pred_i - P_gt_i||^2
    Returns loss and gradient wrt F.
    """
    sel = np.where(M_idx)[0]
    K = max(sel.size, 1)
    if K == 0:
        return 0.0, np.zeros_like(F)

    coords = np.stack([np.tile(np.arange(WID, dtype=np.float32)[None, :], (HEI, 1)).ravel(),
                       np.tile(np.arange(HEI, dtype=np.float32)[:, None], (1, WID)).ravel()], axis=1)  # (HW,2)
    coordsT = coords.T  # (2,HW)
    HW = HEI * WID
    loss = 0.0
    grad = np.zeros_like(F)
    sqrtC = np.sqrt(FCH)

    for i in sel:
        y0, x0 = int(np.round(qpts[i, 1])), int(np.round(qpts[i, 0]))
        y0 = np.clip(y0, 0, HEI-1); x0 = np.clip(x0, 0, WID-1)
        q = F[0, y0, x0]  # (FCH,)
        for t in range(FRAMES):
            keys = F[t].reshape(HW, FCH)             # (HW,FCH)
            s = (keys @ q) / sqrtC                 # (HW,)
            w = softmax(s)                         # (HW,)
            pred = w @ coords                      # (2,)
            diff = pred - traj[i][t]               # (2,)
            loss += np.dot(diff, diff)

            # gradient wrt keys at time t
            # dL/ds = 2 * diff · (w * (coords - pred))
            # diff (2,), (w*(coords-pred)) (HW,2). We need per-a: diff·(w_a*(coord_a-pred)) -> scalar per a.
            weighted = w[:, None] * (coords - pred[None, :])  # (HW,2)
            dL_ds_per_a = diff[0] * weighted[:, 0] + diff[1] * weighted[:, 1]  # (HW,)
            grad_keys = (dL_ds_per_a[:, None] * q[None, :]) / sqrtC  # (HW,FCH)
            grad[t] += grad_keys.reshape(HEI, WID, FCH)

            # gradient wrt query at frame 0
            dL_dq = np.zeros(FCH, dtype=np.float32)
            for a in range(HW):
                dL_dq += dL_ds_per_a[a] * keys[a] / sqrtC
            grad[0, y0, x0] += dL_dq

    return loss / K, grad / K

def semantic_relation_loss(F, F_teacher, M_token):
    """
    L^phy_sem = (1/K^2) sum_ij |R_hat_ij - R_ij|
    M_token is boolean (FRAMES,HEI,WID).
    """
    idx = np.where(M_token.ravel())[0]
    K = idx.size
    if K == 0:
        return 0.0, np.zeros_like(F)

    # gather selected tokens
    Fs = F.reshape(-1, FCH)[idx]           # (K,FCH)
    Ts = F_teacher.reshape(-1, FCH)[idx]   # (K,FCH)

    norm_F = np.linalg.norm(Fs, axis=1, keepdims=True) + 1e-6
    norm_T = np.linalg.norm(Ts, axis=1, keepdims=True) + 1e-6
    f = Fs / norm_F
    t = Ts / norm_T

    Rf = f @ f.T
    Rt = t @ t.T
    diff = Rf - Rt
    loss = np.abs(diff).mean()

    sign = np.sign(diff)                 # (K,K)
    grad_f = np.zeros_like(f)            # (K,FCH)
    for i in range(K):
        # sum over j of sign[i,j] * (f[j] - f[i] * Rf[i,j])
        # plus symmetric term sign[j,i] same value
        g = np.zeros(FCH, dtype=np.float32)
        for j in range(K):
            g += sign[i, j] * (f[j] - f[i] * Rf[i, j])
            g += sign[j, i] * (f[j] - f[i] * Rf[j, i])
        grad_f[i] = g / np.float32(K * K)

    # backprop through normalization: f_i = Fs_i / norm_i
    grad_Fs = (grad_f - f * np.sum(grad_f * f, axis=1, keepdims=True)) / norm_F

    grad = np.zeros_like(F)
    grad.reshape(-1, FCH)[idx] = grad_Fs
    return loss, grad

# -----------------------------------------------------------------------------
# Tiny linear student and training helpers
# -----------------------------------------------------------------------------
def predict_F(Wmat, x):
    # x: (FRAMES,HEI,WID,D) -> (N,D); WID: (FCH,D)
    N, D = x.reshape(-1, x.shape[-1]).shape
    return (x.reshape(-1, D) @ Wmat.T).reshape(FRAMES, HEI, WID, FCH)

def train_variant(x, F_teacher, qpts, traj, M_idx, M_token, F_mse_target,
                  use_mse, use_pix, use_sem, steps=400, lr=0.15):
    D = x.shape[-1]
    Wmat = np.random.randn(FCH, D).astype(np.float32) * 0.05
    vel = np.zeros_like(Wmat)
    mu = 0.9

    records = []
    for step in range(steps):
        F = predict_F(Wmat, x)
        grad = np.zeros_like(Wmat)
        loss = 0.0
        info = {}

        if use_mse:
            d = F - F_mse_target
            loss_mse = (d * d).mean()
            grad_mse = (np.float32(2.0) / np.float32(d.size)) * (x.reshape(-1, D).T @ d.reshape(-1, FCH)).T
            loss += float(loss_mse)
            grad += grad_mse.astype(np.float32)
            info['mse'] = float(loss_mse)

        if use_pix:
            loss_pix, grad_F_pix = pixel_trajectory_loss(F, qpts, traj, M_idx)
            loss += float(loss_pix)
            grad += (x.reshape(-1, D).T @ grad_F_pix.reshape(-1, FCH)).T.astype(np.float32)
            info['pix'] = float(loss_pix)

        if use_sem:
            loss_sem, grad_F_sem = semantic_relation_loss(F, F_teacher, M_token)
            loss += float(loss_sem)
            grad += (x.reshape(-1, D).T @ grad_F_sem.reshape(-1, FCH)).T.astype(np.float32)
            info['sem'] = float(loss_sem)

        # diagnostic metrics (against clean teacher)
        mse_clean = ((F - F_teacher)**2).mean()
        pix_err, _ = pixel_trajectory_loss(F, qpts, traj, M_idx)
        sem_err, _ = semantic_relation_loss(F, F_teacher, M_token)
        info.update({'total': float(loss), 'mse_clean': float(mse_clean),
                     'pix_err': float(pix_err), 'sem_err': float(sem_err)})
        records.append(info)

        # SGD with momentum
        vel = mu * vel - lr * grad
        Wmat += vel

    return F, records

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    out_dir = "runs/2026-07-11-gigaworld-vla-corrector-physis/physisforcing-2606.28128/code/probe_outputs"
    os.makedirs(out_dir, exist_ok=True)

    video, depth, qpts, traj = make_world()
    M, M_idx = physics_mask(qpts, traj, depth)
    F_teacher = make_teacher_features(video)
    x = make_input_features(video)

    # Noisy / incomplete reconstruction target for the plain loss
    F_mse_target = F_teacher + np.random.randn(*F_teacher.shape).astype(np.float32) * 0.25

    # token mask for semantic loss: any frame where physics mask is active
    M_token = M.sum(axis=0) > 0

    variants = [
        ('MSE only', True, False, False),
        ('MSE + pixel', True, True, False),
        ('MSE + semantic', True, False, True),
        ('MSE + pixel + semantic', True, True, True),
    ]

    results = {}
    for name, um, up, us in variants:
        print(f"\n=== training {name} ===")
        F, rec = train_variant(x, F_teacher, qpts, traj, M_idx, M_token, F_mse_target,
                               use_mse=um, use_pix=up, use_sem=us, steps=400, lr=0.15)
        results[name] = rec
        print(f"final mse_clean={rec[-1]['mse_clean']:.4f}  pix_err={rec[-1]['pix_err']:.4f}  "
              f"sem_err={rec[-1]['sem_err']:.4f}")

    # -------------------------------------------------------------------------
    # Save numeric results
    # -------------------------------------------------------------------------
    summary = {n: {k: [r[k] for r in rec] for k in rec[0]} for n, rec in results.items()}
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # -------------------------------------------------------------------------
    # Figures
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    # first frame + physics mask overlay
    im0 = np.clip(video[0].transpose(1, 2, 0), 0.0, 1.0)
    axes[0].imshow(im0)
    axes[0].imshow(M[0], alpha=0.5, cmap='Reds')
    axes[0].set_title('Physics-informative mask (frame 0)')
    axes[0].axis('off')

    # query trajectories
    ax = axes[1]
    ax.imshow(im0)
    for i in np.where(M_idx)[0]:
        ax.plot(traj[i][:, 0], traj[i][:, 1], lw=1.5)
    ax.set_title('Selected reference trajectories')
    ax.axis('off')

    # loss curves
    ax = axes[2]
    for name, rec in results.items():
        ax.plot([r['total'] for r in rec], label=name)
    ax.set_yscale('log')
    ax.set_xlabel('step')
    ax.set_ylabel('training objective')
    ax.set_title('Optimization objective')
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'mask_and_training.png'), dpi=150)
    plt.close(fig)

    # diagnostic metric curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for name, rec in results.items():
        axes[0].plot([r['mse_clean'] for r in rec], label=name)
        axes[1].plot([r['pix_err'] for r in rec], label=name)
        axes[2].plot([r['sem_err'] for r in rec], label=name)
    axes[0].set_title('MSE to clean teacher')
    axes[1].set_title('Trajectory error L^phy_pix')
    axes[2].set_title('Relation error L^phy_sem')
    for ax in axes:
        ax.set_yscale('log')
        ax.set_xlabel('step')
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'metric_curves.png'), dpi=150)
    plt.close(fig)

    print(f"\nOutputs written to {out_dir}")

if __name__ == '__main__':
    main()
