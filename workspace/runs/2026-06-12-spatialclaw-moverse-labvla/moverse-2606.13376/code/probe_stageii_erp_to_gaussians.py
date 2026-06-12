"""
Humble reconstruction of MoVerse Stage II: ERP panorama -> 3D Gaussian scaffold.

This script makes the spherical back-projection, latitude-aware scale initialization,
and angular--inverse-depth residual composition from the paper concrete without
requiring the unreleased MoVerse codebase or a differentiable Gaussian rasterizer.

Run: python probe_stageii_erp_to_gaussians.py
Outputs are written to ../assets/stageii_probe/
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'assets', 'stageii_probe')
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Synthetic ERP panorama + depth (a simple textured room as the "world")
# ---------------------------------------------------------------------------
H, W = 256, 512  # ERP resolution, 2:1 equirectangular
# Longitude / latitude arrays (paper Eq. 14)
theta = (np.arange(W) / W - 0.5) * 2.0 * np.pi  # [-pi, pi]
phi = (0.5 - np.arange(H) / H) * np.pi          # [pi/2, -pi/2]
Theta, Phi = np.meshgrid(theta, phi)

# Direction array (paper Eq. 15): x=cos(phi)sin(theta), y=-sin(phi), z=cos(phi)cos(theta)
D = np.stack([
    np.cos(Phi) * np.sin(Theta),
    -np.sin(Phi),
    np.cos(Phi) * np.cos(Theta)
], axis=-1)  # H x W x 3

# Axis-aligned room: x in [-2,2], y in [-1.2,1.2], z in [-2.5,2.5]
box_min = np.array([-2.0, -1.2, -2.5])
box_max = np.array([2.0, 1.2, 2.5])

# Ray-box intersection distance for every ERP ray
with np.errstate(divide='ignore', invalid='ignore'):
    t_min = (box_min[None, None, :] - 0.0) / D
    t_max = (box_max[None, None, :] - 0.0) / D
t1 = np.minimum(t_min, t_max)
t2 = np.maximum(t_min, t_max)
# Origin is inside the box, so the first forward intersection is the smallest
# positive exit distance across the three axes.
depth = np.min(t2, axis=-1)
# For rays that somehow miss the box, fall back to a far depth.
depth = np.where(np.isfinite(depth) & (depth > 0), depth, 5.0)

# Color by which face was hit (normal of the box plane that limits depth)
face_color = np.zeros((H, W, 3))
arg = np.argmax(t1, axis=-1)  # axis index of the entry plane
# Map axis 0->right(red), 1->floor/ceiling(green), 2->front/back(blue)
face_color[arg == 0] = [0.9, 0.4, 0.4]
face_color[arg == 1] = [0.4, 0.85, 0.4]
face_color[arg == 2] = [0.4, 0.5, 0.95]
# Add a longitude/latitude texture overlay to make parallax visible
hsv = np.stack([
    ((Theta / np.pi) % 1.0 + 1.0) / 2.0,
    0.4 * np.ones_like(Phi),
    0.9 * np.ones_like(Phi)
], axis=-1)
face_color = face_color * 0.7 + hsv_to_rgb(hsv) * 0.3
face_color = np.clip(face_color, 0, 1)

# ---------------------------------------------------------------------------
# 2. ERP-aware Gaussian initialization (paper Sec. 2.3.1)
# ---------------------------------------------------------------------------
H_g, W_g = 64, 128
stride_h, stride_w = H // H_g, W // W_g
theta_g = (np.arange(W_g) / W_g - 0.5) * 2.0 * np.pi
phi_g = (0.5 - np.arange(H_g) / H_g) * np.pi
Theta_g, Phi_g = np.meshgrid(theta_g, phi_g)

# Sample depth/colors at grid centers
u = np.clip((np.arange(H_g) * stride_h + stride_h // 2).astype(int), 0, H - 1)
v = np.clip((np.arange(W_g) * stride_w + stride_w // 2).astype(int), 0, W - 1)
D_sample = depth[np.ix_(u, v)]           # H_g x W_g
C_sample = face_color[np.ix_(u, v)]       # H_g x W_g x 3
valid = np.isfinite(D_sample).ravel()

# Directions for grid centers
d_g = np.stack([
    np.cos(Phi_g) * np.sin(Theta_g),
    -np.sin(Phi_g),
    np.cos(Phi_g) * np.cos(Theta_g)
], axis=-1)  # H_g x W_g x 3

# Spherical back-projection: mu = D * d  (paper Eq. 17)
mu_init = D_sample[..., None] * d_g  # H_g x W_g x 3
mu_init = mu_init.reshape(-1, 3)[valid]
colors = C_sample.reshape(-1, 3)[valid]
phis = Phi_g.ravel()[valid]
depths = D_sample.ravel()[valid]

# Latitude-aware scale: s_k \propto D_k * cos(phi_k) (paper Eq. 18)
base_angular = (2.0 * np.pi / W_g) * 1.6
scale_init = depths * np.cos(phis) * base_angular
scale_init = np.maximum(scale_init, 0.005)  # lower bound near poles
opacity = np.full_like(depths, 0.85)

# ---------------------------------------------------------------------------
# 3. Residual prediction in angular--inverse-depth space (paper Sec. 2.3.3)
# ---------------------------------------------------------------------------
# Synthetic small residuals to show the update rule
def softplus_inv(x, eps=1e-6):
    return np.log(np.expm1(x) + eps)

def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)

rng = np.random.default_rng(42)
N = mu_init.shape[0]
# Keep residuals small and bounded so they act as local corrections around the
# depth-initialized scaffold, as described in the paper.
theta_res = rng.uniform(-0.03, 0.03, size=N)
phi_res = rng.uniform(-0.03, 0.03, size=N)
z_res = rng.uniform(-0.02, 0.02, size=N)
lam_xy, lam_z = 1.0, 1.0

theta_g_flat = Theta_g.ravel()[valid]
phi_g_flat = Phi_g.ravel()[valid]
inv_depth_base = softplus_inv(depths)
inv_depth_new = softplus(inv_depth_base + lam_z * z_res) + 1e-6
D_new = 1.0 / np.clip(inv_depth_new, 1e-3, None)
# Also clamp to a plausible scene depth range so a few near-degenerate rays
# do not dominate the visualization.
D_new = np.clip(D_new, 0.05, 10.0)

theta_new = theta_g_flat + lam_xy * theta_res
phi_new = np.clip(phi_g_flat + lam_xy * phi_res, -np.pi / 2 + 1e-3, np.pi / 2 - 1e-3)
mu_res = np.stack([
    np.cos(phi_new) * np.sin(theta_new),
    -np.sin(phi_new),
    np.cos(phi_new) * np.cos(theta_new)
], axis=-1) * D_new[:, None]

# ---------------------------------------------------------------------------
# 4. Tiny point-splatting renderer for a novel pinhole view
# ---------------------------------------------------------------------------
def look_at(eye, target, up):
    z = eye - target
    z = z / np.linalg.norm(z)
    x = np.cross(up, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=0)
    return R

def render(points, cols, scales, eye, target, img_h=240, img_w=320, f=220.0, tag=''):
    R = look_at(eye, target, np.array([0.0, -1.0, 0.0]))
    cam_pts = (R @ (points - eye).T).T  # N x 3; camera looks toward negative z in this space
    visible = cam_pts[:, 2] < -0.05
    cp = cam_pts[visible]
    cc = cols[visible]
    ss = scales[visible]
    order = np.argsort(cp[:, 2])  # front-to-back (most negative z first)
    cp, cc, ss = cp[order], cc[order], ss[order]

    img = np.ones((img_h, img_w, 3)) * 0.05
    for p, c, s in zip(cp, cc, ss):
        x = p[0] / p[2] * f + img_w / 2
        y = p[1] / p[2] * f + img_h / 2
        r = max(1.5, f * s / p[2])
        # disk splat with soft falloff
        yy, xx = np.ogrid[:img_h, :img_w]
        d2 = (xx - x) ** 2 + (yy - y) ** 2
        if r < 60:
            sigma2 = (r / 2.0) ** 2
            alpha = np.exp(-d2 / (2 * sigma2)) * 0.75
            alpha = np.clip(alpha, 0, 1)
            img = alpha[..., None] * c + (1 - alpha[..., None]) * img
    return np.clip(img, 0, 1)

# ---------------------------------------------------------------------------
# 5. Visualize and save
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].imshow(face_color)
axes[0].set_title('Synthetic ERP panorama (Stage I output)')
axes[0].axis('off')
axes[1].imshow(depth, cmap='turbo')
axes[1].set_title('ERP depth map')
axes[1].axis('off')
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, '01_input_erp_depth.png'), dpi=150)
plt.close(fig)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(mu_init[:, 0], mu_init[:, 1], mu_init[:, 2],
           c=colors, s=5, alpha=0.6)
ax.set_title('Gaussian centers from spherical back-projection')
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
ax.set_box_aspect([1, 1, 1])
fig.savefig(os.path.join(OUT_DIR, '02_gaussian_centers_3d.png'), dpi=150)
plt.close(fig)

# Novel view from a displaced camera
eye = np.array([0.35, 0.0, 0.35])
target = np.array([0.0, 0.0, 0.0])
rend_init = render(mu_init, colors, scale_init, eye, target)
rend_res = render(mu_res, colors, scale_init, eye, target)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(rend_init)
axes[0].set_title('Novel view from initialized scaffold')
axes[0].axis('off')
axes[1].imshow(rend_res)
axes[1].set_title('Novel view after angular--inverse-depth residuals')
axes[1].axis('off')
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, '03_novel_view_residual.png'), dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# 6. Numerical sanity check: cos-latitude scaling trend
# ---------------------------------------------------------------------------
cos_phi = np.cos(phis)
print(f"Saved probe artifacts to {OUT_DIR}")
print(f"Mean scale at equator (|phi|<0.2): {scale_init[np.abs(phis) < 0.2].mean():.4f}")
print(f"Mean scale near poles (|phi|>1.2): {scale_init[np.abs(phis) > 1.2].mean():.4f}")
print(f"Center displacement L2 after residuals: {np.linalg.norm(mu_res - mu_init, axis=1).mean():.4f}")
