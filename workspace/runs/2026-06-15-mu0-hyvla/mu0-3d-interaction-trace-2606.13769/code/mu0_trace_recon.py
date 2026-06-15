"""Reconstruct mu0's B-spline trace target from Appendix B.2.

Run with: python code/probe_venv/bin/python runs/.../mu0_trace_recon.py
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import lstsq
from scipy.interpolate import BSpline

np.random.seed(0)
H, D = 32, 10
# Synthetic 3D future trace
t = np.linspace(0, 1, H)
T = np.stack([np.sin(4*np.pi*t), np.cos(4*np.pi*t), 0.5*t], axis=1).astype(np.float32)
T += np.random.randn(H, 3)*0.02
anchor = np.array([0.,0.,0.], dtype=np.float32)
T_rel = T - anchor

# Build cubic B-spline basis with D control points, clamped at both ends
interior = np.linspace(0, 1, D-2)[1:-1]  # D-4 interior knots
knots = np.concatenate(([0]*4, interior, [1]*4))
assert len(knots) == D + 4

t_grid = np.linspace(0,1,H+1)[1:]
B = np.zeros((H, D))
for j in range(D):
    c = np.zeros(D); c[j]=1
    B[:, j] = BSpline(knots, c, 3)(t_grid)

# Anchor-prepended ridge least-squares fit (Eq. 4 in paper)
B_anch = np.vstack([np.eye(D)[0:1], B])
T_anch = np.vstack([np.zeros((1,3)), T_rel])
lam = 0.2
Gamma = np.eye(D) - np.eye(D, k=1)
A = np.vstack([B_anch, lam*Gamma])
Y = np.vstack([T_anch, np.zeros((D,3))])
P_star, *_ = lstsq(A, Y)
T_hat = B @ P_star

print('P* shape:', P_star.shape)
print('reconstruction MSE:', float(np.mean((T_rel-T_hat)**2)))

fig, ax = plt.subplots(1,2, figsize=(10,4))
ax[0].plot(T_rel[:,0], T_rel[:,1], 'o-', label='raw rel trace (H=32)')
ax[0].plot(T_hat[:,0], T_hat[:,1], 's--', label=f'B-spline recon (D={D})')
ax[0].scatter(P_star[:,0], P_star[:,1], c='k', s=20, label='control points')
ax[0].set_title('mu0 B-spline trace compression')
ax[0].legend(); ax[0].axis('equal')
ax[1].plot(T_rel[:,2], label='raw z')
ax[1].plot(T_hat[:,2], label='recon z')
ax[1].set_title('Depth component'); ax[1].legend()
plt.tight_layout()
import pathlib
out = pathlib.Path(__file__).resolve().parents[2] / 'assets' / 'mu0_bspline_recon.png'
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150)
print('saved', out)
