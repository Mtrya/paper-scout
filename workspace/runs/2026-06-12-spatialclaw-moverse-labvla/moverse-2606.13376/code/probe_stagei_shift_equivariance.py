"""
Humble probe of MoVerse Stage I: topology-aware ERP completion.

MoVerse trains the panoramic generator to be shift-equivariant (paper Eq. 11):
    Roll_delta(epsilon_theta(X)) == epsilon_theta(Roll_delta(X))
This script demonstrates that a circular-padded convolution satisfies the property,
whereas a zero-padded convolution does not, making the seam problem concrete.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'assets', 'stagei_probe')
os.makedirs(OUT_DIR, exist_ok=True)

# A small synthetic latent "panorama": H x W, 2:1 aspect, with a horizontal pattern
H, W = 32, 64
rng = np.random.default_rng(7)
latent = np.sin(np.linspace(0, 4 * np.pi, W))[None, :] * np.ones((H, 1))
latent += 0.2 * rng.normal(size=(H, W))

def roll_horizontal(x, delta):
    return np.roll(x, delta, axis=1)

def conv2d(x, kernel, padding='circular'):
    """Simple 3x3 convolution."""
    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2
    if padding == 'circular':
        x_pad = np.pad(x, ((pad_h, pad_h), (pad_w, pad_w)), mode='wrap')
    elif padding == 'zero':
        x_pad = np.pad(x, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    else:
        raise ValueError(padding)
    out = np.zeros_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            out[i, j] = (x_pad[i:i + kh, j:j + kw] * kernel).sum()
    return out

# A simple edge-detection / smoothing kernel
kernel = np.array([[0.0, -1.0, 0.0],
                   [-1.0, 5.0, -1.0],
                   [0.0, -1.0, 0.0]])

delta = 10
base_circ = conv2d(latent, kernel, 'circular')
shift_then_conv_circ = conv2d(roll_horizontal(latent, delta), kernel, 'circular')
conv_then_shift_circ = roll_horizontal(base_circ, delta)

base_zero = conv2d(latent, kernel, 'zero')
shift_then_conv_zero = conv2d(roll_horizontal(latent, delta), kernel, 'zero')
conv_then_shift_zero = roll_horizontal(base_zero, delta)

error_circ = np.abs(shift_then_conv_circ - conv_then_shift_circ).mean()
error_zero = np.abs(shift_then_conv_zero - conv_then_shift_zero).mean()

print(f"Circular padding shift-equivariance error: {error_circ:.6f}")
print(f"Zero padding shift-equivariance error:     {error_zero:.6f}")

# Visualize
fig, axes = plt.subplots(2, 3, figsize=(12, 7))
im_kwargs = dict(cmap='RdBu_r', vmin=-3, vmax=3)
axes[0, 0].imshow(latent, **im_kwargs); axes[0, 0].set_title('Input latent'); axes[0, 0].axis('off')
axes[0, 1].imshow(base_circ, **im_kwargs); axes[0, 1].set_title('Conv circular (base)'); axes[0, 1].axis('off')
axes[0, 2].imshow(np.abs(base_circ - base_zero), cmap='hot'); axes[0, 2].set_title('|circular - zero| base'); axes[0, 2].axis('off')
axes[1, 0].imshow(conv_then_shift_circ, **im_kwargs); axes[1, 0].set_title(f'Roll({delta}) after circular conv'); axes[1, 0].axis('off')
axes[1, 1].imshow(shift_then_conv_circ, **im_kwargs); axes[1, 1].set_title(f'Circular conv after Roll({delta})'); axes[1, 1].axis('off')
axes[1, 2].imshow(np.abs(conv_then_shift_circ - shift_then_conv_circ), cmap='hot');
axes[1, 2].set_title(f'Difference (err={error_circ:.2e})'); axes[1, 2].axis('off')
fig.suptitle('Stage I topology probe: circular padding preserves horizontal shift equivariance')
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'shift_equivariance_probe.png'), dpi=150)
plt.close(fig)

# Demonstrate ERP periodicity: rolling the panorama should keep the seam invisible
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(latent, cmap='gray'); axes[0].set_title('Original ERP latent'); axes[0].axis('off')
axes[1].imshow(roll_horizontal(latent, W // 3), cmap='gray'); axes[1].set_title(f'Rolled by {W//3}'); axes[1].axis('off')
axes[2].imshow(np.hstack([latent[:, -8:], latent[:, :8]]), cmap='gray'); axes[2].set_title('Left-right seam neighborhood'); axes[2].axis('off')
fig.suptitle('ERP horizontal S1 topology: no privileged boundary')
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'erp_periodicity.png'), dpi=150)
plt.close(fig)

print(f"Saved Stage I probe artifacts to {OUT_DIR}")
