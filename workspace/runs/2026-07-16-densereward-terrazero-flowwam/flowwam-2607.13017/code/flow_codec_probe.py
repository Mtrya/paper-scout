"""
Minimal numpy-only probe of FlowWAM's reversible flow codec (16-bit path).
The full codec in training/reversible_flow_codec.py requires cv2, which is not
installed in this harness. The 16-bit path uses the manual HSV<->RGB helpers,
so we can reproduce the round-trip numerics without cv2.
"""
import numpy as np


def encode16(flow, max_magnitude):
    dx, dy = flow[..., 0], flow[..., 1]
    magnitude = np.sqrt(dx ** 2 + dy ** 2)
    angle = np.arctan2(dy, dx)
    mag_norm = np.clip(magnitude / max_magnitude, 0, 1)
    angle_norm = (angle + np.pi) / (2 * np.pi)
    angle_norm = np.clip(angle_norm, 0, 1)
    h = (angle_norm * 65535).astype(np.uint16)
    s = (mag_norm * 65535).astype(np.uint16)
    v = np.full_like(h, 65535)
    return np.stack([h, s, v], axis=-1)


def hsv16_to_rgb16(hsv):
    h = hsv[..., 0].astype(np.float64) / 65535.0 * 360.0
    s = hsv[..., 1].astype(np.float64) / 65535.0
    v = hsv[..., 2].astype(np.float64) / 65535.0
    c = v * s
    h_prime = h / 60.0
    x = c * (1 - np.abs(h_prime % 2 - 1))
    m = v - c
    r, g, b = np.zeros_like(h), np.zeros_like(h), np.zeros_like(h)
    for lo, hi, rv, gv, bv in [
        (0, 1, c, x, 0), (1, 2, x, c, 0), (2, 3, 0, c, x),
        (3, 4, 0, x, c), (4, 5, x, 0, c), (5, 6, c, 0, x),
    ]:
        mask = (h_prime >= lo) & (h_prime < hi)
        r[mask] = rv[mask] if isinstance(rv, np.ndarray) else rv
        g[mask] = gv[mask] if isinstance(gv, np.ndarray) else gv
        b[mask] = bv[mask] if isinstance(bv, np.ndarray) else bv
    rgb = np.stack([(r + m), (g + m), (b + m)], axis=-1)
    return (rgb * 65535).clip(0, 65535).astype(np.uint16)


def rgb16_to_hsv16(rgb):
    r = rgb[..., 0].astype(np.float64) / 65535.0
    g = rgb[..., 1].astype(np.float64) / 65535.0
    b = rgb[..., 2].astype(np.float64) / 65535.0
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin
    h = np.zeros_like(delta)
    mask_r = (cmax == r) & (delta > 0)
    mask_g = (cmax == g) & (delta > 0)
    mask_b = (cmax == b) & (delta > 0)
    h[mask_r] = 60 * (((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6)
    h[mask_g] = 60 * ((b[mask_g] - r[mask_g]) / delta[mask_g] + 2)
    h[mask_b] = 60 * ((r[mask_b] - g[mask_b]) / delta[mask_b] + 4)
    s = np.where(cmax > 0, delta / cmax, 0)
    return np.stack([
        (h / 360.0 * 65535).clip(0, 65535),
        (s * 65535).clip(0, 65535),
        (cmax * 65535).clip(0, 65535),
    ], axis=-1).astype(np.uint16)


def decode16(rgb, max_magnitude):
    hsv = rgb16_to_hsv16(rgb)
    angle_norm = hsv[..., 0].astype(np.float64) / 65535.0
    mag_norm = hsv[..., 1].astype(np.float64) / 65535.0
    angle = angle_norm * 2 * np.pi - np.pi
    magnitude = mag_norm * max_magnitude
    dx = magnitude * np.cos(angle)
    dy = magnitude * np.sin(angle)
    return np.stack([dx, dy], axis=-1).astype(np.float32)


def roundtrip_test():
    H, W = 240, 320
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    cx, cy = W / 2, H / 2
    dx = -(yy - cy) * 0.3 + 2.0
    dy = (xx - cx) * 0.3 - 1.5
    flow_gt = np.stack([dx, dy], axis=-1).astype(np.float32)
    max_mag = 25.0

    hsv = encode16(flow_gt, max_mag)
    rgb = hsv16_to_rgb16(hsv)
    flow_dec = decode16(rgb, max_mag)

    err = np.abs(flow_gt - flow_dec)
    print(f"shape={flow_gt.shape}, max_mag={max_mag}")
    print(f"mean abs error:  {err.mean():.6f} px")
    print(f"max abs error:   {err.max():.6f} px")
    print(f"99th percentile: {np.percentile(err, 99):.6f} px")


if __name__ == "__main__":
    roundtrip_test()
