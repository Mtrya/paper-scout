"""Geometry helpers: arc-length resampling, polylines, winding, crossings."""

import math

import numpy as np

TABLE_MIN = 0.05
TABLE_MAX = 0.95
ROPE_Z = 0.004


def polyline_len(P):
    P = np.asarray(P, float)
    return float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1)))


def resample(P, npts):
    """Resample a polyline to npts points equally spaced by arc length."""
    P = np.asarray(P, float)
    if P.ndim == 2 and P.shape[1] == 2:
        P = np.hstack([P, np.zeros((len(P), 1))])
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    total = float(seg.sum())
    cs = np.concatenate([[0.0], np.cumsum(seg)])
    out = []
    for i in range(npts):
        t = total * i / (npts - 1)
        j = int(np.searchsorted(cs, t, side="right") - 1)
        j = max(0, min(j, len(seg) - 1))
        f = (t - cs[j]) / seg[j] if seg[j] > 1e-12 else 0.0
        out.append(P[j] + f * (P[j + 1] - P[j]))
    return np.array(out)


def uv_to_path(uv, xy, z=ROPE_Z):
    u, v = uv
    x, y = xy
    return np.array([x, y, z])


def serpentine(x0, y0, x1, n_bends, amp, npts, z=ROPE_Z):
    xs = np.linspace(x0, x1, n_bends + 1)
    pts = [np.array([xs[0], y0, z])]
    for i in range(1, len(xs)):
        pts.append(np.array([xs[i], y0 + (amp if i % 2 else -amp), z]))
    return resample(np.array(pts), npts)


def u_shape(cx, cy, arm, width, npts, z=ROPE_Z):
    """U opening toward +y: down one arm, across the base, up the other."""
    pts = np.array([
        [cx - width / 2, cy + arm, z],
        [cx - width / 2, cy, z],
        [cx + width / 2, cy, z],
        [cx + width / 2, cy + arm, z],
    ])
    return resample(pts, npts)


def in_table(P, margin=0.0):
    P = np.asarray(P, float)
    return bool((P[:, 0] > TABLE_MIN - margin).all() and (P[:, 0] < TABLE_MAX + margin).all()
                and (P[:, 1] > TABLE_MIN - margin).all() and (P[:, 1] < TABLE_MAX + margin).all())


# --------------------------------------------------------------------------
# task-geometry predicates
# --------------------------------------------------------------------------

def point_seg_dist(p, a, b):
    p, a, b = map(lambda v: np.asarray(v, float)[:2], (p, a, b))
    ab = b - a
    L2 = float(np.dot(ab, ab))
    t = 0.0 if L2 < 1e-12 else float(np.clip(np.dot(p - a, ab) / L2, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def dist_to_polyline(p, P):
    P = np.asarray(P, float)
    return min(point_seg_dist(p, P[i], P[i + 1]) for i in range(len(P) - 1))


def cross2(a, b):
    return float(a[0] * b[1] - a[1] * b[0])


def point_line_dist(p, a, b):
    p, a, b = map(lambda v: np.asarray(v, float)[:2], (p, a, b))
    ab = b - a
    n = np.linalg.norm(ab)
    if n < 1e-12:
        return float(np.linalg.norm(p - a))
    return float(abs(cross2(ab, p - a)) / n)


def winding_about(P, axis_xy, axis_z=0.0):
    """Signed winding of a polyline around a vertical axis.

    Each node is projected to the plane and its angle about the axis is
    unwrapped; the winding number is the accumulated turn divided by 2*pi.
    Positive = counter-clockwise when viewed from +z.
    """
    P = np.asarray(P, float)
    c = np.asarray(axis_xy, float)[:2]
    ang = np.arctan2(P[:, 1] - c[1], P[:, 0] - c[0])
    d = np.diff(ang)
    d = (d + np.pi) % (2 * np.pi) - np.pi
    return float(d.sum() / (2 * np.pi))


def crossings_2d(P, skip_adjacent=True):
    """Number of proper self-intersections of the projected polyline.

    The polyline is open -- a rope's two ends are free -- so the first and the
    last segment are *not* neighbours: a rope whose ends cross is the textbook
    case this has to count, and skipping that pair reported 0 for every such
    scene.
    """
    Q = np.asarray(P, float)[:, :2]
    n = len(Q)
    cnt = 0
    for i in range(n - 1):
        for j in range(i + 2, n - 1):
            if skip_adjacent and j == i + 1:
                continue
            if seg_intersect(Q[i], Q[i + 1], Q[j], Q[j + 1]):
                cnt += 1
    return cnt


def _orient(a, b, c):
    return cross2(b - a, c - a)


def seg_intersect(p1, p2, p3, p4):
    """True when two closed 2D segments cross properly (endpoint touches excluded)."""
    d1 = _orient(p3, p4, p1)
    d2 = _orient(p3, p4, p2)
    d3 = _orient(p1, p2, p3)
    d4 = _orient(p1, p2, p4)
    eps = 1e-9
    if ((d1 > eps and d2 < -eps) or (d1 < -eps and d2 > eps)) and \
       ((d3 > eps and d4 < -eps) or (d3 < -eps and d4 > eps)):
        return True
    return False


def angle_diff(a, b):
    d = (a - b + math.pi) % (2 * math.pi) - math.pi
    return abs(d)
