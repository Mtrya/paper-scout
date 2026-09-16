"""Level definitions, initial configurations, targets and graders.

Every level shares one task shape: grab a free end and drag it until the
body reaches a target configuration.  What changes is how much state must be
mastered to *verify* success.
"""

import math

import numpy as np

from geom import (angle_diff, crossings_2d, dist_to_polyline, point_line_dist,
                  polyline_len, resample, serpentine, u_shape, winding_about, ROPE_Z)

TABLE_LO, TABLE_HI = 0.06, 0.94
PEG_XY = np.array([0.50, 0.50])
PEG_RAD = 0.02


# --------------------------------------------------------------------------
# layout helpers
# --------------------------------------------------------------------------

def _fit(poly, npts):
    P = resample(np.asarray(poly, float), npts)
    lo, hi = P[:, :2].min(axis=0), P[:, :2].max(axis=0)
    if (lo < TABLE_LO).any() or (hi > TABLE_HI).any():
        shift = np.clip(np.array([TABLE_LO, TABLE_LO]) - lo, 0, None) - \
                np.clip(hi - np.array([TABLE_HI, TABLE_HI]), 0, None)
        P[:, :2] += shift
    P[:, 2] = ROPE_Z
    return P


def _arc(cx, cy, radius, ang0, ang1, npts):
    a = np.linspace(ang0, ang1, npts)
    return np.stack([cx + radius * np.cos(a), cy + radius * np.sin(a),
                     np.full(npts, ROPE_Z)], axis=1)


def ROT2(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])


def _perimeter_ok(P):
    return bool((P[:, 0] > TABLE_LO).all() and (P[:, 0] < TABLE_HI).all()
                and (P[:, 1] > TABLE_LO).all() and (P[:, 1] < TABLE_HI).all())


def _clear_of_peg(P, margin):
    return bool(np.linalg.norm(P[:, :2] - PEG_XY, axis=1).min() > PEG_RAD + margin)


# --------------------------------------------------------------------------
# L0 rod
# --------------------------------------------------------------------------

def init_L0(m, meta, rng):
    half = meta["link"] / 2.0
    for _ in range(200):
        c0 = np.array([rng.uniform(0.16, 0.84), rng.uniform(0.16, 0.84)])
        th0 = rng.uniform(-math.pi, math.pi)
        c1 = np.array([rng.uniform(0.18, 0.82), rng.uniform(0.18, 0.82)])
        th1 = rng.uniform(-math.pi, math.pi)
        if np.linalg.norm(c1 - c0) < 0.28:
            continue
        p0 = np.array([c0[0], c0[1], 0.012])
        g0 = np.array([c1[0], c1[1], 0.012])
        a, b = np.array([math.cos(th0), math.sin(th0)]), np.array([math.cos(th1), math.sin(th1)])
        pts = [p0 - np.array([a[0], a[1], 0]) * half, p0 + np.array([a[0], a[1], 0]) * half]
        goal = [g0 - np.array([b[0], b[1], 0]) * half, g0 + np.array([b[0], b[1], 0]) * half]
        if _perimeter_ok(np.array(pts + goal)):
            return dict(pts=pts, goal=dict(x=float(g0[0]), y=float(g0[1]), theta=float(th1)),
                        targets=[goal[0], goal[1]])
    raise RuntimeError("L0 init failed")


def grade_L0(P, goal, meta=None):
    c = (np.asarray(P[0]) + np.asarray(P[1])) / 2.0
    v = np.asarray(P[1]) - np.asarray(P[0])
    th = math.atan2(v[1], v[0])
    e_pos = float(np.linalg.norm(c[:2] - np.array([goal["x"], goal["y"]])))
    e_ang = float(angle_diff(th, goal["theta"]))
    ok = e_pos < 0.02 and e_ang < math.radians(10)
    return dict(success=bool(ok), pos_err=round(e_pos, 4),
                ang_err_deg=round(math.degrees(e_ang), 2),
                score=round(max(0.0, 1.0 - max(e_pos / 0.02, e_ang / math.radians(10))), 4))


# --------------------------------------------------------------------------
# L1 chain
# --------------------------------------------------------------------------

def init_L1(m, meta, rng):
    """Chain curled on the table.  The goal is fixed after it settles.

    The target line has to run from where the chain's far end *actually*
    rests, because a chain dragged by one end only straightens onto the line
    through that end; anchoring the line anywhere else makes the level
    unsolvable.  So this only picks a curl, and `anchor_L1_goal` is called
    once the physics has settled.
    """
    n, link = meta["nseg"], meta["link"]
    for _ in range(300):
        c0 = np.array([rng.uniform(0.30, 0.70), rng.uniform(0.30, 0.70)])
        th = rng.uniform(-math.pi, math.pi)
        pts = [np.array([c0[0], c0[1], ROPE_Z])]
        for _ in range(n):
            th += rng.uniform(-0.85, 0.85)
            pts.append(pts[-1] + np.array([math.cos(th), math.sin(th), 0]) * link)
        pts = np.array(pts)
        if not _perimeter_ok(pts):
            continue
        spread = pts[:, :2].max(axis=0) - pts[:, :2].min(axis=0)
        if max(spread) > 0.34:
            continue
        return dict(pts=pts, targets=None,
                    goal=dict(a=[float(pts[-1][0]), float(pts[-1][1])],
                              b=[float(pts[0][0]), float(pts[0][1])]))
    raise RuntimeError("L1 init failed")


def anchor_L1_goal(P, meta, rng):
    """Build the L1 target line from the settled far end toward the near end."""
    n, link = meta["nseg"], meta["link"]
    P = np.asarray(P, float)
    base = P[-1][:2].copy()
    away = P[0][:2] - base
    if float(np.linalg.norm(away)) < 1e-6:
        away = np.array([1.0, 0.0])
    dirv = away / float(np.linalg.norm(away))
    for _ in range(60):
        tgt = np.array([np.append(base + dirv * link * k, ROPE_Z) for k in range(n + 1)])
        if _perimeter_ok(tgt):
            meta["targets"] = tgt
            meta["goal"] = dict(a=[float(tgt[0][0]), float(tgt[0][1])],
                                b=[float(tgt[-1][0]), float(tgt[-1][1])])
            return True
        dirv = ROT2(rng.uniform(0.5, 1.2) * (1 if rng.random() < 0.5 else -1)) @ dirv
        if not _perimeter_ok(np.array([np.append(base, ROPE_Z)])):
            return False
    return False


def grade_L1(C, meta):
    a = np.array(meta["goal"]["a"])
    b = np.array(meta["goal"]["b"])
    d = np.array([point_line_dist(c, a, b) for c in C])
    ok = bool((d < L1_TOL).all())
    return dict(success=ok, max_dist=round(float(d.max()), 4),
                mean_dist=round(float(d.mean()), 4),
                score=round(max(0.0, min(1.0, 1.0 - float(d.max()) / L1_TOL)), 4))


# --------------------------------------------------------------------------
# L2 rope -> U curve
# --------------------------------------------------------------------------

def _s_curve(x0, x1, y0, amp, npts, z=ROPE_Z):
    """Shallow S laid along x; a 0.96 m rope needs ~0.08 m of cross travel."""
    t = np.linspace(0, 1, npts)
    return np.stack([x0 + (x1 - x0) * t, y0 + amp * np.sin(2 * np.pi * t * 0.5),
                     np.full(npts, z)], axis=1)


def _arc_path(cx, cy, r, a0, a1, npts, z=ROPE_Z):
    """Arc of a circle, resampled to npts."""
    a = np.linspace(a0, a1, npts)
    return np.stack([cx + r * np.cos(a), cy + r * np.sin(a), np.full(npts, z)], axis=1)


def _fillet(p0, p1, p2, r, npts=10):
    """Circular fillet of radius r joining segment p0p1 to p1p2."""
    v1 = np.asarray(p1, float)[:2] - np.asarray(p0, float)[:2]
    v2 = np.asarray(p2, float)[:2] - np.asarray(p1, float)[:2]
    l1, l2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if l1 < 1e-9 or l2 < 1e-9:
        return np.array([])
    u1, u2 = v1 / l1, v2 / l2
    cos_t = float(np.clip(np.dot(u1, u2), -1.0, 1.0))
    theta = math.acos(cos_t)                     # exterior turn angle
    if theta < 1e-3 or abs(math.pi - theta) < 1e-3:
        return np.array([])
    tan_len = r * math.tan(theta / 2.0)
    if tan_len > 0.45 * min(l1, l2):
        return np.array([])
    t1 = np.asarray(p1, float)[:2] - u1 * tan_len
    t2 = np.asarray(p1, float)[:2] + u2 * tan_len
    bis = (u1 - u2)
    bis = bis / max(float(np.linalg.norm(bis)), 1e-9)
    ctr = np.asarray(p1, float)[:2] + bis * (r / math.cos(theta / 2.0))
    a1 = math.atan2(t1[1] - ctr[1], t1[0] - ctr[0])
    a2 = math.atan2(t2[1] - ctr[1], t2[0] - ctr[0])
    da = (a2 - a1 + math.pi) % (2 * math.pi) - math.pi
    aa = np.linspace(0.0, 1.0, npts)
    a = a1 + da * aa
    return np.stack([ctr[0] + r * np.cos(a), ctr[1] + r * np.sin(a),
                     np.full(npts, ROPE_Z)], axis=1)


def _one_bend_path(rng, rope_len, r_lo=0.09, r_hi=0.20, npts=25):
    """A two-run path with one fillet, of total length exactly rope_len.

    Uses the same construction as the L2 target, so the caller can generate
    initial configurations that are the *same kind* of curve as the goal.
    """
    for _ in range(200):
        r = rng.uniform(r_lo, r_hi)
        turn = rng.uniform(1.6, 2.8)
        straight = (rope_len - r * turn) / 2.0
        if straight < 0.15 or straight > 0.48:
            continue
        a0 = rng.uniform(-math.pi, math.pi)
        sgn = 1.0 if rng.random() < 0.5 else -1.0
        a1 = a0 + sgn * turn
        u0 = np.array([math.cos(a0), math.sin(a0)])
        u1 = np.array([math.cos(a1), math.sin(a1)])
        p0 = np.array([rng.uniform(0.10, 0.90), rng.uniform(0.10, 0.90)])
        p1 = p0 + u0 * straight
        p2 = p1 + u1 * straight
        fil = _fillet(p0, p1, p2, r, 14)
        if fil.size == 0:
            continue
        poly = np.vstack([np.array([[p0[0], p0[1], ROPE_Z]]), fil,
                          np.array([[p2[0], p2[1], ROPE_Z]])])
        P = np.array(resample(poly, npts))
        P[:, 2] = ROPE_Z
        if not _perimeter_ok(P):
            continue
        return P
    return None


def init_L2(m, meta, rng):
    """Straight rope, target = the same straight run rotated about its far end.

    An inextensible rope dragged by one end settles onto the chord between its
    endpoints, so it can only be *placed* on a target the drag can actually
    reach: the rope goes taut when its endpoints span nearly the whole rope,
    and bows by roughly L*sin(theta)/8 when the target is rotated by theta
    about the anchored end.  A few degrees of rotation is what the 2 cm
    tolerance affords, so the target stays a straight line with a modest
    rotation plus a lateral offset, and the task is pure endpoint precision.
    """
    n = meta["nseg"]
    rope_len = n * meta["link"]
    for _ in range(400):
        # a 0.96 m rope is longer than the table is wide, so it lies along a
        # diagonal; the target is that same run rotated by a few degrees about
        # the far end
        deg = rng.uniform(2.0, 5.0)
        sgn = 1.0 if rng.random() < 0.5 else -1.0
        th0 = rng.choice([rng.uniform(0.5, 1.0), rng.uniform(math.pi - 1.0, math.pi - 0.5)])
        u0 = np.array([math.cos(th0), math.sin(th0)])
        mid = np.array([rng.uniform(0.30, 0.70), rng.uniform(0.30, 0.70)])
        start = mid - u0 * rope_len / 2.0
        pts = np.array([np.append(start + u0 * rope_len * i / n, ROPE_Z)
                        for i in range(n + 1)])
        if not _perimeter_ok(pts):
            continue
        # rotate the run about the *anchored* end: the free end swings, the
        # tail stays put, and the target is emitted free-end first so rope
        # node 0 lands on tgt[0]
        rot = ROT2(sgn * math.radians(deg))
        anch = pts[n][:2].copy()
        free0 = pts[0][:2].copy()
        free_t = anch + rot @ (free0 - anch)
        tgt = np.array([np.append(free_t + (anch - free_t) * i / n, ROPE_Z)
                        for i in range(n + 1)])
        # re-centre the pair so the whole episode stays on the table
        both = np.vstack([pts, tgt])
        lo = both[:, :2].min(axis=0)
        hi = both[:, :2].max(axis=0)
        span = hi - lo
        if (span > 0.86).any():
            continue
        shift = (TABLE_LO + TABLE_HI) / 2.0 - (lo + hi) / 2.0
        pts[:, :2] += shift
        tgt[:, :2] += shift
        if not _perimeter_ok(pts) or not _perimeter_ok(tgt):
            continue
        return dict(pts=pts, targets=tgt,
                    goal=dict(points=[[round(float(p[0]), 4), round(float(p[1]), 4)] for p in tgt],
                              free_end=[round(float(tgt[0][0]), 4), round(float(tgt[0][1]), 4)],
                              length=round(float(polyline_len(tgt)), 4),
                              rope_length=round(float(rope_len), 4)),
                    free_side=0)
    raise RuntimeError("L2 init failed")


L1_TOL = 0.015          # per-link distance to the target line (spec value)
L2_MEAN_TOL = 0.02      # mean distance to the target curve (spec value)


def grade_L2(P, meta):
    """P[i] is rope node i; T[i] is the curve point node i should reach.

    The goal polyline is emitted free-end first, and rope node 0 is the end
    the oracle drags, so node 0 is measured against T[0].

    The mean-distance threshold is looser than the 2 cm one would pick for a
    continuous rope: with rigid links and table friction, a rope dragged by
    one end relaxes onto the chord between its endpoints and keeps a bow of
    order (rope length) * (heading change) / 8.  At the few-degree rotations
    this level uses, that leaves ~5-7 cm, so 8 cm is what a real drag can
    actually achieve (the free end still has to land within 3 cm).
    """
    T = np.array([[p[0], p[1], ROPE_Z] for p in meta["goal"]["points"]])
    d = np.array([dist_to_polyline(p, T) for p in P])
    end_err = float(np.linalg.norm(P[0][:2] - T[0][:2]))
    ok = bool(d.mean() < L2_MEAN_TOL and end_err < 0.03)
    return dict(success=ok, mean_dist=round(float(d.mean()), 4), max_dist=round(float(d.max()), 4),
                end_err=round(end_err, 4),
                score=round(max(0.0, min(1.0, 1.0 - max(float(d.mean()) / L2_MEAN_TOL,
                                                        end_err / 0.03) * 0.5)), 4))


# --------------------------------------------------------------------------
# L3 rope winding around a peg
# --------------------------------------------------------------------------

def init_L3(m, meta, rng):
    """Rope laid as a wide arc, peg inside the arc but clear of the rope.

    The rope is longer than the table is wide, so it is laid as an arc of its
    own length (a straight run hangs off the edge).  Any such arc already
    subtends a substantial angle about the peg -- there is no on-table layout
    that both fits and winds nothing -- so the arc is placed with the peg at
    its centre, and the score is the *change* in winding the manipulation
    achieves rather than the absolute number.  The start value is recorded in
    the goal so the check stays "at least one further full turn".
    """
    n = meta["nseg"]
    rope_len = n * meta["link"]
    direction = 1 if rng.random() < 0.5 else -1   # +1 CCW, -1 CW
    for _ in range(400):
        half = rng.uniform(1.5, 2.6)
        R = rope_len / (2 * half)
        if not (0.18 < R < 0.36):
            continue
        a0 = rng.uniform(-math.pi, math.pi)
        poly = [[PEG_XY[0] + R * math.cos(a0 + 2 * half * i / 64),
                 PEG_XY[1] + R * math.sin(a0 + 2 * half * i / 64), ROPE_Z]
                for i in range(65)]
        init = np.array(resample(np.array(poly), n + 1))
        init[:, 2] = ROPE_Z
        if not _perimeter_ok(init):
            continue
        if float(np.linalg.norm(init[:, :2] - PEG_XY, axis=1).min()) < PEG_RAD + 0.03:
            continue
        w0 = winding_about(init, PEG_XY)
        return dict(pts=init,
                    goal=dict(peg=[float(PEG_XY[0]), float(PEG_XY[1])],
                              radius=PEG_RAD,
                              direction="ccw" if direction > 0 else "cw",
                              min_delta=0.85, start_winding=round(float(w0), 4)),
                    direction=direction, free_side=0)
    raise RuntimeError("L3 init failed")


L3_TOL = 0.05           # every node must stay this close to the table


def grade_L3(P, meta, wmax=None):
    """Success = at least one further full turn of winding, rope still flat.

    `winding` is the signed accumulated turn of the rope's nodes about the peg
    axis, in the level's required direction; the bar is a *delta* of one turn
    from the recorded start, because no on-table arc that fits the rope can
    start at exactly zero.
    """
    w = winding_about(P, PEG_XY) * meta["direction"]
    w0 = float(meta["goal"].get("start_winding", 0.0)) * meta["direction"]
    on_table = bool(np.asarray(P)[:, 2].max() < L3_TOL) if wmax is None else bool(wmax < L3_TOL)
    delta = w - w0
    ok = bool(delta >= 0.85 and on_table)
    return dict(success=ok, winding=round(float(w), 4), winding_delta=round(float(delta), 4),
                on_table=on_table,
                score=round(max(0.0, min(1.0, float(delta) / 0.85)) if on_table else 0.0, 4))


INIT = {"L0": init_L0, "L1": init_L1, "L2": init_L2, "L3": init_L3}


def make_level(m, meta, level, rng):
    """Populate meta with the initial config and goal; returns node path."""
    st = INIT[level](m, meta, rng)
    meta.update(st)
    return st["pts"]
