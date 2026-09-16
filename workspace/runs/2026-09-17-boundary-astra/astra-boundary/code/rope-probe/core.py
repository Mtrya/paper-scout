"""MuJoCo model construction and state plumbing for the rope-probe levels.

Levels:
  L0  rigid rod          - one capsule on a free joint
  L1  articulated chain  - 8 capsules chained by ball joints
  L2  rope on a table    - 24 capsules chained by ball joints
  L3  rope + peg         - L2 plus a static cylinder (no machine addition)

The rope is built as an explicit nested kinematic chain rather than with
`<composite type="cable">`.  MuJoCo 3.13's composite builder emits a jointless
stub body at the root of the chain whose frame becomes an invisible parent for
every rope body, which makes world-frame placement ambiguous and leaks a
frozen body into the state.  Explicit nesting keeps one body per node, each
addressable by name.

Geometry convention (n links -> n+1 nodes):
  body i has a ball joint anchored at node i and a capsule geom reaching
  toward node i+1; body n is the bare final node.
"""

import math

import mujoco
import numpy as np

RAD = 0.004          # rope capsule radius [m]
SEG = 0.04           # rope link length [m]
MASS = 0.005         # rope link mass [kg]
NSEG = 24            # rope links   (=> NSEG+1 nodes)
ROD_HALF = 0.10      # L0 rod half-length [m]
ROD_RAD = 0.008
CHAIN_N = 8          # L1 links
CHAIN_LINK = 0.04
CHAIN_RAD = 0.008
CHAIN_MASS = 0.02
PEG_RAD = 0.02
PEG_H = 0.06
EE_Z = 0.01          # effector sphere centre height [m]
# The task frame is [0,1]x[0,1] on the table; the MuJoCo world centres the
# table on the origin, so every position crossing that boundary is shifted.
WORLD_OFF = 0.5

PHYS = {
    "ts": 0.001,     # integrator timestep [s] (1 kHz: keeps the soft grip stable)
    "it": 80,        # solver iterations
    "fric": 0.35,    # sliding friction (table <-> body)
    # Joint damping is the one knob that decides whether the links can
    # articulate at all.  A ball joint only ever sees the tension the grip
    # passes down the chain -- order 0.01 N*m -- so a damping of 3.0 N*m*s
    # caps the joint rate at ~0.01 rad/s and the chain stops being a chain:
    # it translates and rotates as one rigid piece (measured: shape bitwise
    # unchanged over a 0.3 m drag).  0.02 lets the links follow the pull at
    # the rates the manipulation actually uses while still bleeding off the
    # contact chatter that would otherwise rattle the rope.
    "jdamp": 0.02,   # joint damping [N*m*s]
}


# --------------------------------------------------------------------------
# quaternion helpers
# --------------------------------------------------------------------------

def q_from_x(d):
    """Quaternion (w,x,y,z) that rotates +x onto direction d."""
    d = np.asarray(d, float)
    n = float(np.linalg.norm(d))
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    d = d / n
    c = float(d[0])
    if c > 1 - 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if c < -1 + 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0])
    ax = np.cross(np.array([1.0, 0.0, 0.0]), d)
    ax = ax / np.linalg.norm(ax)
    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, ax, math.acos(max(-1.0, min(1.0, c))))
    return q / np.linalg.norm(q)


def q_inv(q):
    q = np.asarray(q, float)
    return np.array([q[0], -q[1], -q[2], -q[3]]) / np.dot(q, q)


def q_mul(a, b):
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, np.asarray(a, float), np.asarray(b, float))
    n = float(np.linalg.norm(out))
    return out / n if n > 1e-12 else np.array([1.0, 0.0, 0.0, 0.0])


# --------------------------------------------------------------------------
# XML assembly
# --------------------------------------------------------------------------

def _options(cfg):
    return (
        '<option timestep="%s" integrator="implicitfast" cone="elliptic" '
        'iterations="%d" ls_iterations="60" gravity="0 0 -9.81"/>'
        % (cfg["ts"], cfg["it"])
    )


def _world(cfg):
    return (
        '<geom name="table" type="plane" size="0.5 0.5 0.01" pos="0 0 0" '
        'friction="%s 0.01 0.001" condim="3" rgba="0.88 0.88 0.85 1"/>'
        '<body name="ee" mocap="true" pos="0 0 %s">'
        '<geom name="ee_g" type="sphere" size="0.01" rgba="0.90 0.13 0.13 1" '
        'contype="0" conaffinity="0"/></body>' % (cfg["fric"], EE_Z)
    )


def _chain(cfg, n, link, radius, mass, prefix, max_bend=None):
    """Nested chain of n links over n+1 nodes.

    Layout: `<prefix>_root` carries the free joint; every node i is a child
    body with a ball joint, and body i also carries a capsule geom reaching
    toward node i+1.  The last node is a bare anchor.  Because the free joint
    sits on the jointless root, the whole rope is one floating kinematic
    island whose frame we can place directly.
    """
    inner = ""
    for i in range(n, -1, -1):
        px = "0 0 0" if i == 0 else "%s 0 0" % link
        geom = ""
        if i < n:
            geom = (
                '<geom name="{p}_g{i}" type="capsule" fromto="0 0 0 {L} 0 0" '
                'size="{r}" density="0" mass="{m}" friction="{f} 0.01 0.001" '
                'condim="3" rgba="0.22 0.52 0.82 1"/>'
            ).format(p=prefix, i=i, L=link, r=radius, m=mass, f=cfg["fric"])
        else:
            geom = ('<geom name="%s_tip" type="sphere" size="0.0002" density="0" '
                    'mass="1e-6" rgba="0.10 0.35 0.65 1" contype="0" conaffinity="0"/>'
                    % prefix)
        rng = ""
        if max_bend is not None:
            # MuJoCo's default compiler angle unit is degrees; feeding it a
            # radian value silently welds the joint shut (a "50" written as
            # 0.87 leaves 0.87 deg of travel), so pass degrees and say so.
            rng = 'range="0 %.6g" limited="true"' % max_bend
        jt = ('<joint name="%s_j%d" type="ball" damping="%s" %s solreflimit="0.02 2"/>'
              % (prefix, i, cfg["jdamp"], rng))
        inner = (
            '<body name="{p}_{i}" pos="{px}">{jt}{geom}{child}</body>'
        ).format(p=prefix, i=i, px=px, jt=jt, geom=geom, child=inner)
    return ('<body name="%s_root" pos="0 0 0">'
            '<joint name="%s_free" type="free" damping="0.05"/>'
            '<geom name="%s_root_g" type="sphere" size="0.0002" density="0" mass="1e-6" '
            'rgba="0 0 0 0" contype="0" conaffinity="0"/>'
            "%s</body>" % (prefix, prefix, prefix, inner))


def build_level(level, cfg=None):
    """Build the MuJoCo model for a level.  Returns (model, meta)."""
    cfg = dict(PHYS if cfg is None else cfg)
    meta = {"level": level, "cfg": cfg}

    if level == "L0":
        rod = (
            '<body name="rod" pos="0 0 0">'
            '<joint name="rod_j" type="free" damping="0.05"/>'
            '<geom name="rod_g" type="capsule" fromto="-%s 0 0 %s 0 0" size="%s" '
            'density="0" mass="0.05" friction="%s 0.01 0.001" condim="3" '
            'rgba="0.85 0.55 0.18 1"/></body>'
            % (ROD_HALF, ROD_HALF, ROD_RAD, cfg["fric"])
        )
        xml = ("<mujoco>%s<worldbody>%s%s</worldbody></mujoco>"
               % (_options(cfg), _world(cfg), rod))
        m = mujoco.MjModel.from_xml_string(xml)
        meta.update(nodes=[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "rod")],
                    nseg=1, link=2 * ROD_HALF, radius=ROD_RAD, endpoints=[0, 1])
        return m, meta

    if level == "L1":
        n, link, radius, mass, prefix, bend = CHAIN_N, CHAIN_LINK, CHAIN_RAD, CHAIN_MASS, "ch", 50.0
    else:
        n, link, radius, mass, prefix, bend = NSEG, SEG, RAD, MASS, "rp", None

    peg = ""
    if level == "L3":
        peg = (
            '<body name="peg" pos="%s %s 0"><geom name="peg_g" type="cylinder" '
            'size="%s %s" pos="0 0 %s" friction="%s 0.01 0.001" condim="3" '
            'rgba="0.30 0.30 0.34 1"/></body>'
            % (0.0, 0.0, PEG_RAD, PEG_H / 2, PEG_H / 2, cfg["fric"])
        )
    xml = ("<mujoco>%s<worldbody>%s%s%s</worldbody></mujoco>"
           % (_options(cfg), _world(cfg), _chain(cfg, n, link, radius, mass, prefix, bend), peg))
    m = mujoco.MjModel.from_xml_string(xml)
    meta.update(
        nodes=[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "%s_%d" % (prefix, i))
               for i in range(n + 1)],
        root=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "%s_root" % prefix),
        nseg=n, link=link, radius=radius, endpoints=[0, n],
    )
    return m, meta


# --------------------------------------------------------------------------
# state <-> qpos
# --------------------------------------------------------------------------

def place_nodes(m, d, meta, pts):
    """Write world-frame node positions into qpos.

    pts[i] is node i.  The free-joint root is placed at node 0 aligned with
    the first link; every node body then gets its position/orientation
    relative to its parent, so the chain lands exactly on the given polyline.
    """
    pts = [np.asarray(p, float) for p in pts]
    pts = [np.array([p[0] - WORLD_OFF, p[1] - WORLD_OFF, p[2]]) for p in pts]
    if meta["level"] == "L0":
        ids = meta["nodes"]
        c = (pts[0] + pts[1]) / 2.0
        a = int(m.jnt_qposadr[int(m.body_jntadr[ids[0]])])
        buf = np.array(d.qpos)
        buf[a:a + 3] = c
        buf[a + 3:a + 7] = q_from_x(pts[1] - pts[0])
        d.qpos[:] = buf
        mujoco.mj_forward(m, d)
        return

    ids = meta["nodes"]
    n = len(ids) - 1
    if len(pts) == n:
        pts = pts + [pts[-1] + (pts[-1] - pts[-2])]
    assert len(pts) == n + 1, (len(pts), n + 1)
    buf = np.array(d.qpos)
    fa = int(m.jnt_qposadr[int(m.body_jntadr[meta["root"]])])
    w0 = q_from_x(pts[1] - pts[0])
    buf[fa:fa + 3] = pts[0]
    buf[fa + 3:fa + 7] = w0
    prev_q = w0
    for k, b in enumerate(ids):
        a = int(m.jnt_qposadr[int(m.body_jntadr[b])])
        if a + 4 > len(buf):
            raise RuntimeError("qpos overrun at node %d (nq=%d, adr=%d)" % (k, len(buf), a))
        if k == 0:
            buf[a:a + 4] = np.array([1.0, 0.0, 0.0, 0.0])
            continue
        wq = q_from_x(pts[k + 1] - pts[k]) if k + 1 < len(pts) else prev_q
        buf[a:a + 4] = q_mul(q_inv(prev_q), wq)
        prev_q = wq
    d.qpos[:] = buf
    mujoco.mj_forward(m, d)


def _to_task(P):
    P = np.array(P, float, copy=True)
    P[:, 0] += WORLD_OFF
    P[:, 1] += WORLD_OFF
    return P


def node_positions(d, meta):
    """All nodes in the task frame (L0: the two rod endpoints)."""
    if meta["level"] == "L0":
        b = meta["nodes"][0]
        c = np.asarray(d.xpos[b]).copy()
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, np.asarray(d.xquat[b]))
        ax = R.reshape(3, 3)[:, 0]
        return _to_task([c - ax * ROD_HALF, c + ax * ROD_HALF])
    return _to_task([np.asarray(d.xpos[b]).copy() for b in meta["nodes"]])


def node_sites(m, meta):
    """Body + local offset for each grabbable node.

    The rod's nodes are its two tips (local +-x), so a grab lands on an end
    and the lever arm actually rotates the rod.  Everywhere else the node is
    the body origin.
    """
    if meta["level"] == "L0":
        b = meta["nodes"][0]
        return [(b, np.array([-ROD_HALF, 0.0, 0.0])), (b, np.array([ROD_HALF, 0.0, 0.0]))]
    return [(b, np.zeros(3)) for b in meta["nodes"]]


def node_world(d, meta):
    """Grabbable node positions in the task frame."""
    out = []
    for b, off in node_sites(None, meta):
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, np.asarray(d.xquat[b]))
        out.append(np.asarray(d.xpos[b]) + R.reshape(3, 3) @ off)
    return _to_task(out)


def link_centres(d, meta):
    P = node_positions(d, meta)
    return (P[:-1] + P[1:]) / 2.0
