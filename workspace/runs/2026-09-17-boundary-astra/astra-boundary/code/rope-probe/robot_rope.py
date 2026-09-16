#!/usr/bin/env python3
"""robot_rope.py - MuJoCo rope-manipulation probe environment.

A single-file CLI server.  One episode lives in $ROPE_EP (default ./episode):
the model state, the task metadata and a trace of every call.  Each
invocation loads the episode, applies exactly one command, and writes it back
so the next call resumes from the same physics state.

Commands
  init <L0|L1|L2|L3> <seed>   start an episode
  state                       print the observation JSON
  observe                     render a top-down PNG into <episode>/obs/
  move <x> <y>                move the effector (<=5 cm per call), settle 0.5 s
  grab                        attach to the nearest rope node (<=2.5 cm)
  release                     detach
  done                        score and print {"FINAL": ...}

Budget: 200 per episode.  move/grab/release/observe cost 1; state/init are free.
"""

import json
import math
import os
import sys
import time
import traceback

# The GL backend has to be chosen before mujoco is imported, or the renderer
# silently comes up on whatever backend the shell had and can hand back a
# half-initialised frame.
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import mujoco  # noqa: E402

from core import (EE_Z, MASS, PHYS, WORLD_OFF, build_level, link_centres,  # noqa: E402
                  node_positions, node_sites, node_world, place_nodes)
import geom  # noqa: E402
import task as task_mod  # noqa: E402

EP = os.environ.get("ROPE_EP", os.path.join(HERE, "episode"))
ST = os.path.join(EP, "ep_state.json")
TR = os.path.join(EP, "trace.jsonl")
OBS = os.path.join(EP, "obs")

BUDGET = 200
MAX_STEP = 0.05          # metres per move
GRAB_TOL = 0.025         # metres
SETTLE = 0.5             # seconds of physics per move
SPRING_K = 120.0         # grip spring stiffness [N/m]
SPRING_FMAX = 12.0       # grip force cap [N] (keeps the sim stable)
ANG_DAMP = 2.0e-3        # grip rotational damping [N*m*s] (finite-size jaws)


def _grip_gains(m, meta):
    """Critically damped grip gains, sized to the held body's mass.

    The grip is a soft connection, not a rigid one: with k this stiff and a
    2 ms timestep the explicit spring would be at the edge of stability, so
    the force is capped (that cap alone removes the damping term, which is
    why the gains must be chosen from the body mass rather than guessed).
    """
    mb = float(m.body_mass[meta["nodes"][0]]) if meta["level"] == "L0" else MASS
    mb = max(mb, 1e-4)
    k = SPRING_K
    c = 2.0 * math.sqrt(k * mb)
    fmax = max(SPRING_FMAX, 3.0 * mb * 9.81)
    return k, c, fmax


# --------------------------------------------------------------------------
# episode persistence
# --------------------------------------------------------------------------

def load():
    with open(ST) as f:
        return json.load(f)


def save(s):
    tmp = ST + ".tmp"
    with open(tmp, "w") as f:
        json.dump(s, f)
    os.replace(tmp, ST)


def log(s, kind, payload, cost=1):
    s["budget"] -= cost
    rec = {"kind": kind, "payload": payload,
           "sim_t": round(float(s["sim_t"]), 4), "budget_left": s["budget"]}
    with open(TR, "a") as f:
        f.write(json.dumps(rec) + "\n")
    save(s)
    return s["budget"]


def restore(s):
    m, meta = build_level(s["level"], s["cfg"])
    _apply_lighting(m)
    meta.update(s["task"])
    d = mujoco.MjData(m)
    d.qpos[:] = np.array(s["qpos"], float)
    d.qvel[:] = np.array(s["qvel"], float)
    d.mocap_pos[0] = np.array(s["ee"], float) - np.array([WORLD_OFF, WORLD_OFF, 0.0])
    d.time = s["sim_t"]
    mujoco.mj_forward(m, d)
    return m, meta, d


def snapshot(s, m, d, meta, hold):
    s["qpos"] = [float(v) for v in np.asarray(d.qpos)]
    s["qvel"] = [float(v) for v in np.asarray(d.qvel)]
    s["sim_t"] = float(d.time)
    s["hold"] = hold
    P = node_positions(d, meta)
    s["zmax"] = max(float(s.get("zmax", 0.0)), float(P[:, 2].max()))


def advance(s, m, d, meta, seconds, hold=None):
    """Step physics, applying the soft grip spring on the held node."""
    n = int(round(seconds / m.opt.timestep))
    ee = np.asarray(s["ee"], float)
    if hold is None:
        for _ in range(n):
            mujoco.mj_step(m, d)
        return
    b, off = node_sites(m, meta)[hold]
    ee = ee - np.array([WORLD_OFF, WORLD_OFF, 0.0])   # world frame
    ja = int(m.body_jntadr[b])
    if ja < 0 or m.body_jntnum[b] < 1:
        return
    # mj_jntdofadr points at the *rotational* block for ball joints, so the
    # translatable block is jnt_dofadr - body_dofnum + 1 for those.  Anchor on
    # the body's own dof address instead: it is the linear block for both ball
    # and free joints.
    da = int(m.body_dofadr[b])
    nv = len(np.asarray(d.qvel))
    if da + 3 > nv:
        return
    cg_dof = da + 3 if int(m.body_dofnum[b]) > 3 else -1
    k, c, fmax = _grip_gains(m, meta)
    R = np.zeros(9)
    for _ in range(n):
        mujoco.mju_quat2Mat(R, np.asarray(d.xquat[b]))
        Rm = R.reshape(3, 3)
        cg = np.asarray(d.xpos[b])
        p = cg + Rm @ off
        v = np.asarray(d.qvel[da:da + 3])
        f = k * (ee - p) - c * v
        # keep the pull planar: a vertical component presses the body into the
        # table or lifts it, either of which changes the friction that the
        # manipulation depends on
        f[2] = 0.0
        nf = float(np.linalg.norm(f))
        if nf > fmax:
            f = f * (fmax / nf)
        # a real gripper is not a point: it also resists the body spinning in
        # its jaws, and without this term the planar contact makes rotation
        # effectively undamped, so any tangential lead spins the body up
        # without bound instead of converging on a heading
        if cg_dof >= 0:
            w = np.asarray(d.qvel[cg_dof:cg_dof + 3])
            tau_damp = -ANG_DAMP * w
        else:
            tau_damp = np.zeros(3)
        # xfrc_applied is a wrench at the body origin, so the moment that the
        # off-centre grip exerts has to be supplied explicitly
        d.xfrc_applied[b, :3] = f
        d.xfrc_applied[b, 3:] = np.cross(p - cg, f) + tau_damp
        mujoco.mj_step(m, d)
        d.xfrc_applied[b, :] = 0


def ee_pos(m, d, meta):
    """Effector position in the task frame."""
    p = np.asarray(d.xpos[meta["ee"]], float).copy()
    p[0] += WORLD_OFF
    p[1] += WORLD_OFF
    return p


# --------------------------------------------------------------------------
# observation
# --------------------------------------------------------------------------

def render(m, d, out=None, size=384):
    from PIL import Image
    r = mujoco.Renderer(m, size, size)
    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)
    opt.geomgroup[:] = 0
    opt.geomgroup[0] = 1
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    # The table is a 1x1 m plane centred on the MuJoCo origin while the task
    # frame is [0,1] (see WORLD_OFF), so the camera looks at the origin.  The
    # distance is derived from the model's own vertical field of view instead
    # of being hand-tuned: a straight-down view of a plane at the lookat
    # height shows 2*d*tan(fovy/2) metres vertically, so d = 0.5/tan(fovy/2)
    # frames the table edge-to-edge and the small slack keeps the rim inside
    # rather than clipped.
    fovy = float(m.vis.global_.fovy)
    cam.lookat[:] = [0.0, 0.0, 0.0]
    cam.distance = 1.04 * 0.5 / math.tan(math.radians(fovy) / 2.0)
    cam.elevation = -90.0
    cam.azimuth = 90.0
    r.update_scene(d, cam, scene_option=opt)
    img = r.render()
    if out is None:
        os.makedirs(OBS, exist_ok=True)
        n = len([f for f in os.listdir(OBS) if f.endswith(".png")])
        path = os.path.join(OBS, "obs_%03d.png" % n)
    else:
        path = out
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    Image.fromarray(img).save(path)
    return path


def state_json(s, m, d, meta):
    P = node_positions(d, meta)
    out = {
        "level": s["level"], "seed": s["seed"],
        "ee": [round(float(v), 4) for v in ee_pos(m, d, meta)],
        "holding": s["hold"],
        "budget_left": s["budget"], "sim_t": round(float(d.time), 3),
        "paths": [[round(float(p[0]), 4), round(float(p[1]), 4), round(float(p[2]), 4)] for p in P],
        "nodes_xy": [[round(float(p[0]), 4), round(float(p[1]), 4)]
                     for p in node_world(d, meta)],
        "n_segments": meta["nseg"],
        "goal": meta.get("goal", {}),
    }
    if s["level"] == "L0":
        c = (P[0] + P[1]) / 2.0
        v = P[1] - P[0]
        out["rod"] = {"centre": [round(float(c[0]), 4), round(float(c[1]), 4), round(float(c[2]), 4)],
                      "theta": round(float(math.atan2(v[1], v[0])), 4)}
    if s["level"] == "L3":
        if not os.environ.get("ROPE_HIDE_WINDING"):
            out["winding_now"] = round(float(geom.winding_about(P, task_mod.PEG_XY) * meta["direction"]), 3)
        out["peg"] = [float(task_mod.PEG_XY[0]), float(task_mod.PEG_XY[1]), task_mod.PEG_RAD]
    return out


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------

def _apply_lighting(m):
    m.vis.headlight.ambient[:] = 0.55
    m.vis.headlight.diffuse[:] = 0.70
    m.vis.headlight.specular[:] = 0.20


def cmd_init(level, seed):
    import random
    if level not in ("L0", "L1", "L2", "L3"):
        print(json.dumps({"error": "unknown level %r" % level}))
        return
    os.makedirs(EP, exist_ok=True)
    rng = random.Random(seed)
    cfg = dict(PHYS)
    m, meta = build_level(level, cfg)
    _apply_lighting(m)
    init_pts = task_mod.make_level(m, meta, level, rng)
    d = mujoco.MjData(m)
    place_nodes(m, d, meta, init_pts)
    # let it settle onto the table before the episode starts
    for _ in range(int(0.5 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    P = node_positions(d, meta)
    if level == "L1":
        for attempt in range(12):
            if task_mod.anchor_L1_goal(P, meta, rng):
                break
            rng = random.Random(seed * 7919 + attempt)
            init_pts = task_mod.make_level(m, meta, level, rng)
            d = mujoco.MjData(m)
            place_nodes(m, d, meta, init_pts)
            for _ in range(int(0.5 / m.opt.timestep)):
                mujoco.mj_step(m, d)
            P = node_positions(d, meta)
    # L3 must start effectively unwound: retry layouts until it settles clear
    for attempt in range(40):
        w0 = abs(geom.winding_about(P, task_mod.PEG_XY))
        if level != "L3" or w0 <= 0.12:
            break
        rng = random.Random(seed * 1000 + attempt)
        init_pts = task_mod.make_level(m, meta, level, rng)
        d = mujoco.MjData(m)
        place_nodes(m, d, meta, init_pts)
        for _ in range(int(0.5 / m.opt.timestep)):
            mujoco.mj_step(m, d)
        P = node_positions(d, meta)
    ee = np.array([float(P[0][0]), float(P[0][1]), EE_Z])
    d.mocap_pos[0] = ee - np.array([WORLD_OFF, WORLD_OFF, 0.0])
    mujoco.mj_forward(m, d)
    start = [int(meta["nodes"][0]), int(meta["nodes"][-1])]
    s = {"level": level, "seed": int(seed), "cfg": cfg, "budget": BUDGET, "hold": None,
         "ee": [float(v) for v in ee], "zmax": float(P[:, 2].max()),
         "qpos": [float(v) for v in np.asarray(d.qpos)],
         "qvel": [float(v) for v in np.asarray(d.qvel)],
         "sim_t": float(d.time),
         "task": {"nodes": [int(n) for n in meta["nodes"]],
                  "ee": int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ee")),
                  "root": int(meta["root"]) if "root" in meta else int(meta["nodes"][0]),
                  "nseg": int(meta["nseg"]), "link": float(meta["link"]),
                  "goal": meta["goal"], "direction": int(meta.get("direction", 0)),
                  "endpoints": start,
                  "targets": [[float(p[0]), float(p[1]), float(p[2])] for p in meta.get("targets", [])]}}
    save(s)
    with open(TR, "w") as f:
        f.write("")
    print(json.dumps({"ok": True, "level": level, "seed": seed, "budget": BUDGET,
                      "n_segments": meta["nseg"], "goal": meta["goal"]}))


def cmd_state(s, m, d, meta):
    print(json.dumps(state_json(s, m, d, meta)))


def cmd_observe(s, m, d, meta, out=None):
    p = render(m, d, out)
    left = log(s, "observe", {"path": p})
    print(json.dumps({"image": p, "budget_left": left}))


def cmd_move(s, m, d, meta, x, y):
    ee = np.asarray(s["ee"], float)
    tgt = np.array([float(x), float(y), EE_Z])
    delta = tgt[:2] - ee[:2]
    dist = float(np.linalg.norm(delta))
    clipped = dist > MAX_STEP
    if clipped and dist > 1e-9:
        delta = delta / dist * MAX_STEP
    s["ee"] = [float(ee[0] + delta[0]), float(ee[1] + delta[1]), EE_Z]
    d.mocap_pos[0] = np.array(s["ee"], float) - np.array([WORLD_OFF, WORLD_OFF, 0.0])
    mujoco.mj_forward(m, d)
    advance(s, m, d, meta, SETTLE, hold=s["hold"])
    left = log(s, "move", {"from": [round(float(ee[0]), 4), round(float(ee[1]), 4)],
                           "to": [round(float(s["ee"][0]), 4), round(float(s["ee"][1]), 4)],
                           "requested": [round(float(x), 4), round(float(y), 4)],
                           "clipped": clipped})
    print(json.dumps({"ee": [round(float(v), 4) for v in s["ee"]], "clipped": clipped,
                      "holding": s["hold"], "sim_t": round(float(d.time), 3),
                      "budget_left": left}))


def cmd_grab(s, m, d, meta):
    if s["hold"] is not None:
        left = log(s, "grab", {"result": "already_holding"})
        print(json.dumps({"error": "already holding node %d" % s["hold"], "budget_left": left}))
        return
    ee = np.asarray(s["ee"], float)
    P = node_world(d, meta)
    dists = np.linalg.norm(P[:, :2] - ee[:2], axis=1)
    j = int(np.argmin(dists))
    if float(dists[j]) > GRAB_TOL:
        left = log(s, "grab", {"result": "miss", "nearest": j,
                               "dist": round(float(dists[j]), 4), "tol": GRAB_TOL})
        print(json.dumps({"error": "no node within %.3f m (nearest node %d at %.3f m)"
                                   % (GRAB_TOL, j, float(dists[j])), "budget_left": left}))
        return
    s["hold"] = j
    left = log(s, "grab", {"result": "ok", "node": j, "dist": round(float(dists[j]), 4)})
    print(json.dumps({"holding": j, "dist": round(float(dists[j]), 4), "budget_left": left}))


def cmd_release(s, m, d, meta):
    if s["hold"] is None:
        left = log(s, "release", {"result": "not_holding"})
        print(json.dumps({"error": "not holding anything", "budget_left": left}))
        return
    j = s["hold"]
    s["hold"] = None
    advance(s, m, d, meta, 0.2, hold=None)
    left = log(s, "release", {"result": "ok", "node": j})
    print(json.dumps({"holding": None, "budget_left": left}))


def cmd_done(s, m, d, meta):
    P = node_positions(d, meta)
    lvl = s["level"]
    if lvl == "L0":
        res = task_mod.grade_L0(P, meta["goal"], meta)
    elif lvl == "L1":
        res = task_mod.grade_L1(link_centres(d, meta), meta)
    elif lvl == "L2":
        res = task_mod.grade_L2(P, meta)
    else:
        res = task_mod.grade_L3(P, meta, wmax=s.get("zmax"))
    res["FINISHED"] = True
    left = log(s, "done", res, cost=0)
    print(json.dumps({"FINAL": res}))


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def main(argv):
    cmd = argv[1] if len(argv) > 1 else "help"
    if cmd == "help" or cmd == "--help":
        print(__doc__)
        return 0
    if cmd == "init":
        cmd_init(argv[2], int(argv[3]))
        return 0
    if not os.path.exists(ST):
        print(json.dumps({"error": "no episode at %s; run: init <level> <seed>" % EP}))
        return 1
    s = load()
    m, meta, d = restore(s)
    if cmd == "state":
        cmd_state(s, m, d, meta)
    elif cmd == "observe":
        out = None
        if "--out" in argv:
            out = argv[argv.index("--out") + 1]
        cmd_observe(s, m, d, meta, out)
    elif cmd == "move":
        cmd_move(s, m, d, meta, float(argv[2]), float(argv[3]))
    elif cmd == "grab":
        cmd_grab(s, m, d, meta)
    elif cmd == "release":
        cmd_release(s, m, d, meta)
    elif cmd == "done":
        cmd_done(s, m, d, meta)
    else:
        print(json.dumps({"error": "unknown command %r" % cmd}))
        return 1
    if cmd in ("move", "grab", "release", "observe"):
        snapshot(s, m, d, meta, s["hold"])
        save(s)  # log() 已在命令内保存过预算;这里必须再存一次,否则物理状态不跨调用持久
    if s["budget"] <= 0 and cmd in ("move", "grab", "release", "observe"):
        print(json.dumps({"error": "budget exhausted"}))
        return 2
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except Exception:
        traceback.print_exc()
        sys.exit(3)
