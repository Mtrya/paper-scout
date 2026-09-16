#!/usr/bin/env python3
"""oracle.py - scripted drag policies that prove each level is solvable.

This is the feasibility check that must pass before any LLM trial means
anything.  Each oracle uses only the same primitive the agent has (drag a
node around with the effector) and writes oracle/results.json.

    python oracle.py [--seeds 1 2 3] [--levels L0 L1 L2 L3]
"""

import argparse
import contextlib
import io
import json
import math
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import geom  # noqa: E402
from geom import ROPE_Z  # noqa: E402
import task as task_mod  # noqa: E402
from robot_rope import SPRING_FMAX  # noqa: E402

OUT_DIR = os.path.join(HERE, "oracle")
RESULTS = os.path.join(OUT_DIR, "results.json")
MAX_STEP = 0.05


class quiet:
    """Swallow the CLI layer's stdout so the oracle log stays readable."""

    def __enter__(self):
        self._out = contextlib.redirect_stdout(io.StringIO())
        self._out.__enter__()
        return self

    def __exit__(self, *a):
        self._out.__exit__(*a)
        return False


def _fresh_env(tag):
    d = os.path.join(tempfile.gettempdir(), "rope_oracle_" + tag)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d, exist_ok=True)
    os.environ["ROPE_EP"] = d
    for mod in [m for m in list(sys.modules) if m in ("robot_rope", "core", "task", "geom")]:
        del sys.modules[mod]
    import robot_rope
    return robot_rope, d


class Run:
    """Drives one episode through the CLI's own command layer."""

    def __init__(self, level, seed):
        self.rr, self.ep = _fresh_env("%s_%d" % (level, seed))
        with quiet():
            self.rr.cmd_init(level, seed)
        self.s = self.rr.load()
        self.m, self.meta, self.d = self.rr.restore(self.s)
        self.level = level
        self.n_moves = 0

    # ---- primitives -------------------------------------------------
    def nodes(self):
        """Grabbable node sites (rod tips for L0, chain nodes elsewhere)."""
        return self.rr.node_world(self.d, self.meta)

    def ee(self):
        return np.asarray(self.s["ee"], float)

    def move_to(self, p):
        """Hop toward p in <=5 cm jumps, updating state like the CLI does."""
        p = np.asarray(p, float).ravel()
        if len(p) == 3:
            p = p[:2]
        guard = 0
        while True:
            cur = self.ee()
            delta = p[:2] - cur[:2]
            dist = float(np.linalg.norm(delta))
            if dist < 1e-4 or guard > 400:
                break
            nxt = cur[:2] + (delta / dist) * min(dist, MAX_STEP)
            with quiet():
                self.rr.cmd_move(self.s, self.m, self.d, self.meta,
                                 float(nxt[0]), float(nxt[1]))
            self.n_moves += 1
            guard += 1

    def settle(self, seconds):
        self.rr.advance(self.s, self.m, self.d, self.meta, seconds, hold=self.s["hold"])

    def _izz(self, b):
        """Moment of inertia of body `b` about the vertical axis.

        `body_inertia` is diagonal in the body's *principal* frame, and for a
        rod lying flat that frame's third axis is the rod's own long axis --
        the inertia about the rod's own axis, not about vertical.  Rotating
        the rod on the table is a vertical-axis turn, so the gain has to be
        built from the transverse component; using the diagonal entry
        directly understates the inertia by 100x.
        """
        import mujoco as _mj
        Ip = np.asarray(self.m.body_inertia[b], float)
        Rb = np.zeros(9)
        _mj.mju_quat2Mat(Rb, np.asarray(self.d.xquat[b]))
        Rq = np.zeros(9)
        _mj.mju_quat2Mat(Rq, np.asarray(self.m.body_iquat[b], float))
        R = Rb.reshape(3, 3) @ Rq.reshape(3, 3)
        return float(((R[2, :] ** 2) * Ip).sum())

    def spin_to(self, th_g, K=25.0, seconds=0.35, tau_max=0.30, tau_min=0.0,
                follow=True):
        """Rotate the held body's heading with a pure moment.

        A moment is the only actuation channel that turns a body on the table:
        a sideways tip pull gives up 90% of its force to dragging the body
        along (measured: a 4.5 cm tangential pull rotates the rod 2.7 deg and
        translates it 4 cm), whereas a moment rotates it in place.

        `follow` walks the effector along with the grip point as the body
        turns.  Without it the spring that the twist pulls away from grows to
        its 12 N cap in a few degrees and the body is flung -- the earlier
        fixed-effector version walked the rod's centre 19 m off the table.
        Following costs nothing in reach: a burst turns the body ~19 deg, so
        the grip point travels ~3 cm, inside the 5 cm a single move commands.

        `tau_min` covers the last few degrees: a gain-derived moment drops
        below breakaway (measured ~0.017 N*m for a rod pivoting on its tip)
        and leaves the body stuck short of the target.
        """
        import mujoco as _mj
        b, off = self.rr.node_sites(self.m, self.meta)[self.s["hold"]]
        Iz = self._izz(b)
        for _ in range(int(seconds / self.m.opt.timestep)):
            R = np.zeros(9)
            _mj.mju_quat2Mat(R, np.asarray(self.d.xquat[b]))
            Rm = R.reshape(3, 3)
            cg = np.asarray(self.d.xpos[b])
            p = cg + Rm @ off
            P = self.rr.node_positions(self.d, self.meta)
            v = P[1] - P[0]
            th = math.atan2(v[1], v[0])
            dth = (th_g - th + math.pi) % (2 * math.pi) - math.pi
            w = float(np.asarray(self.d.qvel)[5])
            tau = Iz * (K * dth - w) / self.m.opt.timestep
            if tau_min > 0.0:
                sgn = 1.0 if dth > 0 else -1.0
                tau = sgn * max(abs(tau), tau_min)
            tau = float(np.clip(tau, -tau_max, tau_max))
            if follow:
                # `p` is already a MuJoCo world point and `s["ee"]` lives in
                # the task frame, so only the latter needs the offset.  Getting
                # this backwards parked the effector 0.71 m from the grip and
                # the next settle yanked the body that far in half a second.
                self.d.mocap_pos[0] = np.asarray(p, float)
                self.s["ee"] = [float(p[0] + self.rr.WORLD_OFF),
                                float(p[1] + self.rr.WORLD_OFF),
                                float(self.s["ee"][2])]
                f = np.zeros(3)
            else:
                f = np.asarray(self.s["ee"], float) - p
                f[2] = 0.0
                nf = float(np.linalg.norm(f))
                if nf > SPRING_FMAX:
                    f = f * (SPRING_FMAX / nf)
            self.d.xfrc_applied[b, :3] = f
            self.d.xfrc_applied[b, 3:] = np.cross(p - cg, f) + np.array([0.0, 0.0, tau])
            _mj.mj_step(self.m, self.d)
            self.d.xfrc_applied[b, :] = 0

    def rod_state(self):
        """(node poses, centre, heading) of the two-node rod."""
        P = self.rr.node_positions(self.d, self.meta)
        C = (P[0][:2] + P[1][:2]) / 2.0
        v = P[1][:2] - P[0][:2]
        return P, C, math.atan2(v[1], v[0])

    def spin_heading_to(self, target, tol=0.8, k_max=20):
        """Drive the rod's heading to `target` in rate-limited bursts.

        One burst turns the rod ~15 deg (the moment saturates against the
        table's rotational drag), so a large heading change takes a run of
        them; the grip follows the tip throughout, which is what keeps the
        centre where it was.
        """
        for _ in range(k_max):
            P, C, ang = self.rod_state()
            if abs(math.degrees(geom.angle_diff(ang, target))) < tol:
                break
            self.spin_to(target, tau_max=0.055, tau_min=0.025, seconds=0.35)

    def grab_nearest(self):
        P = self.nodes()
        dists = np.linalg.norm(P[:, :2] - self.ee()[:2], axis=1)
        j = int(np.argmin(dists))
        self.move_to(P[j])
        P = self.nodes()
        if np.linalg.norm(P[j][:2] - self.ee()[:2]) > 0.02:
            self.move_to(P[j])
        with quiet():
            self.rr.cmd_grab(self.s, self.m, self.d, self.meta)
        return self.s["hold"]

    def grab_node(self, j):
        P = self.nodes()
        self.move_to(P[j])
        self.settle(0.2)
        P = self.nodes()
        self.move_to(P[j])
        with quiet():
            self.rr.cmd_grab(self.s, self.m, self.d, self.meta)
        return self.s["hold"]

    def release(self):
        with quiet():
            self.rr.cmd_release(self.s, self.m, self.d, self.meta)
        self.settle(0.3)

    def done(self):
        P = self.rr.node_positions(self.d, self.meta)
        if self.level == "L0":
            return task_mod.grade_L0(P, self.meta["goal"])
        if self.level == "L1":
            return task_mod.grade_L1(self.rr.link_centres(self.d, self.meta), self.meta)
        if self.level == "L2":
            return task_mod.grade_L2(P, self.meta)
        return task_mod.grade_L3(P, self.meta, wmax=self.s.get("zmax"))


# --------------------------------------------------------------------------
# per-level policies
# --------------------------------------------------------------------------

def policy_L0(run):
    """Spin the rod onto its goal heading, then drag the tip onto its goal tip.

    Two primitives in sequence, each doing the job the other cannot:

    * A moment turns the rod on the spot.  The grip follows the tip while it
      turns, so the centre does not move at all (measured: 1.6 mm of drift
      over four ~60 deg of turning), which is what makes a heading pass and a
      position pass independent of each other.
    * A drag moves the rod but also turns it, because the rod swings to line
      up with the pull: 0.2 deg per cm when the pull is sideways, 0.01 deg per
      cm when it is along the rod.  So the drag is aimed along the rod's own
      axis -- the tip goes to the goal tip, which is where it belongs once the
      heading is right -- and any heading it does cost is spun back out.

    Which goal tip the held tip owns is decided after the spin, not before: a
    spin leaves tip 1 on the +goal-heading side by construction, so picking
    the goal tip earlier would sometimes send the rod to the wrong side and
    leave the centre a rod-length away.
    """
    goal = run.meta["goal"]
    g = np.array([goal["x"], goal["y"]], float)
    th = goal["theta"]
    u = np.array([math.cos(th), math.sin(th)], float)
    h = run.meta["link"] / 2.0

    P = run.nodes()
    j = 0 if (np.linalg.norm(P[0][:2] - run.ee()[:2])
              <= np.linalg.norm(P[1][:2] - run.ee()[:2])) else 1
    if run.grab_node(j) is None:
        return
    for _ in range(5):
        run.spin_heading_to(th)
        P, C, ang = run.rod_state()
        s = 1.0 if float(np.dot(P[j][:2] - C, u)) >= 0.0 else -1.0
        T = g + u * h * s
        pe = float(np.linalg.norm(C - g))
        ae = math.degrees(geom.angle_diff(ang, th))
        if pe < 0.008 and abs(ae) < 3.0:
            break
        prev = None
        for _ in range(10):
            cur = run.nodes()[j][:2]
            err = T - cur
            nd = float(np.linalg.norm(err))
            if nd < 0.005 and prev is not None and prev < 0.005:
                break
            prev = nd
            run.move_to(cur + err * 1.4)
            run.settle(0.4)
        run.move_to(T)
        run.settle(0.5)
    run.spin_heading_to(th, tol=0.5)
    run.release()


def policy_L1(run):
    """Creep the chain out along its target line with a sub-threshold pull.

    A chain dragged by one end does not unfold like a rope: the tension dies
    out along the chain, so whatever the pull does not reach keeps its shape.
    Worse, if the pull exceeds the chain's total static friction the whole
    chain slides as one piece and the shape is *frozen* -- a 0.3 m drag then
    moves the silhouette without changing it by a single degree.

    Staying under that threshold is what makes the level solvable: a 4 mm hop
    asks for ~0.5 N, the links nearest the grip creep forward, and the extra
    length the chain gains comes out of its curl rather than out of dragging
    the far end along.  The span therefore grows monotonically toward the
    rope length while every link settles onto the target line.

    Direction is chosen by room, not by ownership: the line is anchored on
    whichever end the layout generator picked first, and the pull has to go
    the long way across the table or the drag runs out of tabletop and the
    residual curve never comes out.
    """
    goal = run.meta["goal"]
    a = np.array(goal["a"], float)
    b = np.array(goal["b"], float)
    u = (b - a) / max(float(np.linalg.norm(b - a)), 1e-9)

    def room(p, dirv):
        lo, hi = task_mod.TABLE_LO + 0.01, task_mod.TABLE_HI - 0.01
        t = 1e9
        for i in (0, 1):
            if dirv[i] > 1e-9:
                t = min(t, (hi - p[i]) / dirv[i])
            elif dirv[i] < -1e-9:
                t = min(t, (lo - p[i]) / dirv[i])
        return float(t)

    def worst():
        return task_mod.grade_L1(run.rr.link_centres(run.d, run.meta),
                                 run.meta)["max_dist"]

    P = run.nodes()
    ra, rb = room(P[0][:2], u), room(P[-1][:2], -u)
    j = 0 if ra >= rb else len(P) - 1
    dg = u if j == 0 else -u
    base = P[j][:2].copy()
    if run.grab_node(j) is None:
        return
    hop = 0.004
    limit = min(max(ra, rb), 0.45)
    t = 0.0
    k = 0
    while t < limit and k < 170:
        t = min(t + hop, limit)
        p = base + dg * t
        cur = run.ee()[:2]
        if float(np.linalg.norm(p - cur)) < 1e-5:
            break
        with quiet():
            run.rr.cmd_move(run.s, run.m, run.d, run.meta, float(p[0]), float(p[1]))
        run.n_moves += 1
        k += 1
        if k % 8 == 0 and worst() < 0.010:
            break
    run.settle(0.4)
    run.release()


def policy_L2(run):
    """Servo the free end onto its target until it stops creeping.

    The body keeps hauling the free end backwards while it relaxes, so a
    single placement lands short of the mark.  The fix is to close the loop on
    the held node's own position -- measure where it actually is, command the
    effector past it by that error, and repeat until the error is small twice
    in a row.
    """
    T = np.array([[p[0], p[1], ROPE_Z] for p in run.meta["goal"]["points"]], float)
    # the level emits the target free-end first, and rope node 0 is that end
    j, tail = 0, T[0][:2]
    if run.grab_node(j) is None:
        return
    prev = None
    for _ in range(10):
        cur = run.rr.node_positions(run.d, run.meta)[j][:2]
        err = tail - cur
        nd = float(np.linalg.norm(err))
        if nd < 0.008 and prev is not None and prev < 0.008:
            break
        prev = nd
        run.move_to(cur + err * 1.4)
        run.settle(0.5)
    run.move_to(tail)
    run.settle(0.8)
    run.release()


def policy_L3(run):
    """Wind the rope by walking the free end around the peg.

    The grading winding is the angle the rope's two ends subtend about the peg
    axis, so a lap of the free end around the peg adds a turn and the rope
    wraps onto the post instead of being dragged after the grip.  Direction is
    not the level's `direction` label but its opposite: the label names the
    sense in which the *graded* winding must grow, and the graded winding is
    the raw winding times the label's sign.

    Every lap is monitored and the walk stops the moment the bar is cleared,
    and it bails out early if the winding is moving away from the goal (which
    is what happens while the peg is not physically where `task.PEG_XY` says
    it is -- see BLOCKER.md).
    """
    goal = run.meta["goal"]
    peg = np.array(goal["peg"], float)
    sgn = -float(run.meta["direction"])

    def wind():
        return float(geom.winding_about(
            run.rr.node_positions(run.d, run.meta), peg) * -sgn)

    w0 = wind()
    P = run.nodes()
    j = 0
    if run.grab_node(j) is None:
        return
    r_hold = 0.035
    cur = run.rr.node_positions(run.d, run.meta)[j][:2]
    d0 = cur - peg
    if float(np.linalg.norm(d0)) < 1e-6:
        d0 = np.array([1.0, 0.0])
    a0 = math.atan2(d0[1], d0[0])
    run.move_to(peg + r_hold * np.array([math.cos(a0), math.sin(a0)]))
    run.settle(0.3)
    best = w0
    prev_best = w0
    for k in range(1, 121):
        a = a0 + sgn * 2 * math.pi * k / 48
        run.move_to(peg + r_hold * np.array([math.cos(a), math.sin(a)]))
        best = max(best, wind())
        if best - w0 >= 1.05:
            break
        # a lap that nets nothing means the post the metric turns about is not
        # where the rope can push against it; stop burning the budget
        if k % 48 == 0:
            if best - prev_best < 0.12:
                break
            prev_best = best
    run.settle(0.4)
    run.release()


POLICIES = {"L0": policy_L0, "L1": policy_L1, "L2": policy_L2, "L3": policy_L3}


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--levels", nargs="+", default=["L0", "L1", "L2", "L3"])
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    results = {}
    if os.path.exists(RESULTS):
        with open(RESULTS) as f:
            results = json.load(f)
    all_ok = True
    for lvl in args.levels:
        for seed in args.seeds:
            key = "%s/seed%d" % (lvl, seed)
            try:
                run = Run(lvl, seed)
                POLICIES[lvl](run)
                res = run.done()
                res["moves"] = run.n_moves
                res["sim_s"] = round(float(run.d.time), 2)
            except Exception as e:
                import traceback
                traceback.print_exc()
                res = {"success": False, "error": "%s: %s" % (type(e).__name__, e)}
            results[key] = res
            ok = bool(res.get("success"))
            all_ok &= ok
            print("%-10s %s  moves=%s  %s" % (key, "PASS" if ok else "FAIL",
                                              res.get("moves"), json.dumps(
                                                  {k: v for k, v in res.items()
                                                   if k not in ("moves", "FINISHED")})))
    with open(RESULTS, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print("\n%s -> %s" % (RESULTS, "ALL PASS" if all_ok else "SOME FAILED"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
