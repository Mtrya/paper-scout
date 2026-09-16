#!/usr/bin/env python3
"""gen_perception_cases.py - build the perception cases for the rope family.

Each case is a settled rope scene (rope + peg, built through the same model and
the same placement path an episode uses) plus its ground truth for three
questions a perception front-end has to answer before any policy can act:

    Q1  does the rope encircle the peg?          (yes / no)
    Q2  how many times does the rope cross itself? (0 / 1 / 2+)
    Q3  which free end is nearer the peg?         (E0 / E1)

Output:
    perception/cases.json          - index + ground truth for every case
    perception/cases/<i>/state.json  - the same JSON `robot_rope.py state` prints
    perception/cases/<i>/render.png  - top-down render of the settled scene

    cd code/rope-probe && .venv/bin/python gen_perception_cases.py
"""

import contextlib
import io
import json
import math
import os
import random
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import mujoco  # noqa: E402

import geom  # noqa: E402
import task as task_mod  # noqa: E402
from core import WORLD_OFF, build_level, place_nodes  # noqa: E402
from geom import ROPE_Z  # noqa: E402

OUT_DIR = os.path.join(HERE, "perception")
CASES = os.path.join(OUT_DIR, "cases.json")
SETTLE_S = 0.5
WOUND_TURNS = 0.5        # |winding| at or above this reads as "encircles the peg"


def _episode(tag):
    """Run the CLI's own init against a scratch episode dir; return its state."""
    import robot_rope
    d = os.path.join(tempfile.gettempdir(), "rope_perc_" + tag)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d, exist_ok=True)
    os.environ["ROPE_EP"] = d
    return robot_rope


# --------------------------------------------------------------------------
# scene builders -- each returns the node path to place, before settling
# --------------------------------------------------------------------------

def _spiral(peg, r0, dr, turns, npts, a0=0.0, sgn=1.0):
    """Rope coiled around the peg: r grows with angle so the coils nest."""
    a = np.linspace(0.0, sgn * 2 * math.pi * turns, npts)
    r = r0 + (dr - r0) * (a / a[-1])
    return np.stack([peg[0] + r * np.cos(a0 + a), peg[1] + r * np.sin(a0 + a),
                     np.full(npts, ROPE_Z)], axis=1)


def _arc(peg, r, half, npts, a0):
    a = np.linspace(a0, a0 + 2 * half, npts)
    return np.stack([peg[0] + r * np.cos(a), peg[1] + r * np.sin(a),
                     np.full(npts, ROPE_Z)], axis=1)


def _straight(cx, cy, ang, length, npts):
    u = np.array([math.cos(ang), math.sin(ang)])
    t = np.linspace(-0.5, 0.5, npts)[:, None]
    P = np.array([cx, cy])[None, :] + t * length * u[None, :]
    return np.hstack([P, np.full((npts, 1), ROPE_Z)])


def _shift_inside(P):
    """Slide a path so the whole rope stays on the table."""
    P = np.array(P, float)
    lo, hi = P[:, :2].min(axis=0), P[:, :2].max(axis=0)
    span = hi - lo
    c = np.array([(task_mod.TABLE_LO + task_mod.TABLE_HI) / 2.0] * 2)
    if (span > task_mod.TABLE_HI - task_mod.TABLE_LO).any():
        P[:, :2] *= ((task_mod.TABLE_HI - task_mod.TABLE_LO) / span.max()) * 0.98
        lo, hi = P[:, :2].min(axis=0), P[:, :2].max(axis=0)
    P[:, :2] += c - (lo + hi) / 2.0
    return P


def _waypoints(wp, npts):
    """A hand-laid polyline (a pigtail crosses its own stem) at rope length."""
    P = np.array([[p[0], p[1], ROPE_Z] for p in wp], float)
    return geom.resample(P, npts)


def _layouts(peg, nseg):
    """The case list: label -> node path (rope has nseg+1 nodes)."""
    n = nseg + 1
    return [
        ("coiled_1p5", _spiral(peg, 0.045, 0.055, 1.5, n, a0=0.4, sgn=1.0)),
        ("coiled_2p2", _spiral(peg, 0.040, 0.060, 2.2, n, a0=-1.1, sgn=-1.0)),
        ("coiled_1p0", _spiral(peg, 0.035, 0.050, 1.0, n, a0=2.0, sgn=1.0)),
        ("arc_wide", _arc(peg, 0.26, 2.4, n, -0.7)),
        ("arc_half", _arc(peg, 0.30, 1.4, n, 2.2)),
        ("straight_past", _straight(peg[0] - 0.10, peg[1] + 0.16, 0.15, 0.92, n)),
        ("straight_near", _straight(peg[0] + 0.03, peg[1] - 0.05, 2.6, 0.92, n)),
        ("pigtail_1x", _waypoints(
            [(0.45, 0.30), (0.45, 0.60), (0.62, 0.56), (0.60, 0.40), (0.34, 0.44)], n)),
        ("pigtail_2x", _waypoints(
            [(0.45, 0.30), (0.45, 0.60), (0.60, 0.60), (0.62, 0.45),
             (0.32, 0.40), (0.64, 0.34)], n)),
        ("big_loop", _waypoints(
            [(0.42, 0.28), (0.42, 0.66), (0.66, 0.68), (0.70, 0.40),
             (0.26, 0.36), (0.72, 0.26)], n)),
    ]


# --------------------------------------------------------------------------

def main():
    # regenerate from scratch: stale case dirs from an earlier layout list
    # would otherwise sit next to the current ones with colliding indices
    shutil.rmtree(os.path.join(OUT_DIR, "cases"), ignore_errors=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    rr = _episode("gen")
    m, meta = build_level("L3", dict(rr.PHYS))
    rr._apply_lighting(m)
    # the fields cmd_init would have filled in; state_json reads these
    meta["ee"] = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ee")
    meta["direction"] = 1
    meta["goal"] = {"peg": list(task_mod.PEG_XY), "radius": task_mod.PEG_RAD}
    peg = task_mod.PEG_XY
    entries = []
    for i, (name, path) in enumerate(_layouts(peg, meta["nseg"])):
        P = _shift_inside(path)
        P[:, 2] = ROPE_Z
        d = mujoco.MjData(m)
        place_nodes(m, d, meta, P)
        for _ in range(int(SETTLE_S / m.opt.timestep)):
            mujoco.mj_step(m, d)
        P = rr.node_positions(d, meta)
        w = geom.winding_about(P, peg)
        cross = geom.crossings_2d(P)
        dists = [float(np.linalg.norm(P[k][:2] - peg)) for k in (0, len(P) - 1)]
        case_dir = os.path.join(OUT_DIR, "cases", "%02d_%s" % (i, name))
        os.makedirs(case_dir, exist_ok=True)
        rr.render(m, d, out=os.path.join(case_dir, "render.png"))
        s = {"level": "L3", "seed": -1, "hold": None,
             "budget": rr.BUDGET, "zmax": float(P[:, 2].max())}
        with open(os.path.join(case_dir, "state.json"), "w") as f:
            json.dump(rr.state_json(s, m, d, meta), f, indent=2)
        entries.append({
            "case": "%02d_%s" % (i, name),
            "state": "cases/%02d_%s/state.json" % (i, name),
            "render": "cases/%02d_%s/render.png" % (i, name),
            "answers": {
                "Q1_wound_around_peg": "yes" if abs(w) >= WOUND_TURNS else "no",
                "Q2_self_crossings": "0" if cross == 0 else ("1" if cross == 1 else "2+"),
                "Q3_nearer_free_end": "E0" if dists[0] <= dists[1] else "E1",
            },
            "ground_truth": {
                "winding_about_peg": round(float(w), 4),
                "wound_threshold": WOUND_TURNS,
                "self_crossings": int(cross),
                "end_dist_to_peg": {"E0": round(dists[0], 4), "E1": round(dists[1], 4)},
            },
        })
        print("%-16s wound=%-3s crossings=%d nearer=%s" % (
            name, entries[-1]["answers"]["Q1_wound_around_peg"], cross,
            entries[-1]["answers"]["Q3_nearer_free_end"]))
    with open(CASES, "w") as f:
        json.dump({
            "n_cases": len(entries),
            "scene": "L3 (24-link rope + peg at %s)" % np.round(peg, 3).tolist(),
            "questions": {
                "Q1": "does the rope encircle the peg? |winding| >= %.2f" % WOUND_TURNS,
                "Q2": "how many times does the rope cross itself?",
                "Q3": "which free end is nearer the peg (E0 = node 0, E1 = node 24)?",
            },
            "cases": entries,
        }, f, indent=2)
    print("\n%d cases -> %s" % (len(entries), CASES))
    return 0


if __name__ == "__main__":
    with contextlib.redirect_stdout(sys.stdout):
        sys.exit(main())
