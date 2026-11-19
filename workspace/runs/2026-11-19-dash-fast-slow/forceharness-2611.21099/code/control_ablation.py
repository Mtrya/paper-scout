"""E2: does the fast force loop change what can be identified about the unknown physics?

Same high-level plan, same estimator, same decision rule; the arms differ only in what the controller does at
contact and while lifting:

  force   move_until_contact down (stop at 3 N), 18 mm deeper, grasp, lift(height=0.10, hold_time=1.0)
  guard   move_to the nominal grasp height with contact protection (stop on contact), grasp,
          move_by up 10 cm at the same 2 cm/s, hold(1.0)
  pose    the same as guard but with no contact protection (the no-force control of the paper, none|pose):
          the move runs to its target even through the object

The estimator runs in every arm, so the difference is purely what the trajectory does to the object (shove at contact,
grasp height, tilt while lifting) -- not what is rendered to the model.

Usage
  python control_ablation.py --seeds 9000-9019 --out runs/dev/e2.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _repo_root(cli: str | None = None) -> Path:
    """The ForceHarness clone: --repo, then $FORCEHARNESS_REPO, then the nearest pyproject.toml above this file."""
    import os
    for cand in (cli, os.environ.get("FORCEHARNESS_REPO")):
        if cand:
            p = Path(cand).expanduser().resolve()
            if (p / "pyproject.toml").exists():
                return p
            raise SystemExit(f"not a ForceHarness clone: {p}")
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise SystemExit("cannot find the ForceHarness clone; pass --repo or set FORCEHARNESS_REPO")


def _repo_from_argv() -> str | None:
    import sys
    if "--repo" in sys.argv:
        return sys.argv[sys.argv.index("--repo") + 1]
    return None


REPO = _repo_root(_repo_from_argv())
sys.path.insert(0, str(REPO / "scripts"))
import _bootstrap  # noqa: F401,E402

import numpy as np  # noqa: E402

from forceharness.agent.loop import EpisodeConfig  # noqa: E402
from forceharness.baselines.expert import _pick_yaw  # noqa: E402
from forceharness.eval.runner import _make_sim  # noqa: E402
from forceharness.primitives.library import Primitives  # noqa: E402

EP = dict(scene_noise_pos=0.004, scene_noise_yaw_deg=2.0, f_contact=3.0, f_cap=25.0, kp=150.0, force_sign=-1.0,
          force_noise_std=0.5, force_bias_std=0.3)
TASKS = {"weight_sort": dict(obj="cube", period=90.0), "com_sort": dict(obj="bar", period=180.0)}
ARMS = ("force", "guard", "pose")


def find(sc, name):
    return next(o for o in sc if o.get("name") == name)


def snapshot(sim) -> dict:
    mj, m, d = sim.mujoco, sim.m, sim.d
    out = dict(eef=np.round(sim.eef_pos, 5).tolist(), sim_time=float(sim.sim_time), opening=sim.gripper_opening)
    bids = {}
    bar_id = getattr(sim.env, "bar_body_id", None)
    if bar_id is not None:
        bids["bar"] = int(bar_id)
    for key, nm in (("bar", "bar_root"), ("cube", "cube_main")):
        if key in bids:
            continue
        bid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, nm)
        if bid >= 0:
            bids[key] = int(bid)
    for key, bid in bids.items():
        out[f"{key}_com"] = np.round(d.xipos[bid], 5).tolist()
        out[f"{key}_R"] = np.round(d.xmat[bid].reshape(3, 3), 5).tolist()
    return out


def run_episode(task: str, seed: int, arm: str, out_dir: str, run_name: str) -> dict:
    spec = TASKS[task]
    cfg = EpisodeConfig(task=task, model="scripted", fmt="phys2", vision=False, interface=arm, seed=seed,
                        run_name=run_name, out_dir=out_dir, max_turns=40, max_tool_calls=80, max_sim_time=300.0, **EP)
    sim = _make_sim(task, cfg, "robosuite")
    pr = Primitives(sim, interface=arm, f_contact=3.0, f_cap=25.0,
                    pos_tol=float(sim.task.extra.get("pos_tol", 0.003)), slow_in_contact=bool(sim.task.extra.get("slow_in_contact", False)))
    steps, snaps, yaw_acc = [], [], 0.0

    def call(name, **kw):
        nonlocal yaw_acc
        rep = pr.call(name, **kw)
        if name == "rotate_z":
            yaw_acc += float(kw.get("angle_deg", 0.0))
        steps.append(dict(name=name, args=kw, termination=rep.termination, peak_force=rep.peak_force,
                          n_ticks=len(rep.wrench_rows), contact_fraction=rep.truth.get("contact_fraction"),
                          derived=rep.derived,
                          quantities={q["name"]: q for q in rep.quantities},
                          truth=rep.truth.get("task", {})))
        snaps.append(snapshot(sim))
        return rep

    scene = sim.scene()
    obj, table = find(scene, spec["obj"]), find(scene, "table")
    top = float(table["top_z"])
    home = snapshot(sim)
    call("move_to", x=float(obj["pos"][0]), y=float(obj["pos"][1]), z=top + 0.15)
    yaw = _pick_yaw(float(obj.get("yaw_deg", 0.0)), spec["period"], 0.0)
    call("rotate_z", angle_deg=yaw)
    if arm == "force":
        call("move_until_contact", direction=[0, 0, -1], max_dist=0.2, f_threshold=3.0)
        call("move_by", dx=0, dy=0, dz=0.018)
    else:
        call("move_to", x=float(obj["pos"][0]), y=float(obj["pos"][1]), z=top + 0.02)
    call("grasp")
    if arm == "force":
        call("lift", height=0.10, hold_time=1.0)
    else:
        call("move_by", dx=0, dy=0, dz=0.10, speed=0.02)
        call("hold", seconds=1.0)
    end = snapshot(sim)
    task_state = dict(sim.task_state)
    sim.close()
    return dict(task=task, seed=seed, arm=arm, scene=scene, steps=steps, home=home, end=end,
                truth_episode=task_state)


def parse_seeds(s):
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=list(TASKS))
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--seeds", default="9000-9019")
    ap.add_argument("--repo", default=None, help="path of the ForceHarness clone")
    ap.add_argument("--out", default="runs/dev/e2.jsonl")
    ap.add_argument("--out-dir", default="runs")
    ap.add_argument("--run-name", default="dev/e2")
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w") as f:
        for task in args.tasks:
            for arm in args.arms:
                for seed in parse_seeds(args.seeds):
                    try:
                        rec = run_episode(task, seed, arm, args.out_dir, args.run_name)
                    except Exception as e:
                        rec = dict(task=task, seed=seed, arm=arm, fatal=f"{type(e).__name__}: {e}")
                    f.write(json.dumps(rec, default=float) + "\n")
                    f.flush()
                    n += 1
                    if n % 10 == 0:
                        print(f"  {n} episodes", flush=True)
    print(f"wrote {n} episodes -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
