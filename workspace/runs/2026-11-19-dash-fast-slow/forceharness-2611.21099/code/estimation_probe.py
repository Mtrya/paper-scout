"""E1: are the phys2 payload-mass / COM-offset estimates trustworthy on real control trajectories, and where does
the reported sigma stop being a useful confidence?

Runs the official scripted expert (force interface, `lift` primitive) through the official HiddenPhys tasks at
several force-channel conditions, and records every estimate against simulator ground truth at the moment of the
estimate.  No model calls, no code changes to the harness.

Conditions (arms)
  paper        force_noise_std 0.5 N/tick, force_bias_std 0.3 N   -- the paper's HiddenPhys setting
  bias1n       force_noise_std 0.5 N/tick, force_bias_std 1.0 N   -- a worse but honestly declared force channel
               (the estimator's sigma follows force_bias_std, so its sigma grows with the arm)
  undeclared   paper setting *plus* a constant residual in the working pose that the estimator is not told about
               (+0.5 N along the load axis, +0.02 N*m about the transverse axis): the real-arm failure mode where the
               calibration is right at rest but a residual offset appears once the arm is loaded/moved

Tasks: weight_sort, com_sort, weight_sort_ood, com_sort_ood (the shifted distribution: smaller physical gap).

Usage
  python estimation_probe.py --seeds 9000-9019 --out runs/dev/e1.jsonl
  python estimation_probe.py --tasks weight_sort --arms paper --seeds 9000 9001
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

from forceharness.agent.loop import Episode, EpisodeConfig  # noqa: E402
from forceharness.backends.mujoco.env import ForceSim, WrenchSample  # noqa: E402
from forceharness.baselines.expert import ScriptedAgent  # noqa: E402
from forceharness.eval.runner import _make_sim  # noqa: E402
from forceharness.translation import make_injector  # noqa: E402

# the HiddenPhys protocol constants used by check_sim.py and the expert scripts
EP = dict(scene_noise_pos=0.004, scene_noise_yaw_deg=2.0, f_contact=3.0, f_cap=25.0, kp=150.0, force_sign=-1.0)

ARMS = {
    "paper": dict(noise=0.5, bias=0.3, offset_f=(0.0, 0.0, 0.0), offset_t=(0.0, 0.0, 0.0)),
    "bias1n": dict(noise=0.5, bias=1.0, offset_f=(0.0, 0.0, 0.0), offset_t=(0.0, 0.0, 0.0)),
    "undeclared": dict(noise=0.5, bias=0.3, offset_f=(0.0, 0.0, 0.5), offset_t=(0.0, 0.02, 0.0)),
}

TASKS = ["weight_sort", "com_sort", "weight_sort_ood", "com_sort_ood"]


class DriftedSim(ForceSim):
    """ForceSim with a constant residual added to the calibrated wrench, in the working pose only.

    The residual is applied after the simulator's own tare (which uses raw_wrench), so it models a calibration that is
    correct at rest and wrong once the arm is loaded -- the part of a real force channel that the estimator's
    `bias_std` term cannot know about.  It is a property of this probe, not of the harness.
    """

    offset_f = (0.0, 0.0, 0.0)
    offset_t = (0.0, 0.0, 0.0)

    def wrench(self) -> WrenchSample:
        w = super().wrench()
        of = np.asarray(self.offset_f, dtype=float)
        ot = np.asarray(self.offset_t, dtype=float)
        if np.any(of):
            w.net_force = w.net_force + of
            w.net_force_world = w.net_force_world + of
        if np.any(ot):
            w.net_torque = w.net_torque + ot
            w.net_torque_world = w.net_torque_world + ot
        return w


def snapshot(sim) -> dict:
    """Simulator state right after a primitive: what the estimators were looking at."""
    mj, m, d = sim.mujoco, sim.m, sim.d
    sid = int(sim._ft_site)
    out = dict(eef=np.round(sim.eef_pos, 5).tolist(), quat=np.round(sim.eef_quat, 4).tolist(),
               opening=sim.gripper_opening, sim_time=float(sim.sim_time),
               site=np.round(d.site_xpos[sid], 5).tolist(), R_site=np.round(d.site_xmat[sid].reshape(3, 3), 5).tolist())
    for key in ("bar", "cube_main"):
        bid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, key)
        if bid >= 0:
            out[f"{key}_com"] = np.round(d.xipos[bid], 5).tolist()
            out[f"{key}_R"] = np.round(d.xmat[bid].reshape(3, 3), 5).tolist()
    return out


def run_episode(task: str, seed: int, arm: str, out_dir: str, run_name: str) -> dict:
    cfg_kw = ARMS[arm]
    cfg = EpisodeConfig(task=task, model="scripted", fmt="phys2", vision=False, interface="force", seed=seed,
                        run_name=run_name, out_dir=out_dir, max_turns=40, max_tool_calls=80, max_sim_time=300.0,
                        force_noise_std=cfg_kw["noise"], force_bias_std=cfg_kw["bias"], **EP)
    sim = _make_sim(task, cfg, "robosuite")
    if arm == "undeclared":
        sim.__class__ = DriftedSim
        sim.offset_f = cfg_kw["offset_f"]
        sim.offset_t = cfg_kw["offset_t"]
    ep = Episode(sim, ScriptedAgent(task, "force"), cfg)
    injector = make_injector("phys2")
    snaps, renders, orig = [], [], ep.prims.call

    def call(name, **kw):
        rep = orig(name, **kw)
        snaps.append((name, snapshot(sim)))
        try:
            parts = injector.render(rep, {"state": "", "proprio": "minimal", "log": []})
            text = "\n".join(p.get("text", "") for p in parts if isinstance(p, dict))
        except Exception as e:  # rendering is diagnostic here
            text = f"<render error {e}>"
        renders.append(text)
        return rep

    ep.prims.call = call
    rec = ep.run()
    truth_ep = dict(sim.task_state)
    rows = []
    for (name, snap), rep, text in zip(snaps, ep.reports, renders):
        q = {x["name"]: x for x in rep.quantities}
        rows.append(dict(
            primitive=name, termination=rep.termination, n_ticks=len(rep.wrench_rows),
            snapshot=snap, truth=rep.truth, derived=rep.derived,
            payload_mass=(q.get("payload_mass") or {}).get("value"),
            payload_sigma=(q.get("payload_mass") or {}).get("sigma"),
            payload_n=(q.get("payload_mass") or {}).get("n"),
            payload_flags=(q.get("payload_mass") or {}).get("flags"),
            payload_extra=(q.get("payload_mass") or {}).get("extra"),
            com_r_cm=(q.get("com_offset") or {}).get("value"),
            com_sigma_cm=(q.get("com_offset") or {}).get("sigma"),
            com_n=(q.get("com_offset") or {}).get("n"),
            com_flags=(q.get("com_offset") or {}).get("flags"),
            com_extra=(q.get("com_offset") or {}).get("extra"),
            render=text,
        ))
    sim.close()
    return dict(task=task, seed=seed, arm=arm, success=rec["success"], verdict=rec["verdict"],
                verdict_correct=rec["verdict_correct"], stop=rec["stop_reason"], error=rec["error"],
                n_tool_calls=rec["n_tool_calls"], truth_episode=truth_ep, rows=rows)


def parse_seeds(s):
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=TASKS)
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--seeds", default="9000-9019")
    ap.add_argument("--repo", default=None, help="path of the ForceHarness clone")
    ap.add_argument("--out", default="runs/dev/e1.jsonl")
    ap.add_argument("--out-dir", default="runs")
    ap.add_argument("--run-name", default="dev/e1")
    args = ap.parse_args()
    seeds = parse_seeds(args.seeds)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w") as f:
        for task in args.tasks:
            for arm in args.arms:
                for seed in seeds:
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
