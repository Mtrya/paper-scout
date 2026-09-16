#!/usr/bin/env python3
"""Show-Harness 命名×约定探针驱动器:四条件 × N 回合。

用法: python3 run_trial.py <cond A/B/C/D> <trial_idx>
产物: runs/<cond>/t<idx>/{prompt.txt,codex.log,result.json,episode/}
"""
import json, os, re, subprocess, sys, time

ROOT = os.path.dirname(os.path.abspath(__file__))

PREAMBLE = """You are a general-purpose agent controlling a simulated robot arm through a command-line bridge interface. The simulator is `robot_units.py` in the current working directory.

The robot's end-effector (EE) moves on a 1m x 1m table. There is one block (5cm cube) and a circular goal pad (radius 5cm). Your job: pick up the block and place it so its center lands within 4cm of the pad center.

Interface (each invocation is one bridge call, charged against a shared budget of 250 calls; every call also incurs 0.5 s of decision latency in simulation time):

- `python3 robot_units.py state` -- print the scene state as JSON (numeric EE / block / pad coordinates, and the list of valid action units).
- `python3 robot_units.py observe` -- render the current scene to a PNG and print its path. (`state` already gives complete geometric information.)
- `python3 robot_units.py act <UNIT>` -- execute one action unit.
- `python3 robot_units.py done` -- declare completion; the episode is scored immediately.

Rules:
- Interact with the robot ONLY through the commands above. Do not read, write, or modify any file inside the episode directory (its path is in the SH_EP environment variable), and do not modify robot_units.py. You may create your own scratch files in the current working directory.
- The episode is already initialized. When finished (or to give up), call `python3 robot_units.py done`. If the budget is exhausted the episode ends automatically.
"""

CONVENTION_SEMANTIC = """
The robot accepts these action units:
- `MV_LEFT` / `MV_RIGHT` / `MV_FWD` / `MV_BACK`: move the EE by one fixed step (2 cm) in the table's -x / +x / +y / -y direction respectively. If holding the block, it moves with the EE.
- `PICK`: grasp the block if its center is within grasp tolerance of the EE.
- `PLACE`: release the held block at the current EE position.
"""

CONVENTION_ARBITRARY = """
The robot accepts these action units: {units}. Their effects:
{effects}
"""

NO_CONVENTION = """
The robot accepts a fixed set of action units (listed by `state`). What each unit does is NOT documented here. You will have to figure out the mapping from interaction itself.
"""

def effects_text(cond, unit_map):
    eff_desc = {"minus_x": "move EE one fixed step (2 cm) in -x",
                "plus_x": "move EE one fixed step (2 cm) in +x",
                "plus_y": "move EE one fixed step (2 cm) in +y",
                "minus_y": "move EE one fixed step (2 cm) in -y",
                "pick": "grasp the block if within grasp tolerance",
                "place": "release the held block at the current EE position"}
    return "\n".join(f"- `{u}`: {eff_desc[e]}" for e, u in sorted(unit_map.items(), key=lambda kv: kv[1]))

def main():
    cond, idx = sys.argv[1], int(sys.argv[2])
    assert cond in ("A", "B", "C", "D", "E")
    seed = 100 + idx
    trial_dir = os.path.join(ROOT, "runs", cond, f"t{idx}")
    ep = os.path.join(trial_dir, "episode")
    os.makedirs(ep, exist_ok=True)
    env = dict(os.environ, SH_EP=ep)
    subprocess.run([sys.executable, os.path.join(ROOT, "robot_units.py"), "init", cond, str(seed)],
                   env=env, check=True, capture_output=True)
    with open(os.path.join(ep, "state.json")) as f:
        unit_map = json.load(f)["unit_map"]

    if cond == "A":
        mid = CONVENTION_SEMANTIC
    elif cond == "B":
        mid = ("The robot's movement action units are named `MV_LEFT`, `MV_RIGHT`, `MV_FWD`, `MV_BACK`; "
               "it also has `PICK` and `PLACE` units. Their exact effects are NOT documented here.\n")
    elif cond == "C":
        mid = CONVENTION_ARBITRARY.format(
            units=" / ".join(f"`{u}`" for u in sorted(unit_map.values())),
            effects=effects_text(cond, unit_map))
    else:
        mid = NO_CONVENTION
        if cond == "E":
            mid += ("\nIMPORTANT: this robot does NOT report numeric positions anywhere — `state` gives no coordinates. "
                    "The only way to see the scene is the `observe` PNG. You must infer both the action-unit effects "
                    "and the geometry from the rendered images (top-down view: table x rightward, y upward in the image).\n")

    prompt = PREAMBLE + "\n" + mid + "\nSuccess criterion: `done` reports success=true (block center within 4cm of pad center).\n"
    with open(os.path.join(trial_dir, "prompt.txt"), "w") as f:
        f.write(prompt)

    t0 = time.time()
    with open(os.path.join(trial_dir, "codex.log"), "w") as lf:
        proc = subprocess.run(["codex", "exec", "-s", "workspace-write", prompt],
                              cwd=ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT, timeout=3600)
    wall = time.time() - t0

    done = subprocess.run([sys.executable, os.path.join(ROOT, "robot_units.py"), "done"],
                          env=env, capture_output=True, text=True)
    score = None
    try:
        score = json.loads(done.stdout.strip())["FINAL"]
    except Exception:
        m = re.search(r'"FINAL": (\{.*?\})\s*\}?\s*$', done.stdout)
        if m:
            score = json.loads(m.group(1))
    n_calls = 0
    tr = os.path.join(ep, "trace.jsonl")
    if os.path.exists(tr):
        with open(tr) as f:
            n_calls = sum(1 for _ in f)
    result = {"cond": cond, "trial": idx, "seed": seed, "wall_s": round(wall, 1),
              "codex_rc": proc.returncode, "bridge_calls": n_calls,
              "unit_map": unit_map, "score": score}
    with open(os.path.join(trial_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result))

if __name__ == "__main__":
    main()
