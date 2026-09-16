#!/usr/bin/env python3
"""AGP-probe 试验驱动器:初始化 episode -> codex(astra) 跑任务 -> 收分。

用法: python3 run_trial.py <task> <trial_idx>
产物: runs/<task>/t<idx>/{prompt.txt,codex.log,result.json,episode/}
"""
import json, os, re, subprocess, sys, time

ROOT = os.path.dirname(os.path.abspath(__file__))

PREAMBLE = """You are a general-purpose agent controlling a simulated robot through a command-line bridge interface. The simulator is `robot.py` in the current working directory.

Interface (each invocation is one bridge call, charged against a shared budget of 250 calls):

- `python3 robot.py state` -- print the full robot/scene state as JSON.
- `python3 robot.py observe` -- render the current scene to a PNG and print its path. (`state` already gives complete information.)
- `python3 robot.py move X Y` -- move the end-effector to table coordinates (x, y) in meters. The table is [0,1]x[0,1].
- `python3 robot.py pick` -- grasp a block at the current end-effector position (tolerance ~1.2cm).
- `python3 robot.py place X Y` -- place the held block at (x, y). Blocks are 5cm cubes; a block lands on top of whatever supports it at (x,y); if its center is not fully over its support it topples.
- `python3 robot.py throw VX VZ` -- throw the held puck with launch velocity (vx, vz) in m/s (x horizontal, z vertical). Gravity is 9.81 m/s^2. After each throw the landing position is reported and the puck returns to your hand.
- `python3 robot.py push F` -- (balance task only) apply force F newtons (|F| <= 10) to the cart for 0.1 seconds.
- `python3 robot.py upload_controller PATH` -- upload a Python file defining `control(state) -> float`. During `run`, the simulator calls it at every integration step (250 Hz) and applies its return value as the cart force (clamped to +-10 N). The state dict passed to it has keys x, v, theta, omega, t (theta in radians, 0 = upright).
- `python3 robot.py run SECONDS` -- advance the simulation by SECONDS seconds.
- `python3 robot.py done` -- declare completion; the episode is scored immediately.

Rules:
- Interact with the robot ONLY through the commands above. Do not read, write, or modify any file inside the episode directory (its path is in the AGP_EP environment variable), and do not modify robot.py. You may create your own scratch files in the current working directory.
- Every bridge call also incurs 0.5 s of decision latency in simulation time (the world is frozen during latency).
- The episode is already initialized. When finished (or to give up), call `python3 robot.py done`. If the budget is exhausted the episode ends automatically.
"""

TASKS = {
    "build": (
        "Task: On a 1m x 1m table there are 3 blocks (5cm cubes) at random positions, plus a goal pad whose position is given in the state. Build a single 3-level tower on the pad: all three blocks stacked on top of each other over the pad, stable (nothing toppled).\n"
        "Success criterion: `done` reports success=true (3 blocks stacked on the pad, none toppled)."
    ),
    "throw": (
        "Task: You are holding a puck at x=0.15 (launch height z=0.10 m). A bowl of radius 4cm sits on the floor (z=0) at bowl_x (see state). Throw the puck so that it lands inside the bowl radius. You may retry as many times as you like within budget.\n"
        "Success criterion: at least one throw lands in the bowl."
    ),
    "wind": (
        "Task: You are holding a puck at x=0.15 (launch height z=0.10 m). A bowl of radius 4cm sits on the floor (z=0) at bowl_x (see state). Throw the puck so that it lands inside the bowl radius. You may retry as many times as you like within budget.\n"
        "Success criterion: at least one throw lands in the bowl."
    ),
    "windx": (
        "Task: You are holding a puck at x=0.15 (launch height z=0.10 m). A bowl of radius 4cm sits on the floor (z=0) at bowl_x (see state). Throw the puck so that it lands inside the bowl radius. You may retry as many times as you like within budget.\n"
        "Success criterion: at least one throw lands in the bowl."
    ),
    "balpush": (
        "Task: A pole (length 0.6 m) is hinged on a cart that moves along x. The pole starts at a small angle from upright. Keep the pole continuously upright -- |theta| < 0.2 rad -- for at least 10 consecutive seconds of simulation time, and never let the cart leave |x| <= 1.4 m. Your only actuation is `push`; this robot does NOT support controller upload.\n"
        "Success criterion: `done` reports success=true."
    ),
    "balance": (
        "Task: A pole (length 0.6 m) is hinged on a cart that moves along x. The pole starts at a small angle from upright. Keep the pole continuously upright -- |theta| < 0.2 rad -- for at least 10 consecutive seconds of simulation time, and never let the cart leave |x| <= 1.4 m.\n"
        "Success criterion: `done` reports success=true."
    ),
}


def main():
    task, idx = sys.argv[1], int(sys.argv[2])
    assert task in TASKS
    trial_dir = os.path.join(ROOT, "runs", task, f"t{idx}")
    ep = os.path.join(trial_dir, "episode")
    os.makedirs(ep, exist_ok=True)
    env = dict(os.environ, AGP_EP=ep, AGP_SEED=str(100 + idx))
    subprocess.run([sys.executable, os.path.join(ROOT, "robot.py"), "init", task],
                   env=env, check=True, capture_output=True)

    prompt = PREAMBLE + "\n" + TASKS[task] + "\n"
    with open(os.path.join(trial_dir, "prompt.txt"), "w") as f:
        f.write(prompt)

    t0 = time.time()
    with open(os.path.join(trial_dir, "codex.log"), "w") as lf:
        proc = subprocess.run(["codex", "exec", "-s", "workspace-write", prompt],
                              cwd=ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT, timeout=3600)
    wall = time.time() - t0

    # 强制收分(agent 可能没调 done)
    done = subprocess.run([sys.executable, os.path.join(ROOT, "robot.py"), "done"],
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
    result = {"task": task, "trial": idx, "seed": 100 + idx, "wall_s": round(wall, 1),
              "codex_rc": proc.returncode, "bridge_calls": n_calls, "score": score}
    with open(os.path.join(trial_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
