#!/usr/bin/env python3
"""投掷-非平稳探针驱动器:5 模式 × N 回合。
用法: python3 run_trial.py <mode d0-d4> <trial_idx>
"""
import json, os, re, subprocess, sys, time

ROOT = os.path.dirname(os.path.abspath(__file__))

PROMPT = """You are a general-purpose agent controlling a simulated robot through a command-line bridge interface. The simulator is `robot_throw.py` in the current working directory.

You are holding a puck at x=0.15 (launch height z=0.10 m). A bowl of radius 4cm sits on the floor (z=0) at bowl_x=0.80. Throw the puck so that it lands inside the bowl radius. Gravity is 9.81 m/s^2. After each throw the landing position is reported and the puck returns to your hand. You have at most 15 throws.

Interface:
- `python3 robot_throw.py state` -- print scene state as JSON (including your full landing history).
- `python3 robot_throw.py throw VX VZ` -- throw with launch velocity (vx, vz) in m/s (x horizontal, z vertical).
- `python3 robot_throw.py done` -- end the episode and score.

Rules:
- Interact ONLY through the commands above. Do not read/write files inside the episode directory (path in the TH_EP env var), do not modify robot_throw.py. Scratch files in the working directory are allowed.
- The physics may not be perfectly ideal. Use your landing history wisely.
- When finished (or out of throws), call `done`.

Success criterion: `done` reports success=true (at least one throw inside the bowl radius).
"""

def main():
    mode, idx = sys.argv[1], int(sys.argv[2])
    assert mode in ("d0", "d1", "d2", "d3", "d4")
    seed = 300 + idx
    trial_dir = os.path.join(ROOT, "runs", mode, f"t{idx}")
    ep = os.path.join(trial_dir, "episode")
    os.makedirs(ep, exist_ok=True)
    env = dict(os.environ, TH_EP=ep)
    subprocess.run([sys.executable, os.path.join(ROOT, "robot_throw.py"), "init", mode, str(seed)],
                   env=env, check=True, capture_output=True)
    with open(os.path.join(trial_dir, "prompt.txt"), "w") as f:
        f.write(PROMPT)
    t0 = time.time()
    with open(os.path.join(trial_dir, "codex.log"), "w") as lf:
        proc = subprocess.run(["codex", "exec", "-s", "workspace-write", PROMPT],
                              cwd=ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT, timeout=3600)
    wall = time.time() - t0
    done = subprocess.run([sys.executable, os.path.join(ROOT, "robot_throw.py"), "done"],
                          env=env, capture_output=True, text=True)
    score = None
    try:
        score = json.loads(done.stdout.strip())["FINAL"]
    except Exception:
        m = re.search(r'"FINAL": (\{.*?\})\s*\}?\s*$', done.stdout)
        if m:
            score = json.loads(m.group(1))
    result = {"mode": mode, "trial": idx, "seed": seed, "wall_s": round(wall, 1),
              "codex_rc": proc.returncode, "score": score}
    with open(os.path.join(trial_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result))

if __name__ == "__main__":
    main()
