#!/usr/bin/env python3
"""拦截-延迟探针驱动器:5 速度 × N 回合。

用法: python3 run_trial.py <speed_idx 0-4> <trial_idx>
"""
import json, os, re, subprocess, sys, time

ROOT = os.path.dirname(os.path.abspath(__file__))

PROMPT = """You are a general-purpose agent controlling a simulated robot through a command-line bridge interface. The simulator is `robot_intercept.py` in the current working directory.

A ball moves across a 1m x 1m table at CONSTANT velocity (its exact position and velocity are in the state). Your end-effector (EE) starts at the table center. Your job: catch the ball before it leaves the table by bringing the EE within 3cm of the ball and executing GRAB.

CRITICAL TIME MODEL: the world does NOT freeze between your calls. EVERY bridge call — including `state` — advances the world by 0.5 s of decision latency. The ball keeps moving during your thinking.

Interface (each invocation is one bridge call, charged against a shared budget of 250 calls):
- `python3 robot_intercept.py state` -- print scene state as JSON (EE pos, ball pos/vel, budget).
- `python3 robot_intercept.py act MV_LEFT|MV_RIGHT|MV_FWD|MV_BACK` -- move the EE by one fixed 2cm step (-x/+x/+y/-y).
- `python3 robot_intercept.py act GRAB` -- catch succeeds iff the ball is currently within 3cm of the EE.
- `python3 robot_intercept.py observe` -- render the scene to a PNG (state already gives complete information).
- `python3 robot_intercept.py done` -- end the episode and score.

Rules:
- Interact ONLY through the commands above. Do not read/write files inside the episode directory (path in the IC_EP env var), do not modify robot_intercept.py. Scratch files in the working directory are allowed.
- The EE moves at most 2cm per call and every call costs 0.5 s of world time, so the EE's effective top speed is 4 cm/s. There is no controller upload on this robot.
- The episode is already initialized. When done (or to give up), call `done`.

Success criterion: `done` reports success=true (ball caught before escaping).
"""

PROMPT_J = PROMPT.replace(
    "A ball moves across a 1m x 1m table at CONSTANT velocity (its exact position and velocity are in the state).",
    "A ball moves across a 1m x 1m table at constant SPEED, but its HEADING wanders unpredictably over time "
    "(its current position and velocity are in the state; future headings cannot be known in advance).")

def main():
    arg, idx = sys.argv[1], int(sys.argv[2])
    if arg.startswith("j"):
        speed_idx, jitter = 2, float(arg[1:])
        tag = f"j{arg[1:]}"
    else:
        speed_idx, jitter = int(arg), 0.0
        tag = f"v{arg}"
    assert 0 <= speed_idx <= 4
    seed = 200 + idx
    trial_dir = os.path.join(ROOT, "runs", tag, f"t{idx}")
    ep = os.path.join(trial_dir, "episode")
    os.makedirs(ep, exist_ok=True)
    env = dict(os.environ, IC_EP=ep)
    subprocess.run([sys.executable, os.path.join(ROOT, "robot_intercept.py"), "init", str(speed_idx), str(seed), str(jitter)],
                   env=env, check=True, capture_output=True)
    with open(os.path.join(trial_dir, "prompt.txt"), "w") as f:
        f.write(PROMPT_J if jitter > 0 else PROMPT)
    t0 = time.time()
    with open(os.path.join(trial_dir, "codex.log"), "w") as lf:
        proc = subprocess.run(["codex", "exec", "-s", "workspace-write", PROMPT_J if jitter > 0 else PROMPT],
                              cwd=ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT, timeout=3600)
    wall = time.time() - t0
    done = subprocess.run([sys.executable, os.path.join(ROOT, "robot_intercept.py"), "done"],
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
    result = {"tag": tag, "jitter": jitter, "trial": idx, "seed": seed, "wall_s": round(wall, 1),
              "codex_rc": proc.returncode, "bridge_calls": n_calls, "score": score}
    with open(os.path.join(trial_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result))

if __name__ == "__main__":
    main()
