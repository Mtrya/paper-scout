#!/usr/bin/env python3
"""绳索族探针 astra 驱动器:L0-L3 × N 回合。

用法: python3 run_astra.py <L0|L1|L2|L3> <trial_idx>
环境就绪确认: robot_rope.py 的 CLI 与 oracle 均通过后使用。
"""
import json, os, re, subprocess, sys, time

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(ROOT, ".venv", "bin", "python")

LEVEL_DESC = {
    "L0": "a single rigid rod lying on the table. Task: move the rod so that it matches the goal pose (position and orientation) given in the state.",
    "L1": "an 8-segment chain lying curled on the table. Task: straighten it onto the target line given in the state (the `goal` field defines the line with endpoints `a` and `b`; every chain node must end up close to that line).",
    "L2": "a 24-segment rope lying in an S-like curve. Task: reshape it into the target U-shaped curve shown in the state/observation.",
    "L3": "a rope lying near a fixed vertical peg. Task: wind the rope around the peg at least 0.85 of a full turn. Direction matters (see state).",
    # L3X: 与 L3 完全同 prompt、同环境,仅剥掉 state 里的 winding_now 字段(机制①A/B)
    "L3X": "a rope lying near a fixed vertical peg. Task: wind the rope around the peg at least 0.85 of a full turn. Direction matters (see state).",
}

PROMPT = """You are a general-purpose agent controlling a simulated robot arm through a command-line bridge interface. The simulator is `robot_rope.py` in the current working directory (run it with the bundled interpreter `.venv/bin/python`).

The scene: {level_desc}

Interface (budget {budget} calls; move/grab/release/observe each cost 1, state is free):
- `.venv/bin/python robot_rope.py state` -- print scene state as JSON (effector position, rope node coordinates, goal, budget).
- `.venv/bin/python robot_rope.py observe` -- render a top-down PNG of the scene.
- `.venv/bin/python robot_rope.py move <x> <y>` -- move the effector toward (x, y), at most 5 cm per call; physics settles 0.5 s after each move.
- `.venv/bin/python robot_rope.py grab` -- attach the gripper to the nearest rope node within 2.5 cm.
- `.venv/bin/python robot_rope.py release` -- detach.
- `.venv/bin/python robot_rope.py done` -- end the episode and score.

Rules:
- Interact ONLY through the commands above. Do not read/write files inside the episode directory (path in the ROPE_EP env var), do not modify robot_rope.py or any simulator files. Scratch files in the working directory are allowed.
- The episode is already initialized. When done (or to give up), call `done`.

Success criterion: `done` reports success=true.
"""

def main():
    level, idx = sys.argv[1], int(sys.argv[2])
    assert level in ("L0", "L1", "L2", "L3", "L3X")
    seed = 300 + idx
    trial_dir = os.path.join(ROOT, "runs", level, f"t{idx}")
    ep = os.path.join(trial_dir, "episode")
    os.makedirs(ep, exist_ok=True)
    env = dict(os.environ, ROPE_EP=ep)
    init_level = "L3" if level == "L3X" else level
    if level == "L3X":
        env["ROPE_HIDE_WINDING"] = "1"
    subprocess.run([PY, os.path.join(ROOT, "robot_rope.py"), "init", init_level, str(seed)],
                   env=env, check=True, capture_output=True)
    prompt = PROMPT.format(level_desc=LEVEL_DESC[level], budget=200)
    with open(os.path.join(trial_dir, "prompt.txt"), "w") as f:
        f.write(prompt)
    t0 = time.time()
    with open(os.path.join(trial_dir, "codex.log"), "w") as lf:
        proc = subprocess.run(["codex", "exec", "-s", "workspace-write", prompt],
                              cwd=ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT, timeout=3600)
    wall = time.time() - t0
    done = subprocess.run([PY, os.path.join(ROOT, "robot_rope.py"), "done"],
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
    result = {"tag": level, "trial": idx, "seed": seed, "wall_s": round(wall, 1),
              "codex_rc": proc.returncode, "bridge_calls": n_calls, "score": score}
    with open(os.path.join(trial_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result))

if __name__ == "__main__":
    main()
