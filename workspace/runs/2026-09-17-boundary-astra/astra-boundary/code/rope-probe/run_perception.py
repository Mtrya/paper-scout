#!/usr/bin/env python3
"""绳索形态感知探针:数值 state 通道 vs 渲染图通道。
每个 case × channel 一次 codex 单发问答,机械判分。
用法: python3 run_perception.py <case_name> <numeric|image>
"""
import json, os, re, subprocess, sys, time

ROOT = os.path.dirname(os.path.abspath(__file__))
CASES = os.path.join(ROOT, "perception", "cases")

Q = """Answer three questions about the rope in the scene:
Q1: Is the rope wound around the peg (at least one full turn)? ("yes" or "no")
Q2: How many times does the rope cross over itself? (an integer: 0, 1, 2, ...)
Q3: Which free end of the rope is closer to the peg — end "A" (the first node in the node list / node order) or "B" (the last node)?

Reply with ONLY a JSON object on one line: {"q1": "...", "q2": ..., "q3": "..."}"""

PROMPT_NUM = """You are given the numeric state of a rope-on-table scene (JSON below). The rope is a polyline of nodes in `nodes_xy` (ordered from end A to end B, coordinates in metres on a 1m x 1m table). The peg is a small vertical post at `peg`.

{state}

""" + Q + "\n\n(For Q3, end A = nodes_xy[0], end B = nodes_xy[-1].)"

Q_IMG = """Answer two questions about the rope in the scene:
Q1: Is the rope wound around the peg (at least one full turn)? ("yes" or "no")
Q2: How many times does the rope cross over itself? (an integer: 0, 1, 2, ...)

Reply with ONLY a JSON object on one line: {"q1": "...", "q2": ...}"""

PROMPT_IMG = """You are given a top-down rendered image of a rope-on-table scene at `{img}` (384x384 px; the rope is blue, the peg is the red dot, the table border is black). Read the image file and inspect it visually.

""" + Q_IMG

# Q1 操作化变体:判据与 cases.json 的 |winding| >= 0.5 阈值严格一致
Q1OP = """Answer one question about the rope in the scene:
Q1: Does the rope wrap at least HALF a full turn (>= 180 degrees) around the peg? Picture standing on the peg and walking along the rope from one end to the other: if your facing direction has rotated by 180 degrees or more (in either direction) because of the rope curving around the peg, answer "yes", otherwise "no". Note: a full closed loop is 360 degrees; a rope that merely passes near the peg without curving around it is near 0 degrees.

Reply with ONLY a JSON object on one line: {"q1": "..."}"""

PROMPT_NUM_Q1OP = """You are given the numeric state of a rope-on-table scene (JSON below). The rope is a polyline of nodes in `nodes_xy` (ordered from end A to end B, coordinates in metres on a 1m x 1m table). The peg is a small vertical post at `peg`.

{state}

""" + Q1OP

PROMPT_IMG_Q1OP = """You are given a top-down rendered image of a rope-on-table scene at `{img}` (384x384 px; the rope is blue, the peg is the red dot, the table border is black). Read the image file and inspect it visually.

""" + Q1OP

def sanitize(state):
    s = json.loads(open(state).read())
    keep = {"nodes_xy": s["nodes_xy"], "peg": s["goal"]["peg"] if "goal" in s and "peg" in s["goal"] else s.get("peg")}
    return json.dumps(keep)

def main():
    case, channel = sys.argv[1], sys.argv[2]
    variant = sys.argv[3] if len(sys.argv) > 3 else ""
    cdir = os.path.join(CASES, case)
    out_dir = os.path.join(ROOT, "perception", "runs", case, channel + ("_" + variant if variant else ""))
    os.makedirs(out_dir, exist_ok=True)
    if variant == "q1op":
        if channel == "numeric":
            prompt = PROMPT_NUM_Q1OP.replace("{state}", sanitize(os.path.join(cdir, "state.json")))
        else:
            prompt = PROMPT_IMG_Q1OP.replace("{img}", os.path.join(cdir, "render.png"))
    elif channel == "numeric":
        prompt = PROMPT_NUM.replace("{state}", sanitize(os.path.join(cdir, "state.json")))
    else:
        prompt = PROMPT_IMG.replace("{img}", os.path.join(cdir, "render.png"))
    with open(os.path.join(out_dir, "prompt.txt"), "w") as f:
        f.write(prompt)
    t0 = time.time()
    with open(os.path.join(out_dir, "codex.log"), "w") as lf:
        proc = subprocess.run(["codex", "exec", "-s", "read-only", prompt],
                              cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT, timeout=1800)
    wall = time.time() - t0
    log = open(os.path.join(out_dir, "codex.log"), errors="ignore").read()
    ans = None
    for m in re.finditer(r'\{"q1"[^}]*\}', log):
        try:
            ans = json.loads(m.group(0))
        except Exception:
            pass
    result = {"case": case, "channel": channel, "wall_s": round(wall, 1),
              "codex_rc": proc.returncode, "answer": ans}
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result))

if __name__ == "__main__":
    main()
