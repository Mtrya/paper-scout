#!/usr/bin/env python3
"""拦截探针 oracle:贪心前置拦截策略,验证各速度档可行性(失败归因的前提)。
每 0.5s(一次调用):瞄准 球当前位置 + 球速 × 预估到达时间;到位则 GRAB。
"""
import json, os, subprocess, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
RB = os.path.join(ROOT, "robot_intercept.py")

def call(ep, *args):
    env = dict(os.environ, IC_EP=ep)
    r = subprocess.run([sys.executable, RB] + list(args), env=env, capture_output=True, text=True)
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:
        return {"raw": r.stdout[-200:], "err": r.stderr[-200:]}

def run(speed_idx, seed, jitter=0.0):
    ep = os.path.join(ROOT, "oracle", f"v{speed_idx}j{jitter:g}", f"s{seed}")
    os.makedirs(ep, exist_ok=True)
    call(ep, "init", str(speed_idx), str(seed), str(jitter))
    for _ in range(250):
        st = call(ep, "state")
        if st.get("episode_over"):
            break
        b, ee = st["ball"], st["ee"]
        # 估计球到 EE 当前位置的呼叫数,前置瞄准
        dist = ((b["x"] - ee[0]) ** 2 + (b["y"] - ee[1]) ** 2) ** .5
        calls_ahead = dist / 0.02
        tx, ty = b["x"] + b["vx"] * calls_ahead * 0.5, b["y"] + b["vy"] * calls_ahead * 0.5
        if dist <= 0.025:
            g = call(ep, "act", "GRAB")
            if g.get("caught"):
                break
            continue
        dx, dy = tx - ee[0], ty - ee[1]
        if abs(dx) >= abs(dy):
            unit = "MV_RIGHT" if dx > 0 else "MV_LEFT"
        else:
            unit = "MV_FWD" if dy > 0 else "MV_BACK"
        call(ep, "act", unit)
    fin = call(ep, "done")
    return fin.get("FINAL")

def main():
    out = {}
    for j in (0.0, 5.0, 15.0, 30.0, 60.0):
        for seed in (200, 201, 202, 203, 204):
            fin = run(2, seed, j)
            out[f"j{j:g}_s{seed}"] = fin
            print(f"j{j:g}", seed, fin, flush=True)
    path = os.path.join(ROOT, "oracle", "results_jitter.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

if __name__ == "__main__":
    main()
