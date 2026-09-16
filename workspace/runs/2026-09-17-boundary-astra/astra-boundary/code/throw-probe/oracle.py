#!/usr/bin/env python3
"""投掷探针 oracle:滑窗模型拟合+二分求解,验证各漂移档可辨识性边界。
策略:每投后用最近 W=4 投网格搜索 (wind,k) 最小二乘拟合,再二分求命中 vx。
"""
import json, math, os, subprocess, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
RB = os.path.join(ROOT, "robot_throw.py")
G = 9.81

def call(ep, *args):
    env = dict(os.environ, TH_EP=ep)
    r = subprocess.run([sys.executable, RB] + list(args), env=env, capture_output=True, text=True)
    return json.loads(r.stdout.strip().splitlines()[-1])

def fly_model(vx, vz, wind, k, x0=0.15, z0=0.10):
    dt = 0.0005; x, z = x0, z0
    while z > 0:
        vx += (wind - k * vx * abs(vx)) * dt
        vz += (-G - (k * vz * abs(vz) if vz > 0 else 0.0)) * dt
        x += vx * dt; z += vz * dt
    return x

def fit(attempts, W=4):
    """网格搜索 (wind,k) 拟合最近 W 投。"""
    hist = attempts[-W:]
    best = None
    for wi in range(-20, 201):
        wind = wi / 10.0
        for ki in range(0, 41):
            k = ki / 100.0
            sse = sum((fly_model(a["vx"], a["vz"], wind, k) - a["land_x"]) ** 2 for a in hist)
            if best is None or sse < best[0]:
                best = (sse, wind, k)
    return best[1], best[2]

def solve_vx(wind, k, vz=0.8, target=0.80):
    lo, hi = 0.5, 8.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if fly_model(mid, vz, wind, k) < target: lo = mid
        else: hi = mid
    return (lo + hi) / 2

def run(mode, seed):
    ep = os.path.join(ROOT, "oracle", mode, f"s{seed}")
    os.makedirs(ep, exist_ok=True)
    call(ep, "init", mode, str(seed))
    attempts = []
    for t in range(15):
        if len(attempts) < 2:
            vx = 4.5 if t == 0 else 3.5
        else:
            wind, k = fit(attempts)
            vx = solve_vx(wind, k)
        r = call(ep, "throw", f"{vx:.4f}", "0.8")
        r["vx"], r["vz"] = vx, 0.8
        attempts.append(r)
        if r.get("hit"):
            break
    return call(ep, "done").get("FINAL")

def main():
    out = {}
    for mode in ("d0", "d1", "d2", "d3", "d4"):
        for seed in (300, 301, 302, 303, 304):
            fin = run(mode, seed)
            out[f"{mode}_s{seed}"] = fin
            print(mode, seed, fin, flush=True)
    with open(os.path.join(ROOT, "oracle", "results.json"), "w") as f:
        json.dump(out, f, indent=2)

if __name__ == "__main__":
    main()
