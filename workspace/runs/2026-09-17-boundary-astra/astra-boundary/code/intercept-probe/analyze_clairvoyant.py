#!/usr/bin/env python3
"""全知(clairvoyant)可行性上界:重放带种子抖动的球轨迹,
EE 在 2cm 格点上每调用一步(曼哈顿), GRAB 半径 3cm,
若存在调用 k 与可达 EE 格点 q 使 |q - ball_k| <= 0.03, 则该回合理论上可捕。
这是上界: 它假设 agent 预知全部未来抖动且每次调用都用于移动(不观测)。
用法: python3 analyze_clairvoyant.py
"""
import json, glob, math, random, sys

LAT = 0.5      # 每调用世界前进秒数
STEP = 0.02    # EE 步进 (m)
GRAB_R = 0.03  # GRAB 半径 (m)
SPEEDS = [0.01, 0.02, 0.04, 0.08, 0.16]

def replay_ball(seed, jitter_deg, speed_idx):
    """按 robot_intercept.py 的 RNG 精确重放每调用后的球位(含初始点)。
    返回 [(k, x, y)], k=0 为 init 状态, k>=1 为第 k 次 advance 之后;球出界即止。"""
    rnd = random.Random(seed)
    v = SPEEDS[speed_idx]
    ang = math.radians(rnd.uniform(-30, 30)); y0 = rnd.uniform(0.35, 0.65)
    x, y = 0.02, round(y0, 3)
    vx, vy = round(v * math.cos(ang), 5), round(v * math.sin(ang), 5)
    traj = [(0, x, y)]
    k = 0
    while 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
        if jitter_deg > 0:
            r = random.Random(seed + k * 7919)  # advance 时 sim_t = k*0.5, int(round(sim_t*2)) = k
            a = math.atan2(vy, vx) + r.gauss(0.0, math.radians(jitter_deg))
            sp = math.hypot(vx, vy)
            vx, vy = round(sp * math.cos(a), 5), round(sp * math.sin(a), 5)
        x += vx * LAT; y += vy * LAT
        k += 1
        traj.append((k, x, y))
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            break
    return traj

def reachable_lattice(moves):
    """moves 次移动后 EE 可达格点: (0.5+0.02i, 0.5+0.02j), |i|+|j|<=moves 且同奇偶, 裁剪到 [0,1]。"""
    pts = []
    mmax = int(round(0.5 / STEP))
    for i in range(-mmax, mmax + 1):
        for j in range(-mmax, mmax + 1):
            if abs(i) + abs(j) <= moves and (moves - abs(i) - abs(j)) % 2 == 0:
                pts.append((0.5 + STEP * i, 0.5 + STEP * j))
    return pts

def clairvoyant_feasible(seed, jitter_deg, speed_idx):
    traj = replay_ball(seed, jitter_deg, speed_idx)
    cache = {}
    best = (None, 1e9)
    for k, bx, by in traj:
        if k == 0: continue
        moves = k - 1  # GRAB 在第 k 调用, EE 来自前 k-1 次移动
        if moves not in cache: cache[moves] = reachable_lattice(moves)
        d = min(math.hypot(qx - bx, qy - by) for qx, qy in cache[moves])
        if d < best[1]: best = (k, d)
        if d <= GRAB_R: return True, k, round(d, 4)
    return False, best[0], round(best[1], 4)

def main():
    print(f"{'cond':6} {'trial':5} {'seed':5} {'astra':6} {'clairo':6} best_k/d")
    agg = {}
    for patt, jitter in [("v*", None)] + [(f"j{j}", float(j)) for j in (5, 15, 30, 60)]:
        for f in sorted(glob.glob(f"runs/{patt}/t*/result.json")):
            r = json.load(open(f))
            tag = r["tag"] if "tag" in r else f.split("/")[1]
            if jitter is None and not tag.startswith("v"): continue
            seed = r["seed"]; succ = r["score"]["success"]
            if jitter is None:
                sp = r["score"]["speed"]; si = SPEEDS.index(sp); jit = 0.0
            else:
                si = 2; jit = jitter  # 抖动档球速固定 4cm/s
            ok, k, d = clairvoyant_feasible(seed, jit, si)
            print(f"{tag:6} t{r['trial']:<4} {seed:<5} {str(succ):6} {str(ok):6} k={k} d={d}")
            a = agg.setdefault(tag, [0, 0, 0, 0])
            a[0] += succ; a[1] += ok; a[2] += (ok and not succ); a[3] += (ok and succ)
    print("\n=== 汇总: cond | astra成功 | 全知可捕 | 可捕但失败(agent失分) | 可捕且成功 ===")
    for t, v in agg.items(): print(t, v)

if __name__ == "__main__":
    main()
