#!/usr/bin/env python3
"""拦截-延迟探针 robot server:运动目标 vs 桥接延迟(关键快区间机制)。

球以恒定速度横穿桌面;**每次桥调用(含 state)世界前进 0.5s,不冻结**。
EE 只能通过 4 方向 2cm 步进单元移动(EE 极速 = 4cm/s)。
没有 upload_controller:代码必须过桥,快区间无法在环内闭合。

子命令:
  state        打印状态 JSON(含球的精确位置与速度)
  act <UNIT>   MV_LEFT/MV_RIGHT/MV_FWD/MV_BACK(2cm)/GRAB(球距 EE ≤3cm 则成功)
  observe      渲染 PNG
  done         评分
"""
import json, math, os, random, sys, time

EP = os.environ.get("IC_EP", os.path.dirname(os.path.abspath(__file__)) + "/episode")
ST = os.path.join(EP, "state.json")
TR = os.path.join(EP, "trace.jsonl")
OBS = os.path.join(EP, "obs")
LATENCY = 0.5
STEP = 0.02

def load():
    with open(ST) as f: return json.load(f)

def save(s):
    with open(ST, "w") as f: json.dump(s, f)

def log(s, kind, payload, budget_cost=1):
    s["budget"] -= budget_cost
    rec = {"t_wall": time.time(), "sim_t": round(s["sim_t"], 4), "kind": kind, "payload": payload, "budget_left": s["budget"]}
    with open(TR, "a") as f: f.write(json.dumps(rec) + "\n")
    save(s)
    if s["budget"] <= 0:
        print(json.dumps({"error": "budget exhausted", "final": score(s)}))
        sys.exit(2)

def advance(s, dt):
    """世界前进 dt 秒:球运动(带航向抖动);出界则标记逃逸。"""
    b = s["ball"]
    if s.get("jitter", 0.0) > 0:
        r = random.Random(s["_seed"] + int(round(s["sim_t"] * 2)) * 7919)
        ang = math.atan2(b["vy"], b["vx"]) + r.gauss(0.0, math.radians(s["jitter"]))
        sp = math.hypot(b["vx"], b["vy"])
        b["vx"], b["vy"] = round(sp * math.cos(ang), 5), round(sp * math.sin(ang), 5)
    b["x"] += b["vx"] * dt; b["y"] += b["vy"] * dt
    s["sim_t"] += dt
    if not (0.0 <= b["x"] <= 1.0 and 0.0 <= b["y"] <= 1.0) and not s.get("caught"):
        s["escaped"] = True

def render(s):
    os.makedirs(OBS, exist_ok=True)
    from PIL import Image, ImageDraw
    W = 640; img = Image.new("RGB", (W, W), (250, 250, 248)); d = ImageDraw.Draw(img)
    sc = W / 1.1; ox, oy = 0.05 * sc, 0.05 * sc
    def P(x, y): return (ox + x * sc, oy + (1.0 - y) * sc)
    d.rectangle([P(0, 0), P(1, 1)], outline=(60, 60, 60), width=3)
    b = s["ball"]
    d.ellipse([P(b["x"] - .015, b["y"] - .015), P(b["x"] + .015, b["y"] + .015)], fill=(230, 120, 0), outline=(0, 0, 0))
    d.line([P(b["x"], b["y"]), P(b["x"] + b["vx"], b["y"] + b["vy"])], fill=(230, 180, 120), width=1)
    ex, ey = s["ee"]
    d.line([P(ex - .02, ey), P(ex + .02, ey)], fill=(0, 0, 200), width=2)
    d.line([P(ex, ey - .02), P(ex, ey + .02)], fill=(0, 0, 200), width=2)
    d.text((10, 10), f"t={s['sim_t']:.1f}s ball=({b['x']:.3f},{b['y']:.3f})", fill=(0, 0, 0))
    n = len([f for f in os.listdir(OBS) if f.endswith(".png")])
    path = os.path.join(OBS, f"obs_{n:03d}.png"); img.save(path)
    return path

def init(speed_idx, seed, jitter_deg=0.0):
    os.makedirs(EP, exist_ok=True)
    rnd = random.Random(seed)
    speeds = [0.01, 0.02, 0.04, 0.08, 0.16]
    v = speeds[speed_idx]
    # 球从左边界出发,方向向右,带随机 ±30° 倾角,轨迹穿越中央走廊
    ang = math.radians(rnd.uniform(-30, 30))
    y0 = rnd.uniform(0.35, 0.65)
    s = {"speed": v, "jitter": jitter_deg, "_seed": seed, "sim_t": 0.0, "budget": 250, "ee": [0.5, 0.5], "step": STEP,
         "ball": {"x": 0.02, "y": round(y0, 3), "vx": round(v * math.cos(ang), 5), "vy": round(v * math.sin(ang), 5)},
         "caught": False, "escaped": False, "score": None}
    save(s)
    with open(TR, "w") as f: f.write("")
    print(json.dumps({"ok": True, "speed": v, "budget": s["budget"]}))

def score(s):
    return {"success": bool(s.get("caught")), "escaped": bool(s.get("escaped")),
            "speed": s["speed"], "sim_t": s["sim_t"]}

def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    if cmd == "init": init(int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]) if len(sys.argv) > 4 else 0.0); return
    s = load()
    if cmd == "done":
        sc = score(s); s["score"] = sc
        with open(TR, "a") as f: f.write(json.dumps({"kind": "done", "payload": sc}) + "\n")
        save(s)
        print(json.dumps({"FINAL": sc})); return
    if s.get("caught") or s.get("escaped"):
        print(json.dumps({"episode_over": True, "final": score(s)})); return
    if cmd == "state":
        advance(s, LATENCY); log(s, "state", {}, 1)
        out = {"sim_t": s["sim_t"], "budget": s["budget"], "ee": s["ee"], "ball": s["ball"],
               "units": ["MV_LEFT", "MV_RIGHT", "MV_FWD", "MV_BACK", "GRAB"]}
        print(json.dumps(out)); return
    if cmd == "observe":
        advance(s, LATENCY)
        p = render(s); log(s, "observe", {"path": p})
        print(json.dumps({"image": p, "budget_left": s["budget"]})); return
    if cmd == "act":
        unit = sys.argv[2]
        advance(s, LATENCY)
        if unit in ("MV_LEFT", "MV_RIGHT", "MV_FWD", "MV_BACK"):
            dx, dy = {"MV_LEFT": (-1, 0), "MV_RIGHT": (1, 0), "MV_FWD": (0, 1), "MV_BACK": (0, -1)}[unit]
            s["ee"][0] = round(min(max(s["ee"][0] + dx * STEP, 0.0), 1.0), 4)
            s["ee"][1] = round(min(max(s["ee"][1] + dy * STEP, 0.0), 1.0), 4)
            log(s, "act", {"unit": unit, "ee": s["ee"]})
            print(json.dumps({"ee": s["ee"], "sim_t": round(s["sim_t"], 2)})); return
        if unit == "GRAB":
            b = s["ball"]; ex, ey = s["ee"]
            dist = ((b["x"] - ex) ** 2 + (b["y"] - ey) ** 2) ** .5
            if dist <= 0.03 and not s.get("escaped"):
                s["caught"] = True
                log(s, "act", {"unit": unit, "grab": "ok", "dist": round(dist, 4)})
                print(json.dumps({"caught": True, "dist": round(dist, 4)})); return
            log(s, "act", {"unit": unit, "grab": "fail", "dist": round(dist, 4)})
            print(json.dumps({"caught": False, "dist": round(dist, 4)})); return
        print(json.dumps({"error": f"unknown unit {unit!r}"})); return
    print(__doc__)

if __name__ == "__main__":
    main()
