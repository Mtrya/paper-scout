#!/usr/bin/env python3
"""AGP-probe robot server: 2D 物理沙盒,模拟 AGP 论文的 agent-robot 接口。

每个子命令是一次"机器人接口调用"(对应 AGP 的 observation/action request)。
状态持久化在 episode 目录的 state.json;每次调用追加 trace.jsonl。

子命令:
  state                 打印机器人/场景状态(JSON)
  observe               渲染当前场景 PNG 到 obs/,打印路径
  move X Y              末端移动到 (x,y)(桌面坐标系,米)
  pick                  抓取当前位置处的方块(容差 1.2cm)
  place X Y             把持有的方块放到 (x,y)(会掉落,做支撑/稳定性检查)
  throw VX VZ           以速度 (vx,vz) 抛出手中物体(侧视弹道)
  push F                给小车施加 0.1 秒的力 F(牛顿,|F|<=10);仅 balance 任务
  upload_controller P   上传控制器 python 文件(def control(state)->float);
                        之后 run 期间每个仿真步(250Hz)调用它
  run SECONDS           推进仿真 SECONDS 秒(稀疏指令期每次额外扣 LATENCY 秒决策延迟)
  done                  宣告完成,立即评分并打印结果
"""
import json, math, os, sys, time

EP = os.environ.get("AGP_EP", os.path.dirname(os.path.abspath(__file__)) + "/episode")
ST = os.path.join(EP, "state.json")
TR = os.path.join(EP, "trace.jsonl")
OBS = os.path.join(EP, "obs")
LATENCY = 0.5          # 每次稀疏指令的决策延迟(秒),模拟 agent 回路时延
DT = 0.004
G = 9.81

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

def render(s):
    os.makedirs(OBS, exist_ok=True)
    from PIL import Image, ImageDraw
    W = 640; img = Image.new("RGB", (W, W), (250, 250, 248)); d = ImageDraw.Draw(img)
    task = s["task"]
    if task in ("build", "throw", "wind", "windx"):
        # 俯视:桌子 [0,1]x[0,1] -> 像素
        sc = W / 1.1; ox, oy = 0.05 * sc, 0.05 * sc
        def P(x, y): return (ox + x * sc, oy + (1.0 - y) * sc)
        def rect(x0, y0, x1, y1):
            (ax, ay), (bx, by) = P(x0, y0), P(x1, y1)
            return [min(ax, bx), min(ay, by), max(ax, bx), max(ay, by)]
        d.rectangle(rect(0, 0, 1, 1), outline=(60, 60, 60), width=3)
        if task == "build":
            px, py = s["pad"]
            d.rectangle(rect(px - .045, py - .045, px + .045, py + .045), outline=(30, 130, 30), width=3)
            d.text(P(px - .04, py - .05), "GOAL PAD", fill=(30, 130, 30))
        if task in ("throw", "wind", "windx"):
            bx = s["bowl_x"]; r = s["bowl_r"]
            d.ellipse(rect(bx - r, .5 - r, bx + r, .5 + r), outline=(150, 30, 30), width=3)
            d.text(P(bx - .03, .5 - r - .03), "BOWL", fill=(150, 30, 30))
        for i, b in enumerate(s["blocks"]):
            x, y = b["x"], b["y"]; half = b["size"] / 2
            shade = max(60, 230 - int(b.get("z", 0) * 800))
            col = (shade, shade // 2 + 60, 200 - shade // 3)
            d.rectangle(rect(x - half, y - half, x + half, y + half), fill=col, outline=(0, 0, 0))
            d.text(P(x - .01, y + .004), f"b{i} z={b.get('z',0):.2f}", fill=(255, 255, 255))
        if s.get("puck"):
            p = s["puck"]; d.ellipse([P(p["x"] - .012, p["y"] - .012), P(p["x"] + .012, p["y"] + .012)], fill=(230, 120, 0), outline=(0, 0, 0))
        ex, ey = s["ee"]
        d.line([P(ex - .02, ey), P(ex + .02, ey)], fill=(0, 0, 200), width=2)
        d.line([P(ex, ey - .02), P(ex, ey + .02)], fill=(0, 0, 200), width=2)
        d.text(P(ex + .02, ey + .02), "EE", fill=(0, 0, 200))
    else:  # balance 侧视
        sc = W / 3.0
        def P(x, z): return (W / 2 + x * sc, W - 80 - z * sc)
        d.line([P(-1.5, 0), P(1.5, 0)], fill=(60, 60, 60), width=3)
        cx = s["cart_x"]; th = s["theta"]; L = s["pole_L"]
        d.rectangle([P(cx - .1, 0), P(cx + .1, .08)], fill=(80, 80, 160), outline=(0, 0, 0))
        tipx, tipz = cx + L * math.sin(th), .08 + L * math.cos(th)
        d.line([P(cx, .08), P(tipx, tipz)], fill=(200, 60, 40), width=5)
        d.ellipse([P(tipx - .03, tipz - .03), P(tipx + .03, tipz + .03)], fill=(200, 60, 40))
        d.text((10, 10), f"t={s['sim_t']:.2f}s theta={math.degrees(th):.1f}deg x={cx:.2f}", fill=(0, 0, 0))
    n = len([f for f in os.listdir(OBS) if f.endswith(".png")])
    path = os.path.join(OBS, f"obs_{n:03d}.png"); img.save(path)
    return path

# ---------- 任务初始化 ----------

def init(task):
    os.makedirs(EP, exist_ok=True)
    import random
    rnd = random.Random(int(os.environ.get("AGP_SEED", "7")))
    s = {"task": task, "sim_t": 0.0, "budget": 250, "ee": [0.15, 0.15], "holding": None,
         "blocks": [], "puck": None, "score": None}
    if task == "build":
        for i in range(3):
            s["blocks"].append({"x": round(.25 + .5 * rnd.random(), 3), "y": round(.15 + .25 * rnd.random(), 3),
                                "z": 0.0, "size": .05, "stacked_on": None})
        s["pad"] = [0.7, 0.75]
    elif task in ("throw", "wind", "windx"):
        s["blocks"] = []
        s["puck"] = {"x": 0.15, "y": 0.5, "held": True}
        s["bowl_x"] = 0.80; s["bowl_r"] = 0.04
        s["launch_z"] = 0.10
        default_wind = {"wind": 0.6, "windx": 6.0}.get(task, 0.0)
        s["wind_ax"] = float(os.environ.get("AGP_WIND", default_wind))  # 未知恒定风(不打印)
        s["attempts"] = []
        s["holding"] = "puck"
    elif task in ("balance", "balpush"):
        s.update({"cart_x": 0.0, "cart_v": 0.0, "theta": 0.12, "omega": 0.0,
                  "pole_L": 0.6, "pole_m": 0.2, "cart_m": 1.0, "controller": None,
                  "no_controller": task == "balpush" or os.environ.get("AGP_NO_CONTROLLER") == "1",
                  "upright_time": 0.0, "force_limit": 10.0})
    save(s)
    with open(TR, "w") as f: f.write("")
    print(json.dumps({"ok": True, "task": task, "budget": s["budget"]}))

# ---------- 物理 ----------

def step_balance(s, f, dt):
    th, om, x, v = s["theta"], s["omega"], s["cart_x"], s["cart_v"]
    M, m, L = s["cart_m"], s["pole_m"], s["pole_L"] / 2
    st, ct = math.sin(th), math.cos(th)
    tot = M + m
    tmp = (f + m * L * om * om * st) / tot
    thacc = (G * st - ct * tmp) / (L * (4.0 / 3 - m * ct * ct / tot))
    xacc = tmp - m * L * thacc * ct / tot
    s["omega"] += thacc * dt; s["theta"] += s["omega"] * dt
    s["cart_v"] += xacc * dt; s["cart_x"] += s["cart_v"] * dt

def run_sim(s, seconds):
    ctrl = None
    if s.get("controller_path"):
        ns = {}
        exec(compile(open(s["controller_path"]).read(), s["controller_path"], "exec"), ns)
        ctrl = ns.get("control")
    t_end = s["sim_t"] + seconds
    while s["sim_t"] < t_end:
        if s["task"] in ("balance", "balpush"):
            f = 0.0
            if ctrl:
                f = max(-s["force_limit"], min(s["force_limit"],
                        float(ctrl({"x": s["cart_x"], "v": s["cart_v"],
                                    "theta": s["theta"], "omega": s["omega"], "t": s["sim_t"]}))))
            elif s.get("sparse_force"):
                # 稀疏推力的剩余作用时间
                f = s["sparse_force"]["f"] if s["sim_t"] < s["sparse_force"]["until"] else 0.0
            step_balance(s, f, DT)
            if abs(s["theta"]) < 0.2: s["upright_time"] += DT
            else: s["upright_time"] = 0.0
            if abs(s["cart_x"]) > 1.4: s["failed_fell"] = True
        s["sim_t"] += DT
    s.pop("sparse_force", None)
    save(s)

# ---------- 评分 ----------

def score(s):
    t = s["task"]
    if t == "build":
        pad = s["pad"]; on_pad = [b for b in s["blocks"] if abs(b["x"] - pad[0]) < .05 and abs(b["y"] - pad[1]) < .05]
        heights = sorted((b.get("z", 0) for b in on_pad), reverse=True)
        tower3 = len(on_pad) == 3 and len(heights) == 3 and abs(heights[0] - .10) < .012 and abs(heights[1] - .05) < .012 and abs(heights[2]) < .012
        stable = all(not b.get("toppled") for b in s["blocks"])
        return {"task": t, "success": bool(tower3 and stable), "blocks_on_pad": len(on_pad),
                "levels": len([h for h in heights if h > .01]), "sim_t": s["sim_t"]}
    if t in ("throw", "wind", "windx"):
        att = s.get("attempts", [])
        hits = [a for a in att if a["hit"]]
        return {"task": t, "success": bool(hits), "attempts": len(att), "hits": len(hits),
                "best_err": (min((a["err"] for a in att), default=None)), "sim_t": s["sim_t"]}
    if t in ("balance", "balpush"):
        return {"task": t, "success": s.get("upright_time", 0) >= 10.0 and not s.get("failed_fell"),
                "upright_time": round(s.get("upright_time", 0), 2), "used_controller": bool(s.get("controller_path")),
                "fell": bool(s.get("failed_fell")), "sim_t": s["sim_t"]}

def support_z(s, x, y, ignore=None):
    z = 0.0; sup = None
    for b in s["blocks"]:
        if ignore is not None and b is ignore: continue
        half = b["size"] / 2
        if abs(b["x"] - x) <= half and abs(b["y"] - y) <= half:
            top = b.get("z", 0) + b["size"]
            if top > z: z, sup = top, b
    return z, sup

def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    if cmd == "init": init(sys.argv[2]); return
    s = load()
    if cmd == "state":
        out = {k: v for k, v in s.items() if k not in ("controller", "wind_ax", "no_controller")}
        print(json.dumps(out, default=str)[:2000]); return
    if cmd == "observe":
        p = render(s); log(s, "observe", {"path": p})
        print(json.dumps({"image": p, "budget_left": s["budget"]})); return
    if cmd == "move":
        x, y = float(sys.argv[2]), float(sys.argv[3])
        x = min(max(x, 0.0), 1.0); y = min(max(y, 0.0), 1.0)
        s["ee"] = [x, y]; s["sim_t"] += LATENCY; log(s, "move", {"x": x, "y": y}, 1)
        print(json.dumps({"ee": s["ee"]})); return
    if cmd == "pick":
        ex, ey = s["ee"]
        for b in s["blocks"]:
            if abs(b["x"] - ex) < .012 + b["size"] / 2 and abs(b["y"] - ey) < .012 + b["size"] / 2 and s["holding"] is None:
                s["holding"] = s["blocks"].index(b); s["sim_t"] += LATENCY
                log(s, "pick", {"block": s["holding"]})
                print(json.dumps({"holding": s["holding"]})); return
        s["sim_t"] += LATENCY; log(s, "pick", {"fail": "no block here"})
        print(json.dumps({"error": "no block within reach at EE"})); return
    if cmd == "place":
        x, y = float(sys.argv[2]), float(sys.argv[3])
        h = s["holding"]; s["sim_t"] += LATENCY
        if h is None:
            log(s, "place", {"fail": "not holding"}); print(json.dumps({"error": "not holding anything"})); return
        if h == "puck":
            log(s, "place", {"fail": "use throw"}); print(json.dumps({"error": "puck must be thrown, not placed"})); return
        b = s["blocks"][h]; z, sup = support_z(s, x, y, ignore=b)
        b["x"], b["y"] = x, y; b["z"] = z
        # 稳定性:方块中心必须落在支撑面内
        if sup is not None and (abs(sup["x"] - x) > sup["size"] / 2 or abs(sup["y"] - y) > sup["size"] / 2):
            b["toppled"] = True; b["z"] = 0.0; b["x"] = min(max(x + .07, .03), .97)
        s["holding"] = None
        log(s, "place", {"x": x, "y": y, "z": z, "toppled": b.get("toppled", False)})
        print(json.dumps({"placed_z": z, "toppled": b.get("toppled", False)})); return
    if cmd == "throw":
        vx, vz = float(sys.argv[2]), float(sys.argv[3])
        s["sim_t"] += LATENCY
        if s.get("holding") != "puck":
            log(s, "throw", {"fail": "no puck"}); print(json.dumps({"error": "no puck to throw"})); return
        x, z = s["puck"]["x"], s["launch_z"]; wx = s.get("wind_ax", 0.0)
        dt = 0.001; px, pz, vx_, vz_ = x, z, vx, vz
        while pz > 0:
            px += vx_ * dt; pz += vz_ * dt; vz_ -= G * dt; vx_ += wx * dt
        err = abs(px - s["bowl_x"]); hit = err <= s["bowl_r"]
        s["attempts"].append({"vx": vx, "vz": vz, "land_x": round(px, 4), "err": round(err, 4), "hit": hit})
        s["puck"]["x"] = 0.15; s["holding"] = "puck"  # 回收再试
        log(s, "throw", s["attempts"][-1])
        print(json.dumps({"land_x": round(px, 4), "bowl_x": s["bowl_x"], "err": round(err, 4), "hit": hit,
                          "note": "puck retrieved; you may retry"})); return
    if cmd == "upload_controller":
        if s.get("no_controller"):
            print(json.dumps({"error": "controller upload not supported on this robot"})); return
        path = sys.argv[2]
        src = open(path).read()
        ns = {}
        exec(compile(src, path, "exec"), ns)
        if "control" not in ns:
            print(json.dumps({"error": "controller must define control(state)->float"})); return
        s["controller_path"] = os.path.abspath(path); s["sim_t"] += LATENCY
        log(s, "upload_controller", {"path": s["controller_path"]})
        print(json.dumps({"ok": True, "note": "controller active during run"})); return
    if cmd == "push":
        f = max(-10.0, min(10.0, float(sys.argv[2])))
        s["sim_t"] += LATENCY          # 决策延迟先于执行
        s["sparse_force"] = {"f": f, "until": s["sim_t"] + 0.1}
        run_sim(s, 0.1)
        log(s, "push", {"f": f})
        print(json.dumps({"applied_f": f, "theta_deg": round(math.degrees(s["theta"]), 2), "sim_t": round(s["sim_t"], 3)})); return
    if cmd == "run":
        sec = float(sys.argv[2]); run_sim(s, min(sec, 30.0))
        log(s, "run", {"seconds": sec})
        out = {"sim_t": round(s["sim_t"], 3)}
        if s["task"] in ("balance", "balpush"):
            out.update({"theta_deg": round(math.degrees(s["theta"]), 2), "cart_x": round(s["cart_x"], 3),
                        "upright_time": round(s["upright_time"], 2), "fell": bool(s.get("failed_fell"))})
        print(json.dumps(out)); return
    if cmd == "done":
        sc = score(s); s["score"] = sc; log(s, "done", sc, 0)
        print(json.dumps({"FINAL": sc})); return
    print(__doc__)

if __name__ == "__main__":
    main()
