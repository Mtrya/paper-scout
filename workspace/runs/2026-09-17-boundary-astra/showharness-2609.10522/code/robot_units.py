#!/usr/bin/env python3
"""Show-Harness 命名×约定探针 robot server:2D 桌面,动作只有离散单元。

复刻 Show-Harness (2609.10522) 的 naming×convention 消融:
agent 只能通过裸符号动作单元与机器人交互,解释器提供度量幅度。
条件 A/B 用语义名(MV_LEFT 等),条件 C/D 用任意符号(A-F);
约定是否在 prompt 里说明由 prompt 决定,本文件只管执行。

子命令:
  state            打印状态 JSON(EE/方块/目标垫的数值坐标)
  observe          渲染场景 PNG
  act <UNIT>       执行一个动作单元(移动类 2cm 步进;PICK/PLACE 类)
  done             宣告完成,立即评分
"""
import json, os, sys, time

EP = os.environ.get("SH_EP", os.path.dirname(os.path.abspath(__file__)) + "/episode")
ST = os.path.join(EP, "state.json")
TR = os.path.join(EP, "trace.jsonl")
OBS = os.path.join(EP, "obs")
LATENCY = 0.5

# 效果表:逻辑效果 -> 名称由条件映射决定(见 init 写入的 unit_map)
EFFECTS = ["minus_x", "plus_x", "plus_y", "minus_y", "pick", "place"]

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
    sc = W / 1.1; ox, oy = 0.05 * sc, 0.05 * sc
    def P(x, y): return (ox + x * sc, oy + (1.0 - y) * sc)
    def rect(x0, y0, x1, y1):
        (ax, ay), (bx, by) = P(x0, y0), P(x1, y1)
        return [min(ax, bx), min(ay, by), max(ax, bx), max(ay, by)]
    d.rectangle(rect(0, 0, 1, 1), outline=(60, 60, 60), width=3)
    gx, gy = s["pad"]
    d.ellipse(rect(gx - .05, gy - .05, gx + .05, gy + .05), outline=(30, 130, 30), width=3)
    d.text(P(gx - .045, gy - .06), "GOAL", fill=(30, 130, 30))
    b = s["block"]; half = b["size"] / 2
    d.rectangle(rect(b["x"] - half, b["y"] - half, b["x"] + half, b["y"] + half), fill=(180, 120, 60), outline=(0, 0, 0))
    ex, ey = s["ee"]
    col = (0, 0, 200) if s["holding"] else (200, 0, 0)
    d.line([P(ex - .02, ey), P(ex + .02, ey)], fill=col, width=2)
    d.line([P(ex, ey - .02), P(ex, ey + .02)], fill=col, width=2)
    d.text(P(ex + .02, ey + .02), "EE", fill=col)
    if s["cond"] == "E":
        d.text((10, 10), f"t={s['sim_t']:.1f}s", fill=(0, 0, 0))
    else:
        d.text((10, 10), f"t={s['sim_t']:.1f}s EE=({ex:.3f},{ey:.3f}) holding={s['holding']}", fill=(0, 0, 0))
    n = len([f for f in os.listdir(OBS) if f.endswith(".png")])
    path = os.path.join(OBS, f"obs_{n:03d}.png"); img.save(path)
    return path

def init(cond, seed):
    import random
    os.makedirs(EP, exist_ok=True)
    rnd = random.Random(seed)
    # 布局:EE、方块、目标垫两两至少相隔 0.25m,保证需要多步移动
    def pt():
        return [round(.12 + .76 * rnd.random(), 3), round(.12 + .76 * rnd.random(), 3)]
    while True:
        ee, blk, pad = pt(), pt(), pt()
        def dist(a, b): return ((a[0]-b[0])**2 + (a[1]-b[1])**2) ** .5
        if dist(ee, blk) > .25 and dist(blk, pad) > .25 and dist(ee, pad) > .25:
            break
    if cond in ("A", "B"):
        names = {"minus_x": "MV_LEFT", "plus_x": "MV_RIGHT", "plus_y": "MV_FWD", "minus_y": "MV_BACK",
                 "pick": "PICK", "place": "PLACE"}
    else:
        # 任意符号:字母->效果的指派按种子打乱(同一 seed 下 C/D 配对相同)
        letters = ["A", "B", "C", "D", "E", "F"]
        rnd.shuffle(letters)
        names = dict(zip(EFFECTS, letters))
    s = {"cond": cond, "sim_t": 0.0, "budget": 250, "ee": ee, "holding": False,
         "block": {"x": blk[0], "y": blk[1], "size": .05}, "pad": pad,
         "unit_map": names, "step": 0.02, "score": None}
    save(s)
    with open(TR, "w") as f: f.write("")
    print(json.dumps({"ok": True, "cond": cond, "budget": s["budget"]}))

def score(s):
    b, (gx, gy) = s["block"], s["pad"]
    err = ((b["x"] - gx) ** 2 + (b["y"] - gy) ** 2) ** .5
    return {"cond": s["cond"], "success": bool(err <= 0.04), "final_err": round(err, 4),
            "sim_t": s["sim_t"]}

def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    if cmd == "init": init(sys.argv[2], int(sys.argv[3])); return
    s = load()
    if cmd == "state":
        out = {k: v for k, v in s.items() if k != "unit_map"}
        out["units"] = sorted(s["unit_map"].values())
        if s["cond"] == "E":
            out.pop("ee"); out.pop("block"); out.pop("pad")
            out["note"] = "numeric positions are not available on this robot; use observe"
        print(json.dumps(out, default=str)[:1500]); return
    if cmd == "observe":
        p = render(s); log(s, "observe", {"path": p})
        print(json.dumps({"image": p, "budget_left": s["budget"]})); return
    if cmd == "act":
        unit = sys.argv[2]
        inv = {v: k for k, v in s["unit_map"].items()}
        if unit not in inv:
            print(json.dumps({"error": f"unknown unit {unit!r}; valid units: {sorted(inv)}"})); return
        eff = inv[unit]
        s["sim_t"] += LATENCY
        if eff in ("minus_x", "plus_x", "plus_y", "minus_y"):
            dx, dy = {"minus_x": (-1, 0), "plus_x": (1, 0), "plus_y": (0, 1), "minus_y": (0, -1)}[eff]
            s["ee"][0] = round(min(max(s["ee"][0] + dx * s["step"], 0.0), 1.0), 4)
            s["ee"][1] = round(min(max(s["ee"][1] + dy * s["step"], 0.0), 1.0), 4)
            # 持有时方块随 EE 走
            if s["holding"]:
                s["block"]["x"], s["block"]["y"] = s["ee"]
            log(s, "act", {"unit": unit, "ee": s["ee"]})
            if s["cond"] == "E":
                print(json.dumps({"ok": True, "holding": s["holding"]})); return
            print(json.dumps({"ee": s["ee"], "holding": s["holding"]})); return
        if eff == "pick":
            b = s["block"]; ex, ey = s["ee"]
            if not s["holding"] and abs(b["x"] - ex) <= .02 + b["size"] / 2 and abs(b["y"] - ey) <= .02 + b["size"] / 2:
                s["holding"] = True; s["block"]["x"], s["block"]["y"] = s["ee"]
                log(s, "act", {"unit": unit, "pick": "ok"})
                print(json.dumps({"holding": True})); return
            log(s, "act", {"unit": unit, "pick": "fail"})
            print(json.dumps({"error": "nothing within grasp tolerance here"})); return
        if eff == "place":
            if not s["holding"]:
                log(s, "act", {"unit": unit, "place": "fail"})
                print(json.dumps({"error": "not holding anything"})); return
            s["holding"] = False
            b, (gx, gy) = s["block"], s["pad"]
            err = ((b["x"] - gx) ** 2 + (b["y"] - gy) ** 2) ** .5
            log(s, "act", {"unit": unit, "place_err": round(err, 4)})
            if s["cond"] == "E":
                print(json.dumps({"placed": True})); return
            print(json.dumps({"placed": True, "dist_to_goal": round(err, 4)})); return
    if cmd == "done":
        sc = score(s); s["score"] = sc; log(s, "done", sc, 0)
        print(json.dumps({"FINAL": sc})); return
    print(__doc__)

if __name__ == "__main__":
    main()
