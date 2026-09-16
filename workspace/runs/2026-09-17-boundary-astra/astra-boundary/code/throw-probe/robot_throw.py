#!/usr/bin/env python3
"""投掷-非平稳探针 robot server:few-shot 可辨识性边界(机制②)。

平直投掷(复刻 windx 几何):隐藏未知量 = 恒定风 a_w + 二次阻力 k(2 维未知);
漂移模式下风按随机游走每投一步:N(0, sigma_drift)。估计永远落后于真值。
agent 不知道 wind/k/sigma 的值,只能从落点历史反推。

子命令:
  state          打印状态 JSON(不含隐藏参数;含落点历史)
  throw VX VZ    以 (vx,vz) 抛出,打印落点;puck 自动回收;预算 = 15 投
  done           评分
"""
import json, math, os, random, sys, time

EP = os.environ.get("TH_EP", os.path.dirname(os.path.abspath(__file__)) + "/episode")
ST = os.path.join(EP, "state.json")
TR = os.path.join(EP, "trace.jsonl")
G = 9.81

MODES = {  # mode: (wind_mu, drag_k, drift_sigma)
    "d0": (6.0, 0.05, 0.0),
    "d1": (6.0, 0.05, 0.5),
    "d2": (6.0, 0.05, 1.0),
    "d3": (6.0, 0.05, 2.0),
    "d4": (6.0, 0.05, 4.0),
}

def load():
    with open(ST) as f: return json.load(f)

def save(s):
    with open(ST, "w") as f: json.dump(s, f)

def init(mode, seed):
    os.makedirs(EP, exist_ok=True)
    mu, k, sig = MODES[mode]
    rnd = random.Random(seed)
    wind = mu + rnd.uniform(-0.5, 0.5)
    s = {"mode": mode, "sim_t": 0.0, "budget": 15, "launch_x": 0.15, "launch_z": 0.10,
         "bowl_x": 0.80, "bowl_r": 0.04, "attempts": [],
         "_wind": wind, "_k": k, "_sig": sig, "_rnd_state": rnd.getstate() if False else None,
         "score": None}
    # random 实例不可序列化;用独立文件保存种子推进
    s["_seed"] = seed
    save(s)
    with open(TR, "w") as f: f.write("")
    print(json.dumps({"ok": True, "mode": mode, "budget": s["budget"]}))

def get_rnd(s):
    r = random.Random(s["_seed"] + len(s["attempts"]) * 7919)
    return r

def fly(s, vx, vz, wind):
    k = s["_k"]; x, z = s["launch_x"], s["launch_z"]
    dt = 0.0005
    while z > 0:
        vx += (wind - k * vx * abs(vx)) * dt
        vz += (-G - (k * vz * abs(vz) if vz > 0 else 0.0)) * dt
        x += vx * dt; z += vz * dt
    return x

def score(s):
    att = s["attempts"]; hits = [a for a in att if a["hit"]]
    return {"mode": s["mode"], "success": bool(hits), "attempts": len(att), "hits": len(hits),
            "best_err": min((a["err"] for a in att), default=None)}

def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    if cmd == "init": init(sys.argv[2], int(sys.argv[3])); return
    s = load()
    if cmd == "state":
        out = {k: v for k, v in s.items() if not k.startswith("_")}
        print(json.dumps(out, default=str)[:2000]); return
    if cmd == "throw":
        if s["budget"] <= 0:
            print(json.dumps({"error": "budget exhausted", "final": score(s)})); return
        vx, vz = float(sys.argv[2]), float(sys.argv[3])
        r = get_rnd(s)
        s["_wind"] += r.gauss(0.0, s["_sig"])
        land = fly(s, vx, vz, s["_wind"])
        err = abs(land - s["bowl_x"]); hit = err <= s["bowl_r"]
        rec = {"vx": vx, "vz": vz, "land_x": round(land, 4), "err": round(err, 4), "hit": hit}
        s["attempts"].append(rec); s["budget"] -= 1; s["sim_t"] += 0.5
        with open(TR, "a") as f:
            f.write(json.dumps({"kind": "throw", "payload": rec, "budget_left": s["budget"]}) + "\n")
        save(s)
        print(json.dumps({"land_x": round(land, 4), "err": round(err, 4), "hit": hit,
                          "attempts_left": s["budget"]})); return
    if cmd == "done":
        sc = score(s); s["score"] = sc; save(s)
        with open(TR, "a") as f: f.write(json.dumps({"kind": "done", "payload": sc}) + "\n")
        print(json.dumps({"FINAL": sc})); return
    print(__doc__)

if __name__ == "__main__":
    main()
