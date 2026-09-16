"""回放实验 A 的平衡/推挡轨迹并绘制策略形态图。

物理参数与 robot.py 完全一致(step_balance 的副本),DT=0.004。
balance: 用日志中提取的 PD 增益从 theta0=0.12 重放 11.5s 连续控制。
balpush: 用 trace.jsonl 记录的 (sim_t, f) 推力序列重放(每推力作用 0.1s,间隔世界冻结)。
"""
import json, math, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

for _f in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
           "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"):
    if os.path.exists(_f):
        font_manager.fontManager.addfont(_f)
plt.rcParams["font.family"] = "Noto Sans CJK SC"

DT = 0.004
G = 9.81
CART_M, POLE_M, POLE_L = 1.0, 0.2, 0.6

def step(st, f, dt=DT):
    th, om, x, v = st
    M, m, L = CART_M, POLE_M, POLE_L / 2
    s, c = math.sin(th), math.cos(th)
    tot = M + m
    tmp = (f + m * L * om * om * s) / tot
    thacc = (G * s - c * tmp) / (L * (4.0 / 3 - m * c * c / tot))
    xacc = tmp - m * L * thacc * c / tot
    om += thacc * dt; th += om * dt
    v += xacc * dt; x += v * dt
    return (th, om, x, v)

# ---- balance: 5 个手写 PD 控制器(codex.log 提取) ----
BALANCE_GAINS = {  # trial: (k_x, k_v, k_th, k_om)
    0: (3.0, 5.0, 60.0, 15.0),
    1: (3.0, 5.0, 45.0, 12.0),
    2: (6.0, 10.0, 70.0, 18.0),
    3: (3.0, 5.0, 60.0, 14.0),
    4: (4.0, 7.0, 60.0, 15.0),
}

def replay_balance(kx, kv, kth, kom, T=11.5, th0=0.12):
    st = (th0, 0.0, 0.0, 0.0)
    ts, ths, xs = [0.0], [th0], [0.0]
    t = 0.0
    while t < T:
        th, om, x, v = st
        f = max(-10.0, min(10.0, kx * x + kv * v + kth * th + kom * om))
        st = step(st, f)
        t += DT
        ts.append(t); ths.append(st[0]); xs.append(st[2])
    return np.array(ts), np.array(ths), np.array(xs)

def replay_balpush(th0=0.12, n=98):
    """重放 shell 循环包装的 PD:f=clip(40θ+9ω+1.5x+3v) 每周期重算,作用 0.1s 后冻结 0.5s。
    (开环回放记录的力序列不可行:浮点步数漂移使记录力与回放状态失配,失稳发散;
     反馈重算才是 agent 实际执行的控制律。)"""
    st = (th0, 0.0, 0.0, 0.0)
    ts, ths, xs = [0.0], [th0], [0.0]
    t = 0.0
    for i in range(n):
        th, om, x, v = st
        f = 3.0 if i == 0 else max(-10.0, min(10.0, 40 * th + 9 * om + 1.5 * x + 3 * v))
        for _ in range(25):
            st = step(st, f)
            t += DT
            ts.append(t); ths.append(st[0]); xs.append(st[2])
        t += 0.5
        ts.append(t); ths.append(st[0]); xs.append(st[2])
    return np.array(ts), np.array(ths), np.array(xs)

# ---- 投掷数据(result.json / trace) ----
def load_json(p):
    with open(p) as fh: return json.load(fh)

def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "../../drafts/agp_plots"
    os.makedirs(out_dir, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150})

    # ============ 图 1:任务总览 ============
    tasks = ["build", "throw", "wind", "windx", "balance", "balpush"]
    labels = ["build\n堆塔×3块", "throw\n无风投掷", "wind\n弱风0.6", "windx\n强风6.0",
              "balance\n可上传控制器", "balpush\n仅稀疏推挡"]
    succ, calls = [], []
    for task in tasks:
        rs = sorted(f"runs/{task}/t{i}/result.json" for i in range(5) if os.path.exists(f"runs/{task}/t{i}/result.json"))
        if not rs:
            rs = sorted(f"runs/{task}/t{i}/result.json" for i in range(3) if os.path.exists(f"runs/{task}/t{i}/result.json"))
        js = [load_json(p) for p in rs]
        succ.append(sum(1 for j in js if j["score"]["success"]) / len(js) * 100)
        calls.append(float(np.median([j["bridge_calls"] for j in js])))
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    x = np.arange(len(tasks))
    bars = ax.bar(x, succ, width=0.55, color=["#4C9F70", "#4C9F70", "#4C9F70", "#E8A13A", "#4C9F70", "#E8A13A"])
    for xi, c in zip(x, calls):
        ax.text(xi, 103, f"{c:.0f} 次桥调用", ha="center", va="bottom", fontsize=9, color="#333333")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("成功率 (%)"); ax.set_ylim(0, 118)
    ax.set_title("gpt-6-astra 经 7 命令桥直接当策略:6 任务共 26 个 trial 全部成功")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{out_dir}/agp_overview.png"); plt.close(fig)

    # ============ 图 2:投掷族(探针-辨识-修正) ============
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.7), sharey=True)
    for ax, task, name in zip(axes, ["throw", "wind", "windx"],
                              ["throw:无风", "wind:风 0.6 m/s²", "windx:风 6.0 m/s²"]):
        for i in range(5):
            p = f"runs/{task}/t{i}/episode/trace.jsonl"
            if not os.path.exists(p): continue
            throws = [json.loads(l)["payload"] for l in open(p) if json.loads(l)["kind"] == "throw"]
            for a, pl in enumerate(throws):
                ax.scatter(a + 1, pl["land_x"], color="#C0392B" if not pl["hit"] else "#2E86C1",
                           zorder=3, s=42 if task != "windx" else 60,
                           marker="x" if not pl["hit"] else "o")
        ax.axhspan(0.8 - 0.04, 0.8 + 0.04, color="#4C9F70", alpha=0.25, lw=0)
        ax.axhline(0.8, color="#4C9F70", ls="--", lw=1)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("第几次投掷"); ax.set_xticks([1, 2])
        ax.set_xlim(0.5, 2.5)
    axes[0].set_ylabel("落点 x (m)")
    for ax in axes: ax.set_ylim(0.70, 0.90)
    axes[2].annotate("第 1 投:无风解析解,被风吹偏 6.7cm", xy=(1, 0.867), xytext=(0.58, 0.878),
                     fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.8))
    axes[2].annotate("第 2 投:单次采样辨识风后解析修正,偏差 0.6mm", xy=(2, 0.7994), xytext=(0.56, 0.725),
                     fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.8))
    fig.suptitle("投掷任务落点(碗中心 0.80m,半径 4cm;多次 trial 落点几乎重合)")
    fig.tight_layout(rect=[0, 0, 1, 0.93]); fig.savefig(f"{out_dir}/agp_throw_family.png"); plt.close(fig)

    # ============ 图 3:平衡族(连续 PD vs 稀疏推挡) ============
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.9))
    ax = axes[0]
    for i, (kx, kv, kth, kom) in BALANCE_GAINS.items():
        ts, ths, xs = replay_balance(kx, kv, kth, kom)
        ax.plot(ts, np.degrees(ths), lw=1.2, label=f"t{i}: θ×{kth:.0f} ω×{kom:.0f}")
    ax.set_title("balance:上传的 5 个手写 PD 控制器\n(250 Hz 连续控制,初扰 $\\theta_0$=0.12 rad)")
    ax.set_xlabel("仿真时间 (s)"); ax.set_ylabel("杆倾角 θ (度)")
    ax.set_xlim(0, 5); ax.legend(fontsize=7, loc="upper right")
    ax.axhline(0, color="k", lw=0.5)

    ax = axes[1]
    ts, ths, xs = replay_balpush()
    ax.plot(ts, np.degrees(ths), lw=1.2, color="#C0392B", label="shell 循环 PD(40θ+9ω+1.5x+3v)")
    ax.set_title("balpush:禁控制器后,agent 用 shell 循环\n包装桥命令重建反馈(占空比 16.7%)")
    ax.set_xlabel("仿真时间 (s)"); ax.set_ylabel("杆倾角 θ (度)")
    ax.set_xlim(0, 20); ax.legend(fontsize=8)
    ax.axhline(0, color="k", lw=0.5)
    fig.tight_layout(); fig.savefig(f"{out_dir}/agp_balance_family.png"); plt.close(fig)

    # 保存重放数据供报告引用
    ts, ths, xs = replay_balance(*BALANCE_GAINS[0])
    idx = np.where(np.abs(ths) < np.radians(0.5))[0]
    print("balance t0 settle(<0.5deg):", round(float(ts[idx[0]]), 2), "s" if len(idx) else "n/a")
    ts, ths, xs = replay_balpush()
    print("balpush replay final theta(deg):", round(float(np.degrees(ths[-1])), 4),
          "final x:", round(float(xs[-1]), 4))
    print("saved to", out_dir)

if __name__ == "__main__":
    main()
