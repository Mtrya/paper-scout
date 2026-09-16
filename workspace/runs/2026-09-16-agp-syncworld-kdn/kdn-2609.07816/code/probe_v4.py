"""实验 C v4:drift 组观测 100% 噪声 —— 池化不确定度"饿死"漂移方向的增益。

v3 中 iso 仍凭池化优势整体领先;v4 把漂移组的观测噪声拉满(无可靠观测)。
此时适配漂移需要积累多次噪声观测:iso 的共享 b 被稳定组锚在低位 ->
漂移方向增益过小、永远追不上漂移;diag 的逐通道 p 保持各方向自己的增益。
"""
import json
import numpy as np
from probe_v3 import make_models, D_K, D_V, N_HOT, N_STABLE, DRIFT_EVERY, P_HOT, P_BG, R_RELIABLE, R_NOISY

T = 30000
hot_keys = np.eye(D_K)


def run(seed=0, omega=5e-4):
    rng = np.random.default_rng(seed)
    true_vals = rng.normal(size=(N_HOT, D_V))
    models = make_models(omega, omega)
    curve, drifts = [], []
    for t in range(T):
        if t > 0 and t % DRIFT_EVERY == 0:
            j = N_STABLE + int(rng.integers(D_K - N_STABLE))
            true_vals[j] = rng.normal(size=D_V)
            drifts.append(t)
        u = rng.random()
        if u < P_HOT:
            j = rng.integers(N_HOT)
            k = hot_keys[j]
            v_clean = true_vals[j]
            r = R_RELIABLE if j < N_STABLE else R_NOISY   # drift 组永远噪声
        elif u < P_HOT + P_BG:
            k = rng.normal(size=D_K); k /= np.linalg.norm(k)
            v_clean = rng.normal(size=D_V); r = R_NOISY
        else:
            continue
        v_obs = v_clean + rng.normal(scale=np.sqrt(r) * np.linalg.norm(v_clean) / np.sqrt(D_V), size=D_V)
        for m in models.values():
            m.write(k, v_obs, r)
        if t % 250 == 0:
            rec = {"t": t}
            for n, m in models.items():
                rec[f"{n}_s"] = float(np.mean([(m.read(hot_keys[j]) - true_vals[j]) ** 2 for j in range(N_STABLE)]))
                rec[f"{n}_d"] = float(np.mean([(m.read(hot_keys[j]) - true_vals[j]) ** 2 for j in range(N_STABLE, N_HOT)]))
            curve.append(rec)
    return curve, drifts


if __name__ == "__main__":
    curve, drifts = run()
    with open(__file__.replace(".py", "_curve.json"), "w") as f:
        json.dump({"curve": curve, "drifts": drifts}, f)
    names = ["fixed-0.1", "gated", "kalman-iso", "kalman-diag"]
    print("分组均值(全程 / 仅 drift 后 1000 token 内):")
    for n in names:
        s_all = np.mean([c[f"{n}_s"] for c in curve])
        d_all = np.mean([c[f"{n}_d"] for c in curve])
        d_post = np.mean([c[f"{n}_d"] for c in curve if any(0 < c["t"] - d <= 1000 for d in drifts)])
        print(f"  {n:12s} stable={s_all:.4f}  drift={d_all:.4f}  drift-恢复期={d_post:.4f}")
    print("\n曲线抽样:")
    for c in curve[::8]:
        print(f"t={c['t']:>6}: " + "  ".join(f"{n} s={c[f'{n}_s']:.4f} d={c[f'{n}_d']:.4f}" for n in names))
