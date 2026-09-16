"""实验 C v3:异质通道下的 delta 规则 vs Kalman 增益写入。

v1(稠密键+各向同性):Kalman 无方向信息可用,输给固定小 beta。
v2(one-hot 同质通道):iso 池化 32 通道证据,始终略优于 diag —— 同质统计下
  逐通道追踪没有存在的理由。
v3(本文件):异质通道 —— 一半通道可靠且稳定,另一半通道噪声大且频繁漂移。
  逐方向置信度在这里应当决定性:保护稳定通道、强写漂移通道。
  这正是 KDN 论文声称的机制("有些键方向证据充分,有些仍不确定或已过期")。

Arms: fixed-0.1 / gated(仅当前 r) / kalman-iso / kalman-diag(各自最优 omega)。
指标:分组 MSE(stable 组、drift 组)、漂移后的恢复时间。
"""
import json
import numpy as np

D_K, D_V = 32, 16
T = 30000
N_HOT = D_K
N_STABLE = 16                 # 前 16 个通道:可靠+稳定
DRIFT_EVERY = 2000            # drift 组每 2000 token 漂移一次
P_HOT = 0.55
P_BG = 0.25
R_RELIABLE, R_NOISY = 0.01, 1.0
C_GATE = 0.05

hot_keys = np.eye(D_K)


def make_models(omega_iso, omega_diag):
    class FixedBeta:
        def __init__(self, beta):
            self.S = np.zeros((D_K, D_V)); self.beta = beta
        def write(self, k, v, r):
            self.S += self.beta * np.outer(k, v - self.S.T @ k)
        def read(self, k):
            return self.S.T @ k

    class GatedBeta:
        def __init__(self, c):
            self.S = np.zeros((D_K, D_V)); self.c = c
        def write(self, k, v, r):
            beta = self.c / (r + self.c)
            self.S += beta * np.outer(k, v - self.S.T @ k)
        def read(self, k):
            return self.S.T @ k

    class KalmanIso:
        def __init__(self, omega):
            self.S = np.zeros((D_K, D_V)); self.b = 1.0; self.omega = omega
        def write(self, k, v, r):
            b_hat = self.b + self.omega
            beta = b_hat / (r + b_hat)
            self.S += beta * np.outer(k, v - self.S.T @ k)
            self.b = b_hat / (1 + b_hat / r)
        def read(self, k):
            return self.S.T @ k

    class KalmanDiag:
        def __init__(self, omega):
            self.S = np.zeros((D_K, D_V)); self.p = np.ones(D_K); self.omega = omega
        def write(self, k, v, r):
            p_hat = self.p + self.omega
            gain = p_hat * k / (r + np.sum(p_hat * k * k))
            self.S += np.outer(gain, v - self.S.T @ k)
            self.p = p_hat / (1 + (k * k / r) * p_hat)
        def read(self, k):
            return self.S.T @ k

    return {"fixed-0.1": FixedBeta(0.1), "gated": GatedBeta(C_GATE),
            "kalman-iso": KalmanIso(omega_iso), "kalman-diag": KalmanDiag(omega_diag)}


def run(omega_iso, omega_diag, seed=0, record=False):
    rng = np.random.default_rng(seed)
    true_vals = rng.normal(size=(N_HOT, D_V))
    models = make_models(omega_iso, omega_diag)
    curve, drift_events = [], []
    last_drift_t = {}
    recovery = {n: [] for n in models}
    drifted = set()
    for t in range(T):
        if t > 0 and t % DRIFT_EVERY == 0:
            j = N_STABLE + int(rng.integers(D_K - N_STABLE))  # 只漂移 drift 组
            true_vals[j] = rng.normal(size=D_V)
            drift_events.append(t)
            last_drift_t = {n: t for n in models}
            drifted = {j}
        u = rng.random()
        if u < P_HOT:
            j = rng.integers(N_HOT)
            k = hot_keys[j]
            v_clean = true_vals[j]
            # stable 组永远可靠;drift 组 70% 噪声
            r = R_RELIABLE if j < N_STABLE else (R_NOISY if rng.random() < 0.7 else R_RELIABLE)
        elif u < P_HOT + P_BG:
            k = rng.normal(size=D_K); k /= np.linalg.norm(k)
            v_clean = rng.normal(size=D_V); r = R_NOISY
            j = -1
        else:
            continue
        v_obs = v_clean + rng.normal(scale=np.sqrt(r) * np.linalg.norm(v_clean) / np.sqrt(D_V), size=D_V)
        for n, mdl in models.items():
            mdl.write(k, v_obs, r)
        if record and t % 250 == 0:
            ms = {n: float(np.mean([(m.read(hot_keys[j]) - true_vals[j]) ** 2 for j in range(N_STABLE)]))
                  for n, m in models.items()}
            md = {n: float(np.mean([(m.read(hot_keys[j]) - true_vals[j]) ** 2 for j in range(N_STABLE, N_HOT)]))
                  for n, m in models.items()}
            curve.append({"t": t, **{f"{n}_s": ms[n] for n in models},
                          **{f"{n}_d": md[n] for n in models}})
    final = {}
    for n, m in models.items():
        mse_s = float(np.mean([(m.read(hot_keys[j]) - true_vals[j]) ** 2 for j in range(N_STABLE)]))
        mse_d = float(np.mean([(m.read(hot_keys[j]) - true_vals[j]) ** 2 for j in range(N_STABLE, N_HOT)]))
        final[n] = {"stable": mse_s, "drift": mse_d}
    return final, curve, drift_events


if __name__ == "__main__":
    print("== omega 扫描(末端 MSE: stable / drift)==")
    for w in [1e-4, 5e-4, 1e-3, 5e-3]:
        final, _, _ = run(w, w)
        line = "  ".join(f"{n}={v['stable']:.4f}/{v['drift']:.4f}" for n, v in final.items())
        print(f"omega={w:g}: {line}")
    print("\n== 最优配置全长曲线 ==")
    final, curve, drifts = run(5e-4, 5e-4, record=True)
    with open(__file__.replace(".py", "_curve.json"), "w") as f:
        json.dump({"curve": curve, "drifts": drifts, "final": final}, f)
    names = list(final.keys())
    for c in curve[::4]:
        print(f"t={c['t']:>6}: " + "  ".join(f"{n} s={c[f'{n}_s']:.4f} d={c[f'{n}_d']:.4f}" for n in names))
