"""实验 C:delta 规则 vs Kalman 增益写入 —— 流式关联记忆玩具探针(v2)。

v1 教训:稠密随机键下各向同性 Kalman 没有方向性置信度可用,输给固定小 beta。
v2 用结构化键(one-hot 热键 + 稠密背景干扰键),让"逐方向置信度"成为决定性变量:
  - 热键被反复可靠观测 -> 该方向应被保护(小增益)
  - 新方向/漂移后的方向 -> 应强写入(大增益)
  - 稠密背景键的噪声写入会污染所有方向,但置信方向应受保护

Arms:
  fixed-beta   DeltaNet 式固定写入强度
  gated-beta   只看当前观测可靠性的门控 beta_t = c/(r_t+c)
  kalman-iso   各向同性 KDN(标量 b_t)
  kalman-diag  对角 KDN(逐通道 p_i,Mobius 更新的对角特例)

指标:热键召回 MSE 随时间曲线(按漂移周期分段),漂移后恢复速度。
"""
import json
import numpy as np

rng = np.random.default_rng(0)

D_K, D_V = 32, 16
T = 24000
N_HOT = D_K       # 每个热键一个基方向
P_HOT = 0.55      # 热键写入概率
P_BG = 0.25       # 稠密背景键写入概率(其余 token 空转)
DRIFT_EVERY = 4000
R_RELIABLE, R_NOISY = 0.01, 1.0
P_NOISY = 0.3     # 热键写入中噪声观测的比例(背景键一律噪声)
OMEGA = 0.01
C_GATE = 0.05

hot_keys = np.eye(D_K)                      # one-hot
true_vals = rng.normal(size=(N_HOT, D_V))


def gen_stream():
    for t in range(T):
        if t > 0 and t % DRIFT_EVERY == 0:
            j = rng.integers(N_HOT)
            true_vals[j] = rng.normal(size=D_V)
            yield ("drift",)
        u = rng.random()
        if u < P_HOT:
            j = rng.integers(N_HOT)
            k = hot_keys[j]
            v_clean = true_vals[j]
            r = R_NOISY if rng.random() < P_NOISY else R_RELIABLE
        elif u < P_HOT + P_BG:
            k = rng.normal(size=D_K)
            k /= np.linalg.norm(k)
            v_clean = rng.normal(size=D_V)
            r = R_NOISY
        else:
            continue
        v_obs = v_clean + rng.normal(scale=np.sqrt(r) * np.linalg.norm(v_clean) / np.sqrt(D_V), size=D_V)
        yield ("tok", k, v_obs, r)


class FixedBeta:
    name = "fixed"
    def __init__(self, beta):
        self.S = np.zeros((D_K, D_V)); self.beta = beta
    def write(self, k, v, r):
        self.S += self.beta * np.outer(k, v - self.S.T @ k)
    def read(self, k):
        return self.S.T @ k


class GatedBeta:
    name = "gated"
    def __init__(self, c):
        self.S = np.zeros((D_K, D_V)); self.c = c
    def write(self, k, v, r):
        beta = self.c / (r + self.c)
        self.S += beta * np.outer(k, v - self.S.T @ k)
    def read(self, k):
        return self.S.T @ k


class KalmanIso:
    name = "kalman-iso"
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
    """对角 KDN 的无转移特例(D=I):逐通道精度累加 + omega 回充。"""
    name = "kalman-diag"
    def __init__(self, omega):
        self.S = np.zeros((D_K, D_V))
        self.p = np.ones(D_K)               # 逐通道不确定度
        self.omega = omega
    def write(self, k, v, r):
        p_hat = self.p + self.omega
        gain = p_hat * k / (r + np.sum(p_hat * k * k))     # 逐通道 Kalman 增益
        self.S += np.outer(gain, v - self.S.T @ k)
        self.p = p_hat / (1 + (k * k / r) * p_hat)         # 逐通道后验
    def read(self, k):
        return self.S.T @ k


def recall_mse(models):
    return {name: float(np.mean([(m.read(hot_keys[j]) - true_vals[j]) ** 2
                                 for j in range(N_HOT)]))
            for name, m in models.items()}


def main():
    models = {"fixed-0.1": FixedBeta(0.1), "fixed-0.5": FixedBeta(0.5),
              "gated": GatedBeta(C_GATE),
              "kalman-iso": KalmanIso(OMEGA), "kalman-diag": KalmanDiag(OMEGA)}
    curve, drifts = [], []
    t = 0
    for ev in gen_stream():
        if ev[0] == "drift":
            drifts.append(t); continue
        _, k, v, r = ev
        for m in models.values():
            m.write(k, v, r)
        t += 1
        if t % 250 == 0:
            rec = recall_mse(models); rec["t"] = t
            curve.append(rec)
    with open(__file__.replace(".py", "_curve.json"), "w") as f:
        json.dump({"curve": curve, "drifts": drifts}, f)
    seg = DRIFT_EVERY // 250
    names = [n for n in curve[0] if n != "t"]
    for i in range(0, len(curve), seg):
        blk = curve[i:i + seg]
        line = " ".join(f"{n}={np.mean([c[n] for c in blk]):.4f}" for n in names)
        print(f"t={blk[0]['t']:>6}-{blk[-1]['t']:>6}: {line}")


if __name__ == "__main__":
    main()
