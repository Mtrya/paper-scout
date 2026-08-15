"""CMD prefix 消融: 教师前缀用真值序列(base CMD 风格) vs 学生 rollout(prefix scoring)。

在线性高斯世界里, 因果教师的条件均值 = A x_{t-1} + B c_t, 前缀来源只影响拟合样本的
输入分布。两种变体:
  - base-style: 拟合样本前缀 = 真值 x(教师训练见过的分布), 部署时学生见自己的前缀
  - prefix-style: 拟合样本前缀 = 学生 rollout(与部署一致)
预测: base-style 学生部署误差 > prefix-style(输入分布漂移)。
"""
import numpy as np
from cmd_teacher_mismatch import make_world, controls, deploy_err

def distill_prefix(A, B, sigma, c_tr, x_tr, style, n_iter=8, seed=0):
    n, T = c_tr.shape
    d = A.shape[0]
    W = np.zeros((d, d)); U = np.zeros((d, 1))
    xhat = np.zeros((n, T, d))
    for it in range(n_iter):
        for t in range(T):
            prev = np.zeros((n, d)) if t == 0 else xhat[:, t-1]
            xhat[:, t] = prev @ W.T + c_tr[:, t, None] * U.T
        X, Y = [], []
        for t in range(1, T-1):
            mu = A @ xhat[:, t-1].T + np.outer(B, c_tr[:, t])
            if style == "base":
                feat = np.concatenate([x_tr[:, t-1], c_tr[:, t, None]], axis=1)
            else:
                feat = np.concatenate([xhat[:, t-1], c_tr[:, t, None]], axis=1)
            X.append(feat); Y.append(mu.T)
        X = np.vstack(X); Y = np.vstack(Y)
        theta = np.linalg.lstsq(X, Y, rcond=None)[0]
        W = theta[:d].T; U = theta[d:d+1].T
    return W, U

def gen_true(A, B, sigma, c, seed):
    r = np.random.default_rng(seed)
    n, T = c.shape
    d = A.shape[0]
    x = np.zeros((n, T, d)); x[:, 0] = np.outer(c[:, 0], B) + r.standard_normal((n, d)) * sigma
    for t in range(1, T):
        x[:, t] = x[:, t-1] @ A.T + np.outer(c[:, t], B) + r.standard_normal((n, d)) * sigma
    return x

if __name__ == "__main__":
    import itertools
    c_tr, _ = controls(32, 4000, seed=1)
    c_te, _ = controls(32, 2000, seed=2)
    print(f"{'ar':>5s} {'base-style':>11s} {'prefix-style':>13s} {'d':>7s}")
    for ar in [0.4, 0.6, 0.7, 0.8, 0.9]:
        A, B = make_world(d=8, ar=ar, sigma=0.3, seed=10)
        x_tr = gen_true(A, B, 0.3, c_tr, seed=6)
        Wb, Ub = distill_prefix(A, B, 0.3, c_tr, x_tr, "base", seed=3)
        Wp, Up = distill_prefix(A, B, 0.3, c_tr, x_tr, "prefix", seed=3)
        eb = deploy_err(A, B, 0.3, c_te, Wb, Ub, seed=4)
        ep = deploy_err(A, B, 0.3, c_te, Wp, Up, seed=4)
        print(f"{ar:5.1f} {eb:11.4f} {ep:13.4f} {eb-ep:7.4f}")
