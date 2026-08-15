"""CMD 信息集错配探针 v3(线性高斯世界, 闭式监督)。

核心量:
  1. acausal 梯度比 = ||d mu_bidir / d c_{t+1}|| / ||d mu_bidir / d c_t||
     (双向教师监督对未来控制的敏感度; 因果教师 = 0。CMD claim 的直接量化)
  2. 蒸馏后学生映射 W 与真值动态 A 的偏离 ||W - A||
  3. 部署(因果, 只观察 c_<=t)一步误差 vs 噪声底
  4. prefix corruption 变体(监督目标用扰动前缀, 学生输入保持干净)

世界: x_t = A x_{t-1} + B c_t + eps_t; 控制 c_t = sign(0.7 c_{t-1} + w_t)
监督: mu_causal = A x_{t-1} + B c_t
      mu_bidir  = (Q^-1 + A'Q^-1 A)^-1 [Q^-1 (A x_{t-1} + B c_t) + A'Q^-1 (x_{t+1} - B c_{t+1})]
"""
import numpy as np

def make_world(d=8, ar=0.7, sigma=0.3, seed=0):
    r = np.random.default_rng(seed)
    M = r.standard_normal((d, d))
    A = ar * M / np.max(np.abs(np.linalg.eigvals(M)))
    B = r.standard_normal(d) * 0.8
    return A, B

def controls(T, n, seed):
    r = np.random.default_rng(seed)
    w = r.standard_normal((n, T))
    v = np.zeros((n, T)); v[:, 0] = w[:, 0]
    for t in range(1, T):
        v[:, t] = 0.7 * v[:, t-1] + w[:, t]
    return (v > 0).astype(float) * 2 - 1, w

def teacher_means(A, B, sigma, x_prev, x_next, c_t, c_t1, kind):
    """kind: causal | bidir | bidir_corr(前缀扰动 sigma_c)"""
    d = A.shape[0]
    Qinv = np.eye(d) / sigma**2
    if kind == "causal":
        return A @ x_prev.T + np.outer(B, c_t)
    M = Qinv + A.T @ Qinv @ A
    Minv = np.linalg.inv(M)
    out = np.zeros((d, len(c_t)))
    for i in range(len(c_t)):
        out[:, i] = Minv @ (Qinv @ (A @ x_prev[i] + B * c_t[i])
                            + A.T @ Qinv @ (x_next[i] - B * c_t1[i]))
    return out

def acausal_ratio(A, B, sigma):
    """解析: ||d mu_bidir / d c_{t+1}|| / ||d mu_bidir / d c_t||"""
    d = A.shape[0]
    Qinv = np.eye(d) / sigma**2
    Minv = np.linalg.inv(Qinv + A.T @ Qinv @ A)
    g_next = Minv @ (A.T @ Qinv @ B)      # -d mu / d c_{t+1}
    g_cur = Minv @ (Qinv @ B)             # d mu / d c_t
    return np.linalg.norm(g_next) / np.linalg.norm(g_cur)

def distill(A, B, sigma, c_tr, kind, n_iter=8, corr=0.0, seed=0):
    r = np.random.default_rng(seed)
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
            if kind == "causal":
                mu = A @ xhat[:, t-1].T + np.outer(B, c_tr[:, t])
                feat = np.concatenate([xhat[:, t-1], c_tr[:, t, None]], axis=1)
            else:
                p_prev = xhat[:, t-1]
                if corr > 0:
                    p_prev = p_prev + r.standard_normal((n, d)) * corr
                mu = teacher_means(A, B, sigma, p_prev, xhat[:, t+1], c_tr[:, t], c_tr[:, t+1], "bidir")
                feat = np.concatenate([xhat[:, t-1], c_tr[:, t, None]], axis=1)  # 学生输入保持干净
            X.append(feat); Y.append(mu.T)
        X = np.vstack(X); Y = np.vstack(Y)
        theta = np.linalg.lstsq(X, Y, rcond=None)[0]
        W = theta[:d].T; U = theta[d:d+1].T
    return W, U

def deploy_err(A, B, sigma, c_te, W, U, seed):
    r = np.random.default_rng(seed)
    n, T = c_te.shape
    d = A.shape[0]
    x_true = np.zeros((n, T, d)); xhat = np.zeros((n, T, d))
    for t in range(T):
        prev_t = np.zeros((n, d)) if t == 0 else x_true[:, t-1]
        prev_h = np.zeros((n, d)) if t == 0 else xhat[:, t-1]
        x_true[:, t] = (prev_t @ A.T + np.outer(c_te[:, t], B)
                        + r.standard_normal((n, d)) * sigma)
        xhat[:, t] = prev_h @ W.T + c_te[:, t, None] * U.T
    return np.mean((x_true - xhat)**2)

if __name__ == "__main__":
    import time, itertools
    c_tr, _ = controls(32, 4000, seed=1)
    c_te, _ = controls(32, 2000, seed=2)

    print("解析: acausal 梯度比 vs ar (sigma=0.3)")
    for ar in [0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95]:
        A, B = make_world(d=8, ar=ar, sigma=0.3, seed=10)
        print(f"  ar={ar:.2f}  ratio={acausal_ratio(A, B, 0.3):.4f}")

    print("\n蒸馏对照 (ar=0.7, sigma=0.3):")
    A, B = make_world(d=8, ar=0.7, sigma=0.3, seed=10)
    for kind in ["causal", "bidir", "bidir+corr0.3", "bidir+corr0.8"]:
        corr = 0.3 if kind.endswith("corr0.3") else (0.8 if kind.endswith("corr0.8") else 0.0)
        W, U = distill(A, B, 0.3, c_tr, kind.split("+")[0], corr=corr, seed=3)
        err = deploy_err(A, B, 0.3, c_te, W, U, seed=4)
        print(f"  {kind:14s} ||W-A||={np.linalg.norm(W-A):.4f}  deploy_err={err:.4f}  (noise floor {0.3**2:.4f})")

    print("\n部署误差 vs ar (causal vs bidir):")
    print(f"{'ar':>5s} {'err_c':>7s} {'err_b':>7s} {'d':>7s} {'||Wb-A||':>8s}")
    for ar in [0.4, 0.6, 0.7, 0.8, 0.9]:
        A, B = make_world(d=8, ar=ar, sigma=0.3, seed=10)
        Wc, Uc = distill(A, B, 0.3, c_tr, "causal", seed=3)
        Wb, Ub = distill(A, B, 0.3, c_tr, "bidir", seed=3)
        ec = deploy_err(A, B, 0.3, c_te, Wc, Uc, seed=4)
        eb = deploy_err(A, B, 0.3, c_te, Wb, Ub, seed=4)
        print(f"{ar:5.1f} {ec:7.4f} {eb:7.4f} {eb-ec:7.4f} {np.linalg.norm(Wb-A):8.4f}")
