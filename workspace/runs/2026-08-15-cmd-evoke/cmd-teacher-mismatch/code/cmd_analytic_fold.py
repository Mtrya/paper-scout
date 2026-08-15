"""解析推导 v2:双向教师监督把 acausal 依赖折叠进学生映射 W。

稳态线性化:c_{t+1} = alpha c_t + w_t, Var(w)=s2=1-alpha^2, w 独立于过去。
学生:xhat_{t+1} = W xhat_t + U c_{t+1}。
  Exc := E[xhat c] = (alpha I - W)^{-1} U alpha
  E[xhat_t c_{t+1}] = E[xhat_{t+1} c_t] = alpha Exc(平稳)
  Sigma = W S W^T + s2 U U^T + W (alpha Exc) U^T + U (alpha Exc)^T W^T  (Lyapunov)
双向教师条件均值(用学生自己的未来输出代入):
  mu_b = M^{-1}[ Q^{-1}(A x + B c_t) + A^T Q^{-1}(x_{t+1} - B c_{t+1}) ]
  x_{t+1} = W^2 x + W U c_t + U c_{t+1};投影后 c_{t+1} -> alpha c_t。
  mu_b = P x + Qc c_t
  P   = M^{-1} Q^{-1} A + M^{-1} A^T Q^{-1} W^2
  Qc  = M^{-1} Q^{-1} B + M^{-1} A^T Q^{-1} (W U - B) + M^{-1} A^T Q^{-1} (U - B) alpha
最小二乘投影解:theta = E[f f^T]^{-1} E[f mu^T], f=[x; c_t]
  W_star = (Sigma - Exc Exc^T)^{-1} (Sigma P^T + Exc Qc^T - Exc (Exc^T P^T + Qc^T))
与迭代蒸馏数值解对照。
"""
import numpy as np
from cmd_teacher_mismatch import make_world, controls, distill

def alpha_estimate(seed=99, n=200000, T=20):
    r = np.random.default_rng(seed)
    w = r.standard_normal((n, T)); v = np.zeros((n, T)); v[:, 0] = w[:, 0]
    for t in range(1, T):
        v[:, t] = 0.7 * v[:, t-1] + w[:, t]
    c = (v > 0).astype(float) * 2 - 1
    e1 = c[:, 1:][c[:, :-1] == 1].mean()
    em = c[:, 1:][c[:, :-1] == -1].mean()
    alpha = (e1 - em) / 2
    return alpha, 1 - alpha**2

def analytic_W(A, B, sigma, W, U, alpha, s2):
    d = A.shape[0]
    Qinv = np.eye(d) / sigma**2
    M = Qinv + A.T @ Qinv @ A
    Minv = np.linalg.inv(M)
    Uv = U.ravel()
    Bv = B.ravel()
    try:
        Exc = np.linalg.solve(alpha * np.eye(d) - W, Uv * alpha).reshape(-1, 1)
    except np.linalg.LinAlgError:
        Exc = np.zeros((d, 1))
    rhs = s2 * np.outer(Uv, Uv) + alpha * (W @ Exc @ U.T + U @ Exc.T @ W.T)
    try:
        S = np.linalg.solve(np.eye(d*d) - np.kron(W, W), rhs.ravel()).reshape(d, d)
    except np.linalg.LinAlgError:
        S = np.eye(d) * 0.1
    P = Minv @ Qinv @ A + Minv @ A.T @ Qinv @ (W @ W)
    Qc = (Minv @ Qinv @ Bv
          + Minv @ A.T @ Qinv @ (W @ Uv - Bv)
          + Minv @ A.T @ Qinv @ (Uv - Bv) * alpha).reshape(-1, 1)
    Emu1 = S @ P.T + Exc @ Qc.T
    Emu2 = Exc.T @ P.T + Qc.T
    Finv = np.linalg.inv(S - Exc @ Exc.T)
    W_star = Finv @ (Emu1 - Exc @ Emu2)
    return W_star.T

if __name__ == "__main__":
    alpha, s2 = alpha_estimate()
    print(f"alpha={alpha:.4f} s2={s2:.4f}")
    for ar in [0.6, 0.7, 0.8, 0.9]:
        A, B = make_world(d=8, ar=ar, sigma=0.3, seed=10)
        c_tr, _ = controls(32, 4000, seed=1)
        W, U = distill(A, B, 0.3, c_tr, "bidir", seed=3)
        Ws = analytic_W(A, B, 0.3, W, U, alpha, s2)
        print(f"ar={ar}: numerical ||W-A||={np.linalg.norm(W-A):.4f}  analytic ||W*-A||={np.linalg.norm(Ws-A):.4f}  |W-W*|={np.linalg.norm(W-Ws):.4f}")

def sample_projection(A, B, sigma, W, U, alpha, s2, n=200000, T=40, seed=7):
    """固定 (W,U) 下, 用大量样本估计线性投影解, 与解析式对照。"""
    r = np.random.default_rng(seed)
    d = A.shape[0]
    Qinv = np.eye(d) / sigma**2
    M = Qinv + A.T @ Qinv @ A
    Minv = np.linalg.inv(M)
    # 用真实 sign 过程生成 c 与 xhat
    w = r.standard_normal((n, T)); v = np.zeros((n, T)); v[:, 0] = w[:, 0]
    for t in range(1, T):
        v[:, t] = 0.7 * v[:, t-1] + w[:, t]
    c = (v > 0).astype(float) * 2 - 1
    xhat = np.zeros((n, T, d))
    for t in range(1, T):
        xhat[:, t] = xhat[:, t-1] @ W.T + c[:, t, None] @ U.T
    X, Y = [], []
    for t in range(1, T-1):
        xp, xn = xhat[:, t-1], xhat[:, t+1]
        ct, cn = c[:, t], c[:, t+1]
        mu = np.zeros((n, d))
        for i in range(n):
            mu[i] = Minv @ (Qinv @ (A @ xp[i] + B * ct[i])
                            + A.T @ Qinv @ (xn[i] - B * cn[i]))
        X.append(np.concatenate([xp, ct[:, None]], axis=1)); Y.append(mu)
    X = np.vstack(X); Y = np.vstack(Y)
    theta = np.linalg.lstsq(X, Y, rcond=None)[0]
    return theta[:d].T

if __name__ == "__main__":
    import numpy as np
    alpha, s2 = alpha_estimate()
    for ar in [0.7, 0.9]:
        A, B = make_world(d=8, ar=ar, sigma=0.3, seed=10)
        c_tr, _ = controls(32, 4000, seed=1)
        W, U = distill(A, B, 0.3, c_tr, "bidir", seed=3)
        Ws = analytic_W(A, B, 0.3, W, U, alpha, s2)
        Wsp = sample_projection(A, B, 0.3, W, U, alpha, s2)
        print(f"ar={ar}: analytic-vs-sample |W*-Wsp|={np.linalg.norm(Ws-Wsp):.4f}  iterative-vs-sample |W-Wsp|={np.linalg.norm(W-Wsp):.4f}")

def distill_iters(A, B, sigma, c_tr, n_iter, seed=0):
    from cmd_teacher_mismatch import distill
    return distill(A, B, sigma, c_tr, "bidir", n_iter=n_iter, seed=seed)

if __name__ == "__main__":
    import numpy as np
    from cmd_teacher_mismatch import make_world, controls
    alpha, s2 = alpha_estimate()
    A, B = make_world(d=8, ar=0.8, sigma=0.3, seed=10)
    c_tr, _ = controls(32, 4000, seed=1)
    Ws = analytic_W(A, B, 0.3, *distill(A, B, 0.3, c_tr, "bidir", n_iter=8, seed=3), alpha, s2)
    print("iter convergence at ar=0.8:")
    for k in [2, 4, 8, 16, 32]:
        W, U = distill_iters(A, B, 0.3, c_tr, k, seed=3)
        print(f"  n_iter={k:2d}: ||W-A||={np.linalg.norm(W-A):.4f}  ||W-Wsp||={np.linalg.norm(W-Ws):.4f}")
