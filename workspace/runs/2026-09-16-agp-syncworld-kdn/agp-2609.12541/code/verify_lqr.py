"""验证 balance 沙盒的可镇定性:数值线性化 + scipy LQR + 真实 sim 回放."""
import math
import numpy as np
from scipy.linalg import solve_continuous_are

G = 9.81
DT = 0.004
M, m, L = 1.0, 0.2, 0.3  # cart, pole, half-length


def step(state, f, dt):
    """与 robot.py 的 step_balance 完全一致的积分."""
    th, om, x, v = state
    st, ct = math.sin(th), math.cos(th)
    tot = M + m
    tmp = (f + m * L * om * om * st) / tot
    thacc = (G * st - ct * tmp) / (L * (4.0 / 3 - m * ct * ct / tot))
    xacc = tmp - m * L * thacc * ct / tot
    om += thacc * dt
    th += om * dt
    v += xacc * dt
    x += v * dt
    return (th, om, x, v)


# 数值线性化: state = [theta, omega, x, v]
eps = 1e-6
A = np.zeros((4, 4))
B = np.zeros((4, 1))
for i in range(4):
    for sgn in (1, -1):
        pert = [0.0] * 4
        pert[i] = sgn * eps
        s1 = step(tuple(pert), 0.0, DT)
        col = np.array(s1) / (sgn * eps)
        A[:, i] += col / 2
A = (A - np.eye(4)) / DT  # 离散->连续
for sgn in (1, -1):
    s1 = step((0, 0, 0, 0), sgn * eps, DT)
    B[:, 0] += np.array(s1) / (sgn * eps) / 2
B = B / DT

print("A ="); print(np.round(A, 4))
print("B ="); print(np.round(B, 4))

# 可控性
C = np.hstack([np.linalg.matrix_power(A, i) @ B for i in range(4)])
print("controllability rank:", np.linalg.matrix_rank(C), "/ 4")

# LQR
Q = np.diag([10.0, 1.0, 1.0, 1.0])
R = np.array([[1.0]])
P = solve_continuous_are(A, B, Q, R)
K = np.linalg.inv(R) @ B.T @ P
print("K =", np.round(K, 4))
eig = np.linalg.eigvals(A - B @ K)
print("closed-loop eig max real:", np.round(eig.real.max(), 4))

# 真实 sim 回放: 初始 theta=0.12 (与沙盒一致), 15 秒
state = (0.12, 0.0, 0.0, 0.0)
upright = 0.0
force_limit = 10.0
max_th = 0.0
for i in range(int(15 / DT)):
    f = float(-(K @ np.array(state))[0])
    f = max(-force_limit, min(force_limit, f))
    state = step(state, f, DT)
    max_th = max(max_th, abs(state[0]))
    if abs(state[0]) < 0.2:
        upright += DT
    else:
        upright = 0.0
    if abs(state[2]) > 1.4:
        print(f"FAILED: cart out of bounds at t={i*DT:.2f}s")
        break
print(f"final theta={math.degrees(state[0]):.3f}deg, x={state[2]:.4f}, "
      f"max|theta|={math.degrees(max_th):.2f}deg, trailing upright={upright:.2f}s")
