"""相同训练预算下 causal vs bidir: 收敛瞬态的部署代价。

发现:W=A 是双向监督的良性不动点(学生复刻真值动态时, 平滑均值坍缩为滤波均值,
未来不再携带新信息)。8 轮迭代的偏差是收敛瞬态而非稳态。因此公平对照 =
相同 n_iter 下两种监督的部署误差与 ||W-A||。
"""
import numpy as np
from cmd_teacher_mismatch import make_world, controls, distill, deploy_err

if __name__ == "__main__":
    c_tr, _ = controls(32, 4000, seed=1)
    c_te, _ = controls(32, 2000, seed=2)
    print(f"{'ar':>5s} {'iter':>4s} {'||Wc-A||':>8s} {'||Wb-A||':>8s} {'err_c':>8s} {'err_b':>8s} {'d':>7s}")
    for ar in [0.7, 0.8, 0.9]:
        A, B = make_world(d=8, ar=ar, sigma=0.3, seed=10)
        for k in [1, 2, 4, 8, 16, 32]:
            Wc, Uc = distill(A, B, 0.3, c_tr, "causal", n_iter=k, seed=3)
            Wb, Ub = distill(A, B, 0.3, c_tr, "bidir", n_iter=k, seed=3)
            ec = deploy_err(A, B, 0.3, c_te, Wc, Uc, seed=4)
            eb = deploy_err(A, B, 0.3, c_te, Wb, Ub, seed=4)
            print(f"{ar:5.1f} {k:4d} {np.linalg.norm(Wc-A):8.4f} {np.linalg.norm(Wb-A):8.4f} {ec:8.4f} {eb:8.4f} {eb-ec:7.4f}")
