"""轻量残差 RL:REINFORCE-with-baseline + 成功 BC(λ+) + 失败锚定(λ−),
保留论文式(2)(4)的三个结构成分:掩码/界残差、成功克隆、失败向零锚定;
外加 reference dropout(以 0.25 概率把 nominal chunk 置零,防止 actor 照抄)。

算法偏差说明(对照论文):论文用 chunk 级 TD3+BC(actor 最大化 critic),
此处用二值尝试级回报的 REINFORCE 替代 critic 项(λ_Q 项)。尝试(attempt)
为信用分配单元:σ3 的一次插入尝试(≤200 环境步 = ≤40 个 chunk)或 B 臂的
一个完整 episode。基线为标签的滑动平均。
"""

import numpy as np
import torch
import torch.nn as nn

import base as B
import env as E

IN_DIM = E.N_OBS + B.ACT_DIM   # 14 + 30
RES_BOUNDS = np.array([0.2, 0.2, 1.0])   # B_k:速度维 ±0.2(=0.1m/s,足够数 mm 级
                                         # 持续纠正),夹爪维 ±1.0(可翻转开合决策)
RES_MASK = np.array([1.0, 1.0, 1.0])     # M_k:σ3 全三维可纠正
BM = torch.as_tensor(np.tile(RES_BOUNDS * RES_MASK, B.CHUNK), dtype=torch.float32)


class ResidualActor(nn.Module):
    """f_θ(s, Ā) → U ∈ [-1,1]^{C×3}(tanh)。末层零初始化:未训练的 actor
    输出零残差(论文:"Untrained actors output a zero residual"),探索由外加
    高斯噪声提供。"""
    def __init__(self, hid=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(IN_DIM, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, B.ACT_DIM), nn.Tanh(),
        )
        nn.init.zeros_(self.net[-2].weight)
        nn.init.zeros_(self.net[-2].bias)

    def forward(self, x, nominal):
        return self.net(torch.cat([x, nominal], dim=-1))


def combine_chunk(nominal_chunk, u_chunk):
    """式(2):a = clip(nominal + B⊙M⊙u)。nominal/u: [C,3] numpy。"""
    return np.clip(nominal_chunk + (RES_BOUNDS * RES_MASK) * u_chunk, -1.0, 1.0)


class ResidualLearner:
    def __init__(self, seed=0, lr=3e-4, lam_pos=0.3, lam_neg=0.05,
                     ref_dropout=0.25, update_epochs=3):
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.actor = ResidualActor()
        self.opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.lr, self.lam_pos, self.lam_neg = lr, lam_pos, lam_neg
        self.ref_dropout, self.update_epochs = ref_dropout, update_epochs
        self.baseline = None       # 标签滑动平均
        self.buffer = []           # 全部累计 attempt(重训用)
        self.n_updates = 0

    # ---------- 数据 ----------
    def add_attempt(self, attempt):
        """attempt: {"chunks": [(x14, nom30, U30, sigma)], "label": noisy 0/1,
        "label_true": 0/1}"""
        self.buffer.append(attempt)
        y = attempt["label"]
        self.baseline = y if self.baseline is None else \
            0.95 * self.baseline + 0.05 * y

    # ---------- 损失 ----------
    def _loss(self, attempts, advantages=None):
        xs, noms, us, sigs, advs = [], [], [], [], []
        pos_x, pos_n, pos_u = [], [], []
        neg_x, neg_n = [], []
        for i, att in enumerate(attempts):
            adv = (att["label"] - (self.baseline if self.baseline is not None else 0.0)) \
                if advantages is None else advantages[i]
            for (x, nom, u, sig) in att["chunks"]:
                xs.append(x); noms.append(nom); us.append(u); sigs.append(sig)
                advs.append(adv)
                if att["label"] > 0.5:
                    pos_x.append(x); pos_n.append(nom); pos_u.append(u)
                else:
                    neg_x.append(x); neg_n.append(nom)
        X = torch.as_tensor(np.array(xs), dtype=torch.float32)
        N = torch.as_tensor(np.array(noms), dtype=torch.float32)
        U = torch.as_tensor(np.array(us), dtype=torch.float32)
        S = torch.as_tensor(np.array(sigs), dtype=torch.float32).unsqueeze(1)
        A = torch.as_tensor(np.array(advs), dtype=torch.float32)

        drop = torch.as_tensor(self.rng.random(len(X)) < self.ref_dropout,
                               dtype=torch.float32).unsqueeze(1)
        Nin = N * (1 - drop)                     # reference dropout
        f = self.actor(X, Nin)

        logp = -0.5 * (((U - f) / S) ** 2).sum(dim=1)   # 未 clip 前的高斯对数概率
        # 正优势截断:结局由运气主导时,从失败样本(U≈好纠正+倒霉噪声)上排斥
        # 会反向推毁好的纠正;负样本的学习信号由 λ− 锚定项提供(拉回 nominal)。
        A_pos = torch.clamp(A, min=0.0)
        loss_rl = -(A_pos * logp).mean()

        loss = loss_rl
        if pos_x:
            fP = self.actor(torch.as_tensor(np.array(pos_x), dtype=torch.float32),
                            torch.as_tensor(np.array(pos_n), dtype=torch.float32))
            loss = loss + self.lam_pos * nn.functional.mse_loss(
                fP, torch.as_tensor(np.array(pos_u), dtype=torch.float32))
        if neg_x:
            fN = self.actor(torch.as_tensor(np.array(neg_x), dtype=torch.float32),
                            torch.as_tensor(np.array(neg_n), dtype=torch.float32))
            loss = loss + self.lam_neg * (fN ** 2).mean()
        return loss

    # ---------- online 更新 ----------
    def update(self, new_attempts):
        for _ in range(self.update_epochs):
            self.opt.zero_grad()
            loss = self._loss(new_attempts)
            loss.backward()
            self.opt.step()
        self.n_updates += 1
        return float(loss.detach())

    # ---------- success-reweighted retraining(式 III-D)----------
    def retrain(self, rho=0.25, steps=30, batch=256):
        """D̃ = D⁺ ∪ Sample_ρ(D⁻)(attempt 级,标签为 noisy 标签——系统只能看到它),
        全新 actor 在 D̃ 上重训后部署,buffer 重置为 D̃。"""
        pos = [a for a in self.buffer if a["label"] > 0.5]
        neg = [a for a in self.buffer if a["label"] <= 0.5]
        keep_neg = [a for a in neg if self.rng.random() < rho]
        curated = pos + keep_neg
        if not pos or len(curated) < 2:
            return {"retrained": False, "n_pos": len(pos), "n_neg": len(neg)}
        torch.manual_seed(int(self.rng.integers(0, 2**31)))
        fresh = ResidualActor()   # 默认初始化隐藏层 + 零初始化末层(零残差起步)
        opt = torch.optim.Adam(fresh.parameters(), lr=self.lr)
        labels = np.array([a["label"] for a in curated], dtype=np.float32)
        mu = float(labels.mean())
        for _ in range(steps):
            idx = self.rng.integers(0, len(curated), size=min(8, len(curated)))
            batch_atts = [curated[i] for i in idx]
            advs = [a["label"] - mu for a in batch_atts]
            old = self.actor
            self.actor = fresh   # 复用 _loss 的 dropout/子集逻辑
            loss = self._loss(batch_atts, advantages=advs)
            self.actor = old
            opt.zero_grad(); loss.backward(); opt.step()
        self.actor = fresh
        self.opt = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.buffer = curated
        self.baseline = mu
        return {"retrained": True, "n_pos": len(pos),
                "n_neg_kept": len(keep_neg), "n_neg": len(neg)}
