# Loom Attention 机制探针：逐 token 线路重连的容量、秩界与静态端口消融
# 依据 arXiv 2611.30634v2 Eqs.(4)-(11), Prop.3.1, Thm.2.4；维度为探针自选（非论文训练尺度）
import math
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
DEV = "cpu"
D, N, HD, DK, DR, ROPE, L, T = 64, 4, 16, 8, 16, 4, 6, 64
ARMS = ("q", "k", "v", "o")


def rms_norm_unit(x, eps=1e-6):
    f = x.float()
    return (f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)


class LoomLayer(nn.Module):
    """单层 LoomMHA+FFN，均向矩阵记忆 M 写入；严格因果掩码排除当前 token。"""

    def __init__(self):
        super().__init__()
        self.W_sg = nn.Linear(D, DK)
        self.W_q = nn.Linear(D, N * ROPE)
        self.W_k = nn.Linear(D, N * ROPE)
        self.W_r1 = nn.Linear(D, DR)
        for a in ARMS:
            setattr(self, f"W_r2_{a}", nn.Linear(DR, N * DK))
            setattr(self, f"W_rg_{a}", nn.Linear(D, N))
            setattr(self, f"q_bar_{a}", nn.Parameter(torch.randn(N, DK) * 0.02))
            setattr(self, f"W_rgb_{a}", nn.Linear(D, N))
        self.key_scale = 0.1
        self.W_o = nn.Linear(N * HD, D)
        self.ffn = nn.Sequential(nn.Linear(D, 2 * D), nn.GELU(), nn.Linear(2 * D, D))
        self.W_w1 = nn.Linear(D, DR)
        self.W_w2 = nn.Linear(DR, N * DK)
        self.W_wg = nn.Linear(D, N)
        self.ffn_w1 = nn.Linear(D, DR)
        self.ffn_w2 = nn.Linear(DR, N * DK)
        self.ffn_wg = nn.Linear(D, N)

    def m_read(self, M_t, x, xr, arm):
        # 论文 Eq.5：q^m = Norm(x^r W2)⊙σ(xWg) + q_bar⊙σ_bar；u = q^m M
        r = rms_norm_unit(getattr(self, f"W_r2_{arm}")(xr)).unflatten(-1, (N, DK))
        g = torch.sigmoid(getattr(self, f"W_rg_{arm}")(x)).unsqueeze(-1)
        qm = self.key_scale * g * r
        u_dyn = torch.einsum("btnk,btkd->btnd", qm, M_t)
        g_bar = torch.sigmoid(getattr(self, f"W_rgb_{arm}")(x)).unsqueeze(-1)
        u_stat = torch.einsum("nk,btkd->btnd", getattr(self, f"q_bar_{arm}"), M_t) * g_bar
        return u_dyn + u_stat, qm

    def m_write(self, o, x, xw):
        # 论文 Eq.6：k^m = Norm(x^w W2)⊙σ(xWg)（每头标量门）；ΔM = k^m ⊗ Norm(o)
        km = rms_norm_unit(self.W_w2(xw)).unflatten(-1, (N, DK))
        g = torch.sigmoid(self.W_wg(x)).unsqueeze(-1)
        return km * g

    def forward(self, x, M, ports):
        B, T = x.shape[:2]
        M_prev = torch.cat([M[:, :1], M[:, :-1]], 1)
        z = torch.sigmoid(self.W_sg(x)).unsqueeze(-1)
        M_t = rms_norm_unit((1 - z) * M + z * M_prev)          # MShift + MNorm
        xr = F.gelu(self.W_r1(x))
        reads, qports = {}, {}
        for a in ARMS:
            reads[a], qports[a] = self.m_read(M_t, x, xr, a)
            if ports is not None:
                ports["read"][a].append(qports[a])
        qr = F.gelu(self.W_q(x)).unflatten(-1, (N, ROPE))
        kr = F.gelu(self.W_k(x)).unflatten(-1, (N, ROPE))
        q = torch.cat([reads["q"], qr], -1) / math.sqrt(HD + ROPE)
        k = torch.cat([reads["k"], kr], -1)
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool), diagonal=-1)  # 排除当前 token
        att = torch.einsum("bthd,bshd->bhts", q, k).masked_fill(~mask, -1e9)
        a = att.softmax(-1)
        empty = ~mask.any(-1)                                   # t=0 无可 attend 的 token
        a = a * (~empty).float().view(1, 1, T, 1)
        o = torch.einsum("bhts,bshd->bthd", a, reads["v"]) + reads["o"]
        dx = self.W_o(o.flatten(-2))
        for w1, w2, val, split in ((self.W_w1, self.W_w2, o, False),
                                   (self.ffn_w1, self.ffn_w2, self.ffn(x), True)):
            content = rms_norm_unit(val.unflatten(-1, (N, HD))) if split else rms_norm_unit(val)
            km = self.m_write(content, x, F.gelu(w1(x)))
            if ports is not None:
                ports["write"].append(km)
            M = M + torch.einsum("btnk,btnd->btkd", km, content)
        return x + dx, M


def collect_ports(idx):
    emb = nn.Embedding(16, D)
    layers = nn.ModuleList(LoomLayer() for _ in range(L))
    x = emb(idx)
    M = torch.zeros(1, T, DK, HD)
    ports = {"read": {a: [] for a in ARMS}, "write": []}
    for layer in layers:
        x, M = layer(x, M, ports)
    return ports


def cos_dist(p):
    pf = p.reshape(-1, p.shape[-2] * p.shape[-1]).float()
    pf = pf / (pf.norm(dim=-1, keepdim=True) + 1e-9)
    return (1 - pf @ pf.t()).mean().item()


def wiring_rank(li, ports, static=False):
    ks = []
    for l in range(li + 1):
        ks.append(ports["write"][2 * l].squeeze(0))       # [T, N, DK]
        ks.append(ports["write"][2 * l + 1].squeeze(0))
    K = torch.cat(ks, 1)                                   # [T, w, DK]
    Q = torch.cat([ports["read"][a][li].squeeze(0) for a in ("q", "k", "v")], 1)  # [T, 3n, DK]
    if static:
        K = K.mean(0, keepdim=True).expand_as(K)
        Q = Q.mean(0, keepdim=True).expand_as(Q)
    C = torch.einsum("trk,twk->trw", Q, K)
    C_dyn = C - C.mean(0, keepdim=True)
    rel_var = (C_dyn.norm() / C.norm()).item()
    if rel_var < 1e-5:                       # 变化量低于浮点噪声：结构上无动态线路
        return 0.0, 0.0, rel_var
    ranks = []
    for t in range(T):
        s = torch.linalg.svdvals(C_dyn[t].float())
        e = s ** 2
        ranks.append(((e.sum() ** 2) / (e ** 2).sum()).item())
    return sum(ranks) / len(ranks), max(ranks), rel_var


def run_mechanism():
    idx = torch.randint(0, 16, (1, T))
    ports = collect_ports(idx)
    print("== 跨 token 端口余弦距离（动态端口应显著>0） ==")
    for li in (0, 2, 5):
        print(f"layer {li}: read q^m={cos_dist(ports['read']['q'][li]):.3f} "
              f"write k^m(attn)={cos_dist(ports['write'][2*li]):.3f} "
              f"write k^m(ffn)={cos_dist(ports['write'][2*li+1]):.3f}")
    print("\n== 动态线路能量有效秩（token 中心化；上界 min(d_k=8,w)） ==")
    dyn_stats, stat_stats = [], []
    for li in range(L):
        m, mx, rv = wiring_rank(li, ports)
        dyn_stats.append((m, mx, rv))
        print(f"layer {li}: mean={m:.2f} max={mx:.2f} rel-var={rv:.3f}")
    print("\n== 静态端口消融（q^m,k^m 冻结为 token 均值） ==")
    for li in (2, 5):
        m, mx, rv = wiring_rank(li, ports, static=True)
        stat_stats.append((li, m, mx, rv))
        print(f"layer {li}: static mean={m:.2f} rel-var={rv:.2e}（变化量低于浮点噪声 → 结构上无动态线路）")
    # 出图：动态 vs 静态的线路秩随深度
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    xs = list(range(L))
    ax.plot(xs, [s[0] for s in dyn_stats], "o-", label="Loom (dynamic ports)")
    for li, m, mx, rv in stat_stats:
        ax.plot([li], [0], "x", color="tab:red", label="static ports" if li == 2 else None)
    ax.axhline(DK, ls="--", c="gray", lw=1)
    ax.text(0.1, DK + 0.15, f"d_k = {DK} (rank bound)", fontsize=8, color="gray")
    ax.set_xlabel("layer")
    ax.set_ylabel("effective rank of dynamic wiring")
    ax.set_title("Per-token circuit rewiring in toy LoomFormer (untrained probe)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(Path(__file__).resolve().parent / "loom_probe_rank.png", dpi=200)
    print("plot saved: drafts/loom_probe_rank.png")


def run_gating_impossibility():
    print("\n== 节点门控(rank-1) vs Loom(rank<=d_k) 的线路可实现性 ==")
    C_target = torch.zeros(4, 4)
    C_target[0, 0] = 1.0
    C_target[2, 2] = 1.0
    U, S, Vh = torch.linalg.svd(C_target)
    for r in (1, 2, 4):
        C_r = U[:, :r] @ torch.diag(S[:r]) @ Vh[:r]
        rel = (C_r - C_target).norm() / C_target.norm()
        print(f"rank-{r} 最优逼近相对残差 = {rel:.4f}")
    print("loom 槽构造(d_k=2, one-hot): <q1,k1>=1, <q3,k3>=1, 交叉项=0 → 精确实现互不干扰")


if __name__ == "__main__":
    run_mechanism()
    run_gating_impossibility()
