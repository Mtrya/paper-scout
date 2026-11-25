# Motion Mamba 2.0 机制探针：PCM 写入门控/零初始化读出/文本调制 + Mamba-3 可复位递推
# 依据 arXiv 2611.29897v1 Eqs.(6)-(14)；PCM 用官方 PersistentContentMemory 类，递推为论文 Eq.6 的纯 torch 复推
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

torch.manual_seed(0)

# 官方仓库克隆位置：从本脚本位置向上查找 code/MotionMamba2.0；找不到则给出克隆指引
REPO = None
for parent in Path(__file__).resolve().parents:
    for cand in (parent / "code" / "MotionMamba2.0", parent / "MotionMamba2.0"):
        if cand.is_dir():
            REPO = str(cand)
            break
    if REPO:
        break
if REPO is None:
    raise SystemExit(
        "未找到 MotionMamba2.0 官方仓库。请先 git clone --depth 1 "
        "https://github.com/AIGeeksGroup/MotionMamba2.0 到任意上级 code/ 目录后重跑。"
    )
sys.path.insert(0, REPO)
from motion_mamba2.model.content_memory import PersistentContentMemory  # noqa: E402


def check_pcm():
    print("== A. PersistentContentMemory（官方类，CPU） ==")
    state_dim, d_model, slots = 128, 64, 2
    eta = 0.25
    pcm = PersistentContentMemory(state_dim, d_model, slots,
                                  condition_dim=32, normalize_delta=True,
                                  normalize_slots=True, max_write_rate=eta,
                                  write_gate_context="previous_memory").eval()
    B = 8
    prev = torch.randn(B, slots, d_model)
    summary = torch.randn(B, state_dim)
    delta = torch.randn(B, state_dim) * 5.0
    cond = torch.randn(B, 32)
    with torch.no_grad():
        out = pcm.propose(summary, delta, prev, cond)
        gate = out.write_gate
        print(f"A1 写入门 η 上限: max gate = {gate.max().item():.4f} ≤ η={eta} -> {bool(gate.max() <= eta)}")
        upd = (out.slots - prev).norm(dim=-1)
        bound = eta * (out.proposal - prev).norm(dim=-1) + 1e-6
        print(f"A2 更新有界: ||P_t-P_{{t-1}}|| ≤ η·||proposal-prev|| 逐槽成立 -> {bool((upd <= bound).all())}")
        y0 = torch.randn(B, d_model)
        yP = torch.randn(B, d_model)
        mamba_out = torch.randn(B, d_model)
        rho = pcm.read_gate
        zeta = torch.tensor(-20.0)
        Y = y0 + torch.tanh(rho) * (yP - y0) + torch.sigmoid(zeta) * mamba_out
        print(f"A3 零初始化读出: tanh(ρ_init) = {torch.tanh(rho).item():.2e} → Y 与 Native 输出差 = "
              f"{(Y - (y0 + torch.sigmoid(zeta) * mamba_out)).abs().max().item():.2e}")
        cond2 = torch.randn(B, 32)
        out2 = pcm.propose(summary, delta, prev, cond2)
        dp = (out2.proposal - out.proposal).norm().item()
        dg = (out2.write_gate - out.write_gate).norm().item()
        print(f"A4 同一状态转移、不同文本: proposal 差 = {dp:.3f}, gate 差 = {dg:.3f}（均应显著>0）")
        # 迭代写入：槽内容随会话演化，且每步受 η 约束
        P = torch.zeros(1, slots, d_model)
        max_gate = 0.0
        for t in range(20):
            o = pcm.propose(torch.randn(1, state_dim), torch.randn(1, state_dim), P,
                            torch.randn(1, 32))
            max_gate = max(max_gate, o.write_gate.max().item())
            P = o.slots
        print(f"A5 20 步会话写入: 全程 max gate = {max_gate:.4f} ≤ η -> {max_gate <= eta}; 槽范数 = {P.norm().item():.3f}")


def rot(v, phi):
    # R(phi) 作用于成对坐标：[c -s; s c] 逐对旋转；v: [..., 2m], phi: [..., m]
    a, b = v[..., 0::2], v[..., 1::2]
    c, s = torch.cos(phi), torch.sin(phi)
    return torch.stack((a * c - b * s, a * s + b * c), dim=-1).flatten(-2)


def mamba3_scan(us, Bs, Cs, DTs, As, traps, thetas, D=0.5, state=None):
    # 论文 Eq.6：phi 累积相位、梯形更新、旋转后的 B/C；返回逐 y 与末态
    T, dv = us.shape
    dk = Bs.shape[-1]
    if state is None:
        H = torch.zeros(dv, dk)
        phi = torch.zeros(dk // 2)
        u_prev = torch.zeros(dv)
        B_prev = torch.zeros(dk)
    else:
        H, phi, u_prev, B_prev = state
    ys = []
    for i in range(T):
        phi = phi + torch.pi * DTs[i] * torch.tanh(thetas)
        Bt = rot(Bs[i], phi)
        Ct = rot(Cs[i], phi)
        alpha = torch.exp(DTs[i] * As[i])
        beta = (1 - traps[i]) * DTs[i] * alpha
        gamma = traps[i] * DTs[i]
        H = alpha * H + beta * torch.outer(u_prev, B_prev) + gamma * torch.outer(us[i], Bt)
        ys.append(H @ Ct + D * us[i])
        u_prev, B_prev = us[i], Bt
    return torch.stack(ys), (H, phi, u_prev, B_prev)


def check_resumability():
    print("\n== B. Mamba-3 Eq.6 可复位递推（纯 torch 复推） ==")
    T, dv, dk = 64, 16, 8
    gen = torch.Generator().manual_seed(7)
    def r(*shape): return torch.randn(*shape, generator=gen)
    us, Bs, Cs = r(T, dv), r(T, dk), r(T, dk)
    DTs = torch.rand(T, generator=gen) * 0.09 + 0.001
    As = -torch.rand(T, generator=gen) * 2 - 0.1
    traps = torch.rand(T, generator=gen)
    thetas = r(dk // 2) * 0.1
    y_full, st_full = mamba3_scan(us, Bs, Cs, DTs, As, traps, thetas)
    cut = 32
    y1, st1 = mamba3_scan(us[:cut], Bs[:cut], Cs[:cut], DTs[:cut], As[:cut], traps[:cut], thetas)
    y2, st2 = mamba3_scan(us[cut:], Bs[cut:], Cs[cut:], DTs[cut:], As[cut:], traps[cut:], thetas, state=st1)
    err_y = (torch.cat([y1, y2]) - y_full).abs().max().item()
    err_H = (st2[0] - st_full[0]).abs().max().item()
    err_phi = (st2[1] - st_full[1]).abs().max().item()
    print(f"B1 携带 (H, phi, u_prev, B_prev) 分块续扫 32+32 vs 全程 64：")
    print(f"    max|Δy| = {err_y:.2e}   max|ΔH| = {err_H:.2e}   max|Δphi| = {err_phi:.2e}")
    naive_state = (st1[0], torch.zeros_like(st1[1]), torch.zeros_like(st1[2]), torch.zeros_like(st1[3]))
    y2n, _ = mamba3_scan(us[cut:], Bs[cut:], Cs[cut:], DTs[cut:], As[cut:], traps[cut:], thetas, state=naive_state)
    err_naive = (y2n - y_full[cut:]).abs().max().item()
    print(f"B2 只携带 H 的朴素续扫（phi/u/B 清零）：max|Δy| = {err_naive:.3f}（显著失配 → 这三个额外状态量是精确续扫所必需）")


if __name__ == "__main__":
    check_pcm()
    check_resumability()
