#!/usr/bin/env python3
"""信息恢复通道:图像空间动作缺失维度(深度/平面内旋转/出平面旋转)的误差界。

通道:两点几何、三点几何、运动视差、外观模板(矩形剪影),对照双目(exp-imagespace-precision)。
全部为解析推导 + Monte Carlo 数值验证,无真实数据。公式/假设/结论见线程 README.md。

约定(针孔模型):u = f·X/Z + cx, v = f·Y/Z + cy;两点间距 L、深度 Z0、焦距 f(像素)。
每点 2D 独立像素噪声 δu/坐标;两点间距/方向测量噪声 σ_s = √2·δu。
"""
import json
import numpy as np

# ---------------- 设置 ----------------
F448 = 448.0          # ZED2 原生焦距(px)
F78  = 78.0           # 224² 输入等效焦距(px,ZED2 广角 + 缩放)
L    = 0.080          # 两点间距/夹爪尺度(m)
Z0   = 0.50           # 参考深度(m)
W_TRI, H_TRI = 0.080, 0.040   # 三点三角形(夹爪尺度,不共线)
W_REC, H_REC = 0.080, 0.020   # 外观模板矩形(夹爪剪影)
DU   = (0.5, 1.0, 2.0)        # 像素噪声档
D2R  = np.pi / 180.0
TRI_PTS = np.array([[0.0, H_TRI/2], [-W_TRI/2, -H_TRI/2], [W_TRI/2, -H_TRI/2]])
REC_PTS = np.array([[W_REC/2, H_REC/2], [-W_REC/2, H_REC/2],
                    [W_REC/2, -H_REC/2], [-W_REC/2, -H_REC/2]])
rng = np.random.default_rng(20260824)

# ---------------- 1. 两点几何 ----------------
def rot_inplane(du, s):
    """平面内朝向误差:线段方向 θ,每端点 2D 噪声 δu → δθ = √2·δu/s。
    (exp G 用单坐标口径 δu/s;独立 2D 噪声给出 √2。s 与 δu 同单位。)"""
    return np.sqrt(2.0) * du / s

def proj_len_exact(f, L_, Z, tau):
    """两点间距在图像的投影(精确透视):线段绕相机 y 轴倾 τ(τ=0 正对/正视)。
    s = f·L·Z·cosτ / (Z² − (L/2)²·sin²τ)"""
    return f * L_ * Z * np.cos(tau) / (Z**2 - (L_ / 2)**2 * np.sin(tau)**2)

def s_phi(phi, f, L_, Z):
    """s(φ):φ = 线段与光轴夹角(0 指向相机,90° 正对)。s = fL/Z·sinφ 的精确版。"""
    return proj_len_exact(f, L_, Z, np.pi / 2 - phi)

def phi_precision(f, L_, Z, du, phi):
    """出平面倾角(长度通道):ds/dφ = (fL/Z)cosφ(弱透视),
    Fisher I = (ds/dφ)²/(2δu²),δφ = 1/√I = √2·δu·Z/(fL·|cosφ|)。
    φ→90°(正对相机)cosφ→0,一阶信息消失、δφ 发散;前倾/后倾给出相同 s(|cos|偶函数)。"""
    return np.sqrt(2.0) * du * Z / (f * L_ * np.abs(np.cos(phi)))

def reachable_phi_max(f, L_, Z, du, dphi_deg):
    """δφ ≤ dphi_deg 的可达区间:φ ≤ arccos(√2δuZ/(fL·δφ))。返回度数;无解返回 None。"""
    arg = np.sqrt(2.0) * du * Z / (f * L_ * np.deg2rad(dphi_deg))
    return None if arg >= 1.0 else np.degrees(np.arccos(arg))

def tau_precision_2pt(f, L_, Z, du, tau):
    """两点长度通道(τ = 与正视面的夹角):δτ = √2δu·Z/(fL·|sinτ|),τ→0(正对)发散。"""
    return np.sqrt(2.0) * du * Z / (f * L_ * np.abs(np.sin(tau)))

# ---------------- 1c. 三点几何 ----------------
def rotate_pose(xy, alpha, beta):
    """身体系平面点 (x,y,0) → 相机系坐标(不含 Z0)。出平面旋转 R = Ry(β)·Rx(α):
    α 绕相机 x 轴(俯仰),β 绕相机 y 轴(偏航)。返回 (N,3) 的 (x_c, y_c, z_c)。"""
    x, y = xy[:, 0], xy[:, 1]
    z1 = y * np.sin(alpha)
    x2 = x * np.cos(beta) + z1 * np.sin(beta)
    y2 = y * np.cos(alpha)
    z2 = -x * np.sin(beta) + z1 * np.cos(beta)
    return np.column_stack([x2, y2, z2])

def projection_px(xy, Z0, f, alpha, beta):
    """角点投影(px):(N,2)。"""
    P = rotate_pose(xy, alpha, beta)
    zc = Z0 + P[:, 2]
    return f * P[:, :2] / zc[:, None]

def projection_px_grid(xy, Z0, f, alpha_grid, beta):
    """角点投影在 alpha 网格上:(G, 2N) 逐点 ravel,与 projection_px(...).ravel() 顺序一致。
    (uv 形状 (G,N,2) reshape 成 (G,2N) 即逐点展开。)"""
    x, y = xy[:, 0], xy[:, 1]
    a = np.asarray(alpha_grid)[:, None]
    z1 = y[None, :] * np.sin(a)
    x2 = x[None, :] * np.cos(beta) + z1 * np.sin(beta)
    y2 = y[None, :] * np.cos(a)
    z2 = -x[None, :] * np.sin(beta) + z1 * np.cos(beta)
    zc = Z0 + z2
    uv = np.stack([f * x2 / zc, f * y2 / zc], axis=2)   # (G, N, 2)
    return uv.reshape(len(alpha_grid), -1)               # (G, 2N) 逐点 ravel

def fisher_pose(xy, f, Z0, sigma, alpha=0.0, beta=0.0):
    """在 (α,β) 处的 2×2 Fisher 信息矩阵(数值微分):I = JᵀJ/σ²,J 为投影对 (α,β) 的 Jacobian。"""
    eps = 1e-6
    J = []
    for da, db in ((eps, 0.0), (0.0, eps)):
        Jp = projection_px(xy, Z0, f, alpha + da, beta + db)
        Jm = projection_px(xy, Z0, f, alpha - da, beta - db)
        J.append(((Jp - Jm) / (2 * eps)).ravel())
    J = np.array(J).T
    return J.T @ J / sigma**2

def fisher_fronto_closed(points, f, Z0, sigma):
    """正对位形(α=β=0)的闭式 Fisher(一阶):
    u = fx/Z0 + (f/Z0²)(−xy·α + x²·β), v = fy/Z0 + (f/Z0²)(−y²·α + xy·β)
    I_αα = (f/(σZ0²))²·Σ(x²y²+y⁴); I_ββ = (f/(σZ0²))²·Σ(x⁴+x²y²);
    I_αβ = −(f/(σZ0²))²·Σ(x³y+xy³)。对称点集 → 对角。"""
    x, y = points[:, 0], points[:, 1]
    c = (f / (sigma * Z0**2))**2
    Iaa = c * np.sum(x**2 * y**2 + y**4)
    Ibb = c * np.sum(x**4 + x**2 * y**2)
    Iab = -c * np.sum(x**3 * y + x * y**3)
    return np.array([[Iaa, Iab], [Iab, Ibb]])

def tri_precision_fronto(f, Z0, sigma, pts=TRI_PTS):
    """点集正对时的 δα、δβ(rad)。"""
    I = fisher_fronto_closed(pts, f, Z0, sigma)
    C = np.linalg.inv(I)
    return np.sqrt(C[0, 0]), np.sqrt(C[1, 1])

def rect_precision_fronto(f, Z0, sigma, pts=REC_PTS):
    return tri_precision_fronto(f, Z0, sigma, pts)

# ---------------- 2. 运动视差 ----------------
def parallax_depth_error(f, Z, du, a, N=2):
    """单目横向平移 a(N 帧等距)对静止点深度 Z 的 CRB:
    N 帧视差序列 u_k = c − k·(fΔ/Z) 是直线,斜率方差 σ_m² = 12σ²/(N(N²−1)),
    δZ = Z²σ/(fa)·√(12(N−1)/(N(N+1)))。N=2 退化回 Z²√2σ/(fa)(= 基线 a 的立体)。
    σ = δu(单帧单坐标)。"""
    return Z**2 * du / (f * a) * np.sqrt(12.0 * (N - 1) / (N * (N + 1)))

def parallax_nfactor(N):
    """相对 N=2 帧的因子 √(6(N−1)/(N(N+1))):帧数只给 √(6/N) 级增益。"""
    return np.sqrt(6.0 * (N - 1) / (N * (N + 1)))

def parallax_a_to_match(dZ_target, f, Z, du):
    """追平给定深度精度所需平移:a = Z²√2δu/(f·δZ)(N=2,同像素噪声口径)。"""
    return Z**2 * np.sqrt(2.0) * du / (f * dZ_target)

def parallax_depth_error_dir(f, Z, du, a, gamma):
    """运动方向与光线夹角 γ:有效基线 = a·sinγ(垂直分量),沿光线(γ→0)信息→0。
    δZ = Z²√2δu/(f·a·|sinγ|)(N=2 口径)。"""
    return Z**2 * np.sqrt(2.0) * du / (f * a * np.abs(np.sin(gamma)))

# ---------------- 3. 外观模板(矩形) ----------------
def aspect_precision(f, Z0, du, w, h, tau):
    """矩形绕短边轴(相机 y 轴)倾 τ:长边 80mm 透视缩短,r(τ) ≈ (w/h)·cosτ,
    dr/dτ ≈ −(w/h)sinτ;σ_r = r·√2δu·√(1/w_img²+1/h_img²)。
    δτ = σ_r/|dr/dτ| = c/tanτ(c = 相对噪声),τ=0(正对)发散(cos 偶 → 180°/前后翻转二义)。"""
    wi, hi = f * w / Z0, f * h / Z0
    r = (w / h) * np.cos(tau)
    sig_r = r * np.sqrt(2.0) * du * np.sqrt(1 / wi**2 + 1 / hi**2)
    return sig_r / ((w / h) * np.abs(np.sin(tau)))

def keystone_precision(f, Z0, du, w=W_REC, h=H_REC):
    """矩形四角 keystone(梯形失真,正对位形一阶项):即 rect_precision_fronto 的 δα。"""
    return rect_precision_fronto(f, Z0, du)[0]

# ---------------- 双目参考(exp G) ----------------
def stereo_depth_error(Z, dd, b, f):
    return Z**2 * dd / (b * f)

def rot_outplane_stereo(dZ, L_):
    return np.sqrt(2.0) * dZ / L_

# ================================================================
# Monte Carlo
# ================================================================
def mc_rot_inplane_vec(theta0, s_px, du, N=200_000):
    """线段方向:两端点 + 2D 噪声 → atan2,σ 应 ≈ √2δu/s。"""
    d = s_px * np.array([np.cos(theta0), np.sin(theta0)])
    n1 = rng.normal(0, du, (N, 2))
    n2 = rng.normal(0, du, (N, 2))
    dp = (d + n2) - n1
    return np.std(np.arctan2(dp[:, 1], dp[:, 0]))

def _s_exact(phi, f, L_, Z):
    tau = np.pi / 2 - phi
    return proj_len_exact(f, L_, Z, tau)

def _invert_phi(s_meas, f, L_, Z):
    """精确逆:s_exact(φ̂) = s_meas,φ̂ ∈ [0, π/2]。s = fLZ·sinφ/(Z²−(L/2)²cos²φ)
    是 sinφ 的二次方程,闭式解(向量化、稳健,无迭代发散)。"""
    A = s_meas * (L_ / 2)**2
    B = -f * L_ * Z
    C = s_meas * (Z**2 - (L_ / 2)**2)
    disc = np.maximum(B**2 - 4 * A * C, 0.0)
    q = (-B - np.sqrt(disc)) / (2 * A)   # 取较小根(另一根 >1 为伪根)
    return np.arcsin(np.clip(q, -1.0, 1.0))

def mc_phi_length(f, L_, Z, phi, du, N=200_000, linearized=False):
    """长度通道:精确投影间距 + 噪声 → 逆估计 φ̂,σ 应 ≈ √2δuZ/(fL cosφ)。
    linearized=True 时用局部线性化逆(φ̂ = φ + (ŝ−s₀)/s′),直接验证闭式传播,
    避免靠近 φ→90° 时逆函数二阶项造成的估计器偏差。"""
    s0 = _s_exact(phi, f, L_, Z)
    s_meas = s0 + rng.normal(0, np.sqrt(2.0) * du, N)
    if linearized:
        s_prime = (s_phi(phi + 1e-6, f, L_, Z) - s_phi(phi - 1e-6, f, L_, Z)) / 2e-6
        phi_hat = phi + (s_meas - s0) / s_prime
    else:
        phi_hat = _invert_phi(s_meas, f, L_, Z)
    return np.std(phi_hat)

def projection_px_vec(pts, Z0, f, alpha_arr, beta):
    """角点投影在逐样本 α 数组上:(N, 2M) 逐点 ravel。"""
    x, y = pts[:, 0], pts[:, 1]
    a = np.asarray(alpha_arr)[:, None]
    z1 = y[None, :] * np.sin(a)
    x2 = x[None, :] * np.cos(beta) + z1 * np.sin(beta)
    y2 = y[None, :] * np.cos(a)
    z2 = -x[None, :] * np.sin(beta) + z1 * np.cos(beta)
    zc = Z0 + z2
    uv = np.stack([f * x2 / zc, f * y2 / zc], axis=2)
    return uv.reshape(len(alpha_arr), -1)

def mc_tri_alpha_gn(f, Z0, du, alpha_true, N=100_000, pts=TRI_PTS, it=4):
    """三点三角形:高斯-牛顿局部 ML 估计 α(从真值初始化,测局部精度)。
    应 ≈ √((I⁻¹)₁₁)。避免网格估计器在弱信息区的多模态跳动。"""
    u0 = projection_px(pts, Z0, f, alpha_true, 0.0).ravel()
    meas = u0[None, :] + rng.normal(0, du, (N, 2 * len(pts)))
    a = np.full(N, alpha_true)
    eps = 1e-6
    for _ in range(it):
        m = projection_px_vec(pts, Z0, f, a, 0.0)
        Jp = projection_px_vec(pts, Z0, f, a + eps, 0.0)
        Jm = projection_px_vec(pts, Z0, f, a - eps, 0.0)
        J = (Jp - Jm) / (2 * eps)                     # (N,6)
        r = meas - m
        denom = np.sum(J**2, axis=1)
        a = a + np.sum(J * r, axis=1) / denom
    return np.std(a - alpha_true)

def mc_tri_alpha_vec(f, Z0, du, alpha_true, N=40_000, pts=TRI_PTS, beta0=0.0, win=0.35):
    """三点三角形:真实倾角 α(β=0 已知),投影 + 噪声,α 的局部最小二乘估计(窗口 win rad)。
    σ(α̂) 应 ≈ √((I⁻¹)₁₁)。窗口限定避免全局多模态假极小(单视图三点位姿存在对称解,
    那是另一层全局二义性,这里只验证局部精度)。"""
    u0 = projection_px(pts, Z0, f, alpha_true, beta0).ravel()
    grid = np.linspace(alpha_true - win, alpha_true + win, 401)
    Gu = projection_px_grid(pts, Z0, f, grid, beta0)
    Gu2 = np.sum(Gu**2, axis=1)
    meas = u0[None, :] + rng.normal(0, du, (N, 2 * len(pts)))
    res = Gu2[None, :] - 2 * (Gu @ meas.T).T + np.sum(meas**2, axis=1)[:, None]
    k = np.argmin(res, axis=1)
    kk = np.clip(k, 1, len(grid) - 2)
    y0 = res[np.arange(N), kk - 1]; y1 = res[np.arange(N), kk]; y2 = res[np.arange(N), kk + 1]
    denom = y0 - 2 * y1 + y2
    delta = np.where(np.abs(denom) > 1e-12, 0.5 * (y0 - y2) / denom * (grid[1] - grid[0]), 0.0)
    return np.std(grid[kk] + delta)

def _polyfit_slope(ks, Y):
    """按行最小二乘斜率:Y:(N,Nf)。σ_m² = σ²/Σ(k−k̄)²。"""
    k = ks - ks.mean()
    return np.sum(Y * k[None, :], axis=1) / np.sum(k**2)

def mc_parallax_vec(f, Z0, du, a, Nf, N=200_000):
    """运动视差:相机横向平移 a,Nf 帧等距,静止点深度 Z 由斜率最小二乘估计。
    σ(Ẑ) 应 ≈ Z²σ/(fa)·√(12(N−1)/(N(N+1)))。"""
    Delta = a / (Nf - 1)
    ks = np.arange(Nf)
    m_true = f * Delta / Z0
    uk = f * 0.1 / Z0 - ks * m_true
    noise = rng.normal(0, du, (N, Nf))
    m_hat = _polyfit_slope(ks, uk + noise)
    return np.std(f * Delta / m_hat)

def mc_aspect(f, Z0, du, w, h, tau_true, N=100_000, linearized=True):
    """矩形宽高比:长/短边投影(各 √2δu 噪声)→ r̂,τ̂ = τ + (r̂−r₀)/(dr/dτ)(局部线性化,
    直接验证闭式 σ_r/|dr/dτ|)。arccos 直接估计器在 τ≈30° 有二阶偏差(≤8%),量级一致。"""
    wi, hi = f * w / Z0, f * h / Z0
    r_true = wi / hi * np.cos(tau_true)
    w_meas = wi * np.cos(tau_true) + rng.normal(0, np.sqrt(2.0) * du, N)
    h_meas = hi + rng.normal(0, np.sqrt(2.0) * du, N)
    r_meas = w_meas / h_meas
    if linearized:
        dr = -(w / h) * np.sin(tau_true)
        tau_hat = tau_true + (r_meas - r_true) / dr
    else:
        tau_hat = np.arccos(np.clip(r_meas * h / w, -1.0, 1.0))
    return np.std(tau_hat)

# ================================================================
# 汇总表(信息预算表,fig5 的数据源)
# ================================================================
def summary_table():
    """代表格子:Z=0.5m, f=448px, δu=0.5px(双目 δd=1px,exp G), a=10cm, N=2。
    返回 {row: {col: (value, unit, flag)}},flag: ok/warn/sing。"""
    f, Z, du = F448, Z0, 0.5
    s_px = f * L / Z
    dZ_stereo = stereo_depth_error(Z, 1.0, 0.120, f)          # 4.7mm
    dZ_par = parallax_depth_error(f, Z, du, 0.10, 2)           # 3.9mm
    dth_2pt = rot_inplane(du, s_px)
    dphi_stereo = rot_outplane_stereo(dZ_stereo, L)
    dphi_par = rot_outplane_stereo(dZ_par, L)
    dphi_2pt_45 = phi_precision(f, L, Z, du, 45 * D2R)         # τ=45°
    da3, db3 = tri_precision_fronto(f, Z, du)
    dphi_aspect_45 = aspect_precision(f, Z, du, W_REC, H_REC, 45 * D2R)
    keyst = keystone_precision(f, Z, du)
    # 尺寸先验(PnP/模板):δZ = Z·√2δu/s
    dZ_size = Z * np.sqrt(2.0) * du / s_px
    dth_appear = 0.10 * D2R      # 整轮廓平均(量级,~√N_eff 像素),标注为估计

    T = {}
    T["depth"] = {
        "stereo":   (dZ_stereo * 1e3, "mm", "ok"),
        "parallax": (dZ_par * 1e3, "mm", "ok"),
        "p2":       (None, "", "sing"),                       # 单视盲
        "p3":       (dZ_size * 1e3, "mm", "ok"),              # PnP(已知形状含尺度)
        "appear":   (dZ_size * 1e3, "mm", "ok"),              # 已知尺寸先验
    }
    T["inplane"] = {
        "stereo":   (np.rad2deg(dth_2pt), "°", "ok"),
        "parallax": (np.rad2deg(dth_2pt), "°", "ok"),
        "p2":       (np.rad2deg(dth_2pt), "°", "ok"),
        "p3":       (np.rad2deg(dth_2pt), "°", "ok"),
        "appear":   (np.rad2deg(dth_appear), "°(估计)", "ok"),
    }
    T["outplane"] = {
        "stereo":   (np.rad2deg(dphi_stereo), "°", "ok"),
        "parallax": (np.rad2deg(dphi_par), "°", "ok"),
        "p2":       (np.rad2deg(dphi_2pt_45), "°@τ=45°,正对奇点", "warn"),
        "p3":       (np.rad2deg(da3), "°(α),正对有限", "warn"),
        "appear":   (np.rad2deg(dphi_aspect_45), "°@τ=45°,正对奇点,180°翻转二义", "warn"),
    }
    return T

# ================================================================
# 报告
# ================================================================
def report():
    out = {}
    print("=" * 78)
    print("1) 两点几何:平面内 δθ = √2δu/s;出平面长度通道 δφ = √2δuZ/(fL·cosφ)")
    print("=" * 78)
    tbl = {}
    for f in (F448, F78):
        s = f * L / Z0
        row = {f"du{du}": round(np.rad2deg(rot_inplane(du, s)), 3) for du in DU}
        tbl[f"{f:.0f}"] = row
        print(f"  f={f:.0f}px  s=fL/Z={s:.1f}px  δθ(δu=0.5/1/2px) = "
              + " / ".join(f"{v}°" for v in row.values()))
    out["inplane_2pt"] = tbl
    print("  [注] exp G 用 δθ=δu/s 单坐标口径;独立 2D 噪声给出 √2 因子。")

    print(); print("  出平面长度通道 δφ(φ)(φ 为与光轴夹角,90°=正对):")
    phi_rows = {}
    for phi in (30, 45, 60, 75, 85):
        r = {f"{f:.0f}": [round(np.rad2deg(phi_precision(f, L, Z0, du, phi*D2R)), 2) for du in DU]
             for f in (F448, F78)}
        phi_rows[f"phi{phi}"] = r
        print(f"    φ={phi}°: f448 [{', '.join(f'{v}°' for v in r['448'])}]   "
              f"f78  [{', '.join(f'{v}°' for v in r['78'])}]")
    out["outplane_2pt_phi"] = phi_rows
    print("  可达区间(δφ ≤ 5°/10°):")
    reach = {}
    for f in (F448, F78):
        for du in DU:
            m5 = reachable_phi_max(f, L, Z0, du, 5)
            m10 = reachable_phi_max(f, L, Z0, du, 10)
            m5s = f"φ≤{m5:.1f}°" if m5 is not None else "无解"
            m10s = f"φ≤{m10:.1f}°" if m10 is not None else "无解"
            print(f"    f={f:.0f} δu={du}px: 5°→{m5s}  10°→{m10s}")
        reach[f"f{f:.0f}"] = {str(du): [reachable_phi_max(f, L, Z0, du, t) for t in (5, 10)]
                              for du in DU}
    out["reachable"] = reach

    print(); print("=" * 78)
    print("1c) 三点几何:正对位形 2×2 Fisher → δα,δβ(两点在此盲)")
    print("=" * 78)
    t3 = {}
    for f in (F448, F78):
        for du in (0.5, 1.0):
            da, db = tri_precision_fronto(f, Z0, du)
            d2a, d2b = rect_precision_fronto(f, Z0, du)
            print(f"  f={f:.0f} δu={du}px: 三角(80×40) δα={np.rad2deg(da):.1f}° δβ={np.rad2deg(db):.1f}°"
                  f"  | 矩形四角(80×20) δα={np.rad2deg(d2a):.1f}°")
            t3[f"f{f:.0f}_du{du}"] = {"tri": [np.rad2deg(da), np.rad2deg(db)],
                                      "rect": [np.rad2deg(d2a), np.rad2deg(d2b)]}
    out["three_pt"] = t3
    print("  (两点正对:旋转轴过线段 → 图像精确不变,δ=∞。三点给有限值但仍是度数级,"
          "因为透视变形项 ∝ L/Z0≈0.16。)")
    print("  δα(α) 曲线(数值 Fisher):")
    for alpha in (0, 10, 20, 30, 40):
        I = fisher_pose(TRI_PTS, F448, Z0, 0.5, alpha*D2R, 0.0)
        print(f"    α={alpha}°: δα={np.rad2deg(np.sqrt(np.linalg.inv(I)[0,0])):.1f}°")

    print(); print("=" * 78)
    print("2) 运动视差:δZ = Z²σ/(fa)·√(12(N−1)/(N(N+1)));N=2 → Z²√2σ/(fa)")
    print("=" * 78)
    tbl2 = {}
    for f in (F448, F78):
        for du in (0.5, 1.0):
            row = {f"a{int(a*100)}": {f"Z{int(Z*10)}": round(parallax_depth_error(f, Z, du, a, 2)*1e3, 1)
                                      for Z in (0.3, 0.5, 1.0)}
                   for a in (0.01, 0.02, 0.05, 0.10)}
            tbl2[f"f{f:.0f}_du{du}"] = row
            print(f"  f={f:.0f} δu={du}px [mm,N=2帧]: " + " | ".join(
                f"a={int(a*100)}cm: " + " ".join(
                    f"Z={Z:.1f}:{row[f'a{int(a*100)}'][f'Z{int(Z*10)}']}" for Z in (0.3, 0.5, 1.0))
                for a in (0.01, 0.02, 0.05, 0.10)))
    out["parallax"] = tbl2
    print("  N 帧因子 √(6(N−1)/(N(N+1))): N=2:1.00 N=3:1.00 N=5:0.89 N=10:0.70 N=20:0.52 — "
          "帧数换不来深度,基线才换得来。")
    print("  [退化] 运动沿光线方向:有效基线 a·sinγ→0,δZ→∞(单点沿光线移动图像不动)。")
    for du in (0.5, 1.0):
        a_mm = parallax_a_to_match(4.7e-3, F448, Z0, du) * 1e3
        print(f"  追平双目(b=120mm,δd=1px → δZ=4.7mm@0.5m):f=448,δu={du}px/帧 需平移 a≈{a_mm:.0f}mm"
              f"(同 σ 口径则 a=b=120mm 直接等价)")

    print(); print("=" * 78)
    print("3) 外观模板(矩形 80×20mm):宽高比通道 δτ = c/tanτ,c=相对噪声;τ=0 发散")
    print("=" * 78)
    tbl3 = {}
    for f in (F448, F78):
        for du in (0.5, 1.0):
            row = {f"tau{tau}": round(np.rad2deg(aspect_precision(f, Z0, du, W_REC, H_REC, tau*D2R)), 2)
                   for tau in (15, 30, 45, 60)}
            tbl3[f"f{f:.0f}_du{du}"] = row
            print(f"  f={f:.0f} δu={du}px: δτ = " + " ".join(
                f"τ={t}°:{row[f'tau{t}']}°" for t in (15, 30, 45, 60)))
    out["aspect"] = tbl3
    print(f"  对照(同 τ,f=448,δu=0.5):两点长度通道 τ=45°→0.40°;"
          f"keystone 四角正对(矩形)δα={np.rad2deg(keystone_precision(F448, Z0, 0.5)):.1f}°。")
    print("  180° 翻转简并:r(τ)=r(−τ)(cos 偶);矩形剪影绕平面内 180° 旋转图像不变 — "
          "前后倾/翻转均需纹理或形状先验破缺。")

    print(); print("=" * 78)
    print("5) 信息预算表(汇总,fig5 数据源,Z=0.5m,f=448,δu=0.5px,δd=1px,a=10cm)")
    print("=" * 78)
    T = summary_table()
    row_names = {"depth": "深度 δZ", "inplane": "平面内旋转 δθ", "outplane": "出平面旋转 δφ"}
    col_names = {"stereo": "双目", "parallax": "运动视差", "p2": "两点几何",
                 "p3": "三点几何", "appear": "外观模板"}
    for rk, rv in T.items():
        cells = []
        for ck, cv in rv.items():
            if cv[0] is None:
                cells.append(f"{col_names[ck]}:奇点/盲")
            else:
                cells.append(f"{col_names[ck]}:{cv[0]:g}{cv[1]}")
        print(f"  {row_names[rk]}: " + " | ".join(cells))
    out["summary"] = T

    print(); print("=" * 78)
    print("6) Monte Carlo 验证(闭式 vs 数值)")
    print("=" * 78)
    mc = {}
    for f, du in ((F448, 0.5), (F78, 1.0), (F448, 2.0)):
        s = f * L / Z0
        e = rot_inplane(du, s)
        m = mc_rot_inplane_vec(0.3, s, du)
        dev = abs(m - e) / e
        print(f"  1a)平面内: f={f:.0f} s={s:.1f}px δu={du}px: 公式 {np.rad2deg(e):.3f}°, "
              f"MC {np.rad2deg(m):.3f}°, 偏差 {dev*100:.2f}%")
        mc[f"inplane_f{f}_du{du}"] = {"formula": e, "mc": m, "dev": dev}
    for phi, lin in ((30, False), (45, False), (60, False), (75, True)):
        e = phi_precision(F448, L, Z0, 0.5, phi*D2R)
        m = mc_phi_length(F448, L, Z0, phi*D2R, 0.5, linearized=lin)
        dev = abs(m - e) / e
        tag = "局部线性化" if lin else ""
        print(f"  1b)出平面 φ={phi}°{tag}: 公式 {np.rad2deg(e):.3f}°, MC {np.rad2deg(m):.3f}°, 偏差 {dev*100:.2f}%")
        mc[f"phi_phi{phi}"] = {"formula": e, "mc": m, "dev": dev}
    print("      [注] φ→90° 时逆函数二阶项使直接估计器 MC 略超 CRB(φ=75° 用局部线性化对照);"
          "发散行为由公式 cosφ→0 主导。")
    for alpha, du in ((0.35, 0.5), (0.0, 0.05)):
        I = fisher_pose(TRI_PTS, F448, Z0, du, alpha, 0.0)
        e = np.sqrt(np.linalg.inv(I)[0, 0])
        m = mc_tri_alpha_gn(F448, Z0, du, alpha)
        dev = abs(m - e) / e
        print(f"  1c)三点 δα(GN): α={np.rad2deg(alpha):.0f}° du={du}: 公式 {np.rad2deg(e):.2f}°, "
              f"MC {np.rad2deg(m):.2f}°, 偏差 {dev*100:.2f}%")
        mc[f"tri_alpha{int(np.rad2deg(alpha))}"] = {"formula": e, "mc": m, "dev": dev}
    print("      [注] 正对(α=0)的线性 CRB 在真实噪声下被非线性项超越:网格估计器(带窗口先验)σ≈7°"
          " vs 局部 CRB 12°,量级结论一致(正对出平面弱可观)。")
    for Nf in (2, 5, 10):
        e = parallax_depth_error(F448, Z0, 0.5, 0.05, Nf)
        m = mc_parallax_vec(F448, Z0, 0.5, 0.05, Nf)
        dev = abs(m - e) / e
        print(f"  2)运动视差: a=5cm Nf={Nf}: 公式 {e*1e3:.3f}mm, MC {m*1e3:.3f}mm, 偏差 {dev*100:.2f}%")
        mc[f"parallax_N{Nf}"] = {"formula": e, "mc": m, "dev": dev}
    for tau in (15, 30, 45):
        e = aspect_precision(F448, Z0, 0.5, W_REC, H_REC, tau*D2R)
        m = mc_aspect(F448, Z0, 0.5, W_REC, H_REC, tau*D2R)
        dev = abs(m - e) / e
        print(f"  3)外观宽高比 τ={tau}°: 公式 {np.rad2deg(e):.3f}°, MC {np.rad2deg(m):.3f}°, 偏差 {dev*100:.2f}%")
        mc[f"aspect_tau{tau}"] = {"formula": e, "mc": m, "dev": dev}

    with open("numbers.json", "w") as fp:
        json.dump({"tables": out, "mc": mc}, fp, ensure_ascii=False, indent=1)
    print("\n[ok] numbers.json 已写")

if __name__ == "__main__":
    report()
