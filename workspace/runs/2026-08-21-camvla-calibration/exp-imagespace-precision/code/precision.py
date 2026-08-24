#!/usr/bin/env python3
"""图像空间动作表示的精度几何:闭式公式 + Monte Carlo 验证。

全部为解析推导 + 数值验证,无真实数据。公式/假设/结论见线程 README.md。
本文件输出:
  - 数字表(打印 + numbers.json)
  - 闭式公式与 MC 的偏差报告(mc_report.json)

约定(针孔模型):u = fx*X/Z + cx, v = fy*Y/Z + cy;视差 d = u_L - u_R,
立体深度 Z = b*f/d(b 基线、f 焦距,单位像素)。方形像素取 fx=fy=f。
"""
import json
import numpy as np

# ---------------- 设置 ----------------
# 每项: (名称, 源宽 W, 焦距 f [px], 基线 b [m])
SETTINGS = {
    "ZED2":  dict(W=1280, f=448.0, b=0.120),
    "D435":  dict(W=640,  f=462.0, b=0.050),
    "SIM":   dict(W=640,  f=450.0, b=0.300),   # LIBERO 式 sim 双静态相机(自选基线)
}
VIN = 224          # VLA 输入分辨率
ALPHA = {k: v["W"] / VIN for k, v in SETTINGS.items()}   # 1 输入像素 = α 源像素

# ---------------- 闭式公式 ----------------
def lateral_error(Z, du, f):
    """横向误差 δX = Z·δu/f。δu 与 f 同单位(源像素或输入像素)。"""
    return Z * du / f

def stereo_depth_error(Z, dd, b, f):
    """立体深度误差 δZ = Z²·δd/(b·f)。δd 为视差误差。"""
    return Z**2 * dd / (b * f)

def plane_error(X, Z0, du, f, dh):
    """单目+已知平面(前向平行桌面,深度 Z0,高度误差 dh)。

    X = (u-cx)·Z0/f,故 δX = Z0·δu/f + (X/Z0)·δh;δZ = δh。
    """
    dx = Z0 * du / f + (X / Z0) * dh
    return np.hypot(dx, dh)

def rotation_inplane(du, s):
    """两点标注的平面内朝向误差 δθ ≈ δu/s(rad)。δu、s 同单位,比值与分辨率无关。"""
    return du / s

def rotation_outplane(dZ, L):
    """出平面朝向误差:两端点深度误差(独立,各 δZ)经长度 L 合成 ≈ √2·δZ/L。"""
    return np.sqrt(2.0) * dZ / L

def depth_bias_calib(Z, fb, ff):
    """标定偏差(系统误差,非噪声):δZ/Z ≈ fb + ff(fb=δb/b, ff=δf/f,视差噪声项另计)。"""
    return Z * (fb + ff)

def depth_bias_rot(Z, dtheta_rad, b):
    """双目光轴间相对旋转误差 δθ 造成的深度偏差。

    右相机绕自身 y 轴偏转 δθ 时,视差 d = f(b − Z·tanδθ)/(Z + b·tanδθ),
    一阶展开 δZ ≈ δθ·(Z² + b²)/b = δθ·(Z²/b + b)。经典结果 δZ = Z²δθ/b
    是 Z ≫ b 的远场极限;桌面立体 Z/b ~ 1.7–4,近场修正项 δθ·b 不可忽略。
    机制:偏转同时改变右视点的水平偏移(视差偏置)与点在右相机系内的距离。
    """
    return dtheta_rad * (Z**2 + b**2) / b

def delta_action_rot_error(Delta_mag, dtheta_rad):
    """相机系 delta 对外参旋转误差的敏感度。

    单次执行误差取决于 Δ 与旋转轴的夹角:最坏 |Δ|·θ(Δ ⊥ 轴),随机方向均值
    (π/2)·sin(θ/2)·|Δ| ≈ 0.785·|Δ|·θ。返回随机方向的均值(与 MC 对照),
    最坏情形由 worst 返回。
    """
    mean = (np.pi / 2) * np.sin(dtheta_rad / 2) * Delta_mag
    worst = 2 * np.sin(dtheta_rad / 2) * Delta_mag
    return mean, worst

# ---------------- 针孔投影 / 三角化 ----------------
def K_of(f, cx, cy):
    return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])

def project(P_cam, K):
    p = K @ P_cam
    return p[:2] / p[2]

def P_matrix(R, t, K):
    """投影矩阵 P = K·[R|t],P·[X;1]。"""
    return K @ np.hstack([R, t.reshape(3, 1)])

def triangulate(xL, xR, P_L, P_R):
    """DLT 三角化(最小二乘),返回相机系 3D 点。"""
    A = np.zeros((4, 4))
    for i, (x, P) in enumerate([(xL, P_L), (xR, P_R)]):
        A[2*i]   = x[0]*P[2] - P[0]
        A[2*i+1] = x[1]*P[2] - P[1]
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return X[:3] / X[3]

def ry(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1.0, 0], [-s, 0, c]])

def rz(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])

def stereo_rig(b, f, cx=0.0, cy=0.0, R_R=None):
    """左相机在原点,右相机在 (b,0,0)。返回 (P_L, P_R, K)。"""
    K = K_of(f, cx, cy)
    R_L, t_L = np.eye(3), np.zeros(3)
    if R_R is None:
        R_R = np.eye(3)
    t_R = -R_R @ np.array([b, 0, 0])
    return P_matrix(R_L, t_L, K), P_matrix(R_R, t_R, K), K

# ---------------- Monte Carlo 验证 ----------------
rng = np.random.default_rng(20260821)

def mc_lateral(Z0, X0, du, f, N=200_000):
    """σ(恢复 X) 应 ≈ Z0·δu/f。Z 视为已知(立体/平面给出),仅横向。"""
    K = K_of(f, cx=0.0, cy=0.0)
    P0 = np.array([X0, 0.0, Z0])
    u0 = project(P0, K)
    Xs = []
    for _ in range(N):
        u = u0 + rng.normal(0, du, 2)
        Xs.append((u[0] - 0.0) * Z0 / f)
    return np.std(Xs)

def project_world(P, R, t, K):
    """世界系点 P(左相机系)经 [R|t] 投影:相机系坐标 = R·P + t。"""
    return project(R @ P + t, K)

def mc_stereo(b, f, Z0, X0, sigma_px, N=200_000):
    """独立噪声 σ_px/视图 → 视差噪声 δd = √2·σ_px,δZ = Z²·√2σ_px/(bf)。"""
    P_L, P_R, K = stereo_rig(b, f)
    cR = np.array([b, 0.0, 0.0])
    R_R, t_R = np.eye(3), -cR
    P0 = np.array([X0, 0.0, Z0])
    uL0 = project_world(P0, np.eye(3), np.zeros(3), K)
    uR0 = project_world(P0, R_R, t_R, K)
    Zs, Xs = [], []
    for _ in range(N):
        uL = uL0 + rng.normal(0, sigma_px, 2)
        uR = uR0 + rng.normal(0, sigma_px, 2)
        P = triangulate(uL, uR, P_L, P_R)
        Zs.append(P[2]); Xs.append(P[0])
    return np.std(Zs), np.std(Xs)

def mc_calib_bias(b, f, Z0, X0, frac_b, frac_f, N=100_000):
    """用错误基线/焦距三角化 → 深度均值偏差应 ≈ Z0·(frac_b + frac_f)。"""
    P_L, P_R, K = stereo_rig(b, f)
    b_u, f_u = b*(1+frac_b), f*(1+frac_f)
    P_L_u, P_R_u, K_u = stereo_rig(b_u, f_u)
    cR = np.array([b, 0.0, 0.0])
    P0 = np.array([X0, 0.0, Z0])
    uL0 = project_world(P0, np.eye(3), np.zeros(3), K)
    uR0 = project_world(P0, np.eye(3), -cR, K)
    Zs = []
    for _ in range(N):
        uL = uL0 + rng.normal(0, 0.1, 2)     # 很小的噪声,测的是偏差不是噪声
        uR = uR0 + rng.normal(0, 0.1, 2)
        P = triangulate(uL, uR, P_L_u, P_R_u)
        Zs.append(P[2])
    return np.mean(Zs) - Z0

def mc_rot_bias(b, f, Z0, X0, dtheta, N=100_000):
    """右相机绕 y 轴旋转误差 δθ(真实姿态带偏),用标称姿态三角化。
    深度偏差应 ≈ −Z²·δθ/b(第一阶,轴线上精确)。"""
    P_L, P_R_nom, K = stereo_rig(b, f)                       # 标称(用于三角化)
    R_R_t, t_R_t = ry(dtheta), -ry(dtheta) @ np.array([b, 0, 0])   # 真实姿态
    P0 = np.array([X0, 0.0, Z0])
    uL0 = project_world(P0, np.eye(3), np.zeros(3), K)
    uR0 = project_world(P0, R_R_t, t_R_t, K)                 # 真实图像(真对应)
    Zs = []
    for _ in range(N):
        uL = uL0 + rng.normal(0, 0.05, 2)
        uR = uR0 + rng.normal(0, 0.05, 2)
        P = triangulate(uL, uR, P_L, P_R_nom)                # 标称姿态
        Zs.append(P[2])
    return np.mean(Zs) - Z0

def mc_delta_R(Delta, dtheta, N=200_000):
    """相机系 delta:执行端用带旋转误差的外参 R_err 把 delta 旋到机器人系。
    误差 ≈ |Δ|·δθ(小角弦长)。"""
    R = ry(dtheta)
    errs = []
    for _ in range(N):
        d = rng.normal(size=3); d = Delta * d / np.linalg.norm(d)
        errs.append(np.linalg.norm(R @ d - d))
    return np.mean(errs)

# ---------------- 数字表 ----------------
def report():
    out = {}
    D = np.deg2rad(1.0) / 2
    print("="*78)
    print("1) 横向精度 δX = Z·δu/f")
    print("="*78)
    print(f"   1 输入像素 = α 源像素: α(ZED2)={ALPHA['ZED2']:.2f}  α(D435)={ALPHA['D435']:.2f}  α(SIM)={ALPHA['SIM']:.2f}")
    tbl = {}
    for k, s in SETTINGS.items():
        fin = s["f"] / ALPHA[k]
        row = {"f_src": s["f"], "f_input224": round(fin, 1)}
        for Z in (0.5, 1.0):
            row[f"per_srcpx_Z{Z}"] = round(lateral_error(Z, 1, s["f"]) * 1e3, 2)
            row[f"per_inpx_Z{Z}"] = round(lateral_error(Z, 1, s["f"]) * 1e3 * ALPHA[k], 2)
        tbl[k] = row
    print("  相机  f_src  f_224  δX(1源px)@0.5m   δX(1源px)@1m   δX(1输入px)@0.5m  δX(1输入px)@1m  [mm]")
    for k, r in tbl.items():
        print(f"  {k:5s} {r['f_src']:5.0f} {r['f_input224']:6.1f}  {r['per_srcpx_Z0.5']:10.2f}  {r['per_srcpx_Z1.0']:10.2f}  {r['per_inpx_Z0.5']:15.2f}  {r['per_inpx_Z1.0']:13.2f}")
    out["lateral"] = tbl
    out["alpha"] = ALPHA

    print(); print("="*78)
    print("2) 立体深度精度 δZ = Z²·δd/(b·f)")
    print("="*78)
    tbl = {}
    for k, s in SETTINGS.items():
        row = {}
        for dd in (0.5, 1.0, 2.0):
            row[f"dd{dd}"] = {f"Z{int(Z*10)}": round(stereo_depth_error(Z, dd, s["b"], s["f"])*1e3, 2)
                              for Z in (0.3, 0.5, 1.0)}
        tbl[k] = row
        print(f"  {k:5s} b={s['b']*1000:.0f}mm f={s['f']:.0f}   δd=0.5px: " +
              "  ".join(f"Z={Z}m:{row['dd0.5'][f'Z{int(Z*10)}']}mm" for Z in (0.3, 0.5, 1.0)))
        print(f"        δd=1px : " + "  ".join(f"Z={Z}m:{row['dd1.0'][f'Z{int(Z*10)}']}mm" for Z in (0.3, 0.5, 1.0)))
        print(f"        δd=2px : " + "  ".join(f"Z={Z}m:{row['dd2.0'][f'Z{int(Z*10)}']}mm" for Z in (0.3, 0.5, 1.0)))
    out["stereo"] = tbl

    print(); print("="*78)
    print("3) 单目+已知平面(前向平行桌面 Z0,高度误差 dh)对照")
    print("="*78)
    # 用 ZED2 口径的输入像素预算:δu_in=0.5,α=5.71 → δu_src=2.86px
    k = "ZED2"; du_src = 0.5 * ALPHA[k]
    for Z0 in (0.5, 1.0):
        for dh in (1e-3, 3e-3, 5e-3):
            e0 = plane_error(0.0, Z0, du_src, SETTINGS[k]["f"], dh)
            e1 = plane_error(0.3, Z0, du_src, SETTINGS[k]["f"], dh)
            print(f"  Z0={Z0}m dh={dh*1000:.0f}mm 中心:{e0*1e3:.2f}mm  X=0.3m 偏轴:{e1*1e3:.2f}mm")
    out["plane"] = {"du_src_half_inpx": du_src}

    print(); print("="*78)
    print("4) 旋转可观性:平面内 δθ ≈ δu/s;出平面 √2·δZ/L;单目出平面不可观")
    print("="*78)
    tbl = {}
    for du, tag in ((0.5, "δu=0.5输入px"), (1.0, "δu=1输入px")):
        row = {f"s{s}": round(np.rad2deg(rotation_inplane(du, s)), 2) for s in (10, 30, 80)}
        tbl[tag] = row
        print(f"  {tag}: s=10px: {row['s10']}°   s=30px: {row['s30']}°   s=80px: {row['s80']}°")
    print("  出平面(ZED2,δd=1px):")
    for Z in (0.5, 1.0):
        dZ = stereo_depth_error(Z, 1, SETTINGS["ZED2"]["b"], SETTINGS["ZED2"]["f"])
        for L in (0.08, 0.04, 0.02):
            print(f"    Z={Z}m δZ={dZ*1e3:.1f}mm  L={int(L*1000)}mm: {np.rad2deg(rotation_outplane(dZ, L)):.2f}°")
    out["rotation"] = tbl

    print(); print("="*78)
    print("5) 标定误差传播")
    print("="*78)
    print("  (a) δZ/Z = δb/b + δf/f")
    for frac in (0.01, 0.02, 0.05):
        print(f"    δb/b=δf/f={frac*100:.0f}% → δZ/Z={frac*200:.0f}% ... 单看 δb/b={frac*100:.0f}%: Z=0.5m: {0.5*frac*1e3:.0f}mm, Z=1m: {1.0*frac*1e3:.0f}mm")
    print("  (b) 双目光轴旋转误差 δZ ≈ δθ·(Z²/b + b),θ 为绕光轴垂直轴的偏转")
    for b in (0.05, 0.12, 0.30):
        row = []
        for th in (0.5, 1.0, 2.0):
            row.append(f"δθ={th}°: {depth_bias_rot(0.5, np.deg2rad(th), b)*1e3:.0f}mm@0.5m, {depth_bias_rot(1.0, np.deg2rad(th), b)*1e3:.0f}mm@1m")
        print(f"    b={b*1000:.0f}mm: " + " | ".join(row))
    print("  (c) 相机系 delta(只需 R)对照")
    for th in (0.5, 1.0, 2.0):
        e_mean, e_worst = delta_action_rot_error(0.05, np.deg2rad(th))
        ratio = depth_bias_rot(0.5, np.deg2rad(th), 0.12) / e_worst
        print(f"    δθ={th}°: |Δ|=5cm → 最坏 {e_worst*1e3:.2f}mm(平均 {e_mean*1e3:.2f}mm);"
              f"同误差下立体绝对定位(0.5m,b=0.12m)是它的 {ratio:.0f}×")
    out["calib"] = {"bias": {}, "rot": {}}

    print(); print("="*78)
    print("6) Monte Carlo 验证(闭式 vs 数值)")
    print("="*78)
    mc = {}

    # 1) 横向
    Z0, X0, du, f = 0.5, 0.1, 2.86, SETTINGS["ZED2"]["f"]
    e = lateral_error(Z0, du, f)
    m = mc_lateral(Z0, X0, du, f)
    dev = abs(m - e) / e
    print(f"  1)横向: 公式 {e*1e3:.3f}mm, MC σ {m*1e3:.3f}mm, 相对偏差 {dev*100:.2f}%")
    mc["lateral"] = {"formula": e, "mc": m, "dev": dev}

    # 2) 立体(两视图独立 σ=1px → δd=√2px)
    b, f = SETTINGS["SIM"]["b"], SETTINGS["SIM"]["f"]
    sigma, Z0 = 1.0, 0.5
    dd_eff = np.sqrt(2)*sigma
    e = stereo_depth_error(Z0, dd_eff, b, f)
    mz, mx = mc_stereo(b, f, Z0, X0, sigma)
    devz = abs(mz - e) / e
    print(f"  2)立体: 公式(δd=√2·σ) {e*1e3:.3f}mm, MC σZ {mz*1e3:.3f}mm, 偏差 {devz*100:.2f}%")
    print(f"          横向 σX MC: {mx*1e3:.3f}mm vs 公式 Z0·σ/f: {lateral_error(Z0, sigma, f)*1e3:.3f}mm")
    mc["stereo"] = {"formula": e, "mc": mz, "dev": devz}

    # 5a) 基线/焦距偏差
    for frac_b, frac_f, tag in ((0.02, 0.0, "δb/b=2%"), (0.0, 0.02, "δf/f=2%"), (0.02, 0.02, "both 2%")):
        e = depth_bias_calib(Z0, frac_b, frac_f)
        m = mc_calib_bias(b, f, Z0, X0, frac_b, frac_f)
        dev = abs(m - e) / e
        print(f"  5a){tag}: 公式 {e*1e3:.3f}mm, MC {m*1e3:.3f}mm, 偏差 {dev*100:.2f}%")
        mc[f"calib_bias_{tag}"] = {"formula": e, "mc": m, "dev": dev}

    # 5b) 旋转偏差(轴线上的点,公式在第一阶上精确)
    dth = np.deg2rad(1.0)
    e = depth_bias_rot(Z0, dth, b)
    m = mc_rot_bias(b, f, Z0, 0.0, dth)
    dev = abs(abs(m) - e) / e
    print(f"  5b)旋转 1°(轴上点): 公式 {e*1e3:.3f}mm, MC {abs(m)*1e3:.3f}mm(符号由偏转方向决定), 偏差 {dev*100:.2f}%")
    mc["calib_rot"] = {"formula": e, "mc": m, "dev": dev}

    # 5c) delta 动作(随机方向平均)
    dth = np.deg2rad(1.0)
    e_mean, e_worst = delta_action_rot_error(0.05, dth)
    m = mc_delta_R(0.05, dth)
    dev = abs(m - e_mean) / e_mean
    print(f"  5c)delta |Δ|=5cm 旋错 1°: 公式(方向平均) {e_mean*1e3:.3f}mm, MC {m*1e3:.3f}mm, 偏差 {dev*100:.2f}%")
    print(f"      最坏方向(Δ⊥旋转轴) {e_worst*1e3:.3f}mm")
    mc["delta_R"] = {"formula_mean": e_mean, "formula_worst": e_worst, "mc": m, "dev": dev}

    with open("numbers.json", "w") as fp:
        json.dump({"tables": out, "mc": mc}, fp, ensure_ascii=False, indent=1)
    print("\n[ok] numbers.json 已写")

if __name__ == "__main__":
    report()
