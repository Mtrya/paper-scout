#!/usr/bin/env python3
"""figs.py — 图像空间动作表示精度几何的全部图件(纯本机,CPU)。

图件(均在 figures/):
  fig1_lateral.png        横向精度 vs 深度(1 源像素 / 1 输入像素 @224²)
  fig2_stereo_depth.png   立体深度精度 vs 深度 × 视差误差三档
  fig3_workspace_map.png  核心:(X,Z) 切面 3D 定位误差地图,<1cm/<2mm 边界
  fig4_rotation.png       旋转可观性:平面内(两点)vs 出平面(立体深度差)
  fig5_calib.png          标定误差传播:(a)基线/焦距 (b)光轴旋转 (c)绝对 vs delta
英文标签以规避 CJK 字体问题;数据全部来自 precision.py 的闭式公式。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from PIL import Image, ImageChops

from precision import (SETTINGS, ALPHA, lateral_error, stereo_depth_error,
                       plane_error, rotation_inplane, rotation_outplane,
                       depth_bias_calib, depth_bias_rot, delta_action_rot_error)

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "figure.dpi": 180, "savefig.dpi": 180, "font.family": "DejaVu Sans",
})
DIR = "figures"
import os; os.makedirs(DIR, exist_ok=True)

D2R = np.pi / 180
Z_ = np.linspace(0.15, 1.2, 400)

# ---------- fig1: 横向精度 ----------
def fig1():
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    colors = {"ZED2": "#C0392B", "D435": "#2471A3", "SIM": "#1E8449"}
    for k in ("ZED2", "D435", "SIM"):
        s = SETTINGS[k]
        a = axes[0]; a.plot(Z_, lateral_error(Z_, 1, s["f"]) * 1e3, color=colors[k],
                            label=f"{k} (f={s['f']:.0f} px)")
        b = axes[1]
        b.plot(Z_, lateral_error(Z_, 1, s["f"]) * 1e3 * ALPHA[k], color=colors[k],
               label=f"{k} (α={ALPHA[k]:.2f}, f'={s['f']/ALPHA[k]:.0f} px)")
    for a, title, ylab in (
        (axes[0], "Per 1 source pixel", "lateral err. δX (mm)"),
        (axes[1], "Per 1 VLA input pixel (224\u00b2)", "lateral err. δX (mm)"),
    ):
        a.axhline(10, ls=":", color="k", lw=0.8); a.text(0.16, 10.6, "10 mm (grasp)", fontsize=8)
        a.axhline(2, ls=":", color="k", lw=0.8);  a.text(0.16, 2.6, "2 mm (insert)", fontsize=8)
        a.set_xlabel("depth Z (m)"); a.set_ylabel(ylab); a.set_title(title)
        a.grid(alpha=0.3); a.legend(fontsize=8, loc="upper left")
    fig.suptitle("Lateral precision  δX = Z·δu/f — resolution rescaling is the dominant factor",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); return fig

# ---------- fig2: 立体深度精度 ----------
def fig2():
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
    for ax, k in zip(axes, ("ZED2", "D435", "SIM")):
        s = SETTINGS[k]
        for dd, m in zip((0.5, 1.0, 2.0), ("o", "s", "^")):
            ax.plot(Z_, stereo_depth_error(Z_, dd, s["b"], s["f"]) * 1e3,
                    label=f"δd={dd} px", marker=m, markevery=60, ms=4)
        ax.axhline(10, ls=":", color="k", lw=0.8); ax.axhline(2, ls=":", color="k", lw=0.8)
        ax.set_xlabel("depth Z (m)")
        ax.set_title(f"{k}: b={s['b']*1000:.0f} mm, f={s['f']:.0f} px")
        ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
        ax.set_yscale("log"); ax.set_ylim(0.1, 400)
    axes[0].set_ylabel("depth err. δZ (mm)")
    fig.suptitle("Stereo depth precision  δZ = Z\u00b2·δd/(b·f) — quadratic in Z, inverse in b·f",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); return fig

# ---------- fig3: 工作空间地图(核心) ----------
def total_err_stereo(Z, du_src, dd, b, f):
    return np.hypot(lateral_error(Z, du_src, f), stereo_depth_error(Z, dd, b, f))

def fig3():
    fig, axes = plt.subplots(2, 2, figsize=(10, 8.2))
    X = np.linspace(-0.4, 0.4, 200)
    ZG = np.linspace(0.2, 1.0, 200)
    XX, ZZ = np.meshgrid(X, ZG)

    # 名义预算:δu = 0.5 输入像素(热图 argmax 典型),δd = 1 px(视差典型)
    panels = [
        ("ZED2 stereo", "ZED2", dict(du_in=0.5, dd=1.0)),
        ("D435 stereo", "D435", dict(du_in=0.5, dd=1.0)),
        ("LIBERO-like sim stereo", "SIM", dict(du_in=0.5, dd=1.0)),
    ]
    for ax, (label, k, budget) in zip(axes.flat, panels):
        s = SETTINGS[k]
        du_src = budget["du_in"] * ALPHA[k]
        err = total_err_stereo(ZZ, du_src, budget["dd"], s["b"], s["f"]) * 1e3  # mm
        im = ax.pcolormesh(XX, ZZ, err, cmap="magma", vmin=0, vmax=30,
                           shading="auto")
        for thr, ls, tag in ((2.0, "--", "2 mm insert"), (10.0, "-", "10 mm grasp")):
            e = total_err_stereo(ZG, du_src, budget["dd"], s["b"], s["f"]) * 1e3
            cross = np.where(np.diff(np.sign(e - thr)) != 0)[0]
            zs = sorted([ZG[i] for i in cross])
            if zs:
                zmax = max(zs)
                yline = min(zmax, 1.0)
                ax.axhline(yline, ls=ls, color="white" if thr == 10 else "cyan", lw=1.6)
                tagtxt = tag if zmax <= 1.0 else f"{tag} @Z\u2248{zmax:.2f}m"
                ax.text(0.36, min(yline - 0.025, 0.97), tagtxt, fontsize=8,
                        ha="right", va="bottom",
                        color="white" if thr == 10 else "cyan")
        ax.set_xlim(-0.4, 0.4); ax.set_ylim(0.2, 1.0)
        ax.set_xlabel("lateral X (m)"); ax.set_ylabel("depth Z (m)")
        ax.set_title(f"{label}  · δu=0.5 in-px (α={ALPHA[k]:.2f}), δd={budget['dd']} px")
        fig.colorbar(im, ax=ax, label="3D error (mm)", pad=0.01)
    # 与上面的 zc 计算解耦:重新算每条边界的 Z 位置用于标注(纯显示)
    # 面板 (d):单目+已知平面(前向平行桌面 Z0=0.5m)
    ax = axes.flat[3]
    Z0 = 0.5; du_src = 0.5 * ALPHA["ZED2"]; f = SETTINGS["ZED2"]["f"]
    dhv = np.linspace(0.5, 10.0, 150) * 1e-3
    XXd, DHh = np.meshgrid(X, dhv)
    errd = plane_error(XXd, Z0, du_src, f, DHh) * 1e3
    im = ax.pcolormesh(XXd, DHh * 1e3, errd, cmap="magma", vmin=0, vmax=30, shading="auto")
    cs = ax.contour(XXd, DHh * 1e3, errd, levels=[2.0, 10.0],
                    colors=["cyan", "white"], linestyles=["--", "-"], linewidths=1.4)
    ax.clabel(cs, fmt={2.0: "2 mm", 10.0: "10 mm"}, fontsize=8, colors=["cyan", "white"])
    ax.set_xlim(-0.4, 0.4); ax.set_ylim(0.5, 10)
    ax.set_xlabel("lateral X (m)"); ax.set_ylabel("plane-height err. δh (mm)")
    ax.set_title("Monocular + known plane (Z\u2080=0.5 m, ZED2-like f=448)")
    fig.colorbar(im, ax=ax, label="3D error (mm)", pad=0.01)

    fig.suptitle("3D localization error over (X, Z) — image-space action budget (nominal: "
                 "δu=0.5 input px, δd=1 px)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96]); return fig

# ---------- fig4: 旋转可观性 ----------
def fig4():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
    s_ = np.linspace(5, 120, 300)
    for du, c, ls in ((0.5, "#C0392B", "-"), (1.0, "#2471A3", "--")):
        axes[0].plot(s_, np.rad2deg(rotation_inplane(du, s_)), c=c, ls=ls,
                     label=f"δu = {du} input px")
    axes[0].axhline(1, color="k", lw=0.8, ls=":"); axes[0].text(6, 1.15, "1°", fontsize=8)
    axes[0].axhline(5, color="k", lw=0.8, ls=":"); axes[0].text(6, 5.3, "5°", fontsize=8)
    axes[0].set_yscale("log"); axes[0].set_ylim(0.2, 20)
    axes[0].set_xlabel("image separation s of the two marked points (input px)")
    axes[0].set_ylabel("in-plane orientation err. δθ (deg)")
    axes[0].set_title("Two-point marking: δθ ≈ δu/s")
    axes[0].grid(alpha=0.3, which="both"); axes[0].legend(fontsize=8)

    Z = np.linspace(0.25, 1.0, 300)
    for k, c in (("ZED2", "#C0392B"), ("SIM", "#1E8449")):
        s = SETTINGS[k]
        dZ = stereo_depth_error(Z, 1.0, s["b"], s["f"])
        axes[1].plot(Z, np.rad2deg(rotation_outplane(dZ, 0.08)), c=c,
                     label=f"{k} stereo, L=80 mm")
    axes[1].axhline(1, color="k", lw=0.8, ls=":"); axes[1].text(0.26, 1.15, "1°", fontsize=8)
    axes[1].set_yscale("log"); axes[1].set_ylim(0.3, 40)
    axes[1].set_xlabel("depth Z (m)")
    axes[1].set_ylabel("out-of-plane tilt err. (deg)")
    axes[1].set_title("Out-of-plane from stereo depth: δθ ≈ √2·δZ/L (δd=1 px)")
    axes[1].grid(alpha=0.3, which="both"); axes[1].legend(fontsize=8)
    fig.tight_layout(); return fig

# ---------- fig5: 标定误差传播 ----------
def fig5():
    fig = plt.figure(figsize=(11.5, 3.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.15])
    Z = np.linspace(0.2, 1.2, 300)
    ax = fig.add_subplot(gs[0])
    for frac, c in ((0.01, "#2471A3"), (0.02, "#C0392B"), (0.05, "#7D3C98")):
        ax.plot(Z, depth_bias_calib(Z, frac, frac) * 1e3, c=c,
                label=f"δb/b = δf/f = {frac*100:.0f}%")
    ax.axhline(10, ls=":", color="k", lw=0.8)
    ax.set_xlabel("depth Z (m)"); ax.set_ylabel("depth bias (mm)")
    ax.set_title("(a) baseline & focal bias\nδZ/Z ≈ δb/b + δf/f")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1])
    for b, c in ((0.05, "#2471A3"), (0.12, "#C0392B"), (0.30, "#1E8449")):
        ax.plot(Z, depth_bias_rot(Z, 1 * D2R, b) * 1e3, c=c,
                label=f"b = {b*1000:.0f} mm")
    ax.plot(Z, np.full_like(Z, delta_action_rot_error(0.05, 1*D2R)[1] * 1e3),
            "k--", label="delta action, worst |Δ|=5 cm")
    ax.set_yscale("log"); ax.set_ylim(0.2, 500)
    ax.set_xlabel("depth Z (m)"); ax.set_ylabel("error (mm)")
    ax.set_title("(b) stereo axis rotation δθ=1°\nδZ ≈ δθ·(Z²/b + b)")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[2])
    ths = np.array([0.5, 1.0, 2.0]) * D2R
    b = 0.12; Z0 = 0.5
    abs_err = [depth_bias_rot(Z0, t, b) * 1e3 for t in ths]
    del_err = [delta_action_rot_error(0.05, t)[1] * 1e3 for t in ths]
    w = 0.28; x = np.arange(3)
    ax.bar(x - w/2, abs_err, w, label="image-space absolute (stereo lift, b=120 mm, Z=0.5 m)",
           color="#C0392B")
    ax.bar(x + w/2, del_err, w, label="camera-frame delta (|Δ|=5 cm, needs R only)",
           color="#1E8449")
    for xi, a_, d_ in zip(x, abs_err, del_err):
        ax.text(xi - w/2, a_, f"{a_:.0f}", ha="center", va="bottom", fontsize=8)
        ax.text(xi + w/2, d_, f"{d_:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_yscale("log"); ax.set_ylim(0.2, 200)
    ax.set_xticks(x, ["0.5°", "1°", "2°"])
    ax.set_xlabel("extrinsic rotation error δθ"); ax.set_ylabel("error (mm)")
    ax.set_title(f"(c) same δθ, two action types\nratio {abs_err[1]/del_err[1]:.0f}× at 1°")
    ax.grid(alpha=0.3, which="both", axis="y"); ax.legend(fontsize=7.5)
    fig.tight_layout(); return fig

def crop_white(path):
    """按非白像素 bbox 裁掉白边,另存回原位。"""
    im = Image.open(path).convert("RGB")
    bg = Image.new("RGB", im.size, (255, 255, 255))
    diff = ImageChops.difference(im, bg)
    bbox = diff.getbbox()
    if bbox:
        im = im.crop(bbox)
    im.save(path)
    print(f"  cropped {path} -> {im.size}")

def main():
    for fn in (fig1, fig2, fig3, fig4, fig5):
        fig = fn()
        path = os.path.join(DIR, fn.__name__ + ".png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print("saved", path)
    print("crop pass:")
    for p in sorted(os.listdir(DIR)):
        if p.endswith(".png"):
            crop_white(os.path.join(DIR, p))

if __name__ == "__main__":
    main()
