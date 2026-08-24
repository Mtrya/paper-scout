#!/usr/bin/env python3
"""figs.py — 信息恢复通道误差界的全部图件(纯本机,CPU)。

图件(均在 figures/):
  fig1_two_point.png     两点几何:平面内 δθ(s) + 出平面 δφ(φ) 曲线(发散于正对)
  fig2_three_point.png   三点几何:δα(α)/δβ(β) vs 两点(正对盲/斜视角好),2×2 Fisher
  fig3_motion_parallax.png 运动视差:δZ vs 平移 a × 深度;N 帧因子;运动方向退化
  fig4_appearance.png    外观模板:宽高比通道 δτ(τ) vs 两点长度通道 + 180° 翻转简并示意
  fig5_budget_table.png  信息预算表(核心):缺失维度 × 通道,格子=CRB 数字或奇点标注
MC 标注点读自 numbers.json(recovery.py 写出)。英文标签规避 CJK 字体问题。
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageChops

import recovery as R

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "figure.dpi": 180, "savefig.dpi": 180, "font.family": "DejaVu Sans",
})
DIR = "figures"
import os; os.makedirs(DIR, exist_ok=True)

D2R = np.pi / 180
MC = json.load(open("numbers.json"))["mc"]

# ---------- fig1: 两点几何 ----------
def fig1():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    # (a) 平面内
    s_ = np.linspace(5, 120, 300)
    for du, c in ((0.5, "#2471A3"), (1.0, "#C0392B"), (2.0, "#7D3C98")):
        axes[0].plot(s_, np.rad2deg(R.rot_inplane(du, s_)), c=c,
                     label=f"δu={du} px (√2·δu/s)")
    for (s0, du, key), (m0, m1) in (
            ((71.7, 0.5, "inplane_f448.0_du0.5"), (0.565, 0.565)),
            ((12.5, 1.0, "inplane_f78.0_du1.0"), (6.493, 6.543)),
            ((71.7, 2.0, "inplane_f448.0_du2.0"), (2.261, 2.266))):
        axes[0].plot(s0, np.rad2deg(MC[key]["mc"]), "o", ms=6, mfc="none",
                     mec="#E67E22", mew=1.6)
    axes[0].set_yscale("log"); axes[0].set_ylim(0.15, 30)
    axes[0].set_xlabel("image separation s of 2 marked points (px)")
    axes[0].set_ylabel("in-plane orientation err. δθ (deg)")
    axes[0].set_title("(a) Two-point in-plane: δθ = √2·δu/s")
    axes[0].grid(alpha=0.3, which="both"); axes[0].legend(fontsize=8)
    axes[0].annotate("f=78: s=12.5px", xy=(12.5, 6.5), xytext=(16, 11),
                     fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.8))
    axes[0].annotate("f=448: s=71.7px", xy=(71.7, 0.57), xytext=(52, 2.2),
                     fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.8))

    # (b) 出平面长度通道
    phi = np.linspace(1, 89.9, 400)
    for f, c, ls in ((R.F448, "#C0392B", "-"), (R.F78, "#2471A3", "--")):
        for du, lw in ((0.5, 2.0), (2.0, 1.2)):
            axes[1].plot(phi, np.rad2deg(R.phi_precision(f, R.L, R.Z0, du, phi*D2R)),
                         c=c, ls=ls, lw=lw,
                         label=f"f={f:.0f}, δu={du}px")
    for ph, key in ((30, "phi_phi30"), (45, "phi_phi45"), (60, "phi_phi60")):
        axes[1].plot(ph, np.rad2deg(MC[key]["mc"]), "o", ms=5, mfc="none",
                     mec="#E67E22", mew=1.4)
    axes[1].axvline(90, color="k", lw=0.8, ls=":")
    axes[1].text(86.5, 0.22, "fronto-\nparallel", fontsize=7.5, ha="right", va="bottom")
    # 可达区间(δφ≤5°)
    for f, du in ((R.F448, 0.5), (R.F78, 0.5)):
        m = R.reachable_phi_max(f, R.L, R.Z0, du, 5)
        if m is not None:
            axes[1].axvspan(m, 90, alpha=0.10, color="#C0392B" if f == 448 else "#2471A3")
    axes[1].set_yscale("log"); axes[1].set_ylim(0.2, 300)
    axes[1].set_xlabel("tilt angle φ to optical axis (deg; 90° = facing camera)")
    axes[1].set_ylabel("out-of-plane tilt err. δφ (deg)")
    axes[1].set_title("(b) 2-pt length channel: δφ = √2δu·Z/(fL·|cosφ|)")
    axes[1].grid(alpha=0.3, which="both"); axes[1].legend(fontsize=8)
    axes[1].annotate("1st-order info → 0\nas φ→90° (diverge)\n"
                     "±φ sign ambiguous\n(|cosφ| even)", xy=(82, 8), xytext=(52, 25),
                     fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.8))
    axes[1].annotate("δφ≤5° zone: φ≤83.5° (f=448)\nφ≤49.5° (f=78)", xy=(84, 1.6),
                     xytext=(34, 7), fontsize=8,
                     arrowprops=dict(arrowstyle="->", lw=0.8))
    fig.suptitle("Two-point rotation observability (L=80 mm, Z=0.5 m)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94]); return fig

# ---------- fig2: 三点几何 ----------
def fig2():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    ang = np.linspace(0, 50, 200)
    # (a) 三点 δα(α) / δβ(β)
    for f, c, ls in ((R.F448, "#C0392B", "-"), (R.F78, "#2471A3", "--")):
        da = [np.rad2deg(np.sqrt(np.linalg.inv(R.fisher_pose(R.TRI_PTS, f, R.Z0, 0.5, a*D2R, 0))[0, 0]))
              for a in ang]
        db = [np.rad2deg(np.sqrt(np.linalg.inv(R.fisher_pose(R.TRI_PTS, f, R.Z0, 0.5, 0, a*D2R))[1, 1]))
              for a in ang]
        axes[0].plot(ang, da, c=c, ls=ls, lw=2.0, label=f"δα, f={f:.0f}")
        axes[0].plot(ang, db, c=c, ls=ls, lw=1.1, label=f"δβ, f={f:.0f}")
    axes[0].plot(20, np.rad2deg(MC["tri_alpha20"]["mc"]), "o", ms=6, mfc="none",
                 mec="#E67E22", mew=1.6, label="MC (GN, f=448)")
    axes[0].axhline(5, color="k", lw=0.8, ls=":"); axes[0].text(1, 5.6, "5°", fontsize=8)
    axes[0].annotate("2-pt: blind at fronto\n(rotation axis through segment)", xy=(0.5, 60),
                     xytext=(8, 40), fontsize=8,
                     arrowprops=dict(arrowstyle="->", lw=0.8))
    axes[0].set_yscale("log"); axes[0].set_ylim(1, 300)
    axes[0].set_xlabel("out-of-plane tilt α or β (deg, 0 = fronto-parallel)")
    axes[0].set_ylabel("tilt err. (deg)")
    axes[0].set_title("(a) 3-point 2×2 Fisher: finite at fronto, δ≈6-12°@f=448")
    axes[0].grid(alpha=0.3, which="both"); axes[0].legend(fontsize=7.5)

    # (b) 三点 δβ vs 两点长度通道(第一个出平面 DOF)
    tau = np.linspace(1, 60, 300)
    db3 = [np.rad2deg(np.sqrt(np.linalg.inv(R.fisher_pose(R.TRI_PTS, R.F448, R.Z0, 0.5, 0, t*D2R))[1, 1]))
           for t in tau]
    d2 = np.rad2deg(R.tau_precision_2pt(R.F448, R.L, R.Z0, 0.5, tau*D2R))
    axes[1].plot(tau, d2, "--", color="#1E8449", lw=2.0,
                 label="2-pt length channel δτ (√2δuZ/(fL sinτ))")
    axes[1].plot(tau, db3, "-", color="#C0392B", lw=2.0,
                 label="3-pt δβ (finite at τ=0)")
    axes[1].axhline(5, color="k", lw=0.8, ls=":"); axes[1].text(1, 5.6, "5°", fontsize=8)
    axes[1].set_yscale("log"); axes[1].set_ylim(0.3, 100)
    axes[1].set_xlabel("tilt from fronto-parallel τ (deg)")
    axes[1].set_ylabel("err. (deg)")
    axes[1].set_title("(b) who rescues out-of-plane? 2-pt off-axis, 3-pt at fronto")
    axes[1].grid(alpha=0.3, which="both"); axes[1].legend(fontsize=8)
    axes[1].annotate("3-pt better\nnear fronto", xy=(6, 7), xytext=(14, 18), fontsize=8,
                     arrowprops=dict(arrowstyle="->", lw=0.8))
    axes[1].annotate("2-pt better off-axis\n(0.4° @ 45°)", xy=(45, 0.42), xytext=(30, 2.5),
                     fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.8))
    fig.suptitle("Three-point geometry vs two-point: the second out-of-plane DOF (L=80 mm, Z=0.5 m, δu=0.5 px)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94]); return fig

# ---------- fig3: 运动视差 ----------
def fig3():
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0))
    # (a) δZ vs a
    a_ = np.linspace(0.005, 0.12, 300)
    for Z, c in ((0.3, "#2471A3"), (0.5, "#C0392B"), (1.0, "#1E8449")):
        axes[0].plot(a_*100, R.parallax_depth_error(R.F448, Z, 0.5, a_, 2)*1e3, c=c,
                     label=f"Z={Z} m")
    axes[0].plot(a_*100, R.parallax_depth_error(R.F78, 0.5, 0.5, a_, 2)*1e3, "--",
                 color="#C0392B", label="Z=0.5 m, f=78")
    axes[0].axhline(4.7, color="k", lw=0.8, ls=":")
    axes[0].text(0.6, 5.6, "ZED2 stereo b=120mm, δd=1px", fontsize=7.5)
    axes[0].axvline(8.4, color="k", lw=0.8, ls=":")
    axes[0].text(8.8, 90, "a≈84mm ties stereo\n(δu=0.5px/frame)", fontsize=7.5, rotation=90,
                 va="top")
    axes[0].set_yscale("log"); axes[0].set_ylim(1, 400)
    axes[0].set_xlabel("lateral translation a (cm)")
    axes[0].set_ylabel("depth err. δZ (mm)")
    axes[0].set_title("(a) parallax depth: δZ = Z²√2δu/(fa), 2 frames")
    axes[0].grid(alpha=0.3, which="both"); axes[0].legend(fontsize=8)
    # (b) N 帧因子
    Ns = np.arange(2, 51)
    axes[1].plot(Ns, R.parallax_nfactor(Ns), "-o", ms=3, color="#7D3C98")
    axes[1].plot(Ns, np.sqrt(6.0/Ns), ":", color="#888")
    axes[1].text(30, 0.42, "√(6/N)", fontsize=8, color="#666")
    axes[1].set_ylim(0.15, 1.05)
    axes[1].set_xlabel("number of frames N (total baseline a fixed)")
    axes[1].set_ylabel("δZ(N)/δZ(2)")
    axes[1].set_title("(b) frames barely help: √(6(N−1)/(N(N+1)))")
    axes[1].grid(alpha=0.3, which="both")
    # (c) 运动方向退化
    gam = np.linspace(1, 90, 200)
    axes[2].plot(gam, R.parallax_depth_error_dir(R.F448, 0.5, 0.5, 0.05, gam*D2R)*1e3,
                 color="#C0392B")
    axes[2].axvline(90, color="k", lw=0.8, ls=":"); axes[2].text(84, 260, "⊥ ray", fontsize=8)
    axes[2].set_yscale("log"); axes[2].set_ylim(5, 1000)
    axes[2].set_xlabel("motion direction γ from ray (deg)")
    axes[2].set_ylabel("depth err. δZ (mm)")
    axes[2].set_title("(c) along-ray motion: info → 0 (δZ ∝ 1/sinγ)")
    axes[2].grid(alpha=0.3, which="both")
    fig.suptitle("Motion parallax depth recovery (δu=0.5 px/frame unless noted)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94]); return fig

# ---------- fig4: 外观模板 ----------
def fig4():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    tau = np.linspace(2, 60, 300)
    for f, c, ls in ((R.F448, "#C0392B", "-"), (R.F78, "#2471A3", "--")):
        axes[0].plot(tau, np.rad2deg(R.aspect_precision(f, R.Z0, 0.5, R.W_REC, R.H_REC, tau*D2R)),
                     c=c, ls=ls, lw=2.0, label=f"aspect-ratio δτ, f={f:.0f}")
    axes[0].plot(tau, np.rad2deg(R.tau_precision_2pt(R.F448, R.L, R.Z0, 0.5, tau*D2R)),
                 "-", color="#1E8449", lw=1.6,
                 label="2-pt length channel (f=448)")
    for t_, key in ((15, "aspect_tau15"), (30, "aspect_tau30"), (45, "aspect_tau45")):
        axes[0].plot(t_, np.rad2deg(MC[key]["mc"]), "o", ms=5, mfc="none",
                     mec="#E67E22", mew=1.4)
    axes[0].axhline(5, color="k", lw=0.8, ls=":"); axes[0].text(2, 5.8, "5°", fontsize=8)
    axes[0].set_yscale("log"); axes[0].set_ylim(0.3, 100)
    axes[0].set_xlabel("tilt from fronto-parallel τ (deg)")
    axes[0].set_ylabel("tilt err. (deg)")
    axes[0].set_title("(a) 80×20 mm silhouette: δτ = c/tanτ — worse than 2-pt length\n"
                      "(short side h=20mm → 18px dominates ratio noise)")
    axes[0].grid(alpha=0.3, which="both"); axes[0].legend(fontsize=8)
    axes[0].annotate("τ=0: aspect flat\n(singular)", xy=(2, 60), xytext=(10, 34),
                     fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.8))

    # (b) 180° 翻转简并示意(精确投影的剪影)
    ax = axes[1]
    def sil(tau_d, c):
        pts = R.REC_PTS
        P = R.projection_px(pts, R.Z0, R.F448, 0.0, tau_d*D2R)   # (4,2)
        u, v = P[:, 0], P[:, 1]
        # 归一化到画面
        u = (u - u.min()) / (u.max() - u.min())
        v = (v - v.min()) / (v.max() - v.min())
        poly = list(zip(u, v))
        ax.fill([p[0] for p in poly], [1 - p[1] for p in poly], c, alpha=0.25, ec=c, lw=1.5)
    sil(30, "#C0392B"); sil(-30, "#2471A3"); sil(0, "#888888")
    ax.text(0.17, 0.5, "τ=+30°", color="#C0392B", fontsize=9, ha="center")
    ax.text(0.83, 0.5, "τ=−30°", color="#2471A3", fontsize=9, ha="center")
    ax.text(0.5, 0.92, "same aspect ratio r(±τ);\n180°-flip of +τ equals −τ (ambiguous)\n"
                        "keystone differs in sign — needs a known reference side;\n"
                        "texture/template breaks it", fontsize=8, ha="center")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("(b) 180°-flip degeneracy of a bare silhouette\n(rectangle 80×20, exact perspective)")
    fig.suptitle("Appearance template channel: geometry vs two-point, and what the template actually buys",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94]); return fig

# ---------- fig5: 信息预算表 ----------
def fig5():
    T = R.summary_table()
    rows = ["depth", "inplane", "outplane"]
    cols = ["stereo", "parallax", "p2", "p3", "appear"]
    row_titles = {"depth": "Depth δZ\n(Z=0.5m)", "inplane": "In-plane rot. δθ\n(Z=0.5m)",
                  "outplane": "Out-of-plane rot. δφ\n(Z=0.5m)"}
    col_titles = {"stereo": "Stereo\n(b=120mm,\nδd=1px)", "parallax": "Motion parallax\n(a=10cm,\nδu=0.5px)",
                  "p2": "2-pt geometry", "p3": "3-pt geometry", "appear": "Appearance\n(template)"}
    verdict = {
        "depth": "Buy a baseline: stereo b=120mm or ~8-10cm lateral motion → 4-5mm;\n"
                 "else known-size PnP / template scale prior → ~5mm",
        "inplane": "2-point marking is enough (0.57° @ f=448); zero extra cost.\n"
                   "f=78 → 3.2°: rescale/resolution dominates",
        "outplane": "Hardest: stereo / motion → 4-5° (needs δZ budget); 2-pt only off-axis\n"
                    "(0.8° @ τ=45°, sign-ambiguous); 3-pt finite but 12° @ fronto;\n"
                    "sub-degree needs appearance + shape/texture prior",
    }
    def cell_color(rk, val, unit):
        if val is None:
            return "#F5B7B1"
        v = val
        if rk == "depth":
            return "#D5F5E3" if v < 5 else ("#FDEBD0" if v < 30 else "#F5B7B1")
        return "#D5F5E3" if v < 1 else ("#FDEBD0" if v < 10 else "#F5B7B1")

    nrow, ncol = 3, 5
    fig, ax = plt.subplots(figsize=(13.5, 4.3))
    ax.axis("off")
    w = np.array([1.15, 1.15, 1.0, 1.05, 1.05, 0.9, 3.0])
    x0, y0, hh, h = 0.005, 0.965, 0.15, 0.215
    colx = np.concatenate([[0.0], np.cumsum(w)])
    colx = colx / colx[-1]
    # 表头(独立带,位于数据行上方)
    for j, c in enumerate(cols):
        ax.text((colx[j] + colx[j+1])/2, y0 - hh/2, col_titles[c], ha="center", va="center",
                fontsize=8.5, fontweight="bold", color="#1A5276")
    ax.text((colx[5]+colx[6])/2, y0 - hh/2, "Buy decision & cost", ha="center", va="center",
            fontsize=8.5, fontweight="bold", color="#1A5276")
    for i, rk in enumerate(rows):
        y = y0 - hh - h * (i + 1)
        ax.add_patch(plt.Rectangle((x0, y), colx[1]-x0, h, fc="#EAF2F8", ec="#7F8C8D", lw=0.6))
        ax.text((x0+colx[1])/2, y + h/2, row_titles[rk], ha="center", va="center",
                fontsize=8.5, fontweight="bold")
        for j, c in enumerate(cols):
            xl, xr = colx[j+1], colx[j+2]
            val, unit, flag = T[rk][c]
            fc = cell_color(rk, val, unit)
            ax.add_patch(plt.Rectangle((xl, y), xr-xl, h, fc=fc, ec="#7F8C8D", lw=0.6))
            if val is None:
                txt = "blind\n(single view)"
            elif rk == "outplane" and c in ("p2", "appear"):
                txt = f"{val:.1f}°@τ=45°\nsingular@fronto" + ("\n180°-flip" if c == "appear" else "")
            elif rk == "outplane" and c == "p3":
                txt = f"{val:.1f}°(α)\nfronto finite"
            elif c == "appear" and rk == "inplane":
                txt = f"~{val:.1f}°\n(est.)"
            else:
                unit = "" if (rk == "inplane" and c in ("stereo", "parallax")) else unit
                txt = f"{val:.2g}{unit}"
            ax.text((xl+xr)/2, y + h/2, txt, ha="center", va="center", fontsize=8.0)
        xl, xr = colx[6], colx[7]
        ax.add_patch(plt.Rectangle((xl, y), xr-xl, h, fc="#FBFCFC", ec="#7F8C8D", lw=0.6))
        ax.text((xl+xr)/2, y + h/2, verdict[rk], ha="center", va="center", fontsize=7.6)
    ax.text(0.5, 0.055, "color: green <5 mm / <1°  ·  amber 5-30 mm / 1-10°  ·  red: singular / >30 mm / >10°\n"
            "representative cells at Z=0.5 m, f=448 px, δu=0.5 px; stereo δd=1 px (exp G), "
            "parallax a=10 cm, 2 frames; 3-pt / appearance = known-shape / known-size prior",
            ha="center", va="center", fontsize=7.8, color="#444")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("Information budget: which channel buys each missing dimension "
                 "(δu=0.5px, δd=1px, f=448; green <5mm/<1°, amber 5-30mm/1-10°, red singular/>30mm/>10°)",
                 fontsize=11)
    return fig

def crop_white(path):
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
